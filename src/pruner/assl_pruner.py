import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from decimal import Decimal
import os, copy, time, pickle, numpy as np, math
from .meta_pruner import MetaPruner
import utility
import matplotlib.pyplot as plt
from tqdm import tqdm
from fnmatch import fnmatch, fnmatchcase
pjoin = os.path.join

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, passer):
        super(Pruner, self).__init__(model, args, logger, passer)
        loader = passer.loader
        ckp = passer.ckp
        loss = passer.loss
        self.logprint = ckp.write_log_prune # use another log file specifically for pruning logs
        self.netprint = ckp.write_log_prune

        # ************************** variables from RCAN ************************** 
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = model
        self.loss = loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        # **************************************************************************

        # Reg related variables
        self.reg = {}
        self.delta_reg = {}
        self._init_reg()
        self.iter_update_reg_finished = {}
        self.iter_finish_pick = {}
        self.iter_stabilize_reg = math.inf
        self.hist_mag_ratio = {}
        self.w_abs = {}
        self.wn_scale = {}
        
        # prune_init, to determine the pruned weights
        # this will update the 'self.kept_wg' and 'self.pruned_wg' 
        self._get_kept_wg_L1()
        if self.args.same_pruned_wg_layers:
            self._set_same_pruned_wg()

        # init prune_state
        self.prune_state = "update_reg"

    def _init_reg(self):
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                if self.args.wg == 'weight':
                    self.reg[name] = torch.zeros_like(m.weight.data).flatten().cuda()
                else:
                    shape = m.weight.data.shape
                    self.reg[name] = torch.zeros(shape[0], shape[1]).cuda()

    def _greg_1(self, m, name):
        if self.pr[name] == 0:
            return True
        
        pruned = self.pruned_wg[name]
        if self.args.wg == "channel":
            self.reg[name][:, pruned] += self.args.reg_granularity_prune
        elif self.args.wg == "filter":
            self.reg[name][pruned, :] += self.args.reg_granularity_prune
        elif self.args.wg == 'weight':
            self.reg[name][pruned] += self.args.reg_granularity_prune
        else:
            raise NotImplementedError

        # when all layers are pushed hard enough, stop
        return self.reg[name].max() > self.args.reg_upper_limit
    
    def _get_score(self, m):
        shape = m.weight.data.shape
        if self.args.wg == "channel":
            w_abs = m.weight.abs().mean(dim=[0, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=0)
        elif self.args.wg == "filter":
            w_abs = m.weight.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=1)
        elif self.args.wg == "weight":
            w_abs = m.weight.abs().flatten()
        wn_scale = m.wn_scale.abs() if hasattr(m, 'wn_scale') else [None] * shape[0]
        return w_abs, wn_scale

    def _set_same_pruned_wg(self):
        '''Set pruned_wg of some layers to the same indeces. Useful in pruning the last layer in a residual block.
        '''
        index = 0
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers) and self.pr[name] > 0:
                for p in self.args.same_pruned_wg_layers:
                    if fnmatch(name, p):
                        if isinstance(index, int):
                            n_wg = len(self._get_score(m)[0])
                            n_pruned = min(math.ceil(self.pr[name] * n_wg), n_wg - 1) # do not prune all
                            index = np.random.permutation(n_wg)[:n_pruned]
                        self.pruned_wg[name] = index
                        self.kept_wg[name] = [x for x in range(n_wg) if x not in index]
                        self.logprint('==> Set same pruned_wg for "%s"' % name) 
                        break
    
    def _update_reg(self):
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                cnt_m = self.layers[name].layer_index
                pr = self.pr[name]
                if name in self.iter_update_reg_finished.keys():
                    continue

                if self.total_iter % self.args.print_interval == 0:
                    self.logprint("[%d] Update reg for layer '%s'. Pr = %s. Iter = %d" 
                        % (cnt_m, name, pr, self.total_iter))
                
                # get the importance score (L1-norm in this case)
                self.w_abs[name], self.wn_scale[name] = self._get_score(m)
                
                # update reg functions, two things: 
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                finish_update_reg = self._greg_1(m, name)

                # check prune state
                if finish_update_reg:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[name] = self.total_iter
                    self.logprint("==> [%d] Just finished 'update_reg'. Iter = %d" % (cnt_m, self.total_iter))

                    # check if all layers finish 'update_reg'
                    self.prune_state = "stabilize_reg"
                    for n, mm in self.model.named_modules():
                        if isinstance(mm, nn.Conv2d) or isinstance(mm, nn.Linear):
                            if n not in self.iter_update_reg_finished:
                                self.prune_state = "update_reg"
                                break
                    if self.prune_state == "stabilize_reg":
                        self.iter_stabilize_reg = self.total_iter
                        self.logprint("==> All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)
                        # self._save_model(mark='just_finished_update_reg')
                    
                # after reg is updated, print to check
                if self.total_iter % self.args.print_interval == 0:
                    self.logprint("    reg_status: min = %.5f ave = %.5f max = %.5f" % 
                                (self.reg[name].min(), self.reg[name].mean(), self.reg[name].max()))
                    w_abs = self.w_abs[name].data.cpu().numpy()
                    wn_scale = self.wn_scale[name].data.cpu().numpy()
                    avg_mag_pruned = np.mean(w_abs[self.pruned_wg[name]])
                    avg_mag_kept   = np.mean(w_abs[self.kept_wg[name]])
                    avg_scale_pruned = np.mean(wn_scale[self.pruned_wg[name]])
                    avg_scale_kept   = np.mean(wn_scale[self.kept_wg[name]])
                    self.logprint("    average weight magnitude: pruned %.6f kept %.6f" % (avg_mag_pruned, avg_mag_kept))
                    self.logprint("    average weight scale:     pruned %.6f kept %.6f" % (avg_scale_pruned, avg_scale_kept))


    def _apply_reg(self):
        for name, m in self.model.named_modules():
            if name in self.reg and self.pr[name] > 0:
                reg = self.reg[name] # [N, C]
                m.wn_scale.grad += reg[:, 0] * m.wn_scale
                bias = False if isinstance(m.bias, type(None)) else True
                if bias:
                    m.bias.grad += reg[:, 0] * m.bias

    def _merge_wn_scale_to_weights(self):
        '''Merge the learned weight normalization scale to the weights.
        '''
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers) and hasattr(m, 'wn_scale'):
                m.weight.data = F.normalize(m.weight.data, dim=(1,2,3)) * m.wn_scale.view(-1,1,1,1)
                self.logprint(f'Merged weight normalization scale to weights: {name}')

    def _resume_prune_status(self, ckpt_path):
        raise NotImplementedError

    def _save_model(self, acc1=0, acc5=0, mark=''):
        state = {'iter': self.total_iter,
                'prune_state': self.prune_state, # we will resume prune_state
                'arch': self.args.arch,
                'model': self.model,
                'state_dict': self.model.state_dict(),
                'iter_stabilize_reg': self.iter_stabilize_reg,
                'acc1': acc1,
                'acc5': acc5,
                'optimizer': self.optimizer.state_dict(),
                'reg': self.reg,
                'hist_mag_ratio': self.hist_mag_ratio,
                'ExpID': self.logger.ExpID,
        }
        self.save(state, is_best=False, mark=mark)

    def prune(self):
        self.total_iter = 0
        if self.args.resume_path:
            self._resume_prune_status(self.args.resume_path)
            self._get_kept_wg_L1() # get pruned and kept wg from the resumed model
            self.model = self.model.train()
            self.logprint("Resume model successfully: '{}'. Iter = {}. prune_state = {}".format(
                        self.args.resume_path, self.total_iter, self.prune_state))
        
        while True:
            finish_prune = self.train() # there will be a break condition to get out of the infinite loop
            if finish_prune:
                return copy.deepcopy(self.model)
            self.test()

# ************************************************ The code below refers to RCAN ************************************************ #
    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            self.total_iter += 1

            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            
            # --- @mst: update reg factors and apply them before optimizer updates
            if self.total_iter % self.args.print_interval == 0:
                self.logprint("")
                self.logprint("Iter = %d [prune_state = %s, method = %s] " % (self.total_iter, self.prune_state, self.args.method) + "-"*40)
            if self.prune_state == "update_reg" and self.total_iter % self.args.update_reg_interval == 0:
                self._update_reg()
            if self.args.apply_reg: # reg can also be not applied, as a baseline for comparison
                self._apply_reg()
            # ---

            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()  

            # @mst: exit of reg pruning loop
            if self.prune_state == "stabilize_reg" and self.total_iter - self.iter_stabilize_reg >= self.args.stabilize_reg_interval:
                self.logprint(f"'stabilize_reg' is done. Iter {self.total_iter}. Testing...")
                self.test()
                self._merge_wn_scale_to_weights()
                self._prune_and_build_new_model()
                self.logprint(f"Pruned and built a new model, go to 'finetune'. Testing...")
                self.test()
                return True              

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        is_train = self.model.training
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('Evaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                logstr = '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {}) [prune_state = {}, method = {}]'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                        self.prune_state, 
                        self.args.method
                    )
                self.ckp.write_log(logstr)
                self.logprint(logstr)

        self.ckp.write_log('Forward: {:.2f}s'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        # if not self.args.test_only:
        #     self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

        if is_train:
            self.model.train()

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]