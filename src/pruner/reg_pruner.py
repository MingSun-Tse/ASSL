import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from decimal import Decimal
import os, copy, time, pickle, numpy as np, math
from .meta_pruner import MetaPruner
from utils import plot_weights_heatmap, get_n_flops_, get_n_params_
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
        self.logprint = ckp.write_log
        self.netprint = ckp.write_log

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
        
        # prune_init, to determine the pruned weights
        # this will update the 'self.kept_wg' and 'self.pruned_wg' 
        if self.args.method in ['GReg-1']:
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

    def _get_score(self, m):
        shape = m.weight.data.shape
        if self.args.wg == "channel":
            w_abs = m.weight.abs().mean(dim=[0, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=0)
        elif self.args.wg == "filter":
            w_abs = m.weight.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=1)
        elif self.args.wg == "weight":
            w_abs = m.weight.abs().flatten()
        return w_abs

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
    
    def _set_same_pruned_wg(self):
        '''Set pruned_wg of some layers to the same indeces. Useful in pruning the last layer in a residual block.
        '''
        index = 0
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers) and self.pr[name] > 0:
                for p in self.args.same_pruned_wg_layers:
                    if fnmatch(name, p):
                        if isinstance(index, int):
                            n_wg = len(self._get_score(m))
                            n_pruned = min(math.ceil(self.pr[name] * n_wg), n_wg - 1) # do not prune all
                            index = np.random.permutation(n_wg)[:n_pruned]
                        self.pruned_wg[name] = index
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
                self.w_abs[name] = self._get_score(m)
                
                # update reg functions, two things: 
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                if self.args.method == "GReg-1":
                    finish_update_reg = self._greg_1(m, name)
                else:
                    self.logprint("Wrong '--method' argument, please check.")
                    exit(1)

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

    def _apply_reg(self):
        for name, m in self.model.named_modules():
            if name in self.reg and self.pr[name] > 0:
                reg = self.reg[name] # [N, C]
                if self.args.wg in ['filter', 'channel']:
                    if reg.shape != m.weight.data.shape:
                        reg = reg.unsqueeze(2).unsqueeze(3) # [N, C, 1, 1]
                elif self.args.wg == 'weight':
                    reg = reg.view_as(m.weight.data) # [N, C, H, W]
                m.weight.grad += reg * m.weight
    
    def _resume_prune_status(self, ckpt_path):
        raise NotImplementedError
        # state = torch.load(ckpt_path)
        # self.model = state['model'].cuda()
        # self.model.load_state_dict(state['state_dict'])
        # self.optimizer = optim.SGD(self.model.parameters(), 
        #                         lr=self.args.lr_pick if self.args.__dict__.get('AdaReg_only_picking') else self.args.lr_prune, 
        #                         momentum=self.args.momentum,
        #                         weight_decay=self.args.weight_decay)
        # self.optimizer.load_state_dict(state['optimizer'])
        # self.prune_state = state['prune_state']
        # self.total_iter = state['iter']
        # self.iter_stabilize_reg = state.get('iter_stabilize_reg', math.inf)
        # self.reg = state['reg']
        # self.hist_mag_ratio = state['hist_mag_ratio']

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
        # get the statistics of unpruned model
        n_params_original_v2 = get_n_params_(self.model)
        n_flops_original_v2 = get_n_flops_(self.model, img_size=self.args.patch_size, n_channel=3)
        
        self.total_iter = 0
        if self.args.resume_path:
            self._resume_prune_status(self.args.resume_path)
            self._get_kept_wg_L1() # get pruned and kept wg from the resumed model
            self.model = self.model.train()
            self.logprint("Resume model successfully: '{}'. Iter = {}. prune_state = {}".format(
                        self.args.resume_path, self.total_iter, self.prune_state))
        
        while True:
            finish_prune = self.train() # there will be a break condition to get out of the infinite loop
            self.test()
            if finish_prune:
                
                # get the statistics of pruned model and print
                n_params_now_v2 = get_n_params_(self.model)
                n_flops_now_v2 = get_n_flops_(self.model, img_size=self.args.patch_size, n_channel=3)
                self.logprint("==> n_params_original_v2: {:>7.4f}M, n_flops_original_v2: {:>7.4f}G".format(n_params_original_v2/1e6, n_flops_original_v2/1e9))
                self.logprint("==> n_params_now_v2:      {:>7.4f}M, n_flops_now_v2:      {:>7.4f}G".format(n_params_now_v2/1e6, n_flops_now_v2/1e9))
                ratio_param = (n_params_original_v2 - n_params_now_v2) / n_params_original_v2
                ratio_flops = (n_flops_original_v2 - n_flops_now_v2) / n_flops_original_v2
                compression_ratio = 1.0 / (1 - ratio_param)
                speedup_ratio = 1.0 / (1 - ratio_flops)
                self.logprint("==> reduction ratio -- params: {:>5.2f}% (compression {:>.2f}x), flops: {:>5.2f}% (speedup {:>.2f}x)".format(ratio_param*100, compression_ratio, ratio_flops*100, speedup_ratio))
                
                return copy.deepcopy(self.model)

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
                self.logprint("Iter = %d [prune_state = %s, method = %s] " 
                    % (self.total_iter, self.prune_state, self.args.method) + "-"*40)
            if self.prune_state == "update_reg" and self.total_iter % self.args.update_reg_interval == 0:
                self._update_reg()
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

            # --- @mst: exit of reg pruning loop
            if self.prune_state == "stabilize_reg" and self.total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                self._prune_and_build_new_model()
                self.logprint("'stabilize_reg' is done. Pruned, go to 'finetune'. Iter = %d" % self.total_iter)
                return True
            # ---
                

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
    
    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
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
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {}) [prune_state = {}, method = {}]'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                        self.prune_state, 
                        self.args.method
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]