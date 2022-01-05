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
from .utils import get_score_layer, pick_pruned_layer
pjoin = os.path.join
tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)

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

        # init prune_state
        self.prune_state = 'update_reg'
        if args.greg_mode in ['part'] and args.same_pruned_wg_layers and args.same_pruned_wg_criterion in ['reg']:
            self.prune_state = "ssa" # sparsity structure alignment
            self._get_kept_wg_L1(align_constrained=True)
        
        # init pruned_wg/kept_wg if they can be determined right at the begining
        if args.greg_mode in ['part'] and self.prune_state in ['update_reg']:
            self._get_kept_wg_L1(align_constrained=True) # this will update the 'self.kept_wg', 'self.pruned_wg', 'self.pr'

    def _init_reg(self):
        for name, m in self.model.named_modules():
            if name in self.layers:
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

    def _greg_penalize_all(self, m, name):
        if self.pr[name] == 0:
            return True
        
        if self.args.wg == "channel":
            self.reg[name] += self.args.reg_granularity_prune
        elif self.args.wg == "filter":
            self.reg[name] += self.args.reg_granularity_prune
        elif self.args.wg == 'weight':
            self.reg[name] += self.args.reg_granularity_prune
        else:
            raise NotImplementedError

        # when all layers are pushed hard enough, stop
        return self.reg[name].max() > self.args.reg_upper_limit

    def _update_reg(self, skip=[]):
        for name, m in self.model.named_modules():
            if name in self.layers:                
                if name in self.iter_update_reg_finished.keys():
                    continue
                if name in skip:
                    continue

                # get the importance score (L1-norm in this case)
                out = get_score_layer(m, wg='filter', criterion='wn_scale')
                self.w_abs[name], self.wn_scale[name] = out['l1-norm'], out['wn_scale']
                
                # update reg functions, two things:
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                if self.args.greg_mode in ['part']:
                    finish_update_reg = self._greg_1(m, name)
                elif self.args.greg_mode in ['all']:
                    finish_update_reg = self._greg_penalize_all(m, name)

                # check prune state
                if finish_update_reg:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[name] = self.total_iter
                    self.logprint(f"==> {self.layer_print_prefix[name]} -- Just finished 'update_reg'. Iter {self.total_iter}. pr {self.pr[name]}")

                    # check if all layers finish 'update_reg'
                    prune_state = "stabilize_reg"
                    for n, mm in self.model.named_modules():
                        if isinstance(mm, self.LEARNABLES):
                            if n not in self.iter_update_reg_finished:
                                prune_state = ''
                                break
                    if prune_state == "stabilize_reg":
                        self.prune_state = 'stabilize_reg'
                        self.iter_stabilize_reg = self.total_iter
                        self.logprint("==> All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)

    def _apply_reg(self):
        for name, m in self.model.named_modules():
            if name in self.layers and self.pr[name] > 0:
                reg = self.reg[name] # [N, C]
                m.wn_scale.grad += reg[:, 0] * m.wn_scale
                bias = False if isinstance(m.bias, type(None)) else True
                if bias:
                    m.bias.grad += reg[:, 0] * m.bias

    def _merge_wn_scale_to_weights(self):
        '''Merge the learned weight normalization scale to the weights.
        '''
        for name, m in self.model.named_modules():
            if name in self.layers and hasattr(m, 'wn_scale'):
                m.weight.data = F.normalize(m.weight.data, dim=(1,2,3)) * m.wn_scale.view(-1,1,1,1)
                self.logprint(f'Merged weight normalization scale to weights: {name}')

    def _resume_prune_status(self, ckpt_path):
        raise NotImplementedError

    def _save_model(self, filename):
        savepath = f'{self.ckp.dir}/model/{filename}'
        ckpt = {
            'pruned_wg': self.pruned_wg,
            'kept_wg': self.kept_wg,
            'model': self.model,
            'state_dict': self.model.state_dict(),
        }
        torch.save(ckpt, savepath) 
        return savepath

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

            # @mst: print
            if self.total_iter % self.args.print_interval == 0:
                self.logprint("")
                self.logprint(f"Iter {self.total_iter} [prune_state: {self.prune_state} method: {self.args.method} compare_mode: {self.args.compare_mode} greg_mode: {self.args.greg_mode}] " + "-"*40)

            # @mst: regularization loss: sparsity structure alignment (SSA)
            if self.prune_state in ['ssa']:
                n = len(self.constrained_layers)
                soft_masks = torch.zeros(n, self.args.n_feats, requires_grad=True).cuda()
                hard_masks = torch.zeros(n, self.args.n_feats, requires_grad=False).cuda()
                cnt = -1
                for name, m in self.model.named_modules():
                    if name in self.constrained_layers:
                        cnt += 1
                        _, indices = torch.sort(m.wn_scale.data)
                        n_wg = m.weight.size(0)
                        n_pruned = n_pruned = min(math.ceil(self.pr[name] * n_wg), n_wg - 1) # do not prune all
                        thre = m.wn_scale[indices[n_pruned]]
                        soft_masks[cnt] = torch.sigmoid(m.wn_scale - thre)
                        hard_masks[cnt] = m.wn_scale >= thre
                loss_reg = -torch.mm(soft_masks, soft_masks.t()).mean()
                loss_reg_hard = -torch.mm(hard_masks, hard_masks.t()).mean().data # only as an analysis metric, not optimized
                if self.total_iter % self.args.print_interval == 0:
                    logstr = f'Iter {self.total_iter} loss_recon {loss.item():.4f} loss_reg (*{self.args.lw_spr}) {loss_reg.item():6f} (loss_reg_hard {loss_reg_hard.item():.6f})'
                    self.logprint(logstr)
                loss += loss_reg * self.args.lw_spr

                # for constrained Conv layers, at prune_state 'ssa', do not update their regularization co-efficients
                if self.total_iter % self.args.update_reg_interval == 0:
                    self._update_reg(skip=self.constrained_layers)

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            
            # @mst: update reg factors and apply them before optimizer updates
            if self.prune_state in ['update_reg'] and self.total_iter % self.args.update_reg_interval == 0:
                self._update_reg()

            # after reg is updated, print to check
            if self.total_iter % self.args.print_interval == 0:
                self._print_reg_status()
        
            if self.args.apply_reg: # reg can also be not applied, as a baseline for comparison
                self._apply_reg()
            # --

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

            # @mst: at the end of 'ssa', switch prune_state to 'update_reg'
            if self.prune_state in ['ssa'] and self.total_iter == self.args.iter_ssa:
                self._get_kept_wg_L1(align_constrained=True) # this will update the pruned_wg/kept_wg for constrained Conv layers
                self.prune_state = 'update_reg'
                self.logprint(f'==> Iter {self.total_iter} prune_state "ssa" is done, get pruned_wg/kept_wg, switch to {self.prune_state}.')

            # @mst: exit of reg pruning loop
            if self.prune_state in ["stabilize_reg"] and self.total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                self.logprint(f"==> 'stabilize_reg' is done. Iter {self.total_iter}.About to prune and build new model. Testing...")
                self.test()
                
                if self.args.greg_mode in ['all']:
                    self._get_kept_wg_L1(align_constrained=True)
                    self.logprint(f'==> Get pruned_wg/kept_wg.')

                self._merge_wn_scale_to_weights()
                self._prune_and_build_new_model()
                path = self._save_model('model_just_finished_prune.pt')
                self.logprint(f"==> Pruned and built a new model. Ckpt saved: '{path}'. Testing...")
                self.test()
                return True            

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def _print_reg_status(self):
        self.logprint('************* Regularization Status *************')
        for name, m in self.model.named_modules():
            if name in self.layers and self.pr[name] > 0:
                logstr = [self.layer_print_prefix[name]]
                logstr += [f"reg_status: min {self.reg[name].min():.5f} ave {self.reg[name].mean():.5f} max {self.reg[name].max():.5f}"]
                out = get_score_layer(m, wg='filter', criterion='wn_scale')
                w_abs, wn_scale = out['l1-norm'], out['wn_scale']
                pruned, kept = pick_pruned_layer(score=wn_scale, pr=self.pr[name], sort_mode='min')
                avg_mag_pruned, avg_mag_kept = np.mean(w_abs[pruned]), np.mean(w_abs[kept])
                avg_scale_pruned, avg_scale_kept = np.mean(wn_scale[pruned]), np.mean(wn_scale[kept])
                logstr += ["average w_mag: pruned %.6f kept %.6f" % (avg_mag_pruned, avg_mag_kept)]
                logstr += ["average wn_scale: pruned %.6f kept %.6f" % (avg_scale_pruned, avg_scale_kept)]
                logstr += [f'Iter {self.total_iter}']
                logstr += [f'cstn' if name in self.constrained_layers else 'free']
                logstr += [f'pr {self.pr[name]}']
                self.logprint(' | '.join(logstr))
        self.logprint('*************************************************')
        
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
                logstr = '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {}) [prune_state: {} method: {} compare_mode: {} greg_mode: {}]'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                        self.prune_state, 
                        self.args.method,
                        self.args.compare_mode,
                        self.args.greg_mode,
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