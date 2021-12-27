import torch
import torch.nn as nn
import copy
import time
import numpy as np
import utility
from tqdm import tqdm
from utils import _weights_init, _weights_init_orthogonal, orthogonalize_weights
from .meta_pruner import MetaPruner


# refer to: A Signal Propagation Perspective for Pruning Neural Networks at Initialization (ICLR 2020).
# https://github.com/namhoonlee/spp-public
def approximate_isometry_optimize(model, mask, lr, n_iter, wg='weight', print=print):
    def optimize(w):
        '''Approximate Isometry for sparse weights by iterative optimization
        '''
        flattened = w.view(w.size(0), -1) # [n_filter, -1]
        identity = torch.eye(w.size(0)).cuda() # identity matrix
        w_ = torch.autograd.Variable(flattened, requires_grad=True)
        optim = torch.optim.Adam([w_], lr)
        for i in range(n_iter):
            loss = nn.MSELoss()(torch.matmul(w_, w_.t()), identity)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if not isinstance(mask, type(None)):
                w_ = torch.mul(w_, mask[name]) # not update the pruned params
            w_ = torch.autograd.Variable(w_, requires_grad=True)
            optim = torch.optim.Adam([w_], lr)
            if i % 10 == 0:
                print('[%d/%d] approximate_isometry_optimize for layer "%s", loss %.6f' % (i, n_iter, name, loss.item()))
        return w_.view(m.weight.shape)
    
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w_ = optimize(m.weight)
            m.weight.data.copy_(w_)
            print('Finished approximate_isometry_optimize for layer "%s"' % name)

def exact_isometry_based_on_existing_weights(model, print=print):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w_ = orthogonalize_weights(m.weight)
            m.weight.data.copy_(w_)
            print('Finished exact_isometry for layer "%s"' % name)

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, passer):
        super(Pruner, self).__init__(model, args, logger, passer)
        ckp = passer.ckp
        self.logprint = ckp.write_log_prune # use another log file specifically for pruning logs
        self.netprint = ckp.write_log_prune

        # ************************** variables from RCAN ************************** 
        loader = passer.loader
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = model

        self.error_last = 1e8
        # **************************************************************************

    def prune(self):
        self._get_kept_wg_L1()
        self.logprint(f"==> Before _prune_and_build_new_model. Testing...")
        self.test()
        self._prune_and_build_new_model()
        self.logprint(f"==> Pruned and built a new model. Testing...")
        self.test()
        mask = self.mask if self.args.wg == 'weight' else None

        if self.args.reinit:
            if self.args.reinit in ['default', 'kaiming_normal']:
                self.model.apply(_weights_init) # completely reinit weights via 'kaiming_normal'
                self.logprint("==> Reinit model: default ('kaiming_normal' for Conv/FC; 0 mean, 1 std for BN)")

            elif self.args.reinit in ['orth', 'exact_isometry_from_scratch']:
                self.model.apply(lambda m: _weights_init_orthogonal(m, act=self.args.activation)) # reinit weights via 'orthogonal_' from scratch
                self.logprint("==> Reinit model: exact_isometry ('orthogonal_' for Conv/FC; 0 mean, 1 std for BN)")

            elif self.args.reinit == 'exact_isometry_based_on_existing':
                exact_isometry_based_on_existing_weights(self.model, print=self.logprint) # orthogonalize weights based on existing weights
                self.logprint("==> Reinit model: exact_isometry (orthogonalize Conv/FC weights based on existing weights)")

            elif self.args.reinit == 'approximate_isometry': # A Signal Propagation Perspective for Pruning Neural Networks at Initialization (ICLR 2020)
                approximate_isometry_optimize(self.model, mask=mask, lr=self.args.lr_AI, n_iter=10000, print=self.logprint) # 10000 refers to the paper above; lr in the paper is 0.1, but not converged here
                self.logprint("==> Reinit model: approximate_isometry")
            
            else:
                raise NotImplementedError
            
        return copy.deepcopy(self.model)
    
    def test(self):
        is_train = self.model.training
        torch.set_grad_enabled(False)

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
                logstr = '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {}) [method: {} compare_mode: {}]'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                        self.args.method,
                        self.args.compare_mode,
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