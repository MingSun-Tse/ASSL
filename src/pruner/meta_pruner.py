import torch
import torch.nn as nn
import copy
import time
import numpy as np
from math import ceil, sqrt
from collections import OrderedDict
from utils import strdict_to_dict
from fnmatch import fnmatch, fnmatchcase
from layer import Layers
                
class MetaPruner:
    def __init__(self, model, args, logger, passer):
        self.model = model
        self.args = args
        self.logger = logger
        self.logprint = logger.log_printer.logprint if logger else print
        self.netprint = logger.log_printer.netprint if logger else print
        
        # set up layers
        layers = Layers(model)
        self.layers = layers.layers
        self.learnable_layers = layers.learnable_layers
        self._max_len_ix = layers._max_len_ix
        self._max_len_name = layers._max_len_name

        # pruning related
        self.kept_wg = {}
        self.pruned_wg = {}
        self.get_pr() # set up pr for each layer
        
    def _pick_pruned(self, w_abs, pr, mode="min"):
        if pr == 0:
            return []
        w_abs_list = w_abs.flatten()
        n_wg = len(w_abs_list)
        n_pruned = min(ceil(pr * n_wg), n_wg - 1) # do not prune all
        if mode == "rand":
            out = np.random.permutation(n_wg)[:n_pruned]
        elif mode == "min":
            out = w_abs_list.sort()[1][:n_pruned]
            out = out.data.cpu().numpy()
        elif mode == "max":
            out = w_abs_list.sort()[1][-n_pruned:]
            out = out.data.cpu().numpy()
        return out

    def _next_learnable_layer(self, model, name, mm):
        '''get the next conv or fc layer name
        '''
        if hasattr(self.layers[name], 'block_index'):
            block_index = self.layers[name].block_index
            if block_index == self.n_conv_within_block - 1:
                return None
        
        ix = self.layers[name].layer_index # layer index of current layer
        type_ = mm.__class__.__name__ # layer type of current layer
        for name, layer in self.layers.items():
            if layer.layer_type == type_ and layer.layer_index == ix + 1: # for now, requires the same layer_type for wg == 'channel'. TODO: generalize this
                return name
        return None
    
    def _prev_learnable_layer(self, model, name, mm):
        '''get the previous conv or fc layer name
        '''
        if hasattr(self.layers[name], 'block_index'):
            block_index = self.layers[name].block_index
            if block_index in [None, 0, -1]: # 1st conv, 1st conv in a block, 1x1 shortcut layer
                return None
        
        ix = self.layers[name].layer_index # layer index of current layer
        for name, layer in self.layers.items():
            if layer.layer_index == ix - 1:
                return name
        return None

    def _next_bn(self, model, mm):
        just_passed_mm = False
        for m in model.modules():
            if m == mm:
                just_passed_mm = True
            if just_passed_mm and isinstance(m, nn.BatchNorm2d):
                return m
        return None
   
    def _replace_module(self, model, name, new_m):
        '''
            Replace the module <name> in <model> with <new_m>
            E.g., 'module.layer1.0.conv1'
            ==> model.__getattr__('module').__getattr__("layer1").__getitem__(0).__setattr__('conv1', new_m)
        '''
        obj = model
        segs = name.split(".")
        for ix in range(len(segs)):
            s = segs[ix]
            if ix == len(segs) - 1: # the last one
                if s.isdigit():
                    obj.__setitem__(int(s), new_m)
                else:
                    obj.__setattr__(s, new_m)
                return
            if s.isdigit():
                obj = obj.__getitem__(int(s))
            else:
                obj = obj.__getattr__(s)
    
    def _get_n_filter(self, model):
        '''
            Do not consider the downsample 1x1 shortcuts.
        '''
        n_filter = OrderedDict()
        for name, m in model.named_modules():
            if name in self.layers:
                if not self.layers[name].is_shortcut:
                    ix = self.layers[name].layer_index
                    n_filter[ix] = m.weight.size(0)
        return n_filter
    
    def _get_layer_pr(self, name):
        '''Example: '[0-4:0.5, 5:0.6, 8-10:0.2]'
                    6, 7 not mentioned, default value is 0
        '''
        layer_index = self.layers[name].layer_index
        pr = self.args.stage_pr[layer_index]

        # if layer name matchs the pattern pre-specified in 'args.skip_layers', skip it (i.e., pr = 0)
        for p in self.args.skip_layers:
            if fnmatch(name, p):
                pr = 0
        return pr
    
    def get_pr(self):
        self.pr = {}
        if self.args.stage_pr: # stage_pr may be None (in the case that base_pr_model is provided)
            for name, m in self.model.named_modules():
                if isinstance(m, self.learnable_layers):
                    self.pr[name] = self._get_layer_pr(name)
        else:
            assert self.args.base_pr_model
            state = torch.load(self.args.base_pr_model)
            self.pruned_wg_pr_model = state['pruned_wg']
            self.kept_wg_pr_model = state['kept_wg']
            for k in self.pruned_wg_pr_model:
                n_pruned = len(self.pruned_wg_pr_model[k])
                n_kept = len(self.kept_wg_pr_model[k])
                self.pr[k] = float(n_pruned) / (n_pruned + n_kept)
            self.logprint("==> Load base_pr_model successfully and inherit its pruning ratio: '{}'".format(self.args.base_pr_model))

    def _get_kept_wg_L1(self):
        '''Decide kept (or pruned) weight group by L1-norm sorting.
        ''' 
        wg = self.args.wg
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                shape = m.weight.data.shape
                if wg == "filter":
                    score = m.weight.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=1)
                elif wg == "channel":
                    score = m.weight.abs().mean(dim=[0, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=0)
                elif wg == "weight":
                    score = m.weight.abs().flatten()
                else:
                    raise NotImplementedError
                
                self.pruned_wg[name] = self._pick_pruned(score, self.pr[name], self.args.pick_pruned)
                self.kept_wg[name] = [i for i in range(len(score)) if i not in self.pruned_wg[name]]
                
                format_str = "[%{}d] %{}s -- got pruned wg by L1 sorting (%s), pr %s".format(self._max_len_ix, self._max_len_name)
                logtmp = format_str % (self.layers[name].layer_index, name, self.args.pick_pruned, self.pr[name])
                self.netprint(logtmp)

    def _get_kept_filter_channel(self, m, name):
        if self.args.wg == "channel":
            kept_chl = self.kept_wg[name]
            next_learnable_layer = self._next_learnable_layer(self.model, name, m)
            if not next_learnable_layer:
                kept_filter = range(m.weight.size(0))
            else:
                kept_filter = self.kept_wg[next_learnable_layer]
        
        elif self.args.wg == "filter":
            kept_filter = self.kept_wg[name]
            prev_learnable_layer = self._prev_learnable_layer(self.model, name, m)
            if (not prev_learnable_layer) or self.pr[prev_learnable_layer] == 0: 
                # In the case of SR networks, tail, there is an upsampling via sub-pixel. 'self.pr[prev_learnable_layer] == 0' can help avoid it. 
                # Not using this, the code will report error.
                kept_chl = range(m.weight.size(1))
            else:
                kept_chl = self.kept_wg[prev_learnable_layer]
        return kept_filter, kept_chl

    def _prune_and_build_new_model(self):
        if self.args.wg == 'weight':
            self._get_masks()
            return

        new_model = copy.deepcopy(self.model)
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                kept_filter, kept_chl = self._get_kept_filter_channel(m, name)
                
                # copy weight and bias
                bias = False if isinstance(m.bias, type(None)) else True
                if isinstance(m, nn.Conv2d):
                    kept_weights = m.weight.data[kept_filter][:, kept_chl, :, :]
                    new_layer = nn.Conv2d(kept_weights.size(1), kept_weights.size(0), m.kernel_size,
                                    m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                elif isinstance(m, nn.Linear):
                    kept_weights = m.weight.data[kept_filter][:, kept_chl]
                    new_layer = nn.Linear(in_features=len(kept_chl), out_features=len(kept_filter), bias=bias).cuda()
                
                new_layer.weight.data.copy_(kept_weights) # load weights into the new module
                
                if bias:
                    kept_bias = m.bias.data[kept_filter]
                    new_layer.bias.data.copy_(kept_bias)
                
                # load the new conv
                self._replace_module(new_model, name, new_layer)

                # get the corresponding bn (if any) for later use
                next_bn = self._next_bn(self.model, m)

            elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                new_bn = nn.BatchNorm2d(len(kept_filter), eps=m.eps, momentum=m.momentum, 
                        affine=m.affine, track_running_stats=m.track_running_stats).cuda()
                
                # copy bn weight and bias
                if self.args.copy_bn_w:
                    weight = m.weight.data[kept_filter]
                    new_bn.weight.data.copy_(weight)
                if self.args.copy_bn_b:
                    bias = m.bias.data[kept_filter]
                    new_bn.bias.data.copy_(bias)
                
                # copy bn running stats
                new_bn.running_mean.data.copy_(m.running_mean[kept_filter])
                new_bn.running_var.data.copy_(m.running_var[kept_filter])
                new_bn.num_batches_tracked.data.copy_(m.num_batches_tracked)
                
                # load the new bn
                self._replace_module(new_model, name, new_bn)

        self.model = new_model

        # print the layer shape of pruned model
        Layers(new_model)
        # n_filter = self._get_n_filter(self.model)
        # logtmp = '{'
        # for ix, num in n_filter.items():
        #     logtmp += '%s:%d, ' % (ix, num)
        # logtmp = logtmp[:-2] + '}'
        # self.logprint('n_filter of pruned model: %s' % logtmp)
    
    def _get_masks(self):
        '''Get masks for unstructured pruning
        '''
        self.mask = {}
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                mask = torch.ones_like(m.weight.data).cuda().flatten()
                pruned = self.pruned_wg[name]
                mask[pruned] = 0
                self.mask[name] = mask.view_as(m.weight.data)
        self.logprint('Get masks done for weight pruning')