import torch
import torch.nn as nn
import copy
import time
import numpy as np
from math import ceil, sqrt
from collections import OrderedDict
from utils import strdict_to_dict
from fnmatch import fnmatch, fnmatchcase
from layer import LayerStruct
from .utils import pick_pruned_model, get_constrained_layers
                
class MetaPruner:
    def __init__(self, model, args, logger, passer):
        self.model = model
        self.args = args
        self.logger = logger
        self.logprint = logger.log_printer.logprint if logger else print
        self.netprint = logger.log_printer.netprint if logger else print
        
        # set up layers
        self.LEARNABLES = (nn.Conv2d, nn.Linear) # the layers we focus on for pruning
        layer_struct = LayerStruct(model, self.LEARNABLES)
        self.layers = layer_struct.layers
        self._max_len_ix = layer_struct._max_len_ix
        self._max_len_name = layer_struct._max_len_name

        # set up pr for each layer
        self.get_pr()

        # pick pruned and kept weight groups
        self.constrained_layers = get_constrained_layers(self.layers, self.args.same_pruned_wg_layers)
        print(f'constrained: {self.constrained_layers}')

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
        if self.args.compare_mode in ['global']:
            pr = 1e-20 # a positive value, will be replaced. Its current role is to indicate this layer will be considered for pruning
        elif self.args.compare_mode in ['local']:
            layer_index = self.layers[name].layer_index
            pr = self.args.stage_pr[layer_index]

        # if layer name matchs the pattern pre-specified in 'args.skip_layers', skip it (i.e., pr = 0)
        for p in self.args.skip_layers:
            if fnmatch(name, p):
                pr = 0
        return pr
    
    def get_pr(self):
        """Get layer-wise pruning ratio for a model
        """
        self.pr = {}
        if self.args.stage_pr: # 'stage_pr' may be None (in the case that 'base_pr_model' is provided)
            self.pr['model'] = self.args.stage_pr
            for name, m in self.model.named_modules():
                if isinstance(m, self.LEARNABLES):
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
        self.pruned_wg, self.kept_wg = pick_pruned_model(self.layers, self.pr, 
                                                        wg=self.args.wg, 
                                                        criterion=self.args.prune_criterion,
                                                        compare_mode=self.args.compare_mode,
                                                        sort_mode=self.args.pick_pruned)
        
        num_pruned_constrained = 0
        for name, layer in self.layers.items():
            self.pr[name] = len(self.pruned_wg[name]) / len(layer.score)
            if name in self.constrained_layers:
                num_pruned_constrained += len(self.pruned_wg[name])
        
        # adjust pruned/kept/pr for constrained conv layers
        for name, layer in self.layers.items():
            if name in self.constrained_layers:
                num_pruned = int(num_pruned_constrained / len(self.constrained_layers))
                self.pr[name] = num_pruned / len(layer.score)
                order = self.pruned_wg[name] + self.kept_wg[name]
                self.pruned_wg[name], self.kept_wg[name] = order[:num_pruned], order[num_pruned:]
                print(f'{name} is a constrained layer. Adjust its pruned/kept indices.')
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
            if isinstance(m, self.LEARNABLES):
                kept_filter, kept_chl = self._get_kept_filter_channel(m, name)
                
                # decide if renit the current layer
                reinit = False
                for rl in self.args.reinit_layers:
                    if fnmatch(name, rl):
                        reinit = True
                        break
                
                # get number of channels (can be manually assigned)
                num_chl = self.args.layer_chl[name] if name in self.args.layer_chl else len(kept_chl)

                # copy weight and bias
                bias = False if isinstance(m.bias, type(None)) else True
                if isinstance(m, nn.Conv2d):
                    new_layer = nn.Conv2d(num_chl, len(kept_filter), m.kernel_size,
                                    m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                    if not reinit:
                        kept_weights = m.weight.data[kept_filter][:, kept_chl, :, :]

                elif isinstance(m, nn.Linear):
                    kept_weights = m.weight.data[kept_filter][:, kept_chl]
                    new_layer = nn.Linear(in_features=len(kept_chl), out_features=len(kept_filter), bias=bias).cuda()
                
                if not reinit:
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
        LayerStruct(new_model, self.LEARNABLES)
    
    def _get_masks(self):
        '''Get masks for unstructured pruning
        '''
        self.mask = {}
        for name, m in self.model.named_modules():
            if isinstance(m, self.LEARNABLES):
                mask = torch.ones_like(m.weight.data).cuda().flatten()
                pruned = self.pruned_wg[name]
                mask[pruned] = 0
                self.mask[name] = mask.view_as(m.weight.data)
        self.logprint('Get masks done for weight pruning')