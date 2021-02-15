import torch
import torch.nn as nn
import copy
import time
import numpy as np
from math import ceil, sqrt
from collections import OrderedDict
from utils import strdict_to_dict

class Layer:
    def __init__(self, name, size, layer_index, res=False, layer_type=None):
        self.name = name
        self.size = []
        for x in size:
            self.size.append(x)
        self.layer_index = layer_index
        self.layer_type = layer_type
        self.is_shortcut = True if "downsample" in name else False
        if res:
            self.stage, self.seq_index, self.block_index = self._get_various_index_by_name(name)
    
    def _get_various_index_by_name(self, name):
        '''Get the indeces including stage, seq_ix, blk_ix.
            Same stage means the same feature map size.
        '''
        global lastest_stage # an awkward impel, just for now
        if name.startswith('module.'):
            name = name[7:] # remove the prefix caused by pytorch data parallel

        if "conv1" == name: # TODO: this might not be so safe
            lastest_stage = 0
            return 0, None, None
        if "linear" in name or 'fc' in name: # Note: this can be risky. Check it fully. TODO: @mingsun-tse
            return lastest_stage + 1, None, None # fc layer should always be the last layer
        else:
            try:
                stage  = int(name.split(".")[0][-1]) # ONLY work for standard resnets. name example: layer2.2.conv1, layer4.0.downsample.0
                seq_ix = int(name.split(".")[1])
                if 'conv' in name.split(".")[-1]:
                    blk_ix = int(name[-1]) - 1
                else:
                    blk_ix = -1 # shortcut layer  
                lastest_stage = stage
                return stage, seq_ix, blk_ix
            except:
                print('!Parsing the layer name failed: %s. Please check.' % name)
                
class MetaPruner:
    def __init__(self, model, args, logger, passer):
        self.model = model
        self.args = args
        self.logger = logger
        self.logprint = logger.log_printer.logprint
        self.accprint = logger.log_printer.accprint
        self.netprint = logger.log_printer.netprint
        self.test = lambda net: passer.test(passer.test_loader, net, passer.criterion, passer.args)
        self.train_loader = passer.train_loader
        self.criterion = passer.criterion
        self.save = passer.save
        self.is_single_branch = passer.is_single_branch
        self.learnable_layers = (nn.Conv2d, nn.Linear) # Note: for now, we only focus on weights in Conv and FC modules, no BN.
        
        self.layers = OrderedDict()
        self._register_layers()

        arch = self.args.arch
        if arch.startswith('resnet'):
            # TODO: add block
            self.n_conv_within_block = 0
            if args.dataset == "imagenet":
                if arch in ['resnet18', 'resnet34']:
                    self.n_conv_within_block = 2
                elif arch in ['resnet50', 'resnet101', 'resnet152']:
                    self.n_conv_within_block = 3
            else:
                self.n_conv_within_block = 2

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
        elif mode == "max":
            out = w_abs_list.sort()[1][-n_pruned:]
        return out

    def _register_layers(self):
        '''
            This will maintain a data structure that can return some useful 
            information by the name of a layer.
        '''
        ix = -1 # layer index, starts from 0
        self._max_len_name = 0
        layer_shape = {}
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                if "downsample" not in name:
                    ix += 1
                layer_shape[name] = [ix, m.weight.size()]
                self._max_len_name = max(self._max_len_name, len(name))
                
                size = m.weight.size()
                res = True if self.args.arch.startswith('resnet') else False
                self.layers[name] = Layer(name, size, ix, res, layer_type=m.__class__.__name__)
        
        self._max_len_ix = len("%s" % ix)
        print("Register layer index and kernel shape:")
        format_str = "[%{}d] %{}s -- kernel_shape: %s".format(self._max_len_ix, self._max_len_name)
        for name, (ix, ks) in layer_shape.items():
            print(format_str % (ix, name, ks))

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
    
    def _get_layer_pr_vgg(self, name):
        '''Example: '[0-4:0.5, 5:0.6, 8-10:0.2]'
                    6, 7 not mentioned, default value is 0
        '''
        layer_index = self.layers[name].layer_index
        pr = self.args.stage_pr[layer_index]
        if str(layer_index) in self.args.skip_layers:
            pr = 0
        return pr

    def _get_layer_pr_resnet(self, name):
        '''
            This function will determine the prune_ratio (pr) for each specific layer
            by a set of rules.
        '''
        wg = self.args.wg
        layer_index = self.layers[name].layer_index
        stage = self.layers[name].stage
        seq_index = self.layers[name].seq_index
        block_index = self.layers[name].block_index
        is_shortcut = self.layers[name].is_shortcut
        pr = self.args.stage_pr[stage]

        # for unstructured pruning, no restrictions, every layer can be pruned
        if self.args.wg != 'weight':
            # do not prune the shortcut layers for now
            if is_shortcut:
                pr = 0
            
            # do not prune layers we set to be skipped
            layer_id = '%s.%s.%s' % (str(stage), str(seq_index), str(block_index))
            for s in self.args.skip_layers:
                if s and layer_id.startswith(s):
                    pr = 0

            # for channel/filter prune, do not prune the 1st/last conv in a block
            if (wg == "channel" and block_index == 0) or \
                (wg == "filter" and block_index == self.n_conv_within_block - 1):
                pr = 0
        
        return pr
    
    def get_pr(self):
        if self.is_single_branch(self.args.arch):
            get_layer_pr = self._get_layer_pr_vgg
        else:
            get_layer_pr = self._get_layer_pr_resnet

        self.pr = {}
        if self.args.stage_pr: # stage_pr may be None (in the case that base_pr_model is provided)
            for name, m in self.model.named_modules():
                if isinstance(m, self.learnable_layers):
                    self.pr[name] = get_layer_pr(name)
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
        if self.args.base_pr_model and self.args.inherit_pruned == 'index':
            self.pruned_wg = self.pruned_wg_pr_model
            self.kept_wg = self.kept_wg_pr_model
            self.logprint("==> Inherit the pruned index from base_pr_model: '{}'".format(self.args.base_pr_model))
        else:    
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
                    
                    # compare the pruned weights picked by L1-sorting vs. other criterion which provides the base_pr_model (e.g., OBD)
                    if self.args.base_pr_model:
                        intersection = [x for x in self.pruned_wg_pr_model[name] if x in self.pruned_wg[name]]
                        intersection_ratio = len(intersection) / len(self.pruned_wg[name]) if len(self.pruned_wg[name]) else 0
                        logtmp += ', intersection ratio of the weights picked by L1 vs. base_pr_model: %.4f (%d)' % (intersection_ratio, len(intersection))
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
            if not prev_learnable_layer:
                kept_chl = range(m.weight.size(1))
            else:
                if self.layers[name].layer_type == self.layers[prev_learnable_layer].layer_type:
                    kept_chl = self.kept_wg[prev_learnable_layer]
                
                else: # current layer is the 1st fc, the previous layer is the last conv
                    last_conv_n_filter = self.layers[prev_learnable_layer].size[0]
                    last_conv_fm_size = int(m.weight.size(1) / last_conv_n_filter) # feature map spatial size. 36 for alexnet
                    self.logprint('last_conv_feature_map_size: %dx%d (before fed into the first fc)' % (sqrt(last_conv_fm_size), sqrt(last_conv_fm_size)))
                    last_conv_kept_filter = self.kept_wg[prev_learnable_layer]
                    kept_chl = []
                    for i in last_conv_kept_filter:
                        tmp = list(range(i * last_conv_fm_size, i * last_conv_fm_size + last_conv_fm_size))
                        kept_chl += tmp
        
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
        n_filter = self._get_n_filter(self.model)
        logtmp = '{'
        for ix, num in n_filter.items():
            logtmp += '%s:%d, ' % (ix, num)
        logtmp = logtmp[:-2] + '}'
        self.logprint('n_filter of pruned model: %s' % logtmp)
    
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