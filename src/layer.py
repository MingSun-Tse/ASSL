import torch
import torch.nn as nn
import copy
import numpy as np
from collections import OrderedDict

class Layer:
    def __init__(self, name, size, layer_index, module, res=False, layer_type=None):
        self.name = name
        self.size = []
        for x in size:
            self.size.append(x)
        self.layer_index = layer_index # deprecated, use 'index' instead, keep it to maintain back-compatibility
        self.index = layer_index
        self.module = module
        self.layer_type = layer_type
        self.is_shortcut = True if "downsample" in name else False
        # if res:
        #     self.stage, self.seq_index, self.block_index = self._get_various_index_by_name(name)
    
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
                print('! Parsing the layer name failed: %s. Please check.' % name)

class LayerStruct:
    def __init__(self, model, LEARNABLES):
        self.model = model
        self.LEARNABLES = LEARNABLES
        self.register_layers()
        
    def register_layers(self):
        """This will maintain a data structure that can return some useful information by the name of a layer.
        """
        self.layers = OrderedDict()
        self._max_len_name = 0

        ix = -1 # layer index, starts from 0
        layer_shape = {}
        for name, m in self.model.named_modules():
            if isinstance(m, self.LEARNABLES):
                if "downsample" not in name:
                    ix += 1
                layer_shape[name] = [ix, m.weight.size()]
                self._max_len_name = max(self._max_len_name, len(name))
                size = m.weight.size()
                self.layers[name] = Layer(name, size, ix, module=m, res=True, layer_type=m.__class__.__name__)
        
        self._max_len_ix = len("%s" % ix)
        self.num_layers = ix + 1
        
        print("Register layer index and kernel shape:")
        format_str = "[%{}d] %{}s -- kernel_shape: %s".format(self._max_len_ix, self._max_len_name)
        for name, (ix, ks) in layer_shape.items():
            print(format_str % (ix, name, ks))