import torch, torch.nn as nn
from collections import OrderedDict
from fnmatch import fnmatch, fnmatchcase
import math, numpy as np

tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)


def get_score_layer(module, wg='filter', criterion='l1-norm'):
    r"""Get importance score for a layer.

    Return:
        A dict that has key 'score', whose value is a list
    """
    # -- define any scoring scheme here as you like
    shape = module.weight.data.shape
    if wg == "channel":
        l1 = module.weight.abs().mean(dim=[0, 2, 3]) if len(shape) == 4 else module.weight.abs().mean(dim=0)
    elif wg == "filter":
        l1 = module.weight.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else module.weight.abs().mean(dim=1)
    elif wg == "weight":
        l1 = module.weight.abs().flatten()
    # --

    out = {}
    out['l1-norm'] = tensor2list(l1)
    if hasattr(module, 'wn_scale'): 
        out['wn_scale'] = tensor2list(module.wn_scale.abs())
    out['score'] = out[criterion]
    return out

def pick_pruned_layer(score, pr=None, threshold=None, sort_mode='min'):
    r"""Get the indices of pruned weight groups in a layer.
    """
    assert sort_mode in ['min', 'rand', 'max']
    score = np.array(score)
    num_total = len(score)
    if sort_mode in ['rand']:
        assert pr is not None
        num_pruned = min(math.ceil(pr * num_total), num_total - 1) # do not prune all
        order = np.random.permutation(num_total).tolist()
    else:
        num_pruned = min(math.ceil(pr * num_total), num_total - 1) if threshold is None else len(np.where(score < threshold)[0])
        if sort_mode in ['min', 'ascending']:
            order = np.argsort(score).tolist()
        elif sort_mode in ['max', 'descending']:
            order = np.argsort(score)[::-1].tolist()
    pruned, kept = order[:num_pruned], order[num_pruned:]
    return pruned, kept

def pick_pruned_model(layers, pr, wg='filter', criterion='l1-norm', compare_mode='local', sort_mode='min'):
    r"""Pick pruned weight groups for a model.
    Args:
        layers: an OrderedDict, key is layer name

    Return:
        pruned (OrderedDict): key is layer name, value is the pruned indices for the layer
        kept (OrderedDict): key is layer name, value is the kept indices for the layer
    """
    assert sort_mode in ['rand', 'min', 'max'] and compare_mode in ['global', 'local']
    pruned, kept, all_scores = OrderedDict(), OrderedDict(), []

    # iter to get importance score for each layer
    for name, layer in layers.items():
        out = get_score_layer(layer.module, wg=wg, criterion=criterion)
        score = out['score']
        layer.score = score
        if pr[name] > 0: # pr > 0 indicates we want to prune this layer so its score will be included in the <all_scores>
            all_scores += score

        if compare_mode in ['local']:
            assert isinstance(pr, dict)
            pruned[name], kept[name] = pick_pruned_layer(score, pr[name], sort_mode=sort_mode)
            print(f'{name} -- pruned: {pruned[name]} kept: {kept[name]}')
    
    if compare_mode in ['global']:
        num_total = len(all_scores)
        num_pruned = min(math.ceil(pr['model'] * num_total), num_total - 1) # do not prune all
        if sort_mode == 'min':
            threshold = sorted(all_scores)[num_pruned] # in ascending order
        elif sort_mode == 'max':
            threshold = sorted(all_scores)[::-1][num_pruned] # in decending order
        print(f'#all_scores: {len(all_scores)} threshold:{threshold:.6f}')

        for name, layer in layers.items():
            if pr[name] > 0:
                if sort_mode in ['rand']:
                    pass
                elif sort_mode in ['min', 'max']:
                    pruned[name], kept[name] = pick_pruned_layer(layer.score, pr=None, threshold=threshold, sort_mode=sort_mode)
            else:
                pruned[name], kept[name] = [], list(range(len(layer.score)))
            print(f'{name} -- pruned: {pruned[name]} kept: {kept[name]}')
    return pruned, kept

def get_constrained_layers(layers, constrained_layers_pattern):
    constrained_layers = []
    for name, _ in layers.items():
        for p in constrained_layers_pattern:
            if fnmatch(name, p):
                constrained_layers += [name]
    return constrained_layers