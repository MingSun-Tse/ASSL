from . import l1_pruner, assl_pruner

# when new pruner implementation is added in the 'pruner' dir, update this dict to maintain minimal code change.
# key: pruning method name, value: the corresponding pruner
pruner_dict = {
    'L1': l1_pruner,
    'ASSL': assl_pruner,
}