import torch
import copy

import utility
import data
import model
import loss
from option import args

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

# @mst: select different trainers corresponding to different methods
if args.method in ['']:
    from trainer import Trainer
elif args.method in ['kd']:
    from trainer_kd import TrainerKD as Trainer
elif args.method in ['prune']:
    from trainer import Trainer
    from pruner import pruner_dict

# @mst: KD
def set_up_teacher(args, checkpoint, T_model, T_weights, T_n_resblocks, T_n_feats):
    # update args
    args = copy.deepcopy(args) # avoid modifying the original args
    args.model = T_model
    args.n_resblocks = T_n_resblocks
    args.n_feats = T_n_feats

    # set up model
    global model
    model = model.Model(args, ckp=None)
    
    # load pretraiend weights
    ckpt = torch.load(T_weights)
    model.model.load_state_dict(ckpt)
    checkpoint.write_log('==> Set up teacher successfully, pretrained weights: "%s"' % T_weights)
    return model
    
def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            
            # @mst: different methods require different model settings
            if args.method in ['']: # original setting
                _model = model.Model(args, checkpoint)
            elif args.method in ['kd']:
                _model_S = model.Model(args, checkpoint)
                _model_T = set_up_teacher(args, checkpoint, args.T_model, args.T_weights, args.T_n_resblocks, args.T_n_feats)
                _model = [_model_T, _model_S]
            elif args.method in ['prune']:
                _model = model.Model(args, checkpoint)
                pruner = pruner_dict[args.pruner].Pruner(_model, args, checkpoint, passer=None)
                _model = pruner.prune() # get the pruned model as initialization for later finetuning

            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
