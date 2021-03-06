import torch
import copy

import utility
import data
import model
import loss
from option import args
from utils import get_n_flops_, get_n_params_
from torchsummaryX import summary

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

# @mst: select different trainers corresponding to different methods
if args.method in ['']:
    from trainer import Trainer
elif args.method in ['KD']:
    from trainer_kd import TrainerKD as Trainer
elif args.method in ['L1', 'GReg-1', 'ASSL']:
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
    global model, checkpoint
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
            elif args.method in ['KD']:
                _model_S = model.Model(args, checkpoint)
                _model_T = set_up_teacher(args, checkpoint, args.T_model, args.T_weights, args.T_n_resblocks, args.T_n_feats)
                _model = [_model_T, _model_S]
            elif args.method in ['L1', 'GReg-1', 'ASSL']:
                _model = model.Model(args, checkpoint)
                class passer: pass
                passer.ckp = checkpoint
                passer.loss = loss.Loss(args, checkpoint) if not args.test_only else None
                passer.loader = loader
                pruner = pruner_dict[args.method].Pruner(_model, args, logger=None, passer=passer)

                # get the statistics of unpruned model
                # height_lr = 1280 // args.scale[0]
                # width_lr  = 720  // args.scale[0]
                # n_params_original_v2 = get_n_params_(_model)
                # n_flops_original_v2 = get_n_flops_(_model, img_size=(height_lr, width_lr), n_channel=3, count_adds=False, idx_scale=args.scale[0])

                _model = pruner.prune() # get the pruned model as initialization for later finetuning
                
                # get the statistics of pruned model and print
                height_lr = 1280 // args.scale[0]
                width_lr  = 720  // args.scale[0]
                dummy_input = torch.zeros((1, 3, height_lr, width_lr)).cuda()

                # temporarily change the print fn
                import builtins, functools
                flops_f = open(checkpoint.get_path('model_complexity.txt'), 'w+')
                original_print = builtins.print
                builtins.print = functools.partial(print, file=flops_f, flush=True)
                summary(_model, dummy_input, {'idx_scale': args.scale[0]})
                builtins.print = original_print

                # @mst: old code for printing FLOPs etc. Deprecated now. Will be removed.
                # n_params_now_v2 = get_n_params_(_model)
                # n_flops_now_v2 = get_n_flops_(_model, img_size=(height_lr, width_lr), n_channel=3, count_adds=False, idx_scale=args.scale[0])
                # checkpoint.write_log_prune("==> n_params_original_v2: {:>7.4f}M, n_flops_original_v2: {:>7.4f}G".format(n_params_original_v2/1e6, n_flops_original_v2/1e9))
                # checkpoint.write_log_prune("==> n_params_now_v2:      {:>7.4f}M, n_flops_now_v2:      {:>7.4f}G".format(n_params_now_v2/1e6, n_flops_now_v2/1e9))
                # ratio_param = (n_params_original_v2 - n_params_now_v2) / n_params_original_v2
                # ratio_flops = (n_flops_original_v2 - n_flops_now_v2) / n_flops_original_v2
                # compression_ratio = 1.0 / (1 - ratio_param)
                # speedup_ratio = 1.0 / (1 - ratio_flops)
                # checkpoint.write_log_prune("==> reduction ratio -- params: {:>5.2f}% (compression {:>.2f}x), flops: {:>5.2f}% (speedup {:>.2f}x)".format(ratio_param*100, compression_ratio, ratio_flops*100, speedup_ratio))
                
                # reset checkpoint and loss
                args.save = args.save + "_FT"
                checkpoint = utility.checkpoint(args)
                
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            
            if args.greg_mode in ['all']: 
                checkpoint.save(t, epoch=0, is_best=False)
                print(f'==> Regularizing all wn_scale parameters done. Checkpoint saved. Exit!')
                import shutil
                shutil.rmtree(checkpoint.dir) # this folder is not really used, so removed here
                exit(0)
            
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
