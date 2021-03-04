from torchsummaryX import summary
from importlib import import_module
import torch
import model
from model import edsr
from option import args
import sys
# checkpoint = utility.checkpoint(args)
# my_model = model.Model(args)

Net = import_module('model.' + args.model.lower())
net = eval("Net.%s" % args.model.upper())(args).cuda()

# net = edsr.EDSR(args)
height_lr = 1280 // args.scale[0]
width_lr  = 720  // args.scale[0]
input = torch.zeros((1, 3, height_lr, width_lr)).cuda()

summary(net, input)
