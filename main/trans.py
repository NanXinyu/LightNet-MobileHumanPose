from pickle import TRUE
from pyexpat import model
from re import I
from tabnanny import verbose
#from x2paddle.convert import pytorch2paddle
import torch
import torch.nn
import onnx
import numpy as np
#import sys
#from model import get_pose_net
from torch.nn.parallel.data_parallel import DataParallel
#from lpnet_ski_concat import LpNetSkiConcat
from collections import  OrderedDict
from model import get_pose_net

#model = LpNetSkiConcat((256, 256), 18)
model = get_pose_net('LPSKI', False, 21)
#model = DataParallel(model).cuda()

ckpt = torch.load("./output/model_dump/snapshot_10.pth.tar")
new_ckpt = OrderedDict()
for key,param in ckpt['network'].items():
    name = key[7:]
    new_ckpt[name] = param
    # if(key.startswith('module.')):
    #     ckpt[key[7:]] = param
    #     ckpt.pop(key)

model .load_state_dict( new_ckpt) 
model.eval()

input_names =  ['input']
output_names = ['output']
#input_data = np.random.rand(1,3,256,256).astype("float32")
#pytorch2paddle(model,save_dir='pd_model_trace',jit_type="trace",input_examples=[torch.tensor(input_data)])

x  = torch.randn(size=(1,3,256,256))#,requires_grad=TRUE)
torch.onnx.export(model,x,'10.onnx',input_names=input_names,output_names=output_names,verbose='True')
