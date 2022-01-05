import sys,os
sys.path.append('../')
import torch

model = torch.load('best.pt', map_location='cuda')['model'].float()
#print(model.keys())
model.eval()
input_shape = torch.rand([1, 3, 640, 640])
input_shape = input_shape.cuda()
model(input_shape)
torch.jit.trace(model, input_shape).save('best.torchscript.pt')

model = torch.load('small_model_all.pt', map_location='cuda')['model'].float()
#print(model.keys())
model.eval()
input_shape = torch.rand([1, 3, 640, 640])
input_shape = input_shape.cuda()
model(input_shape)
torch.jit.trace(model, input_shape).save('small_model_all.torchscript.pt')


