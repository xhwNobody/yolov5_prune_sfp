import sys,os
sys.path.append('../')
import torch
model = torch.load('best.pt', map_location='cpu')['model'].float()
#print(model.keys())
model.eval()
input_shape = torch.rand([1, 3, 640, 640])
input_shape = input_shape#.cuda()
model(input_shape)
torch.jit.trace(model, input_shape).save('best_cpu.torchscript.pt')

