---
title: "Grad-CAM Using Pytorch"
date: 2019-02-26T22:39:40+08:00
categories: ["Snippet"]
tags: ["pytorch", "gradcam"]
toc: true
math: false
---

# 簡介

之前在打一個 regression 的比賽，發現 [Grad-CAM](https://arxiv.org/abs/1610.02391) 是一個很好用的可視化工具。於是我在網路上找了一下 Pytorch 的 implementation，找到 kazuto1011 實現得不錯。只是程式碼有點過於複雜，且只適用於 classificaiton 問題。所以我修改了他的程式碼，並包上 [Context Manager](http://book.pythontips.com/en/latest/context_managers.html)，來讓程式碼更符合我的風格。

# 原理

待補

# 實作

## Grad-CAM

{{< highlight py "linenos=table,noclasses=false" >}}
class GradCam:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.fmap_pool = dict()
        self.grad_pool = dict()

        def forward_hook(module, input, output):
            self.fmap_pool[module] = output.detach().cpu()
        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[module] = grad_out[0].detach().cpu()
        
        for layer in layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
            self.hooks.append(layer.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        assert layer in self.layers, f'{layer} not in {self.layers}'
        fmap_b = self.fmap_pool[layer] # [N, C, fmpH, fmpW]
        grad_b = self.grad_pool[layer] # [N, C, fmpH, fmpW]

        grad_b = F.adaptive_avg_pool2d(grad_b, (1, 1)) # [N, C, 1, 1]
        gcam_b = (fmap_b * grad_b).sum(dim=1, keepdim=True) # [N, 1, fmpH, fmpW]
        gcam_b = F.relu(gcam_b)

        return gcam_b
{{< /highlight >}}

## Guided Backpropogation

{{< highlight py "linenos=table,noclasses=false" >}}
class GuidedBackPropogation:
    def __init__(self, model):
        self.model = model
        self.hooks = []

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return tuple(grad.clamp(min=0.0) for grad in grad_in)

        for name, module in self.model.named_modules():
            self.hooks.append(module.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)
    
    def get(self, layer):
        return layer.grad.cpu()
{{< /highlight >}}

## Utility

{{< highlight py "linenos=table,noclasses=false" >}}
def colorize(tensor, colormap=plt.cm.jet):
    '''Apply colormap to tensor
    Args:
        tensor: (FloatTensor), sized [N, 1, H, W]
        colormap: (plt.cm.*)
    Return:
        tensor: (FloatTensor), sized [N, 3, H, W]
    '''
    tensor = tensor.clamp(min=0.0)
    tensor = tensor.squeeze(dim=1).numpy() # [N, H, W]
    tensor = colormap(tensor)[..., :3] # [N, H, W, 3]
    tensor = torch.from_numpy(tensor).float()
    tensor = tensor.permute(0, 3, 1, 2) # [N, 3, H, W]
    return tensor

def normalize(tensor, eps=1e-8):
    '''Normalize each tensor in mini-batch like Min-Max Scaler
    Args:
        tensor: (FloatTensor), sized [N, C, H, W]
    Return:
        tensor: (FloatTensor) ranged [0, 1], sized [N, C, H, W]
    '''
    N = tensor.size(0)
    min_val = tensor.contiguous().view(N, -1).min(dim=1)[0]
    tensor = tensor - min_val.view(N, 1, 1, 1)
    max_val = tensor.contiguous().view(N, -1).max(dim=1)[0]
    tensor = tensor / (max_val + eps).view(N, 1, 1, 1)
    return tensor
{{< /highlight >}}

## Example

{{< highlight py "linenos=table,noclasses=false" >}}
import random

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision.models import densenet121
from torchvision.transforms import functional as tf

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')
model = densenet121(pretrained=True).to(device)
model.eval()

img = Image.open('./samples/cat_dog.png')
img = img.convert('RGB').resize((224, 224))
img = tf.to_tensor(img) # [3, 224, 224]
img = tf.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inp_b = img.unsqueeze(dim=0) # [N, 3, 224, 224]
inp_b = inp_b.to(device)

# 243: boxer
# 283: tiger cat
# grad_b = torch.zeros_like(out_b, device=device)
# grad_b[:, out_b.argmax(dim=1)] = +1.0
# out_b.backward(gradient=grad_b)

with GradCam(model, [model.features]) as gcam:
    out_b = gcam(inp_b) # [N, C]
    out_b[:, 283].backward()

    gcam_b = gcam.get(model.features) # [N, 1, fmpH, fmpW]
    gcam_b = F.interpolate(gcam_b, [224, 224], mode='bilinear', align_corners=False) # [N, 1, inpH, inpW]
    save_image(normalize(gcam_b), './gcam.png')


with GuidedBackPropogation(model) as gdbp:
    inp_b = inp_b.requires_grad_() # Enable recording inp_b's gradient
    out_b = gdbp(inp_b)
    out_b[:, 283].backward()

    grad_b = gdbp.get(inp_b) # [N, 3, inpH, inpW]
    grad_b = grad_b.mean(dim=1, keepdim=True) # [N, 1, inpH, inpW]
    save_image(normalize(grad_b), './grad.png')


mixed = gcam_b * grad_b
mixed = normalize(mixed)
save_image(mixed, './mixed.png')
{{< /highlight >}}

圖片待補