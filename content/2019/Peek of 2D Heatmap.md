---
title: "Peek of 2D Heatmap"
date: 2019-01-14T01:21:36+08:00
categories: ["Snippet"]
tags: ["pytorch", "heatmap", "peek"]
toc: true
math: false
---

{{< highlight py "linenos=table,noclasses=false" >}}
def peek2d(lbl):
    '''
    Args:
        lbl: (FloatTensor) sized [N, 4, H, W]
    Return:
        kpt: (FloatTensor) sized [N, 4, 2]
    '''
    N, _, H, W = lbl.size()
    device = lbl.device
    lbl = lbl.view(N, 4, H * W)
    loc = lbl.argmax(dim=2) # [N, 4]
    yy, xx = loc / W, loc % W # [N, 4], [N, 4]
    kpt = torch.stack((xx, yy), dim=2) # [N, 4, 2]
    kpt = kpt.float() / torch.FloatTensor([W, H]).to(device)
    return kpt
{{< /highlight >}}
