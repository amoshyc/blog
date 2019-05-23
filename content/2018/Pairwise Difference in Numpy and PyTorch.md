---
title: "Pairwise Difference in Numpy and PyTorch"
date: 2018-10-03T15:13:10+08:00
categories: ["Snippet"]
tags: ["difference", "pairwise", "pytorch", "numpy"]
toc: true
math: false
---

# 問題

給定維度 `[N, D]` 的矩陣 `A`，與 `[M, D]` 的矩陣 `B`，輸出維度 `[N, M, D]` 的矩陣 `C`，其中 `C[i, j]` 代表向量 `A[i]` 與向量 `B[j]` 的差，即 `A[i] - B[j]`。

# 矩陣化

最直覺的做法，兩個 `for` 迭代一下，簡單：

{{< highlight python "linenos=table,noclasses=false" >}}
for i, a in enumerate(A):
    for j, b in enumerate(B):
        C[i, j] = a - b
{{< / highlight >}}

但這個的計算速度非常非常慢，我們可以將這個計算改寫成矩陣的型式，讓 Numpy/Pytorch 可以利用 SIMD/GPU 來加速計算。

我們想讓結果是兩個矩陣 `C1`, `C2` 相減，即 `C = C1 - C2`，由此避掉大量的 indexing。其中，`C1`, `C2` 的維度都是 `[N, M, D]`。至此問題變成 `C1`, `C2` 分別是什麼。畫一下圖，發揮一下空間感，可以發現：

1. `C1` 是 `A` 的所有向量排在第 `0, 2` 個維度後，往第 `2` 個維度複製 `M` 次。
2. `C2` 是 `B` 的所有向量排在第 `1, 2` 個維度後，往第 `0` 個維度複製 `N` 次。

# Numpy 實現

利用 `np.expand_dims`, `np.broadcast_to`，程式碼可以寫成：

{{< highlight python "linenos=table,noclasses=false" >}}
import numpy as np

N, M, D = 8, 6, 3
A = np.random.rand(N, D)
B = np.random.rand(M, D)

C1 = np.broadcast_to(np.expand_dims(A, 1), (N, M, D))
C2 = np.broadcast_to(np.expand_dims(B, 0), (N, M, D))
C = C1 - C2
{{< / highlight >}}

# Pytorch 實現

類似於 Numpy 的實現，Pytorch 使用 `unsqueeze`, `expand` 來實現。不過 Pytorch 習慣將 Channel 放在第 0 個維度，所以細節有所不同：

{{< highlight python "linenos=table,noclasses=false" >}}
import torch

N, M, D = 8, 6, 3
device = torch.device('cuda:0') # or 'cpu'

A = torch.rand(D, N, device=device)
B = torch.rand(D, M, device=device)

C1 = A.unsqueeze(2).expand(D, N, M)
C2 = B.unsqueeze(1).expand(D, N, M)
C = C1 - C2
{{< / highlight >}}

# IOU

同樣的方法可以類推至 pairwise IOU 的計算：


{{< highlight python "linenos=table,noclasses=false" >}}
def pairwise_iou(A, B):
    '''
    Args
        A: (FloatTensor) first set of boxes in xyxy format, sized [N, 4]
        B: (FloatTensor) second set of boxes in xyxy format, sized [M, 4]
    Return
        C: (FloatTensor) C[i, j] is the iou of A[i] and B[j], sized [N, M]
    '''
    N, M = A.size(0), B.size(0)
    A = A.unsqueeze(1).expand(N, M, 4)
    B = B.unsqueeze(0).expand(N, M, 4)

    Ix = torch.min(A[..., 2], B[..., 2]) - torch.max(A[..., 0], B[..., 0])
    Iy = torch.min(A[..., 3], B[..., 3]) - torch.max(A[..., 1], B[..., 1])
    I = Ix.clamp(min=0) * Iy.clamp(min=0)
    Aa = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    Ab = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
    U = Aa + Ab - I

    return I / U
{{< / highlight >}}