---
title: "Numpy 中實現二維高斯分佈"
date: 2018-03-03T15:46:53+08:00
categories: ["Snippet"]
tags: ["2d", "gaussian", "numpy", "normal"]
toc: true
math: true
---

# 前情

最近讀了 Pose Estimation 相關的論文，發現一些 Bottom Up 的方法 [^1] [^2] 會直接生成各個 Keypoints 會哪，中間不經過 RPN 等方法。而生成的 Confidence Map 的 Ground Truth 是使用高斯分佈 (Gaussian Distribution) 來指示位置。但我翻了一下文檔，`numpy` 似乎沒有提供**生成**二維高斯分佈的函式，只提供從高斯分佈**取樣**的函式，於是我模彷了 `skimage.draw` 的 API，寫了一個函式。

[^1]: [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050)
[^2]: [Associative Embedding: End-to-End Learning for Joint Detection and Grouping](https://arxiv.org/abs/1611.05424)

# 實作原理

二維高斯分佈就是兩個一維高斯分佈取[外積](https://en.wikipedia.org/wiki/Outer_product#Definition_(matrix_multiplication))。於是我分別對 row 與 col 各生成一個高斯分佈，函數 domain 為 $$ [sigma-3sigma, sigma+3sigma] $$，因為是整數域，共 $$ 6sigma + 1 $$ 個值。然後將這兩個分佈使用 `np.outer` 即為所求。

# 程式碼

{{% admonition title="Hint!" color="blue" %}}
此函式不支援 double 型態的 mu 與 sigma！
{{% /admonition %}}

{{< highlight python "linenos=table,noclasses=false" >}}
def gaussian2d(mu, sigma, shape=None):
    """Generate 2d gaussian distribution coordinates and values.

    Parameters
    --------------
    mu: tuple of int
        Coordinates of center, (mu_r, mu_c)
    sigma: tuple of int
        Intensity of the distribution, (sigma_r, sigma_c)
    shape: tuple of int, optional
        Image shape which is used to determine the maximum extent
        pixel coordinates, (r, c)

    Returns
    --------------
    rr, cc: (N,) ndarray of int
        Indices of pixels that belong to the distribution
    gaussian: (N, ) ndarray of float
        Values of corresponding position. Ranges from 0.0 to 1.0.

    .. warning::

        This function does NOT support mu, sigma as double.
    """
    (r, c), (sr, sc), (H, W) = mu, sigma, shape
    rr = np.arange(r - 3 * sr, r + 3 * sr + 1)
    cc = np.arange(c - 3 * sc, c + 3 * sc + 1)
    rr = rr[(rr >= 0) & (rr < H)]
    cc = cc[(cc >= 0) & (cc < W)]
    gr = np.exp(-0.5 * ((rr - r) / sr)**2) / (np.sqrt(2 * np.pi) * sr)
    gc = np.exp(-0.5 * ((cc - c) / sc)**2) / (np.sqrt(2 * np.pi) * sc)
    g = np.outer(gr, gc).ravel()
    rr, cc = np.meshgrid(rr, cc)
    rr = rr.ravel()
    cc = cc.ravel()
    return rr, cc, g
{{< /highlight >}}


# 範例

{{< highlight python "noclasses=false" >}}
import numpy as np
from PIL import Image

img = np.zeros((100, 100), dtype=np.float32)
rr, cc, g = gaussian2d([50, 50], [3, 3], shape=img.shape)
img[rr, cc] = np.maximum(img[rr, cc], g / g.max())
rr, cc, g = gaussian2d([55, 55], [3, 3], shape=img.shape)
img[rr, cc] = np.maximum(img[rr, cc], g / g.max())
rr, cc, g = gaussian2d([20, 20], [3, 3], shape=img.shape)
img[rr, cc] = np.maximum(img[rr, cc], g / g.max())

# Save Image
img = np.uint8(img * 255)
Image.fromarray(img).save('./out.jpg')
{{< /highlight >}}
{{< figure src="https://i.imgur.com/ix9ugHS.jpg" width="500">}}

其中要注意的是函式的值可能太小（例如 `sigma=1` 時，函式值最大為 0.5），可以考慮將之調整。例如上段程式碼就是將每個高斯分佈的最大值縮放成 1。


# Pytorch

{{< highlight python "linenos=table,noclasses=false" >}}

def gaussian2d(mu, sigma, shape):
    (r, c), (sr, sc), (H, W) = mu, sigma, shape
    pi = torch.tensor(math.pi)
    rr = torch.arange(r - 3 * sr, r + 3 * sr + 1).float()
    cc = torch.arange(c - 3 * sc, c + 3 * sc + 1).float()
    rr = rr[(rr >= 0) & (rr < H)]
    cc = cc[(cc >= 0) & (cc < W)]
    gr = torch.exp(-0.5 * ((rr - r) / sr)**2) / (torch.sqrt(2 * pi) * sr)
    gc = torch.exp(-0.5 * ((cc - c) / sc)**2) / (torch.sqrt(2 * pi) * sc)
    g = torch.ger(gr, gc).view(-1)
    rr, cc = torch.meshgrid(rr.long(), cc.long())
    rr = rr.contiguous().view(-1)
    cc = cc.contiguous().view(-1)
    return rr, cc, g
{{< /highlight >}}
