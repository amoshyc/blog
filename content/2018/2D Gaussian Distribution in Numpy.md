---
title: "Numpy 中二維高斯分佈"
date: 2018-03-03T15:46:53+08:00
categories: ["Snippet"]
tags: ["2d", "gaussian", "numpy", "normal"]
toc: true
math: true
---

# 前提提要

最近讀了 Pose Estimation 相關的論文，發現一些 Bottom Up 的方法 [^1] [^2] 會直接生成各個 Keypoints 會哪，中間不經過 RPN 等方法。而生成的 Confidence Map 的 Ground Truth 是使用高斯分佈來指示位置。但我翻了一下文檔，`numpy` 似乎沒有提供**生成**二維高斯分佈的函式，只提供從高斯分佈**取樣**的函式，於是我模彷了 `skimage.draw` 的 API，寫了一個函式。

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
    mu_r, mu_c = mu
    sigma_r, sigma_c = sigma

    R, C = 6 * sigma_r + 1, 6 * sigma_c + 1
    r = np.arange(-3 * sigma_r, +3 * sigma_r + 1) + mu_r
    c = np.arange(-3 * sigma_c, +3 * sigma_c + 1) + mu_c
    if shape:
        r = np.clip(r, 0, shape[0] - 1)
        c = np.clip(c, 0, shape[1] - 1)

    coef_r = 1 / (sigma_r * np.sqrt(2 * np.pi))
    coef_c = 1 / (sigma_c * np.sqrt(2 * np.pi))
    exp_r = -1 / (2 * (sigma_r**2)) * ((r - mu_r)**2)
    exp_c = -1 / (2 * (sigma_c**2)) * ((c - mu_c)**2)

    gr = coef_r * np.exp(exp_r)
    gc = coef_c * np.exp(exp_c)
    g = np.outer(gr, gc)

    r = r.reshape(R, 1)
    c = c.reshape(1, C)
    rr = np.broadcast_to(r, (R, C))
    cc = np.broadcast_to(c, (R, C))
    return rr.ravel(), cc.ravel(), g.ravel()
{{< /highlight >}}

# 範例

{{< highlight python "noclasses=false" >}}
import numpy as np
from PIL import Image

img = np.zeros((100, 100), dtype=np.float32)
rr, cc, g = gaussian2d([50, 50], [4, 4], shape=img.shape)
img[rr, cc] = g / g.max()
rr, cc, g = gaussian2d([20, 40], [5, 2], shape=img.shape)
img[rr, cc] = g / g.max()
rr, cc, g = gaussian2d([0, 0], [1, 3], shape=img.shape)
img[rr, cc] = g / g.max()

# Save Image
img = np.uint8(img * 255)
Image.fromarray(img).save('./out.jpg')
{{< /highlight >}}
{{< figure src="https://i.imgur.com/uz8pX8l.jpg" width="400" >}}

其中要注意的是函式的值可能太小（例如 `sigma=1` 時，函式值最大為 0.5），可以考慮將之 normalize，如程式碼所示。