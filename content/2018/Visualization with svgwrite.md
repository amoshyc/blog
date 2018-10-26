---
title: "使用 Svgwrite 來可視化圖片"
date: 2018-10-21T00:27:00+08:00
categories: ["Snippet"]
tags: ["svg", "visualization"]
toc: true
math: false
---

# Introduction

許多 CV 問題都需要可視化，不管是可視化原始的資料，還是模型的預測結果，可視化總讓人能以直觀的方法了解你 code 有沒有寫錯，或模型是不是有什麼弱點。而有一大類的可視化不是生成圖表，而是必須在圖上標記，例如 Human Pose Estimation 與 Object Detection。

目前大家的做法是利用 OpenCV, Pillow 或 Skimage 直接把 edge, keypoint, text 畫在該圖片上，即直接修改該圖片的 pixel。也有不少人是用 matplotlib 來做可視化。但這兩種方法我都不太喜歡，後者是輸出的圖片會有各種留白，一直找不到方法把他們全部去掉；前者則有失真的問題：當你把圖片放大時，你會發現可視化的部份會變得模糊，線條尤為明顯，如下圖：

TODO

另外還有一些原因，上述的函式庫不容易畫出理想的圖形，例如 Skimage 畫不少太小的圓形、Pillow 沒法指定矩形的邊的粗細等等。為了解決這些問題（沒辦法，我就是受不了），我決定使用 svgwrite 來做這種「**在圖片上標記**」的可視化 [^1]，也就是說我將圖片內嵌到 SVG 中，然後再加入一些 SVG 的 element，例如 Circle, Rect, Text 等來做可視化 [^2]。

SVG 有著許多優點，例如他是向量圖形所以可視化部份不會有失真的問題，而且他內建的那個 element 有著許多參數可以調整，舉凡顏色、粗細等他都有著類似於 CSS 的一致 API，也因此，他可以使用半透明顏色。另外 SVG 內部可以使用 `<g></g>` 來任意嵌套，這在疊圖之類的事情還挺方便的。

這篇文的目的要記錄一些輔助我可視化函式，以防每次我都要重新想重新打，同時也給其他人參考。目前我是傾向使用 function-based 的設計而非 OOP，這樣在使用上會比較方便，這些函式應該全部放在同一個檔案，例如 `svg.py` 中，然後 `import svg` 來使用。

# Basic

Hint:

1. `g` 可以任意嵌套
2. 使用 `x, y` 而不是 `r, c`，原點在左上

{{< highlight python "linenos=table,noclasses=false" >}}
from math import ceil
from io import BytesIO
from base64 import b64encode

import numpy as np
from PIL import Image
import svgwrite as sw


def g(elems):
    '''
    '''
    g = sw.container.Group()
    for elem in elems:
        g.add(elem)
    return g


def pil(img):
    '''
    '''
    w, h = img.size
    buf = BytesIO()
    img.save(buf, 'png')
    b64 = b64encode(buf.getvalue()).decode()
    href = 'data:image/png;base64,' + b64
    elem = sw.image.Image(href, (0, 0), width=w, height=h)
    return elem


def img(img):
    '''
    '''
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    return pil(img)


def save(elems, fname, size, per_row=-1, padding=2, pad_val=None):
    '''
    '''
    n_elem = len(elems)
    elems = [g.copy() for g in elems]
    imgH, imgW = size
    per_row = n_elem if per_row == -1 else per_row
    per_col = ceil(n_elem / per_row)
    gridW = per_row * imgW + (per_row - 1) * padding
    gridH = per_col * imgH + (per_col - 1) * padding

    svg = sw.Drawing(size=[gridW, gridH])
    if pad_val:
        svg.add(sw.shapes.Rect((0, 0), (gridW, gridH), fill=pad_value))
    for i in range(n_elem):
        c = (i % per_row) * (imgW + padding)
        r = (i // per_row) * (imgH + padding)
        elems[i].translate(c, r)
        svg.add(elems[i])

    with open(str(fname), 'w') as f:
        svg.write(f, pretty=True)


def to_png(src_path, dst_path, scale=2):
    '''
    '''
    import cairosvg
    pass

########################################

def bboxs(bboxes, c='red', **extra):
    def transform(bbox):
        x, y, w, h = bbox
        args = {
            'insert': (round(x), round(y)),
            'size': (round(w), round(h)),
            'stroke': c,
            'stroke_width': 2,
            'fill_opacity': 0.0
        }
        return sw.shapes.Rect(**args)
    return g([transform(bbox) for bbox in bboxes])

{{< /highlight >}}


# Keypoint

# Object Detection

# Segmentation


[^1]: 如果只是要畫畫圖表，我還是使用 matplotlib （記得指定 style 為 seaborn）然後輸出成 SVG（是的 matplotlib 可以輸出 svg，如果你還不知道的話記得嘗試看看）。
[^2]: 如果是無法使用 SVG 的場合，例如 Facebook, Imgur, Google Docs，我會先用 cairosvg 將 SVG 轉成 PNG（scale 為 2），再做處理。
