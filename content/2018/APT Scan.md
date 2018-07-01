---
title: "APT Scan: Image Correction"
date: 2018-06-30T23:33:36+08:00
categories: ["Side Project"]
tags: []
toc: true
math: true
---

# Abstract

# Symbols

# Method

APT-Scan 的流程如下：

1. 找出 $$I\_(src)$$ 中目標的 4 個頂點。
2. 估計出轉正後新圖 $$I\_(dst)$$ 的大小 $$W = (T + B)//2, H=(L + R)//2$$，其中 $$T, B, L, R$$ 為 $$I\_(src)$$ 中目標四條邊的邊長。
3. 目標的 4 個頂點轉正後的座標一定會是新圖的 4 個角落，即他們在 $$C\_(dst)$$ 的座標為 $$(0, 0), (W, 0), (W, H), (0, H)$$。
4. 解出 $$C\_(src)$$ 與 $$C\_(dst)$$ 之間的關係：Homography Matrix $$H$$。
5. 使用 Warp 將 $$I\_(src)$$ 轉成 $$I\_(dst)$$。

# Detail

## Corners Detection

## Homography

我們會用到簡單的齊次座標，在此簡介一下，簡單來講齊次座標是為了延遲除法的運算所使用的座標表示法。至於為什麼我們要延遲除法呢，這是因為除法的運算成本相對於加減乘的運算是昂貴的，使用齊次座標可以把許多需要除法的運算轉成矩陣相剩乘，在當代的架構上可以有效的加速運算。

在齊次座標中，一個 2D 直角座標的點 $$(x, y)$$ 會用 3 個數來表示。如果在齊次座標中看到向量 $$(a, b, c)^T$$，那他實際是指直角座標中的 $$(a//c, b//c)$$。而這會造成一個有趣的事實：直角座標中任一點都會有不只一種的寫法，例如直角座標的點 $$(2, 3)$$ 用齊次座標可以寫成 $$(2, 3, 1)^T$$ 也可以寫成 $$(4, 6, 2)^T$$。

----

電腦視覺告訴我們座標從 $$C\_(src)$$ 轉換到 $$ C\_(dst) $$ 是一個 Homography Transform(=Projective Transform=Perspective Transform)，而且這個 Transform 在齊次座標下是一個線性變換。更詳細地說，對 $$C\_(src)$$ 中的任一點 $$(x\_(src),y\_(src))$$ 與 $$C\_(dst)$$ 中的**對應點** $$(x\_(dst), y\_(dst))$$ 有以下關係式：

{{< am >}}
((x_(dst)), (y_(dst)), (1))
=
(
    (h_(11), h_(12), h_(13)),
    (h_(21), h_(22), h_(23)),
    (h_(31), h_(32), 1)
)
((x_(src)), (y_(src)), (1))
{{< /am >}}

其中，中間那個矩陣就是 Homography Matrix $$H$$。我們的目標就是求出這個矩陣，共 8 個未知數。就讓我們先將右式乘開吧：

{{< am >}}
((x_(dst)), (y_(dst)), (1))
=
(
    (h_(11) x_(src) + h_(12) y_(src) + h_(13)),
    (h_(21) x_(src) + h_(22) y_(src) + h_(23)),
    (h_(31) x_(src) + h_(32) y_(src) + 1)
)
{{< /am >}}

依據齊次座標的特性，我們可以得到：

{{< am >}}
x_(dst) = (h_(11) x_(src) + h_(12) y_(src) + h_(13)) / (h_(31) x_(src) + h_(32) y_(src) + 1) \n
y_(dst) = (h_(21) x_(src) + h_(22) y_(src) + h_(23)) / (h_(31) x_(src) + h_(32) y_(src) + 1)
{{< /am >}}

整理可得：

{{< am >}}
h_(11) x_(src) + h_(12) y_(src) +
h_(13) - h_(31) x_(src) x_(dst) -
h_(32) y_(src) x_(dst) = x_(dst) \n

h_(21) x_(src) + h_(22) y_(src) +
h_(23) - h_(31) x_(src) y_(dst) -
h_(32) y_(src) y_(dst) = y_(dst)
{{< /am >}}

寫成針對未知數的矩陣型式：

{{< am >}}
(
    (x_(src), y_(src), 1, 0, 0, 0, -x_(src) x_(dst), -y_(src) x_(dst)),
    (0, 0, 0, x_(src), y_(src), 1, -x_(src) y_(dst), -y_(src) y_(dst)),
    ( ,  ,  ,  ,  ,  ,  ,  ),
    ( ,  ,  ,  ,  ,  ,  ,  ),
    ( ,  ,  ,  ,  ,  ,  ,  ),
    ( ,  ,  ,  ,  ,  ,  ,  ),
    ( ,  ,  ,  ,  ,  ,  ,  ),
    ( ,  ,  ,  ,  ,  ,  ,  )
)
((h_(11)), (h_(12)), (h_(13)), (h_(21)), (h_(22)), (h_(23)), (h_(31)), (h_(32)))
=
(
    (x_(dst)), (y_(dst)), ( ), ( ), ( ), ( ), ( ), ( )
)
{{< /am >}}

一組對應點 $$(x\_(src), y\_(src)) (x\_(dst), y\_(dst))$$ 我們可以得到二條關於未知數的方程式。也就是說我們**只需要四組對應點就可以解出所有的未知數**；另外如果你有四組以上的對應點，那這個方程還是可以利用最小平方法解出。

## Warp

當我們得到兩個座標系 $$C\_(src), C\_(dst)$$ 後，我們就可以使用 Warp 將 $$I\_(src)$$ 轉成 $$I\_(dst)$$。Warp 操作基本就是給定座標之間的關係函式 $$f:C\_(dst) -> C\_(src)$$，說明結果圖片中的每個 pixel 對應到原圖中哪個 pixel，然後針對結果圖片中的每個 pixel，把對應的原圖位置的顏色複製過來。底下給出 pseudo code：

{{< highlight python "noclasses=false" >}}
res = np.zeros((H, W, 3))
for r in range(H):
    for j in range(W):
        src_r, src_c = f(r, c)
        res[i, j] = inp[src_r, src_c]
{{< /highlight >}}

在這個專案中，$$f$$ 就是我們前面求出的 Homography Matrix 的反矩陣 $$H^(-1)$$，$$H^(-1)$$ 將 $$C\_(dst)$$ 的點轉到了 $$C\_(src)$$ 中。

## 實作

然後你就發現 scikit-image 直接內建上述的操作，我們只需要找出 4 組對應點，剩下的事 scikit-image 都幫你搞定了（OpenCV 也內建，不過我比較愛用 scikit-image），scikit-image 的範例可以看 [這裡](http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_geometric.html#sphx-glr-auto-examples-xx-applications-plot-geometric-py)。其中要注意的是他範例中的變數名稱跟本文是反過來的。本文用 src 指原圖，而 dst 指轉正後的結果，與範例相反。我猜這是想減少最後 warp 時所須的反矩陣運算吧。
