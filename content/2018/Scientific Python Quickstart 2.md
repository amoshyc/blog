---
title: "Python 科學計算快速入門 2/2"
date: 2018-03-21T11:38:58+08:00
categories: [教學, Python 科學計算快速入門]
tags: []
toc: true
math: false
---

# Numpy

Tensor on CPU~
```
pip install numpy
conda install -c anaconda numpy -n <venv>
```

{{< highlight py "linenos=inline,noclasses=false" >}}
import numpy as np
x = np.zeros((10, 10), dtype=np.float32)
print(x.shape) # (10, 10)
print(x.dtype) # dtype('float32')

''' Other dtypes:
np.bool
np.uint8
np.int32
'''
{{< /highlight >}}

## Creation

{{< highlight py "linenos=inline,noclasses=false" >}}
x = np.array([1, 2, 3])
x = np.uint8([1, 2, 3])

x = np.arange(0.0, 10.0, 2) # array([0., 2., 4., 6., 8.])
x = np.zeros((10, 10, 10), dtype=np.float32)
x = np.ones((10, 10, 10), dtype=np.float32)
{{< /highlight >}}

{{% admonition title="Hint!" color="blue" %}}
Python 習慣上是 Row Major，所以每一筆資料是一個 row，這有別於數學習慣或 Matlab
{{% /admonition %}}

所以 10 張 256x256 的 RGB 資料會放成：
{{< highlight py "linenos=inline,noclasses=false" >}}
xs = np.zeros((10, 256, 256, 3), dtype=np.uint8)
{{< /highlight >}}


## Arithmetics

ND-array 有對運算子重載，預設都是 element-wise。

{{< highlight py "linenos=inline,noclasses=false" >}}
A = np.arange(6).reshape(2, 3)
B = -A
C = A.T
d = A.flatten() # or A.ravel()
A + B
A - 2
A * B
A / B
A @ C # matrix multiplication
{{< /highlight >}}

## Indexing

跟 python 內建的 slice 不同，numpy 一般情況下是不會回傳 copy，回傳的是 view。

{{< highlight py "linenos=inline,noclasses=false" >}}
A = np.arange(12).reshape(4, 3)
A[1:3, 0:2]
A[[0, 2, 3], [0, 1, 2]] = [1, 2, 3]
{{< /highlight >}}

{{< highlight py "linenos=inline,noclasses=false" >}}
c1 = (A > 5)        # ndarray of bool
c2 = (A == 1)       # ndarray of bool

~c1                 # not
c1 & c2             # and
c1 | c2             # or

A[c1]
A[A > 5]
A[c1 | c2] = -1
{{< /highlight >}}

## Broadcasting

{{< highlight py "linenos=inline,noclasses=false" >}}
A = np.arange(12).reshape(4, 3)
b = np.arange(4).reshape(4, 1)
c = np.arange(3).reshape(1, 3)

A + b
A + c
A - 2
{{< /highlight >}}

## Functions

axis：沿著哪一個軸做運算，可以是負數。預設情況下這個軸會消失掉。

{{< highlight py "linenos=inline,noclasses=false" >}}
A = np.arange(6).reshape(2, 3)
np.sum(A)           # 預設是 element-wise
np.sum(A, axis=0)   # 沿第 0 軸，即 row
np.sum(A, axis=1)   # 沿第 1 軸，即 col
{{< /highlight >}}

{{< highlight py "linenos=inline,noclasses=false" >}}
np.mean()
np.sum()
np.sort()
np.max()
np.min()
np.argsort()
np.amax()
np.amin()
np.nonzero()
np.where()
np.transpose()
np.broadcast_to()
{{< /highlight >}}


# Pillow (fork of PIL)

The basic image io library

```
pip install pillow
conda install -c anaconda pillow -n <venv>
```

{{< highlight py "linenos=inline,noclasses=false" >}}
from PIL import Image
img = Image.open('./test.png')
img = img.convert('RGB')
img = img.resize((256, 256)
img.save('./out.jpg')
{{< /highlight >}}

{{< highlight py "linenos=inline,noclasses=false" >}}
from PIL import Image
img = Image.open('./test.png')
img = img.convert('RGB')
img = img.resize((256, 256))

x = np.array(img)
print(x.shape)              # (256, 256, 3)
print(x.dtype)              # np.uint8
print(x.min(), x.max())     # 0, 255

img = Image.fromarray(x)    # x should be np.uint8
img.save(...)
{{< /highlight >}}


# Matplotlib

Matplotlib 是 Python 社群中最主流的繪圖 library，但他的學習曲線比較陡陗。不管是顯示圖片，還是可視化各種統計資料，都有對應的函式。

```
pip install matplotlib
conda install -c anaconda matplotlib -n <venv>
```

預設使用 tkinter 作為 backend，如果要搭配 Jupyter 使用，可以使用 Jupyter 的 Magic: `%matplotlib inline`
兩套通行的 API，一套模彷 Matlab，一套是 OOP，我推薦使用後者。
關於這個的討論可以參考 [這篇文章](http://pbpython.com/effective-matplotlib.html) 與在 Hacker News 上引起的 [討論](https://news.ycombinator.com/item?id=14668706)。

{{< highlight py "linenos=inline,noclasses=false" >}}
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn') # since default style is ugly
from PIL import Image

img_rgb = Image.open('./rin.png')
img_gray = img_rgb.convert('L')
x_rgb = np.array(img_rgb)
x_gray = np.array(img_gray)

fig, axes = plt.subplots(nrows=1, ncols=2, dpi=150)
axes[0].imshow(x_rgb) # or img_rgb
axes[0].axis('off')
axes[0].set_title('RGB')
axes[1].imshow(x_gray, cmap='gray')
axes[1].axis('off')   # or img_gray
axes[1].set_title('Gray')
fig.tight_layout()
plt.show()
{{< /highlight >}}

![Imgur](https://i.imgur.com/loarbbB.png)


# Scikit-Image

提供了許多 Computer Vision 相關的函式，包含 Sobel, CLAHE, Contour, Snake, SIFT, Watershed, etc. [官方範例](http://scikit-image.org/docs/stable/auto_examples/index.html)。提供了一套簡單的讀寫圖片的 API。另外，也有針對在 ndarray 上繪製幾何圖形的函式。

```
pip install scikit-image
conda install -c anaconda scikit-image -n <venv>
```

{{< highlight py "linenos=inline,noclasses=false" >}}
import matplotlib.pyplot as plt
plt.style.use('seaborn') # since default style is ugly

from skimage import io
from skimage import color
from skimage import filters
from skimage import feature

img_rgb = io.imread('./rin.png') # ndarray
img_gray = color.rgb2gray(img_rgb)
img_sobel = filters.sobel(img_gray)
img_canny = feature.canny(img_gray, sigma=3)

fig, axes = plt.subplots(nrows=2, ncols=2, dpi=150)
axes[0, 0].imshow(img_rgb)
axes[0, 1].imshow(img_gray, cmap='gray')
axes[1, 0].imshow(img_sobel, cmap='jet')
axes[1, 1].imshow(img_canny, cmap='inferno')
for r in range(2):
    for c in range(2):
        axes[r, c].axis('off')
fig.tight_layout()
plt.show()
{{< /highlight >}}

![Imgur](https://i.imgur.com/vkEglfX.png)

# OpenCV

OpenCV 的 Python Wrapper，速度比前面的函式庫快一些，但 API 比較不 pythonic，仍然有許多人喜歡用這個。相對之下，他的文檔比較少，且有歷史包袱：圖片讀進來的格式是 BGR 而不是 RGB，如果該圖片要丟到其他 python 的函式庫，要先轉成 RGB。

```
pip install opencv-python
conda install -c anaconda opencv-python -n <venv>
```

{{< highlight py "linenos=inline,noclasses=false" >}}
import cv2

img = cv2.imread(...) # ndarray, BGR
img = img[..., ::-1]  # 反轉 color channel，即變成 RGB 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

{{< /highlight >}}

# Scikt-learn

實作了非常多的 Learning Algorithm，常見的演算法除了 DNN 以外，都在這找得到。例如 SVM, Cross Validation, Random Forest, K-Means, PCA, etc。提供了統一的界面，非常方便同時測多個 model。很可惜只能跑在 CPU 上，官方範例中繪圖使用 matplotlib。就算是 DNN 的專案，我們仍然會使用 scikit-learn 底下的函式，例如 `train_test_split`, `GridSearchCV`, etc。文檔非常詳盡。

```
pip install scikit-learn
conda install -c anaconda scikit-learn -n <venv>
```

{{< highlight py "linenos=inline,noclasses=false" >}}
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print(X.shape)      # (150, 4): 150 samples, each with 4 features

pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)

print(X_pca.shape)  # (150, 2)

fig, ax = plt.subplots(dpi=150)
for i in range(3):
    ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_names[i])
ax.legend()
ax.set_title('PCA of iris dataset')
fig.tight_layout()
plt.show()
{{< /highlight >}}

![Imgur](https://i.imgur.com/YZkATZc.png)


# Pathlib

處理路徑的函式庫，是標準函式庫的一部份，不需要裝，但只支援 python 3.4+。仍然是一個新興的函式庫，使用的人不多，因為許多函式庫想同時支援各個版本的 python。

{{< highlight py "linenos=inline,noclasses=false" >}}
>>> from pathlib import Path
>>> p = Path('.')
>>> p
PosixPath('.')
>>> p = p.resolve()
>>> p
PosixPath('/home/amoshyc/workspace/test')
>>> q = (p / 'main.py')
>>> q.exists()
True
>>> q = p.parent
>>> q
PosixPath('/home/amoshyc/workspace')
>>> (q / 'ckpt').mkdir() # /home/amoshyc/workspace/ckpt/
>>> str(q)
'/home/amoshyc/workspace'
>>> q.glob('**/*.py')
<generator object Path.glob at 0x7f111eec3ca8>
>>> sorted(list(q.glob('**/*.py')))[:3]
[PosixPath('/home/amoshyc/workspace/CPsolution/source/conf.py'), 
PosixPath('/home/amoshyc/workspace/CPsolution/venv/bin/rst2html.py'), 
PosixPath('/home/amoshyc/workspace/CPsolution/venv/bin/rst2html4.py')]
{{< /highlight >}}


# Homework

