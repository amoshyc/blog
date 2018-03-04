---
title: "在 Fedora 25 安裝 Matlab R2016a"
date: 2018-02-22T14:48:22+08:00
categories: ["Fedora", "Matlab"]
tags: ["fedora", "matlab", "r2016a", "imshow"]
toc: true
math: false
---

# Introduction

學校部份課堂會需要使用 Matlab，包含 Computer Vision, Machine Learning, etc。因為我電腦是裝雙系統，之前 Matlab 都裝在 Windows 下，並在 Windows 下使用，但實在覺得切換系統太麻煩了，而且針對一份作業我都是先用 Python 寫一個版本再用 Matlab 寫，當結果不一致時得不斷切換系統來 debug，太花時間。於是我決定裝 Matlab 裝到 Fedora 下，我就可以整天都待在 Fedora 上了。

# Installation

1. 下載 linux 版的 Matlab, network.lic 跟說明文件，學校一般都有提供。

![](https://imgur.com/yqHD2uq.png)

2. Mount 該 iso 檔。這裡你有兩種方法，一個是傳統地用 `mount` 指令，另一個是在 Nautilus 中，直接在 iso 上點兩下（預設是用 Disk Image Mounter 開啟）。這裡使用第二種方法。這時可以從 Other locations 中看到 Matlab 被 mount 了。

![](https://imgur.com/Zx4jUcA.png)
![](https://imgur.com/iVIfxaj.png)

3. 開啟 Terminal，`cd` 到 `/run/media/<user>/MATLAB_R2016A/` （這是預設 mount 的位置），執行 `./install`。注意這裡不要加 `sudo` （加了會失敗，尚不清楚原因）。之後 matlab 安裝視窗就會出現。照著安裝步驟走即可。 但因為沒給 sudo 權限，無法安裝到 `/usr/local/` 等位置，所以我安裝到 `~/Matlab`。

![](https://imgur.com/2CNMpZB.png)
![](https://imgur.com/E0alIo7.png)

# Desktop Application

不想每次都從 command line 開啟，可以從 [這裡]( <https://gist.github.com/amoshyc/98c1405c86788be291d026416487fe03) 下載兩個檔案，並將：

* `matlab.desktop` 放到 `~/.local/share/applications/`
* `matlab_icon.png` 放到 `~/.local/share/icons/`

這樣即可在 Application 中看到 Matlab。

![](https://imgur.com/mogWngx.png)

# Conclusion

成功在 fedora 25 安裝 Matlab R2016a 了。得益於 Gnome Shell 中 Nautilus 的強大，安裝是很簡單的。雖然用指令 `mount` 來裝也不會複雜多少就是了。

![](https://imgur.com/B6b0VQo.png)

# `imshow` Bug

如果像我一樣 `imshow` 會出現問題

```
'.../R2016a/bin/glnxa64/libmwosgserver.so':
.../R2016a/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6:
version 'CXXABI_1.3.8' not found (required by /lib64/libGLU.so.1)
```

那解決方法是 **uncomment**  `/path/to/matlab/bin/.matlab7rc.sh` 中所有的

```
LDPATH_PREFIX='$MATLAB/sys/opengl/lib/$ARCH'
```

總共有三個。之後重啟 Matlab 即可。