---
title: "在 Fedora 安裝 Zotero"
date: 2018-06-07T15:58:14+08:00
categories: ["Linux"]
tags: ["fedora", "zotero", "desktop file"]
toc: true
math: false
---

# 背景

最近因為在做計畫，大量地看論文，然後發現只使用 Chrome 的書籤已經不夠我管理這堆論文了，所以打算使用 [Zotero](https://www.zotero.org/) 來管理。Zotero 是一個文獻管理軟體，不過目前為止我還沒要用到他生成 Reference 的功能，只是拿他來管理論文。跟他同性質的軟體包含 EndNote, Mendeley 等，不過看來看去，覺得 Zotero 應該是比較好的，而且有出 Linux 版本，讓我可以在 Fedora 上使用。

# 安裝

[官方指示](https://www.zotero.org/support/installation)

## 下載

從[官網](https://www.zotero.org/download/)下載針對 Linux 的壓縮檔即可，是用 Java 寫的。解壓後執行該資料夾中的 `zotero`（不是 `zotero-bin` 也不是 `zotero.jar`）。我個人解壓後的資料夾是 `~/Zotero_linux-x86_64/`，不過你當然可以選你想要的地方。

## 建立 Desktop File

我想將 Zotero 加到 GNOME 的 Favorite 中，讓我方便快速啟動。方法為將 Zotero 的 Desktop File 加到 `~/.local/shared/applications/`。照著官方指示做，在 Zotero 的資料夾中執行：

```
$ ./set_launcher_icon
$ ln -s ~/Zotero_linux-x86_64/zotero.desktop ~/.local/share/applications/zotero.desktop
```

前者會將 Zotero Desktop File 的 Icon 欄位設定好。後者在目標資料夾建一個 symlink 到 Zotero 的 Desktop File。這樣你應該就可以在 GNOME 中直接搜到 Zotero 了，然後就可以右鍵將他加進 Favorite。

{{< figure src="https://i.imgur.com/hrypQGL.png" width="400">}}


## 安裝 Chrome Extention

Zotero 的 Browser Extention 讓你可以在瀏覽網頁時就將網頁加進 Zetero，他能夠自動辨別是不是在看 arXiv 論文之類的，然後自動幫你解析論文的資訊。


# 結論

{{< figure src="https://i.imgur.com/g6wESqz.png" width="700">}}

事實上到目前為止，我還沒找到一個理想的方式來管理讀過的論文，原因有幾個：

1. 我讀論文時，喜歡印下來，看紙本，然後在上面註記，包含寫感想與 Highlight。
2. 我又想我的註記可以數位化，方便搜索與查找。
3. 但論文都是 pdf，尤其我讀的論文八成以上來自 [arXiv](https://arxiv.org/)。
4. pdf 閱讀器之間沒有統一註記的格式，例如我用 GNOME Evince 註記的 pdf，無法被 pdf.js 或 adobe pdf reader 正確解析。
5. 如果把每篇讀過的論文都打成部落格文章又太花時間。

總之，我目前為止打算先使用 Zoteror 管理我讀過的論文，註記的部份再想想看。
