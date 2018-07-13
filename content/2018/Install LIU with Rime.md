---
title: "使用 Rime 在各大平臺安裝嘸蝦米輸入法"
date: 2018-04-19T13:28:26+08:00
categories: ["Linux"]
tags: ["rime", "windows", "linux", "android", "嘸蝦米輸入法", "嘸蝦米", "同文輸入法"]
toc: true
math: false
---

# 前情提要

身為一個國小三年級就跟著我媽學嘸蝦米輸入法的人，我中文輸入法可以打到每分鐘 80 字左右。嘸蝦米只使用 26 個英文字母，很少選字，打出來的結果是確定性的（即不會根據你以前打過的字調整輸出結果），但最大的問題是他是專有軟體，且對 Linux 沒有很好的支援。在將我的主作業系統換到 Fedora 後，我得找一個方法讓我快樂地打嘸蝦米。而我找到的最終解答是使用 [Rime](http://rime.im/) 這個由佛振創立的 Open Source 輸入法框架。

Rime 預設不支援嘸蝦米，但在 Rime 要創一個輸入法是簡單的，尤其嘸蝦米這種其於碼表的輸入法。所需要的資料、設定檔我在幾年前就已經弄好了，但一直沒公開，只自己使用，畢竟感覺會有版權問題。但昨天看到這篇[文章](https://opinion.udn.com/opinion/story/11723/3091600)後，我決定還是來貢獻一下，於是就有了這篇文章。

Rime 可以執行在各大平臺上：Windows, Linux, Mac, Android, etc。但我本身只用也只有 Linux, Android, Windows，所以在下面只講解如何在這幾個平臺使用 Rime 安裝嘸蝦米。

# Linux

底下以 Fedora 27 (GNOME, Wayland) 為例，當代的 Linux 預設的中文輸入法框架應該都是 [ibus](https://github.com/ibus/ibus)，應該都只要小修改即可使用。ibus 本身是一個框架，而 Rime 也是一個框架，透過 Rime 的 ibus 版本，我們可以將 Rime 整合進 ibus 裡。架構圖如下：

```
ibus:
    英語（美式）
    漢語（Rime）:
        嘸蝦米
        注音
----------
ibus 透過「設定／地區與語言」下的「輸入來源」調整，按 Super + Space 切換輸入法。
Rime 透過 ~/.config/ibus/rime 的設定檔調整，按 Ctrl + ` 切換。
```

你也許會問為什麼不搞個基於 ibus 的嘸蝦米輸入法就好，還要嵌套個 Rime 呢？有兩個原因，第一是 ibus 讀嘸蝦米碼表的速度非常慢，每次切換成嘸蝦米時都得等上個幾秒，這是不可接受的。第二個原因是 rime 在各大平臺都有，我弄好一個基於 rime 的嘸蝦米輸入法，可以同時在 Linux, Windows, Android 上使用，這是 ibus 做不到的。

## 安裝 Rime

```
sudo dnf install ibus-rime
```

其他 Linux Distribution 可以參考 Rime 的[文檔](https://github.com/rime/home/wiki/RimeWithIBus)。

安裝成功後，請至 GNOME 的「設定／地區與語言」將「輸入來源」新增「漢語（臺灣）／漢語（Rime）」。結果如下：

{{< figure src="https://i.imgur.com/3rZzLYy.png" width="700">}}

之後透過 GNOME 右上角的選單，讓 ibus 選擇 Rime。

{{< figure src="https://i.imgur.com/ndThCyf.png" width="200">}}

Rime 第一次啟動時會先「部署」，即 Rime 會根據他的設定檔（位於 `~/.config/ibus/rime`）生成執行時必要的資料。如果沒有跳出一個視窗說 "Rime is under maintenance"，請手動按同一個選單中的「部署」。至此 Rime 就成功安裝了，你現在可以使用 Rime 自帶的幾個輸入法，你可以透過按多次的 ``Ctrl + ` `` 再按 `Enter` 來切換 Rime 內部的輸入法。

## 加入嘸蝦米

身為蝦米樂園的一份子，那些自帶的輸入法我們是不需要的（除了注音），而這可以透過 `~/.config/ibus/rime/default.yaml` 設定。但直接去更改設定檔是一件危險的事情，當你改完覺得不好想還原卻忘記原本是怎麼寫的就麻煩了。因些 Rime 提供了一個機制，將你要覆寫的東西寫到 `default.custom.yaml` 去。

而 `default.custom.yaml` 與其對應的嘸蝦米 myliu 我幫大家準備好了，都放在我的 [Github Repo](https://github.com/amoshyc/myliu) 裡。請將 repo 中除了 README.md 外的所有檔案放至 `~/.config/ibus/rime/` 中。

{{< figure src="https://i.imgur.com/oFTIzmv.png">}}

然後將 Rime 重新 **部署** 即可。成功的話，右上角的輸入法選單你就會看到

{{< figure src="https://i.imgur.com/qOFQ0mN.png" width="200">}}

要注意是我這個嘸蝦米是沒辦法反查注音的！我之前有嘗試要加進這個功能（Rime 支援這個功能）但失敗了，如果有誰成功還煩請告之一下。另外這個嘸蝦米使用的碼表是我從網路上載的，原先是嘸蝦米 gcin 版本用的，我將之轉成 Rime 的格式。另外，我的 `default.custom.yaml` 指示了 Rime 只會顯示嘸蝦米與注音。

至此，你就可以用漂亮的界面快樂地打嘸蝦米了。同時，你可以使用 `Super + Space` 切換成英語、``Ctrl + ` `` 切換成注音。

{{< figure src="https://i.imgur.com/J3PEut0.png" title="用整合進 GNOME 的界面打嘸蝦米">}}


# Android

Rime 有被人 port 到 Android 上，名稱叫「同文輸入法」，可以在 Google Play 上[找到](https://play.google.com/store/apps/details?id=com.osfans.trime&hl=zh_TW)。感謝 osfans~
![Imgur](https://i.imgur.com/l52aptT.png)

安裝成功後，開啟同文輸入法。

1. 啟用輸入法
2. 選取同文輸入法
3. 部署
4. 將 `myliu.dict.yaml` 與 `myliu.schema.yaml` 移至「內建儲存空間/rime」。
    這個資料夾似乎因為權限的關係，從電腦看似乎是看不見的，但用手機內建的檔案管理就看得到。
    因此如果你也沒看到這個資料夾，可以先將檔案移至 Downloads，再用手機內建的檔案管理移到 rime。
5. 部署
6. 在「輸入／方案」中選嘸蝦米
7. 開始使用

{{< figure src="https://i.imgur.com/UN67vbR.jpg" width="400" title="左圖：選擇嘸蝦米。右圖：使用中（配色為孤寺）">}}


# 最後

這篇文章以嘸蝦米在 Fedora 上使用 ibus-rime 打成。
嘸蝦米不死，只是凋零！
