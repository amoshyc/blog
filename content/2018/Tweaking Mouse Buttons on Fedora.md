---
title: "在 Fedora 上調整滑鼠按鍵"
date: 2018-12-18T20:06:22+08:00
categories: ["Linux"]
tags: ["fedora", "mouse", "button"]
toc: true
math: false
---

# 需求

Chrome 的分頁可以用滑鼠中鍵關閉，這比花時間瞄準分頁上的 `X` 方便，於是我按中鍵的頻率比按右鍵的頻率高上許多，但滑鼠的中鍵必不好按，這讓我決定交換我滑鼠的中鍵與右鍵。

# 方法

在不同的 GNOME Backend 上用不同的方法。

## Xorg

```
xmodmap -e "pointer = 1 3 2"
```

## Wayland

待補
