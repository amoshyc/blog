---
title: "工作站 bash 的顏色渲染"
date: 2018-02-23T11:34:06+08:00
categories: ["Linux"]
tags: ["bash", "csh", "syntax hightlight"]
toc: true
math: false
---

# 前情提要

學校工作站預設的 shell 是 csh，用的很不習慣，而且也有一些 bug。
可以切換成 bash 但沒有顏色渲染，今天查了一下資料，筆記一下。

# 顏色渲染

更改 ls 的顏色，在 .bashrc 中加入

{{< highlight bash "noclasses=false" >}}
alias ls='ls --color=auto'
alias ll='ls -alh --color=auto'
{{< /highlight >}}

更改 prompt 顏色，這個指令來自 這，一樣在 .bashrc 中加入

{{< highlight bash "noclasses=false" >}}
PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '
{{< /highlight >}}

# 結果

![](https://i.imgur.com/laCv1G6.png)