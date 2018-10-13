---
title: "About Site"
date: 2018-03-04T20:45:49+08:00
categories: ["Misc."]
tags: ["hugo", "github", "zmd", "theme", "site", "docs"]
toc: true
math: false
---

# TL;DR

這個網站是使用 [hugo](https://gohugo.io/) 架設，使用我自己寫的 theme。在此我記錄一下架設的步驟，主要有三個：**初使化**、**設定 Github**、**設定 theme**。另外，因為這個 theme 我是設計給自己使用，所以有一些地方寫得比較死，如果你有任何建議，歡迎 PR。這個網站架設在 Github Page 上，並使用同一個 repo 同時儲存與管理原始碼（你寫的 md 與 hugo 的設定檔）與渲染出來的 html。

# 初使化

我假設你是在 Linux 下操作，Windows/Mac 我不熟，平常也沒在用。

1. `hugo new site <name>`：初化化一個 hugo 專案。
2. `cd <name>`：之後所有操作都是在這個資料夾下。
3. `git init`：讓這專案使用 git 管理。

# 設定 Github

1. 在 Github 新增一個 repo，並在這個 repo 的 Setting 中，將 Github Page 設為從 docs/ 顯示。
2. `git remote add origin <repo>`：<repo> 是 1. 的 SSH/HTTPS 位置。這個指令將專案的 origin 設為剛新增的 repo。
3. `git pull origin master`：將 LICENSE 等預設內容拉下來。

# 設定 theme

分成 5 個小步驟。

## 下載

透過 `git submodule add https://github.com/amoshyc/hugo-zmd-theme themes/zmd` 將我的 zmd theme 複製下來，儲存在 `theme/zmd`，並做為這個專案的一個 submodule。當然，你可以直接用我 theme 的 zip，但這會造成一旦 theme 更新，你無法透過 `git pull` 更新，得自己手動更新。

另一個方法是 clone theme，然後建立 softlink：
```
git clone https://github.com/amoshyc/hugo-zmd-theme
cd <blog_dir>/themes/
ln -s <path_to_theme_folder> ./zmd
```

## 改 config

將 zmd 下的 `themes/zmd/exampleSite/config.toml` 的內容加到你的 `config.toml` 中。裡面有我的 theme 的預設參數，如果你的 `config.toml` 中已經有相同的欄位，我是建議直接改掉，來確保 zmd theme 可以正確運作。所以在操作前，先備份你的 `config.toml` 會是不錯的選擇。

## 改 archetypes

將 `themes/zmd/exampleSite/archetypes/default.md` 的所有內容複製到你的 `archetypes/default.md` 中。archetypes 代表當你創立新的文章時，文章的預設內容與 front-matter。預設情況下，因為你的 archetypes 的優先權比 theme 中的 archetypes 高，hugo 會去使用預設的 `archetypes/default.md`。這一步你也可以改成刪掉你的 `archetypes/default.md`，刪掉後 hugo 渲染時會去使用 theme 的 archetypes。

## abouts

將 `themes/zmd/exampleSite/content/` 下的內容複製到你的 `content/` 下，或你也可以彷造同樣的結構創立檔案。其中 `abouts/` 是一個特殊的資料夾，裡面有一個檔案 `me.jpg` 是必不可少的，請將之換成你的圖片。另外，所有在 `abouts/` 下的文章都會被特殊分類，顯示於網站的 ABOUT 下。

## 與 Github 的結合

如果你前面都沒有出錯，那當你下指令 `hugo` 時，HTML 會被產生至 `docs/` 資料夾中。並且，你可以透過平常的 `git add`, `git commit`, `git push` 流程將你整個專案推到 Github 上。因為我並不想每次都在想 commit message，畢竟就是寫個文章或改改字，所以我建立了 makefile 來加速這個流程：

{{< highlight sh "noclasses=false" >}}
MSG = "Build at $(shell /bin/date '+%Y-%m-%d %H-%M-%S')"
upload:
	rm -rf ./docs && hugo
	git add -A
	git commit -m $(MSG)
	git push origin master
{{< /highlight >}}

每次要推資料至 Github 時只要下 `make upload` 即可。

{{% admonition title="Hint!" color="blue" %}}
記得 Github Repo 要設定成從 `docs/` 生成 Github Page
{{% /admonition %}}

# 開始寫文章

到此設定就完成了，你可以開始使用 hugo 與 zmd theme 來完成你的網站了。
以後每次要寫新文章，流程為：

0. `cd blog`：假設 blog 是你 repo 的資料夾
1. `hugo new <path/to/post.md>` 新增文章
2. 使用編輯器編輯 `<path/to/post.md>`
3. 同時，使用指令 `hugo serve`，結果即時渲染於 http://localhost:1313/blog/
4. 撰寫完畢，使用 `make upload` 將資料推到 Github。
5. 一次都沒問題的話，稍等個幾分鐘後就可以在你 repo 的 Github Page 看到結果。

hugo-zmd-theme 有一些特有的東西與設定，可以讓你的文章內容更有多樣性，詳情請參考 [這篇文](../2017/markdown-cheatsheet.html) 與他的 Markdown [原始碼](https://raw.githubusercontent.com/amoshyc/blog/master/content/2017/Markdown%20Cheatsheet.md)
