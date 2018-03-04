---
title: "Markdown Cheatsheet"
date: 2017-12-09T13:55:37+08:00
categories: ["Cheatsheet"]
tags: ["markdown", "cheatsheet"]
toc: true
math: true
---


# Standard Markdown


## Inline Markup

常見的 Markdown 行內標記包含**粗體**、*斜體* 、~~刪除~~、`程式碼`、Footnote[^1] 與[超連結](https://example.org/ "tooltip")。他們使用方法分別為：

* **粗體** ：`**粗體**` 或 `__粗體__`。
* *斜體* ：`*斜體* ` 或 `_斜體_`。
* ~~刪除~~ ：`~~刪除~~`。
* `程式碼` ：將內容用 \` 包起來。
* [^1]：`[^id]`，其連結的位置為 `[^id]: xxx`。
* [超連結](https://example.org/)：`[text](link “tooltip")`，`"tooltip"` 可以不寫。

其中要注意的是斜體後面得有空白，造成中文書寫時會有多餘空白，因此書寫中文時建議使用粗體代替。在這個 theme 中，**所有**的換行都會被保留，這有別於其他 theme，我認為這樣的書寫比較符合我自己的習慣。如果使用本 theme，建議將你的編輯器調成 wrap line 的模式。

[^1]: https://example.org/

## Image

![Fedora 26](https://i.imgur.com/QzfKyxm.jpg)

```
![Fedora 26](https://i.imgur.com/QzfKyxm.jpg)
```

這種圖片寬度固定為 theme 的寬度（預設是 700px）。

## Code Block

Code Block 一般用來放置一些簡短的、**不需要**顏色渲染的程式碼，例如在 conda 中創造一個虛擬環境的指令為：

```
conda create -n venv python=3.6
source activate venv
```

## Block Quote

> Block Quote 一般用來放要強調的東西。
> 也可以用來放詩詞等。
>
> > 它可以巢狀使用，但很少用到。也可以包含其他東西，像是 List:
> > 1. list item 1
> > 2. list item 2

## Table

| a | b | c | d | e |
|---|---|---|---|---|
| 1 | 2 | 3 | 4 | 5 |
| 5 | 4 | 3 | 2 | 1 |
| 6 | 6 | 6 | 6 | 6 |

## Hr

分隔線之前的文字

-------------

分隔線之後的文字

## List

### Unordered List

* 第 1 級
* 第 1 級
* another list
    * 第 2 級
    * 第 2 級
* 第 1 級

### Ordered List

1. 第 1 級
2. 第 1 級
3. another list
    1. 第 2 級
    2. 第 2 級
4. 第 1 級



# Hugo Shortcodes

## Highlight

用來顯示顏色渲染的程式碼，可以指定：

1. 顯示行號或模式
2. 加亮顯示哪些行
3. 行數起始值

無行號模式：

{{< highlight python "noclasses=false" >}}
from pprint import pprint

TC = int(input())
for _ in range(TC):
    cards = []
    for i in range(5):
        nums = [int(x) for x in input().split()]
        cards.append(set(nums))
{{< / highlight >}}

Inline 模式，方便觀看（尤其在手機上），但不適合複製程式碼（選取時會選到行號）：

{{< highlight python "linenos=inline,hl_lines=3 5-7,linenostart=199,noclasses=false" >}}
from pprint import pprint

TC = int(input())
for _ in range(TC):
    cards = []
    for i in range(5):
        nums = [int(x) for x in input().split()]
        cards.append(set(nums))
{{< / highlight >}}

Table 模式：

{{< highlight python "linenos=table,hl_lines=3 5-7,linenostart=199,noclasses=false" >}}
from pprint import pprint

TC = int(input())
for _ in range(TC):
    cards = []
    for i in range(5):
        nums = [int(x) for x in input().split()]
        cards.append(set(nums))
{{< / highlight >}}


## Gists

Hugo 內建，可以引入外部 gist：

{{< gist spf13 7896402 >}}

## Figure

{{< figure src="https://i.imgur.com/KdlKPx8.jpg" title="人類衰退之後" width="600" >}}

與 Image 的差別在於可以指定大小，但仍不能超出這個 theme 的寬度（預設 700px），並且可以指定圖片的 caption。

## Youtube

{{< youtube _J73EqmpZ7k >}}

## Tweet

{{< tweet 877500564405444608 >}}


# Custom Shortcodes

## Jupyter Notebook

Jupyter Notebook 是非常常見的程式碼格式，易於分享並重現結果，在 ML 領域中大量被使用。本 theme 支援顯示 notebook，首先透過 `nbconvert` 將 notebook 轉成 html。

```
jupyter nbconvert --to html notebook.ipynb
```

將產生的 html 移到與文章相同的資料夾，然後使用 shortcode:

```
{{</* jupyter "notebook.html" */>}}
```

結果如下：

{{< jupyter "markdown-cheatsheet-notebook.html" >}}

{{% admonition title="Hint!" color="blue" %}}
檔名中的空格要 `urlize` ，即全小寫、空白換成 `-`！
{{% /admonition %}}


## Admonition

警示窗是好用的東西，可以指定標題與顏色，顏色可以為任意 css 顏色或預定義的顏色。4 個預定義顏色如下：

{{% admonition title="Warning!" color="red" %}}
Don't panic! This is predefined color red.
{{% /admonition %}}

{{% admonition title="Hint!" color="blue" %}}
Color blue
{{% /admonition %}}

{{% admonition title="title" color="yellow" %}}
Color yellow
{{% /admonition %}}

{{% admonition title="中文標題" color="teal" %}}
Color teal
{{% /admonition %}}

## Color Square

You can use this to show some colors {{< colorsq "#f44336" >}}.


## Expansion

用來放一些隱藏的東西，例如題解的程式碼。但每篇文只支援最多 8 個。

{{% expansion "Click to Open" %}}
test
test
test
{{% /expansion %}}

{{% expansion "程式碼" %}}
{{< highlight python "linenos=table,linenostart=199,noclasses=false" >}}
from pprint import pprint

TC = int(input())
for _ in range(TC):
    cards = []
    for i in range(5):
        nums = [int(x) for x in input().split()]
        cards.append(set(nums))
{{< / highlight >}}
{{% /expansion %}}


# Math Formula

數學公式是不可少的。本 theme 支援兩種數學公式表示法：Latex & Asciimath。個人建議盡量使用 Asciimath，表達不了時再轉用 Latex，因為 Asciimath 比 Latex 好寫太多了。

## Latex

Inline Latex code should be wrapped in `$[` and `]$`.
Block Latex code in `{{</* tex */>}}` short code.

`$[N_1]$` produces $[N_1]$

```
{{</* tex */>}}
x = \frac{-b \pm \sqrt{c^2 - 4ac}}{2a}
{{</* /tex */>}}
```
produces

{{< tex >}}
x = \frac{-b \pm \sqrt{c^2 - 4ac}}{2a}
{{< /tex >}}

## AsciiMath

Inline AsciiMath code should be wrappedn in `$$` and `$$`.
Block AsciiMath code in `{{</* am */>}}` short code.

`$$N_1$$` produces $$N_1$$ .

```
{{</* am */>}}
x = (-b +- sqrt(c^2 - 4ac)) / (2a)
{{</* /am */>}}
```
produces

{{< am >}}
x = (-b +- sqrt(c^2 - 4ac)) / (2a)
{{< /am >}}