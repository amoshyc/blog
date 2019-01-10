---
title: "Python 科學計算快速入門 1/2"
date: 2018-03-20T11:07:11+08:00
categories: [教學, Python 科學計算快速入門]
tags: []
toc: true
math: false
---

# Introduction

Python

- Interpreter
- 一行一行執行
- Use indention
- Python 3.6

# Virtual Environment

主流是 [Conda](https://conda.io/docs/index.html)，共分兩個版本：Anaconda, Miniconda。前者預設就安裝了非常多的 package，所以要非常大的硬碟空間（3GB+）[^1]。但大部份 package 我們都不會用，所以推薦使用 Miniconda。

安裝方法為去[官方](https://conda.io/miniconda.html)下載 bash，並執行，一般來說會將 conda 的執行檔加進系統的 bash：

![Imgur](https://i.imgur.com/UEfGHFG.png)
![Imgur](https://i.imgur.com/BdeSAbN.png)
![Imgur](https://i.imgur.com/dBmtUQn.png)

[^1]: https://conda.io/docs/user-guide/install/index.html

# Editor/IDE

{{< figure src="https://i.imgur.com/uYc2fTG.png" title="ides" width="500" >}}

![Imgur](https://i.imgur.com/GPTlb2u.png)

# Input/Output

{{< highlight python "linenos=table,noclasses=false" >}}
input()
_ = input()
x = input('> ')         # x is str
x = int(input('> '))    
{{< /highlight >}} 

{{< highlight python "linenos=table,noclasses=false" >}}
loss = 2e-1
acc = 0.98
print(loss)
print(type(loss))
print('loss:', loss, 'acc:', acc)
print('loss: {:.2f}, acc: {:.2f}'.format(loss, acc))
print(f'loss: {loss:.2f}, acc: {acc:.2f}') # format literal
print('loss: %.2f, acc: %.2f' % (loss, acc))
# 0.2
# <class 'float'>
# loss: 0.2 acc: 0.98
# loss: 0.20, acc: 0.98
# loss: 0.20, acc: 0.98
# loss: 0.20, acc: 0.98
{{< /highlight >}} 

{{< highlight python "linenos=table,noclasses=false" >}}
# This is a comment
'''doc or long
comment'''
{{< /highlight >}} 

# Variable

## Types

{{< highlight python "linenos=table,noclasses=false" >}}
a = 123456789123456789      # int（任意精度）
b = 1e-16                   # float
c = 1.0 + 1.0j              # complex
d = (True or False)         # boolean

e = [1, 2, 3]               # list
f = (1, 2, 3)               # tuple
g = {'a': 1, 'b': 2}        # dict
h = 'string'                # str

l = None                    # None
{{< /highlight >}} 


## Usage

{{< highlight python "linenos=table,noclasses=false" >}}
2 ^ 1                   # xor: 3
(1 + 2) - 3 * 4         # +-*: -9
5 / 3                   # true division: 1.667 (float)
5 // 3                  # division: 1 (int)
5 % 3                   # mod: 2
i ** 5                  # pow: 32
float(1)                # 1.0 (float)
int(1.5)                # 1 (int)
{{< /highlight >}} 

### List

mutable

{{< highlight python "linenos=table,noclasses=false" >}}
l = list('abc')         # l: ['a', 'b', 'c']
l[1] = 'd'              # l: ['a', 'd', 'c']
l.append(1)             # l: ['a', 'd', 'c', 1]
l.extend([2, 3])        # l: ['a', 'd', 'c', 1, 2, 3]
{{< /highlight >}}

### Tuple

immutable

{{< highlight python "linenos=table,noclasses=false" >}}
t = tuple('abc')        # t: ('a', 'b', 'c')
t[1] = 'd'              # error
x, y, z = t             # x: 'a', y: 'b', z: 'c'
a, b = 1, 2             # parenthesis omitted
{{< /highlight >}}

### Indexing & Slicing
 
 - index 從 `0` 開始數，可正可負
 - `start:end:step`, `step` is optional, 
 - default to `[0:len():1]`
 - slice return a **copy**

{{< highlight python "linenos=table,noclasses=false" >}}
l = [0, 1, 2, 3, 4]
len(l)                  # 5
l[0] + l[-1]            # 0 + 4 = 4
l[2:-1]                 # [2, 3]
l[0:5:2]                # [0, 2, 4]
l[::-1]                 # [4, 3, 2, 1, 0]
{{< /highlight >}}

### Dict

mutable

{{< highlight python "linenos=table,noclasses=false" >}}
d = dict({'a': 0, 'b': 2})
d['a'] = 3              # d: {'a': 0, 'b': 2}
d['c']                  # error
'c' in d                # False
d.update({'a': 3})      # d: {'a': 3, 'b': 2}
d.items()               # [('a', 0), ('b', 2)]
d.keys()                # ['a', 'b']
d.values()              # [0, 2]
{{< /highlight >}}


### Str

immutable

{{< highlight python "linenos=table,noclasses=false" >}}
a = '單引號'
b = "雙引號"
c = "It's time"         # mix
d = r'\n\t\b'           # raw string
e = '''多
行'''
f = '-'.join(['abc', 'def', 'xyz']) # abc-def-xyz
g = 'a b c'.split()     # ['a', 'b', 'c']
h = len('a b c')        # 5
i = 'pp' in 'apple'     # True
a[0] = '我'             # error
{{< /highlight >}}

## Management

Python is **strong**, **dynamic** and has GC system.

{{< highlight python "linenos=table,noclasses=false" >}}
'a' + 1     # error
x = 's'
x = 1       # Valid
{{< /highlight >}}

> `a = value`: 在記憶體中建立值為 value 的物件，讓變數名稱 a 指向他
> `a = b`: 讓變數名稱 a 指向「變數名稱 b 所指的物件」

![Imgur](https://i.imgur.com/Tdj6WnD.png)

{{< highlight python "linenos=table,noclasses=false" >}}
a = 998 # id(a) = 0x7f1f564d27d0
b = 999 # id(b) = 0x7f1f564d2a50
a = b   # a, b = 999, 999; id(a), id(b) = 0x7f1f564d2a50, 0x7f1f564d2a50
b = 997 # a, b = 999, 997; id(a), id(b) = 0x7f1f564d2a50, 0x7f1f564ec330
b = a   # a, b = 999, 999; id(a), id(b) = 0x7f1f564d2a50, 0x7f1f564d2a50
{{< /highlight >}}

Python 為了效率，預先將數字 -5 ~ 256 都先建好物件了，所以會有一個有趣現象

{{< highlight python "linenos=table,noclasses=false" >}}
a = 1
b = 1
print(id(a) == id(b)) # True
a = 1000
b = 1000
print(id(a) == id(b)) # False
{{< /highlight >}}

## Scope

有別於 C/C++/JAVA，Python 變數的 scope 是看函式。

{{< highlight python "linenos=table,noclasses=false" >}}
b = 1

def f():
    if True:
        a = 1
    print(a) # Valid

    # 告訴 python 之後我要使用全域命名空間中的那個變數 b
    # 不加的話，python 會視為創造了 local 變數 b
    global b
    b = 2

f()
print(b)    # 2
{{< /highlight >}}

# Control Flow

## If

### Syntax

{{< highlight python "linenos=table,noclasses=false" >}}
if ...:
    pass
elif ...:
    pass
elif ...:
    pass
else:
    pass
{{< /highlight >}}

### Details

{{< highlight python "linenos=table,noclasses=false" >}}
False, 0, 0.0, '', [], (), {}   # False Condition
and, or, not                    # Operations
<, >, <=, =>, !=, is, is not    # Comparison
a = (1 if 2 > 1 else 0)         # Ternary operator

data = []
if len(data) and data[0] == 1:  # Short-circuit evaluation
    pass # i.e. ';' in C
else:
    pass
{{< /highlight >}}

`==` 與 `is` 的差別在於 `==` 是比較值是不是相同的；而 `is` 是比較是不是指向記憶體中同一個物件，即他們的 `id()` 是不是相同的。如果你寫過 javascript，這個相同於 `==` 與 `===`。

{{< highlight python "linenos=table,noclasses=false" >}}
a = 1000
b = 1000

a == b              # True
a is b              # False
id(a) == id(b)      # False

b = a

a == b              # True
a is b              # True
id(a) == id(b)      # True
{{< /highlight >}}

`is` 的比較速度比 `==` 快上許多，如果再綜合之前所說的，Python 會將 -5 ~ 256 這幾個值都先建好物件，那如果我們知道變數的值在這個範圍，是可以用 `is` 來取代 `==` 的，但強烈 **不建議** 這麼做，因為如果一不小心，變數的值超出這個範圍，你的程式就壞掉了，而且這個範圍會隨著你的平臺，使用的 python 版本而變化。

唯一的例外是測試一個變數是不是 `None` 時，`None` 也是一個常駐記憶體的常量，因為每個 python 版本都一定會這個變數，因為在這個情況下，我們會使用 `is` 而不是 `==`。

{{< highlight python "linenos=table,noclasses=false" >}}
a = None
b = None

a == b              # True
a is b              # True
id(a) == id(b)      # True
a == None
{{< /highlight >}}

所以程式碼中常會看到

{{< highlight python "linenos=table,noclasses=false" >}}
if xxx is None:
    pass

if xxx is not None:
    pass
{{< /highlight >}}

## For, While

### Syntax

{{< highlight py "linenos=table,noclasses=false" >}}
for ... in <iterable>:
    # do something
    # break, continue, etc

while ...:
    # do something
    # break, continue, etc
{{< /highlight >}}

### Frequently Used Patterns

{{< highlight py "linenos=table,noclasses=false" >}}
''' Loop certain times '''
print(list(range(3))) # [0, 1, 2]
for _ in range(3):
    pass

''' Iterate items of a list '''
A = [1, 2, 3]
for a in A: # af if a = A[i]
    a = 2 * a   # will NOT modify A
# A: [1, 2, 3]

''' Iterate indices of a list '''
A = [1, 2, 3]
for i in range(len(A)):
    A[i] = 2 * A[i]
# A: [2, 4, 6]

''' Iterate indices, items of a list '''
A = [1, 2, 3]
for i, a in enumerate(A):
    A[i] = 2 * a
# A: [2, 4, 6]
{{< /highlight >}}

{{< highlight py "linenos=table,noclasses=false" >}}
''' Iterate items of 2 lists '''
A = [1, 2, 3]
B = [3, 2, 1]
C = [a + b for a, b in zip(A, B)] # [4, 4, 4]

''' Iterate over dict '''
C = {'a': 0, 'b': 1}
for k, v in C.items(): # k, v is copies
    C[k] = 2 * v
# C: {'a': 0, 'b': 2}

''' List comprehension'''
a = [0, 1, 2, 3, 4, 5]          
b = [(x % 2) for x in a]         # [0, 1, 0, 1, 0, 1]
c = [x for x in a if x > 3]      # [4, 5]
d = sum([1 for x in a if x > 3]) # 2


{{< /highlight >}}

-------

For with else

{{< highlight py "linenos=table,noclasses=false" >}}
for ...:
    if ...:
        pass
else:
    pass
{{< /highlight >}}

對等的 C 寫成

{{< highlight cpp "linenos=table,noclasses=false" >}}
bool flag = False;
for (...) {
    if (...) {
        flag = True;
        ...
    }
}
if (!flag) {
    ...
}
{{< /highlight >}}


## With

{{< highlight py "linenos=table,noclasses=false" >}}
''' something that need to be closed '''
with open('./test.txt', 'r') as f:
    ...

# equivalent to
f = open('./test.txt', 'r')
...
f.close()
{{< /highlight >}}


# Function

## Syntax

{{< highlight py "linenos=table,noclasses=false" >}}
def f(p1, p2, n1=123, n2=456):
    """Doc string of this function.
    Arguments in Python is **passed by assignment**.

    Parameters
    ----------
    p1, p2: positional arguments
        Necessary arguments and placed before any named arguments.
    n1, n2: named arguments, optional
        Arguments with default value, can be specified by name.
    """
    p1.append(1)    # valid, is visible outside (i.e. a now is [1])
    p1 = []         # valid, p1 now points to another list
    p2.append(2)    # error, because b is immutable
    p2 = ''         # valid, p2 now points to a empty str

    return ...      # optional

''' invoked by '''
a, b = [], ()
f(a, b)                   # p1 = [], p2 = [], n1 = 123, n2 = 456
f(a, b, 789, -1)          # p1 = [], p2 = [], n1 = 789, n2 = -1
f(a, b, n2=-1)            # p1 = [], p2 = [], n1 = 123, n2 = -1
f(a, b, n2=-1, n1=789)    # p1 = [], p2 = [], n1 = 789, n2 = -1

''' Unpacking arguments '''
args = ([], ())
kwargs = {'n1': 789, 'n2': -1}
f(*args, **kwargs)
{{< /highlight >}}

## Multi Returned Value

{{< highlight py "linenos=table,noclasses=false" >}}
def ae_loss(y_pred):
    loss1 = ...
    loss2 = ...
    return loss1, loss2 # i.e. return a tuple (loss1, loss2)

''' invoked by '''
losses = ae_loss(...)  # losses is a tuple
l1, l2 = ae_loss(...)  # general way
l1, _ = ae_loss(...)   # discard loss2
_, l2 = ae_loss(...)   # discard loss1
ae_loss(...)           # discard return values
{{< /highlight >}}

## Generator

{{< highlight py "linenos=table,noclasses=false" >}}
''' Instead of '''
def read_all_imgs(img_paths):
    imgs = []
    for path in img_paths:
        imgs.append(load_img(p))
    return imgs

imgs = read_all_imgs(...) # Too large to fit in memory
for img in imgs:
    convert(img)

''' Use '''
def gen_imgs(img_paths):
    for path in img_paths:
        img = load_img(p)
        yield img

img_gen = gen_imgs(...) # type: <class 'generator'>
for img in img_gen:
    convert(img)
{{< /highlight >}}

{{< highlight py "linenos=table,noclasses=false" >}}
# Some builtin functions are generator,
# you can convert generator to values using
list(range(...))
list(enumerate(...))
list(zip(...))
{{< /highlight >}}

## Built-in functions
{{< highlight py "linenos=table,noclasses=false" >}}
range(start, end, step) # [start, start + step, start + 2 * step, ...]
enumerate(A)            # [(0, A[0]), (1, A[1]), (2, A[2]), ...]   
zip(A, B, C)            # [(A[0], B[0], C[0]), (A[1], B[1], C[1]), ...]

any(A)                  # True if any item in A is not False
all(A)                  # True if all items in A are not False
sorted(A)               # return sorted data, will NOT modify arguments
sum(A)                  # sum
len(A)                  # length
max(A)                  # max
max(a, b, ...)          # max
min(A)                  # min
min(a, b, ...)          # min
round(1.5)              # 2
int(1.5)                # 1
int('123')              # 123
float('123')            # 123.0
hex(255)                # 0xff
{{< /highlight >}}

# Classes

{{< highlight py "linenos=table,noclasses=false" >}}
class Classifier(object):
    ''' Don't forget the first argument is self
    '''
    def __init__(self, var=None):
        self.var1 = var
        self.var2 = None

    def public_method(self, x):
        return self._privateMethod() + x

    def _private_method(self):
        return self.var1

clf = Classifier(var=2)
print(clf.public_method(5)) # 7
{{< /highlight >}}


# PEP 8 & Exception

Style Guide.

{{< highlight py "linenos=table,noclasses=false" >}}

class ClassName(object):
    def __init__(self):
        pass

    def public_method(self):
        pass

    def _private_method(self):
        pass

def function_name(...):
    CONSTANT_VAR = 1
    local_var = 1
    dir_ = 2

{{< /highlight >}}

一些常見的 linters 有 `autopep8`, `yapf`。

{{< highlight py "linenos=table,noclasses=false" >}}
try:
    f = open(...)
except FileNotFoundError as e:
    print(e)
finally:
    pass
{{< /highlight >}}

{{< highlight py "linenos=table,noclasses=false" >}}
def f(x):
    if x is None:
        raise Exception('abc')

f(None)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "<stdin>", line 3, in f
# Exception: abc
{{< /highlight >}}


# Example

從 `data.rec` 中取出各筆資料，存成 json 格式。

```
@GAISRec:
@U:http://travel.ettoday.net/article/1.htm
@T:外星人回到地球了 ETtoday.net Come home
@B:
記者吳光中／台北報導 熄燈3年半，ETtoday回來了，這次不叫「達康」，改叫「達內」...
@GAISRec:
@U:http://travel.ettoday.net/article/3.htm
@T:2011年第46屆電視金鐘奨完整入圍名單
@B:
2011年金鐘獎，ETtoday舉行臉書直播，帶網友看熱鬧也看門道。....
@GAISRec:
@U:http://travel.ettoday.net/article/7.htm
@T:豪雨特報持續 北部持續濕涼
@B:
生活中心／台北報導 受到秋颱奈格外圍環流的影響，... 
```

假設 `data.rec` 太大造成無法整個讀進記憶體，我們只能一行一行讀入資料。那我們該如果抽取出各筆資料呢？ 相信大家都會寫，而我的解法是：

1. 遇到 `@GAISRec:`，代表一筆資料的開始
2. 遇到 `@U:`，取出 url
3. 遇到 `@T:`，取出 title
4. 遇到 `@B:`，什麼都不做
5. 上述都不成立：取出 body，且這筆資料結束


{{< highlight py "linenos=table,noclasses=false" >}}
import json

def extract(f):
    for line in f:
        line = line.strip() # 清掉字串前後的空格、換行
        if line.startswith('@GAISRec:'):
            record = dict()
        elif line.startswith('@U:'):
            record['url'] = line[3:]
        elif line.startswith('@T:'):
            record['title'] = line[3:]
        elif line.startswith('@B:'):
            pass
        else:
            record['body'] = line
            yield record


with open('./data.rec', 'r') as f:
    for i, record in enumerate(extract(f)):
        print(i)
        print(record)
        print()

        # save as json
        filepath = f'./{i:08d}.json'
        with open(filepath, 'w') as f:
            json.dump(record, f, ensure_ascii=False)
{{< /highlight >}}


![Imgur](https://i.imgur.com/qjEEaDY.png)

在這個例子中，可以看到：

1. `with` 的使用
2. `yield` 的使用
3. `enumerate` 的使用
4. format literal 的使用
5. slice 的使用
6. 變數 scope 是看函式