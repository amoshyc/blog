---
title: "Randomized Algorithm HW #2"
date: 2019-04-17T21:07:43+08:00
categories: ["NTHU"]
tags: []
toc: false
math: true
---

107062566
Yu-Cheng Huang 黃鈺程
amoshuangyc@gmail.com

# Problem #1

{{< am >}}
"Assume " X ~ "Uniform"[a, b] " then " "Var"[X] = (b - a)^2 / 12 \n
Var[X] \n
= E[X^2] - E[X]^2 \n
= [int_b^a j^2 Pr(X=j) \ dj] - ((a + b) / 2)^2 \n
= [int_b^a j^2 (1 / (b - a)) dj] - ((a + b) / 2)^2 \n
= (1 / (b - a)) \int_b^a j^2 dj - ((a + b) / 2)^2 \n
= (1 / (b - a)) [1/3 j^3 + C]_a^b - ((a + b) / 2)^2 \n
= (b - a)^2 / 12 \n
{{< /am >}}

**a**

{{< am >}}
Var[X] = (2k - 0)^2 / 12 = (4k^2) / 12 = k^2/3
{{< /am >}}

**b**

{{< am >}}
Var[X] = (k - (-k))^2 / 12 = (4k^2) / 12 = k^2/3
{{< /am >}}

**c**

{{< am >}}
Var[X + t] \n
= E[((X + t) - E[X + t])^2] \n
= E[(X + t - E[X] - E[t])^2] \n
= E[(X + t - E[X] - t)^2] \n
= E[(X - E[X])^2]
= Var[X]
{{< /am >}}

# Problem #2

{{< am >}}
"Independence" => E[X]E[Y] = E[XY] \n
\n
Var[X - Y] \n
= E[(X - Y)^2] - E[X - Y]^2 \n
= E[X^2 - 2XY + Y^2] - (E[X] - E[Y])^2 \n
= E[X^2] - 2E[XY] + E[Y^2] - (E[X]^2 - 2E[X]E[Y] + E[Y]^2) \n
= (E[X^2] - E[X]^2) + (E[Y^2] - E[Y]^2)\n
= Var[X] + Var[Y]
{{< /am >}}

# Problem #3

{{< am >}}
"premise. " Pr(X_i = 1) = 1//n \n
"premise. " E[X] = n * (1//n) = 1 \n
"premise. " E[X^2] \n
= E[(sum_(i=1)^n X_i)^2] 
= E[(sum_(i=1)^n X_i)(sum_(j=1)^n X_j)] \n
= E[sum_(i=1)^n sum_(j=1)^n X_i X_j]
= sum_(i=1)^n sum_(j=1)^n E[X_i X_j] \n
= (sum_(i=1)^n E[X_i]) + (sum_(i=1)^n sum_(j != i) E[X_i X_j]) \n
= 1 + (sum_(i=1)^n sum_(j != i) Pr(X_i=1 nn X_j=1)) \n
= 1 + (sum_(i=1)^n sum_(j != i) ((n-2)!) / (n!)) \n
= 1 + (sum_(i=1)^n sum_(j != i) 1/(n (n-1))) \n
= 1 + (1/(n (n-1)) sum_(i=1)^n sum_(j != i) 1) \n
= 1 + 1 
= 2 \n
\n \n
Var[X] \n
= E[X^2] - E[X]^2 \n
= 2 - 1 \n
= 1
{{< /am >}}

# Problem #4

**a**

{{< am >}}
E[Z_i] \n
= Pr(Z_i = 1) " where " Z_i " is indicator of edge (u, v)" \n
= Pr("u is pressed, v is pressed") + Pr("u is not pressed, v is not pressed") \n
= 1/4 + 1/4 = 1/2
{{< /am >}}

**b**

{{< am >}}
E[Z_i Z_j] \n
= sum_((i, j) in S) i j * Pr(Z_i=i, Z_j=j) " where S={(0, 0), (0, 1), (1, 0), (1, 1)}" \n
= 1 * Pr(Z_i=1, Z_j=1) \n
= 1 * (1/2 * 1/2)\n
= 1/4
{{< /am >}}

**c**

{{< am >}}
Cov[Z_i, Z_j] \n
= E[Z_i Z_j] - E[Z_i]E[Z_j] \n
= 1/4 - 1/2 * 1/2 = 0
{{< /am >}}

**d**

{{< am >}}
"premise. " Var[Z_i] = E[Z_i^2] = E[Z_i]^2 - 1^2 * 1/2 - (1/2)^2 = 1/4 \n
\n
Pr(abs(Z - E[Z]) >= n) \n
<= (Var[Z]) / n^2 \n
= 1/n^2 ( sum_(i=1)^n Var[Z_i] + 2 sum_(i=1)^n sum_(j != i) Cov[Z_i, Z_j]) \n
= 1/n^2 (1/4 - 0) \n
= 1/(4n^2)
{{< /am >}}

# Problem #5

{{< am >}}
lim_(n->oo) Pr(abs(X_1 + X_2 + ... + X_n) > n epsilon) \n
<= lim_(n->oo) 1/(n^2 epsilon^2) Var[X_1 + X_2 + ... + X_n] \n
= lim_(n->oo) 1/(n^2 epsilon^2) (n * sigma^2) \n
= lim_(n->oo) (sigma^2) / (n epsilon^2) \n
= 0
{{< /am >}}

# Problem #6

{{< am >}}
X = " indicator of prob " 1/k " \n
=> E[X] = Pr[X=1] = 1/k \n
\n
Pr(X >= k * E[X]) \n
= Pr(X >= k * 1/k) \n
= Pr(X >= 1) = 1/k
{{< /am >}}

# Problem #7

{{< am >}}
X = "discrete RV ranges [0, inf] and " \n
Pr(X = i) = c / i^3 " where c in a constant" \n
\n
E[X] \n
= sum_(i=0)^oo i * Pr(X=i) \n
= sum_(i=0)^oo 1/i^2 \n
=> " converges (p series with p=2)"\n
\n
Var[X] \n
= E[X^2] - E[X]^2 \n
= sum_(i=0)^oo i^2 * 1/i^3 - pi / 6 \n
= sum_(i=0)^oo 1/i - pi / 6 \n
=> " diverges (harmonic series)"
{{< /am >}}