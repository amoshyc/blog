---
title: "Reproducible PyTorch"
date: 2019-01-10T18:53:38+08:00
categories: ["Snippet"]
tags: ["pytorch", "reproducible", "reproducibility"]
toc: true
math: false
---

將以下程式碼加在程式的 entry point，透過設定 seed 的方式，來讓整個訓練過程能夠複現。只在 pytorch 1.0 測試過，不保證其他版本也有相同效果。

{{< highlight py "linenos=table,noclasses=false" >}}
seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
{{< /highlight >}}
