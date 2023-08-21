# Document-Fingerprint-Algorithms

文档指纹算法具有显著提升文档分类、检索或对比等任务效率的潜在前景，并可保障敏感文件的隐私性，因此，本项目旨在总结归纳出当前可在文档领域中使用的哈希指纹算法以及基于这些指纹的距离计算和相似度值计算方法，从而便于评估和对比这些算法的有效性。

具体的算法测试细节可见`指纹生成算法-测试细节.docx`

# 目录

- [文件目录](#文件目录)
- [文档哈希指纹生成算法](#文档哈希指纹生成算法)
  - [Simhash](#simhash)
  - [Minhash](#minhash)
  - [Karp-Rabin](#karp-rabin)
  - [Winnowing](#winnowing)
  - [Fuzzy Hash](#fuzzy-hash)
  - [FlyHash](#flyhash)
- [文档哈希指纹距离/相似度计算方法](#文档哈希指纹距离-相似度计算方法)
  - [汉明距离](#汉明距离)
  - [Jaccard相似度](#jaccard相似度)
  - [余弦相似度](#余弦相似度)

# 文件目录

```
│  fingerprint.py  # 整合哈希指纹生成和相似度计算/距离计算方法。
│  hashes.py  # 生成文档指纹的哈希算法集合
│  similarities.py  # 文档指纹的相似度计算算法集合
│  utils.py  # 基础函数库
│  README.md
│  requirements.txt
│  关于EMD和WMD.pdf
│  指纹生成算法-测试细节.docx
│
├─tests
│    dict.py  # 字典转字符串用Simhash生成指纹，用汉明距/编辑距离计算相似度
│    Simhash.py  # 使用simhash库实现文本指纹计算、相似度计算
│    Simhash+LSH.py  # simhash计算文本指纹，LSH+汉明距+余弦（TFIDF）计算文本列表相似度
│    Simhash+LSH-2.py  # 使用simhash库实现文本指纹计算、相似度计算和查询
│    测试.py  # 使用simhash库实现文本指纹计算和相似度计算
│  
│    Minhash.py  # MinHash和Jaccard计算集合的指纹和相似度
│  
│    Karp-Rabin.py  # Karp-Rabin哈希+汉明距，计算文本相似度
│    Karp-Rabin2.py  # Karp-Rabin哈希的另一种逻辑
│  
│    Winnowing.py  # winnowing计算文本指纹
│    Winnowing2.py  # Winnowing.py中函数拆分写了，还多一个基于汉明距的相似度计算
│    set.py  # Winnowing计算集合的指纹+jaccard相似度
│  
│    OCR.py  # Fuzzy hashes & flyhash 两个哈希算法
│
│    N-gram+LSH.py  # N-gram+汉明距计算文本相似度
│  
│    run_wmd.py  # 基于WMD的文本距离计算
```

# 文档哈希指纹生成算法

## Simhash

**Simhash**是一种用于文本相似性比较的哈希算法， google 用来处理海量文本去重的算法，可以将一个文档转换成一个 64 位的字节（特征字）。Simhash可以基于文本内容的特征向量，对相似的文本生成相似的哈希值（具有较好的局部变化容忍性，对于细微差异的文本可以生成相似的指纹）。

### 算法步骤
1. **分词**：过滤标点等，移除停用词，提取n个特征关键词来表征文本
2. **哈希**：通过hash算法将分词转换为hash值
3. **加权**：分词权重与分词哈希值相乘
4. **累加**：将加权哈希值累加形成一个序列串
5. **二值化**：将序列串转化为0-1串
6. **比较**：（相似度计算-汉明距离）

### 应用场景
一般是对 **文本大于500+** 的内容提前指纹做相似度比较，如果文本较短的话，相似度就会有较大的偏差。

### 参考教程
[海量数据去重之SimHash算法简介和应用 - 墨天轮 (modb.pro)](https://www.modb.pro/db/115729)

[simhash - 大辉_FFf - 博客园 (cnblogs.com)](https://www.cnblogs.com/do-your-best/p/9846174.html) （包含相似度计算）

[simhash - 码农教程 (manongjc.com)](http://www.manongjc.com/detail/22-ogzpqhxcibhxigw.html)

[(107条消息) 文本相似度计算——Simhash算法（python实现）*simhash文本相似度*Trisyp的博客-CSDN博客](https://blog.csdn.net/Trisyp/article/details/113623966?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-113623966-blog-104106867.235%5Ev38%5Epc_relevant_anti_t3_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-113623966-blog-104106867.235%5Ev38%5Epc_relevant_anti_t3_base&utm_relevant_index=2)

## Minhash

Minhash是一种用于**集合**相似性比较的哈希算法，可以用于处理文本的相似性。通过对文本内容进行集合表示，然后使用Minhash算法生成指纹。相似的文本内容会生成相似的Minhash指纹。主要用于计算两个集合的Jaccard相似度，其中Jaccard相似度表示两个集合的交集大小与并集大小的比值。

### 算法步骤
1. **分词**：首先对文件进行分词或处理，将文件拆分成一系列元素（如词语、字符等）。
2. **哈希**：对每个元素进行哈希，得到一个哈希值。
3. **矩阵化**：构建一个signature matrix（签名矩阵），其中每一列代表一个哈希函数对所有元素的哈希值进行映射。
4. **Minhash**: 对于每个哈希函数，找到其对应列中的最小哈希值，并组成一个指纹向量。
5. **相似度计算**：使用Jaccard相似度来比较两个文件的指纹相似度。Jaccard相似度是两个集合的交集大小除以并集大小，用来度量两个集合之间的相似性。

### 应用场景

适用于大规模数据集合的相似性问题，**适合set数据类型**。Minhash算法是一个近似算法，相似性的计算结果并不是完全精确的。

### 参考教程
[文本内容相似度计算方法：minhash – 标点符 (biaodianfu.com)](https://www.biaodianfu.com/minhash.html)

[最小哈希签名（MinHash）简述-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/2162118)

[文本去重之MinHash算法 - pathenon - 博客园 (cnblogs.com)](https://www.cnblogs.com/pathenon/archive/2012/07/17/2595778.html)

[minHash最小哈希原理 - stardsd - 博客园 (cnblogs.com)](https://www.cnblogs.com/sddai/p/6110704.html)

### 注

minhash和simhash都属于局部敏感哈希（Local Sensitive Hash）。一般的哈希算法对于相似文本的哈希结果可能差别非常大，局部敏感哈希在普通哈希的基础上保留了一定程度的相似性，即相似文本的哈希结果距离较小。

## Karp-Rabin

Karp-Rabin是一种字符串匹配算法，用于在一个长文本中高效地查找一个短模式的出现位置。该算法是由Michael O. Rabin和Richard M. Karp于1987年提出的。

Karp-Rabin算法的核心思想是通过哈希函数对模式和文本中的子串进行哈希计算，然后比较哈希值来判断是否匹配。它使用了一种滑动窗口的方法，在文本中滑动一个与模式长度相同的窗口，通过哈希函数计算窗口内子串的哈希值，并将其与模式的哈希值进行比较。

### 算法步骤
1. 计算模式的哈希值和第一个窗口子串的哈希值。
2. 在文本中滑动窗口，每次滑动一个字符，并通过哈希函数计算新窗口子串的哈希值。
3. 将新窗口子串的哈希值与模式的哈希值进行比较，如果相等，则说明找到了一个匹配。
4. 如果哈希值不相等，继续滑动窗口，重复步骤3。

### 应用场景

主要用于字符串匹配，因此适合处理字符串类型的数据，比如str

### 参考教程
[Rabin–Karp 算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/563551141?utm_id=0)

[Rabin-Karp算法 - 简书 (jianshu.com)](https://www.jianshu.com/p/e20994e5e33c)

[Rabin-Karp算法概述 - ChristianL - 博客园 (cnblogs.com)](https://www.cnblogs.com/christianl/p/13747580.html)

### 局限性

Karp-Rabin算法生成的指纹可能无法准确反映字符串的相似性，尤其是当两个字符串的相似部分位于不同位置时。Karp-Rabin算法是一种局部敏感哈希方法，它主要关注于字符串的局部特征。如果两个字符串在不同的位置有相似的子串，那么它们的指纹可能会有很大的不同，导致计算的相似度不太准确；

存在哈希冲突的潜在问题，即不同的子串可能有相同的哈希值，导致误判（如str1和2直接完全相等)；

由于滑动窗口的大小和基数、大素数等参数都是固定的，因此如果两个文本具有相同的滑动窗口，它们的指纹将会相同，而且str1和2文本长度较小；

Karp-Rabin算法生成的文本指纹在相同位置具有相同的滑动窗口内容时，其指纹将会相同，导致相似度为1.0。为增加指纹的唯一性，可以尝试调整滑动窗口的大小或选择不同的哈希函数。

## Winnowing

Winnowing算法是一种用于文本数据的局部散列算法，主要用于文本去重和查找近似重复的文本片段。该算法基于散列函数，可以快速地生成文本的指纹，从而方便地进行文本相似性的计算。Winnowing可以有效地识别重复的文本段落或者检测抄袭文本。

基本思想是将文本分成固定大小的滑动窗口，在每个窗口内使用散列函数计算散列值，然后在所有窗口中选择散列值最小的一个作为该位置的指纹。通过这种方式，Winnowing算法可以过滤掉文本中的噪声和不重要的信息，保留重要的特征，从而实现文本去重和相似性计算。

### 算法步骤
1. 预处理文本
2. 将文本分成固定大小的滑动窗口，窗口大小为k。
3. 在每个窗口内对文本片段使用哈希函数计算哈希值。
4. 在所有窗口中选择散列值最小的一个作为该位置的指纹。
5. 最后，得到的指纹序列即为文本的指纹，可以用于文本去重和相似性计算。

### k-grams+Winnowing算法步骤
1. **生成k-grams**：将原始文本切分成连续的k个字符的子串，这些子串被称为k-grams。k-grams是文本的局部特征表示，能够捕捉文本的局部信息。
2. **计算k-grams的哈希值**：对于生成的每个k-gram子串，使用哈希函数将其转换为哈希值。这一步将k-grams映射为哈希值，缩小了指纹的维度。
3. **选取局部最小哈希值**：在生成的哈希值列表中，选择局部最小的哈希值作为该位置的指纹。具体做法是定义一个滑动窗口，在窗口内找到最小的哈希值，并记录下来。随着窗口滑动，不断更新最小值，得到文本的指纹。
4. **连接指纹**：将不同位置的指纹连接起来，形成整个文本的指纹表示。这样，每个文本都会有一串指纹值，代表了文本的局部特征。
5. **存储指纹**：将生成的文本指纹存储起来，用于后续的文本相似度计算和比较。

### 应用场景

具有较好的时间和空间复杂度，适用于大规模文本数据的处理。

### 参考教程

[(107条消息) 【文本相似性计算】winnowing算法_夜谷子的博客-CSDN博客](https://blog.csdn.net/weixin_43098787/article/details/82837867)

[(107条消息) winnowing 算法 -- 提取文档指纹特征*winnowing算法*ouprince的博客-CSDN博客](https://blog.csdn.net/qq_32023541/article/details/82382808)

[(107条消息) 基于K-gram的winnowing特征提取剽窃查重检测技术（概念篇）*kgramhash*君的名字的博客-CSDN博客](https://blog.csdn.net/chichoxian/article/details/53115067?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-2-53115067-blog-82382808.235%5Ev38%5Epc_relevant_anti_t3_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-2-53115067-blog-82382808.235%5Ev38%5Epc_relevant_anti_t3_base&utm_relevant_index=3)

### 注

k-grams+Winnowing算法主要用于文本相似度计算，特别是对于较长文本的相似度比较。通过局部特征的表示，可以捕捉文本的相似部分，而Winnowing算法的局部最小哈希选择策略能够进一步减少指纹的维度，提高计算效率。这种算法适用于大规模文本相似度比较和查重等应用场景。

## Fuzzy Hash

ssdeep 是一个用来计算context triggered piecewise hashes(CTPH) 基于文本的分片哈希算法，同样也可以叫做模糊哈希Fuzzy hashes。CTPH可以匹配同源文档（相似文档），这样的文档可能有一些顺序相同的字节，尽管这些字节可能在一个序列中长度和内容都不尽相同。ssdeep 对文本长度有要求，如果要生成有意义的结果，最好文本长度不小于 4096。

ssdeep的主要原理是使用一个弱哈希计算文件局部内容，在特定条件下对文件进行分片，然后使用一个强哈希对文件每片计算哈希值，取这些值的一部分并连接起来，与分片条件一起构成一个模糊哈希结果。使用一个字符串相似性对比算法判断两个模糊哈希值的相似度有多少，从而判断两个文件的相似程度。

### 参考教程

[CTPH工作原理](https://www.sciencedirect.com/science/article/pii/S1742287606000764?via%3Dihub)

## FlyHash

FlyHash 是一种LSH算法，它将输入数据映射到稀疏哈希嵌入，其中哈希嵌入的维度远大于输入，并将输入数据的位置保留在哈希嵌入中。

### 应用场景

FlyHash被设计为计算成本低廉，但内存效率低。它适用于散列中小型数据 （~10-1000）到大哈希嵌入（~100-10000）。


# 文档哈希指纹距离/相似度计算方法

Jaccard相似度适用于集合数据的相似度计算，汉明距离适用于等长字符串的相似度计算，Simhash相似度适用于保持文本信息的压缩指纹计算，Minhash相似度适用于大规模数据集的近似相似度计算，Karp-Rabin相似度适用于文本数据的快速相似度估计，Winnowing相似度适用于滑动窗口处理的文本相似度计算。

## 汉明距离

汉明距离（Hamming distance）是一种衡量两个等长字符串之间差异的度量方法。它用于计算两个字符串在相同位置上不同字符的个数。换句话说，汉明距离衡量了将一个字符串转换为另一个字符串所需的最小替换次数。

### 算法步骤
1. 首先，将两个字符串对齐，使它们的长度相同。如果两个字符串长度不同，需要在较短的字符串末尾补充空字符，直到两个字符串的长度相等。
2. 然后，逐个比较两个字符串在相同位置上的字符。如果两个字符不相同，则汉明距离加1；如果两个字符相同，则汉明距离不变。
3. 最后，将所有不相同字符的个数相加，得到最终的汉明距离。

### 注

如果是**不等长字符串**：常见的做法是先将它们转换成等长的特征向量，然后再计算特征向量之间的相似度。常见的方法包括使用N-gram特征或TF-IDF特征等。

N-gram特征：N-gram是一种将文本切分为连续的N个字符或单词的方法。对于每个字符串，我们可以将其切分为N-gram，并统计每个N-gram在字符串中出现的次数，形成一个特征向量。然后，通过比较两个特征向量之间的相似度，可以得到两个不等长字符串的相似度。

## Jaccard相似度

Jaccard相似度是一种用于计算两个集合之间相似性的指标，它衡量两个集合的交集元素与并集元素之间的比例。Jaccard相似度的取值范围在0到1之间，其中0表示两个集合没有共同的元素，1表示两个集合完全相同。

Jaccard相似度在数据处理和信息检索中广泛应用，特别适用于处理文本数据、集合数据、网络图等。在文件指纹相似度计算中，Jaccard相似度也常用于衡量两个文件指纹之间的相似性。

### 算法步骤
1. 首先，计算两个集合的交集，即两个集合中共有的元素。
2. 然后，计算两个集合的并集，即两个集合中所有的元素，包括重复的元素。
3. 最后，用交集的大小除以并集的大小，得到Jaccard相似度。

## 余弦相似度

余弦相似度（Cosine Similarity）是一种常用的相似度计算方法，它可以用于衡量两个向量之间的相似程度。在文本相似度计算中，可以将文本表示成向量，然后使用余弦相似度来比较两个文本的相似性。

### 算法步骤

假设有两个向量A和B，它们分别是n维向量（可以是词频向量、TF-IDF向量等），那么余弦相似度可以通过以下公式计算：

Similarity = A · B / (||A|| \* ||B||)

其中，A · B表示向量A和向量B的点积（内积），||A||表示向量A的模长，||B||表示向量B的模长。

余弦相似度的取值范围在[-1, 1]之间，取值越接近1表示两个向量越相似，取值越接近-1表示两个向量越不相似，取值为0表示两个向量正交（无相似性）。
