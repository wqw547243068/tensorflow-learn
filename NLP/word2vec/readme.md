
# 词向量

## Wordvec

### 简介
word2vec 是 Google 于 2013 年开源推出的一个用于获取 word vector 的工具包，它简单、高效，因此引起了很多人的关注。对word2vec数学原理感兴趣的可以移步word2vec 中的数学原理详解，这里就不具体介绍。word2vec对词向量的训练有两种方式，一种是CBOW模型，即通过上下文来预测中心词；另一种skip-Gram模型，即通过中心词来预测上下文。其中CBOW对小型数据比较适合，而skip-Gram模型在大型的训练语料中表现更好。
两种模型结构如下：
![graph](https://pic3.zhimg.com/80/v2-ec1758da5fe00e7bb6d5f73524f19d4c_hd.jpg)

### 源码
- word2vec的源码github上可以找到点[这里](https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)，这里面已经实现了对英文的训练

### 步骤
对中文的训练步骤有：
- 对文本进行分词，采用的jieba分词
- 将语料中的所有词组成一个列表，为构建词频统计，词典及反转词典。因为计算机不能理解中文，我们必须把文字转换成数值来替代。
- 构建skip-Gram模型需要的训练数据：由于这里采用的是skip-Gram模型进行训练，即通过中心词预测上下文。因此中心词相当于x，上下文的词相当于y。这里我们设置上下文各为一个词，假设我要对“恐怕 顶多 只 需要 三年 时间”这段话生成样本，我们应该通过“顶多”预测“恐怕”和“只”；通过“只”预测“顶多”和“需要”依次下去即可。最终的训练样本应该为（顶多，恐怕），（顶多，只），（只，顶多），（只，需要），（需要，只），（需要，三年）。

### gensim实现
- `sudo pip install gensim`
- [用gensim训练word2vec](https://zhuanlan.zhihu.com/p/29200034)

### tensorflow实现
- 参考地址：[使用tensorflow实现word2vec中文词向量的训练](https://zhuanlan.zhihu.com/p/28979653)
- 经过大约三小时的训练后，使用s-TNE把词向量降至2维进行可视化，部分词可视化结果如下：
![effect](https://pic3.zhimg.com/80/v2-091fc27bb5a5bbd942da4b702d580199_hd.jpg)

随机对几个词进行验证，得到的结果为：
>
Nearest to 萧炎: 萧炎, 他, 韩枫, 林焱, 古元, 萧厉, 她, 叶重,
Nearest to 灵魂: 灵魂, 斗气, 触手可及, 乌钢, 探头探脑, 能量, 庄严, 晋阶,
Nearest to 火焰: 火焰, 异火, 能量, 黑雾, 火苗, 砸场, 雷云, 火海,
Nearest to 天阶: 天阶, 地阶, 七品, 相媲美, 斗帝, 碧蛇, 稍有不慎, 玄阶,
Nearest to 云岚宗: 云岚宗, 炎盟, 魔炎谷, 磐门, 丹塔, 萧家, 叶家, 花宗,
Nearest to 乌坦城: 乌坦城, 加玛, 大殿, 丹域, 兽域, 大厅, 帝国, 内院,
Nearest to 惊诧: 惊诧, 惊愕, 诧异, 震惊, 惊骇, 惊叹, 错愕, 好笑,

# 其它
