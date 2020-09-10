##

## 相关技术及概念
 
### 1. Distant Supervision
- 简介
  ```
  通过将知识库与非结构化文本对齐来自动构建大量训练数据，减少模型对人工标注数据的依赖，增强模型跨领域适应能力。

  Distant Supervision 的提出主要基于以下假设：两个实体如果在知识库中存在某种关系，则包含该两个实体的非结构化句子均能表示出这种关系。

  例如，“Steve Jobs”, "Apple"在 Freebase 中存在 founder 的关系，则包含这两个实体的非结构文本“Steve Jobs was the co-founder and CEO of Apple and formerly Pixar.”可以作为一个训练正例来训练模型。
  ```
- 步骤
  ```
  1. 从知识库中抽取存在关系的实体对

  2. 从非结构化文本中抽取含有实体对的句子作为训练样例
  ```
- https://www.dazhuanlan.com/2019/09/22/5d876517af55b/  主要介绍distant supervision
- https://www.jiqizhixin.com/articles/2017-04-05-6
  
### 2. SPO三元组
- 包括
  ```
  Subject
  Predicate
  Object
  ```
- https://www.sohu.com/a/340787983_787107
- 例子
  ```
  如 O
  何 O
  演 O
  好 O
  自 O
  己 O
  的 O
  角 O
  色 O
  ， O
  请 O
  读 O
  《 O
  演 O
  员 O
  自 O
  我 O
  修 O
  养 O
  》 O
  《 O
  喜 B-SUBJ
  剧 I-SUBJ
  之 I-SUBJ
  王 I-SUBJ
  》 O
  周 B-OBJ
  星 I-OBJ
  驰 I-OBJ
  崛 O
  起 O
  于 O
  穷 O
  困 O
  潦 O
  倒 O
  之 O
  中 O
  的 O
  独 O
  门 O
  秘 O
  笈 O
  ```

### 3. 失败例子

- 依存语法分析 https://python.ctolib.com/SeanLee97-TripleIE.html

## 基于BERT的关系抽取

### 1. 参考
- https://blog.csdn.net/weixin_42001089/article/details/97657149  后半部分
- https://kexue.fm/archives/6736#%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96  参考code