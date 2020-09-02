## 概念
```
命名实体识别（英文：Named Entity Recognition），简称NER，是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等，以及时间、数量、货币、比例数值等文字。
```

## 1. fastText

## 2. HanLP

### 1. 词性及实体标注格式

- http://www.hankcs.com/nlp/part-of-speech-tagging.html#h2-8

### 2. 命名实体识别选项

```
    newSegment()支持下列多种模式，默认使用viterbi
    维特比 (viterbi)：效率和效果的最佳平衡。也是最短路分词，HanLP最短路求解采用Viterbi算法
    双数组trie树 (dat)：极速词典分词，千万字符每秒（可能无法获取词性，此处取决于你的词典）
    条件随机场 (crf)：分词、词性标注与命名实体识别精度都较高，适合要求较高的NLP任务
    感知机 (perceptron)：分词、词性标注与命名实体识别，支持在线学习
    N最短路 (nshort)：命名实体识别稍微好一些，牺牲了速度
```

### 3. 其它

- 1. 离线安装需要下载部分static到安装包
  ```
  下载 https://file.hankcs.com/hanlp/hanlp-1.7.8-release.zip 到 ${PYTHONPATH}/site-packages/pyhanlp/static/hanlp-1.7.8-release.zip
  下载 https://file.hankcs.com/hanlp/data-for-1.7.5.zip 到 ${PYTHONPATH}/site-packages/pyhanlp/static/data-for-1.7.8.zip
  ```

### 4. 参考
- https://www.cnblogs.com/fonttian/p/9819779.html

### 5. 总结
#### 1. 优点
- 方便快捷
#### 2. 不足
- 依赖java环境
- 无法根据业务需求对垂直领域不同实体进行细分
- 更多偏向词性标注

## 3. BERT
### 1. 参考
- https://eliyar.biz/nlp_chinese_bert_ner/
  
### 2. 

## 4. 特殊实体识别

- 电话号码
- 身份证号
- 网址
- qq及微信号码
- 银行卡号