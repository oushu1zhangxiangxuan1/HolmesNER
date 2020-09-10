from pyhanlp import HanLP, SafeJClass
# from pyhanlp import JClass

"""
reference:
https://www.cnblogs.com/fonttian/p/9819779.html
"""

sentences = [
    "北川景子参演了林诣彬导演的《速度与激情3》",
    "林志玲亮相网友:确定不是波多野结衣？",
    "龟山千广和近藤公园在龟山公园里喝酒赏花",
]


def raw_seg():
    """
    newSegment()支持下列多种模式，默认使用viterbi
    维特比 (viterbi)：效率和效果的最佳平衡。也是最短路分词，HanLP最短路求解采用Viterbi算法
    双数组trie树 (dat)：极速词典分词，千万字符每秒（可能无法获取词性，此处取决于你的词典）
    条件随机场 (crf)：分词、词性标注与命名实体识别精度都较高，适合要求较高的NLP任务
    感知机 (perceptron)：分词、词性标注与命名实体识别，支持在线学习
    N最短路 (nshort)：命名实体识别稍微好一些，牺牲了速度
    """
    seg = HanLP.newSegment()
    for st in sentences:
        print(seg.seg(st))

    seg_crf = HanLP.newSegment("crf")
    for st in sentences:
        print(seg_crf.seg(st))

    """
    # viterbi
    [北川景子/nrj, 参演/v, 了/ule, 林诣彬/nr, 导演/nnt, 的/ude1, 《/w, 速度/n, 与/cc, 激情/n, 3/m, 》/w]
    [林志玲/nr, 亮相/vi, 网友/n, :/w, 确定/v, 不是/c, 波多/nrf, 野/b, 结衣/nz, ？/w]
    [龟山/nz, 千/m, 广/a, 和/cc, 近藤/nz, 公园/n, 在/p, 龟山/nz, 公园/n, 里/f, 喝酒/vi, 赏花/nz]
    # crf
    [北川景子/nrj, 参演/v, 了/u, 林诣彬/nr, 导演/n, 的/u, 《/w, 速度/n, 与/c, 激情/n, 3/m, 》/w]
    [林志玲/nr, 亮相/v, 网友/n, :/w, 确定/v, 不/d, 是/v, 波多野/n, 结衣/n, ？/w]
    [龟/v, 山/n, 千/m, 广/q, 和/c, 近藤/a, 公园/n, 在/p, 龟山公园/ns, 里/f, 喝/v, 酒/n, 赏/v, 花/n]
    """


def enable_seg():
    seg = HanLP.newSegment()

    # 中文人名识别
    seg = seg.enableNameRecognize(True)

    # 音译人名识别
    seg = seg.enableTranslatedNameRecognize(True)

    # 日语人名识别
    seg = seg.enableJapaneseNameRecognize(True)

    # 地名识别
    seg = seg.enablePlaceRecognize(True)

    # 机构名识别
    seg = seg.enableOrganizationRecognize(True)

    for st in sentences:
        print(seg.seg(st))


def result_format():
    HanLP.Config.ShowTermNature = False
    seg = HanLP.newSegment()
    print(seg.seg(sentences[0]))
    HanLP.Config.ShowTermNature = True
    seg = HanLP.newSegment()
    term_list = seg.seg(sentences[0])
    print(term_list)
    print([str(i.word) for i in term_list])
    print([str(i.nature) for i in term_list])


def url_recognition():
    # URL 识别
    text = """HanLP的项目地址是https://github.com/hankcs/HanLP，
            发布地址是https://github.com/hankcs/HanLP/releases，
            我有时候会在www.hankcs.com上面发布一些消息，
            我的微博是http://weibo.com/hankcs/，会同步推送hankcs.com的新闻。
            听说.中国域名开放申请了,但我并没有申请hankcs.中国,因为穷……
                """

    Nature = SafeJClass("com.hankcs.hanlp.corpus.tag.Nature")
    # Term = SafeJClass("com.hankcs.hanlp.seg.common.Term")
    URLTokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.URLTokenizer")

    term_list = URLTokenizer.segment(text)
    print(term_list)
    for term in term_list:
        if term.nature == Nature.xu:
            print(term.word)


def number_recognition():
    # 演示数词与数量词识别
    sentences = [
        "十九元套餐包括什么",
        "九千九百九十九朵玫瑰",    
        "壹佰块都不给我",
        "９０１２３４５６７８只蚂蚁",
        "牛奶三〇〇克*2",
        "ChinaJoy“扫黄”细则露胸超2厘米罚款",
    ]

    seg = HanLP.newSegment().enableNumberQuantifierRecognize(True)

    print("\n========== 演示数词与数量词 开启 ==========\n")
    for st in sentences:
        print(seg.seg(st))
    print("\n========== 演示数词与数量词 默认未开启 ==========\n")
    print(HanLP.newSegment().seg(sentences[0]))


def main():
    # raw_seg()

    # result_format()

    # enable_seg()

    url_recognition()

    number_recognition()


if "__main__" == __name__:
    main()
