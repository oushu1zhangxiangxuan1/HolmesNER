import tensorflow as tf
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(
#     os.path.dirname(__file__), "../bert")))

from bert import tokenization
from est_cls import InputFeatures

vocab_file = "/home/johnsaxon/github.com/Entity-Relation-Extraction/pretrained_model/chinese_L-12_H-768_A-12/vocab.txt"


def sequence_tokenizer(examples, examples_predicate, token_label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a TFRecord file."""

    input_ids = []
    input_mask = []
    segment_ids = []

    for (ex_index, example) in enumerate(examples):
        print("building example %d : %s" % (ex_index, example))
        for predicate_id in examples_predicate[ex_index]:
            feature = build_single_predict_data(
                ex_index, example, predicate_id, token_label_list, max_seq_length, tokenizer)
            input_ids.append(feature.input_ids)
            input_mask.append(feature.input_mask)
            segment_ids.append(feature.segment_ids)

    features = dict()
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    features["segment_ids"] = segment_ids

    return features


def build_single_predict_data(ex_index, example, predicate_id, token_label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    print("in build_single_predict_data")
    tf.logging.info("in build_single_predict_data")

    token_label_map = {}
    for (i, label) in enumerate(token_label_list):
        token_label_map[label] = i

    # TODO:
    # 对文本中存在英文单词的情况识别不好
    # 遇到未登录词会直接失败，需要提前处理，先转回成[UNK]
    # 预测完返回识别的时候也需要通过这个index去处理
    # text_token = list(example)
    tokens_a = tokenizer.tokenize(example)

    tokens = []
    input_ids = []
    segment_ids = []

    text_input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
    text_token_len = len(text_input_ids)

    input_ids.extend(tokenizer.convert_tokens_to_ids(["[CLS]"]))
    input_ids.extend(text_input_ids)
    input_ids.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))

    segment_ids.extend([0]*len(input_ids))

    # bert_tokenizer.convert_tokens_to_ids(["[SEP]"]) --->[102]
    bias = 1  # 1-100 dict index not used
    input_ids.extend([predicate_id + bias]*text_token_len)
    segment_ids.extend([1]*text_token_len)

    input_ids.append(tokenizer.convert_tokens_to_ids(["[SEP]"])[0])  # 102
    segment_ids.append(1)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if ex_index < 5:
        # print("*** Example ***")
        # print("example: %s" % (example))
        # print("input_ids: %s" %
        #                 " ".join([str(x) for x in input_ids]))
        # print("input_mask: %s" %
        #                 " ".join([str(x) for x in input_mask]))
        # print("segment_ids: %s" %
        #                 " ".join([str(x) for x in segment_ids]))
        
        tf.logging.info("*** Example ***")
        tf.logging.info("example: %s" % (example))
        tf.logging.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=None)
    return feature


def get_tokenizer():
    return tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=False)


def get_labels():
    return ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期',
            '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自',
            '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站',
            '邮政编码', '面积', '首都']


def get_token_labels():
    BIO_token_labels = ["[Padding]", "[category]", "[##WordPiece]", "[CLS]", "[SEP]", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "O"]  # id 0 --> [Paddding]
    return BIO_token_labels


def main():

    examples = [
        "《中国风水十讲》是2007年华夏出版社出版的图书，作者是杨文衡",
        "你是最爱词:许常德李素珍/曲:刘天健你的故事写到你离去后为止",
        "《苏州商会档案丛编第二辑》是2012年华中师范大学出版社出版的图书，作者是马敏、祖苏、肖芃"
    ]

    predicates = [
        ['作者', '出版社'],
        ['作者', '出版社'],
        ['作者', '出版社'],
    ]

    predicate_label_list = get_labels()

    predicates_ids = []
    for preds in predicates:
        ids = []
        for pred in preds:
            ids.append(predicate_label_list.index(pred))
        predicates_ids.append(ids)

    token_label_list = get_token_labels()

    token_label_id2label = {}
    for (i, label) in enumerate(token_label_list):
        token_label_id2label[i] = label

    max_seq_length = 128

    tokenizer = get_tokenizer()

    sess = tf.compat.v1.Session()

    model_path = "/home/johnsaxon/github.com/Entity-Relation-Extraction/output/saved_model/seq/1600064798"
    # tf.saved_model.loader.load(
    tf.compat.v1.saved_model.loader.load(
        sess,
        [tf.saved_model.SERVING],
        model_path
    )

    features = sequence_tokenizer(
        examples, predicates_ids, token_label_list, max_seq_length, tokenizer)

    predicate_prediction = sess.graph.get_tensor_by_name("predicate_loss/ArgMax:0")

    predicate_probabilities = sess.graph.get_tensor_by_name("predicate_loss/Softmax:0")

    token_label_predictions = sess.graph.get_tensor_by_name("token_label_loss/ArgMax:0")

    predicate_prediction_result, predicate_probabilities_result, token_label_predictions_result = sess.run(
        (predicate_prediction, predicate_probabilities, token_label_predictions),
        feed_dict={
            'input_ids:0': features['input_ids'],
            'input_mask:0': features['input_mask'],
            'segment_ids:0': features['segment_ids']
        }
    )
    sess.close()

    for token_label_prediction in token_label_predictions_result:
        token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction)
        print(token_label_output_line)
        print("\n")


def create_seq_session():

    g = tf.Graph()

    sess = tf.compat.v1.Session(graph=g)

    model_path = "/home/johnsaxon/github.com/Entity-Relation-Extraction/output/saved_model/seq/1600064798"
    tf.compat.v1.saved_model.loader.load(
        sess,
        [tf.saved_model.SERVING],
        model_path
    )

    

    predicate_prediction = sess.graph.get_tensor_by_name("predicate_loss/ArgMax:0")

    predicate_probabilities = sess.graph.get_tensor_by_name("predicate_loss/Softmax:0")

    token_label_predictions = sess.graph.get_tensor_by_name("token_label_loss/ArgMax:0")

    return (predicate_prediction, predicate_probabilities, token_label_predictions), sess


if "__main__" == __name__:
    # main()
    f, s = create_seq_session()
    _, s2 = create_seq_session()
    s.close()
    s2.close()
