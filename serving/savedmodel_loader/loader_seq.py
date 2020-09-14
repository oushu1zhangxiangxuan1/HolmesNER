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
        tf.logging.info("building example %d : %s" % (ex_index, example))
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


def build_single_predict_data_old(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""\

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

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

    # label_list = example.label.split(" ")
    # label_ids = _predicate_label_to_id(label_list, label_map)

    # if ex_index < 5:
    #     tf.logging.info("*** Example ***")
    #     tf.logging.info("guid: %s" % (example.guid))
    #     tf.logging.info("tokens: %s" % " ".join(
    #         [tokenization.printable_text(x) for x in tokens]))
    #     tf.logging.info("input_ids: %s" %
    #                     " ".join([str(x) for x in input_ids]))
    #     tf.logging.info("input_mask: %s" %
    #                     " ".join([str(x) for x in input_mask]))
    #     tf.logging.info("segment_ids: %s" %
    #                     " ".join([str(x) for x in segment_ids]))
    #     tf.logging.info("label_ids: %s" %
    #                     " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=[0]*len(label_map),
        is_real_example=True)
    return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def build_single_predict_data(ex_index, example, predicate_id, token_label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    token_label_map = {}
    for (i, label) in enumerate(token_label_list):
        token_label_map[label] = i

    text_token = list(example[0])

    tokens_b = [predicate_id] * len(text_token)

    _truncate_seq_pair(text_token, tokens_b, max_seq_length - 3)

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in text_token:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # bert_tokenizer.convert_tokens_to_ids(["[SEP]"]) --->[102]
    bias = 1  # 1-100 dict index not used
    # for token in tokens_b:
    #     # add  bias for different from word dict
    #     input_ids.append(predicate_id + bias)
    #     segment_ids.append(1)
    input_ids.extend([predicate_id + bias]*len(text_token))
    segment_ids.extend([1]*len(text_token))

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
        segment_ids=segment_ids)
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

    max_seq_length = 128

    tokenizer = get_tokenizer()

    sess = tf.Session()

    model_path = "/home/johnsaxon/github.com/Entity-Relation-Extraction/output/saved_model/seq/1600064798"

    tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        model_path
    )

    features = sequence_tokenizer(
        examples, predicates_ids, token_label_list, max_seq_length, tokenizer)

    predicate_prediction = sess.graph.get_tensor_by_name("predicate_loss/ArgMax:0")

    predicate_probabilities = sess.graph.get_tensor_by_name("predicate_loss/Softmax:0")

    token_label_predictions = sess.graph.get_tensor_by_name("token_label_loss/ArgMax:0")

    prediction = sess.run(
        # 'tensorflow/serving/predict',
        # feed_dict={
        #         'input_ids:0': tf.convert_to_tensor(features['input_ids']),
        #         'input_mask:0': tf.convert_to_tensor(features['input_mask']),
        #         'segment_ids:0': tf.convert_to_tensor(features['segment_ids'])
        #     }
        (predicate_prediction, predicate_probabilities, token_label_predictions),
        feed_dict={
            'input_ids:0': features['input_ids'],
            'input_mask:0': features['input_mask'],
            'segment_ids:0': features['segment_ids']
        }
    )

    print(prediction)


def create_session():

    # examples = [
    #     "《中国风水十讲》是2007年华夏出版社出版的图书，作者是杨文衡",
    #     "你是最爱词:许常德李素珍/曲:刘天健你的故事写到你离去后为止",
    #     "《苏州商会档案丛编第二辑》是2012年华中师范大学出版社出版的图书，作者是马敏、祖苏、肖芃"
    # ]

    # label_list = get_labels()

    # max_seq_length = 128

    # tokenizer = get_tokenizer()

    sess = tf.Session()

    # meta_file = '/home/johnsaxon/github.com/Entity-Relation-Extraction/output/predicate_classification_model/epochs6/model.ckpt-487.meta'
    # ckpt_path = '/home/johnsaxon/github.com/Entity-Relation-Extraction/output/predicate_classification_model/epochs6'

    # saver = tf.train.import_meta_graph(meta_file)
    # saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

    # graph = tf.get_default_graph()

    # inputs_ids = graph.get_tensor_by_name('input_ids')
    # input_mask = graph.get_tensor_by_name('input_mask')
    # segment_ids = graph.get_tensor_by_name('segment_ids')

    model_path = "/Users/johnsaxon/test/github.com/Entity-Relation-Extraction/output/1599723701"

    tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        model_path
    )

    fetches = sess.graph.get_tensor_by_name("loss/Sigmoid:0")
    # fetches = sess.graph.get_tensor_by_name("TPUPartitionedCall:1")
    # fetches = sess.graph.get_tensor_by_name("probabilities")

    return fetches, sess


if "__main__" == __name__:
    main()
