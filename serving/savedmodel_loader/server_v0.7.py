import json
from flask import Flask
from flask import request
# import tensorflow as tf
# from bert import modeling
# from bert import optimization
# from bert import tokenization
import est_cls
from loader import create_session
from loader_seq import create_seq_session
from loader import string_tokenizer
from loader_seq import sequence_tokenizer
from loader_seq import get_token_labels
import loader
import tensorflow as tf
# import numpy as np

app = Flask(__name__)

# fetches, sess = create_session()

# fetches_seq, sess_seq = create_seq_session()

fetches, sess = None, None

fetches_seq, sess_seq = None, None

label_list = est_cls.get_labels()

tokenizer = loader.get_tokenizer()

max_seq_length = 128
max_raw_str_len = (max_seq_length - 3)//2

# print("max_raw_str_len: ", max_raw_str_len)

token_label_list = get_token_labels()
token_label_id2label = {}
for (i, label) in enumerate(token_label_list):
    token_label_id2label[i] = label


@app.route('/')
def home():
    return "Welcome to Relation Extraction System"


@app.route('/predict_v1', methods=['POST'])
def predict_v1():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    # 对不同数据进行切分，切分到小于max_raw_str_len
    res = []
    for doc in json_data["data"]:
        segments = getSegments(doc)

        # raw seg
        # token 
        # token_unk
        # predicate_ids

        print("Total {} segments:\n{}".format(len(segments), segments))

        features = string_tokenizer(segments, label_list, max_seq_length, tokenizer)

        result = sess.run(
            fetches,
            feed_dict={
                    'input_ids:0': features['input_ids'],
                    'input_mask:0': features['input_mask'],
                    'segment_ids:0': features['segment_ids']
                    }
        )

        predicate_ids = []

        for prediction in result:
            prediction = prediction.tolist()
            predicate_id = []
            for idx, class_probability in enumerate(prediction):
                predicate_predict = []
                if class_probability > 0.5:
                    predicate_predict.append(label_list[idx])
                    predicate_id.append(idx)
            
            predicate_ids.append(predicate_id)
        # TODO: 其中某句话可能没有predicate，此时应该去除

        seq_segments, seq_predicate_ids = [], []
        for seg, pred in zip(segments, predicate_ids):
            if len(pred)>0:
                seq_segments.append(seg)
                seq_predicate_ids.append(pred)
        

        triples = []
        if len(seq_segments)>0:
            features_seq = sequence_tokenizer(seq_segments, seq_predicate_ids, label_list, max_seq_length, tokenizer)

            print("\nfeatures_seq:\n{}\n\n".format(features_seq))

            seq_result = sess_seq.run(
                # TODO: 是否可以只跑fetches_seq【3】
                fetches_seq,
                feed_dict={
                        'input_ids:0': features_seq['input_ids'],
                        'input_mask:0': features_seq['input_mask'],
                        'segment_ids:0': features_seq['segment_ids']
                        }
            )

            seq_label_result = seq_result[2]

            # for token_label_prediction in seq_label_result:
            #     token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction)
            #     print("raw out:\n{}".format(token_label_prediction))
            #     print("label out:\n{}".format(token_label_output_line))
            #     print("\n")

            res_index = 0
            for (i, src) in enumerate(seq_segments):
                for predicate_id in seq_predicate_ids[i]:
                    pred = label_list[predicate_id]
                    trps = getTriples_v1(seq_label_result[res_index], src ,pred)
                    triples.extend(trps)
                    res_index += 1

            # print("triples: {}".format(triples))

        res.append(triples)

    return json.dumps(res, ensure_ascii=False)


@app.route('/predict/<data>', methods=['GET'])
def predict_single(data):
    response = []

    predict_test_data = [
        # "《中国风水十讲》是2007年华夏出版社出版的图书，作者是杨文衡",
        # "你是最爱词:许常德李素珍/曲:刘天健你的故事写到你离去后为止",
        # "《苏州商会档案丛编第二辑》是2012年华中师范大学出版社出版的图书，作者是马敏、祖苏、肖芃",
        "广州珂妃化妆品有限公司创立于2005年，是一家集研发、生产、销售、服务为一体的多元化化妆品集团公司",
        "吕雅堂（1907-）安徽寿县人，1926年10月南京中央军校第六期，后任国民革命军徐州剿总骑兵第一旅副旅长",
    ]
    # num_actual_predict_examples = len(predict_test_data)

    features = string_tokenizer(predict_test_data, label_list, max_seq_length, tokenizer)

    result = sess.run(
        fetches,
        feed_dict={
                'input_ids:0': features['input_ids'],
                'input_mask:0': features['input_mask'],
                'segment_ids:0': features['segment_ids']
                }
    )

    for prediction in result:
        prediction = prediction.tolist()
        print("\n\n prediction:\n{}".format(prediction))
        for idx, class_probability in enumerate(prediction):
            predicate_predict = []
            if class_probability > 0.5:
                print(label_list[idx])
        predicate_predict = []
        predicate_predict.append(label_list[prediction.index(max(prediction))])
        res = {"relations": predicate_predict}
        response.append(res)

    return json.dumps(response)


@app.route('/pipline/<data>', methods=['GET'])
def predict_pipline(data):
    response = []

    predict_test_data = [
        # "《中国风水十讲》是2007年华夏出版社出版的图书，作者是杨文衡",
        # "你是最爱词:许常德李素珍/曲:刘天健你的故事写到你离去后为止",
        # "《苏州商会档案丛编第二辑》是2012年华中师范大学出版社出版的图书，作者是马敏、祖苏、肖芃",
        "广州珂妃化妆品有限公司创立于2005年，是一家集研发、生产、销售、服务为一体的多元化化妆品集团公司",
        "吕雅堂（1907-）安徽寿县人，1926年10月南京中央军校第六期，后任国民革命军徐州剿总骑兵第一旅副旅长",
    ]
    # num_actual_predict_examples = len(predict_test_data)

    features = string_tokenizer(predict_test_data, label_list, max_seq_length, tokenizer)

    result = sess.run(
        fetches,
        feed_dict={
                'input_ids:0': features['input_ids'],
                'input_mask:0': features['input_mask'],
                'segment_ids:0': features['segment_ids']
                }
    )

    predicate_ids = []

    for prediction in result:
        prediction = prediction.tolist()
        print("\n\n prediction:\n{}".format(prediction))

        predicate_id = []
        for idx, class_probability in enumerate(prediction):
            predicate_predict = []
            if class_probability > 0.5:
                predicate_predict.append(label_list[idx])
                predicate_id.append(idx)
        
        predicate_ids.append(predicate_id)
        res = {"relations": predicate_predict}
        response.append(res)

    features_seq = sequence_tokenizer(predict_test_data, predicate_ids, label_list, max_seq_length, tokenizer)
    seq_result = sess_seq.run(
        fetches_seq,
        feed_dict={
                'input_ids:0': features_seq['input_ids'],
                'input_mask:0': features_seq['input_mask'],
                'segment_ids:0': features_seq['segment_ids']
                }
    )

    seq_label_result = seq_result[2]

    for token_label_prediction in seq_label_result:
        token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction)
        print("raw out:\n{}".format(token_label_prediction))
        print("label out:\n{}".format(token_label_output_line))
        print("\n")

    res_index = 0
    triples = []
    for (i, src) in enumerate(predict_test_data):
        for predicate_id in predicate_ids[i]:
            pred = label_list[predicate_id]
            # sub, obj = getS
            # obj = 
            trps = getTriples_v1(seq_label_result[res_index], src ,pred)
            triples.extend(trps)
            res_index += 1
        

    # print("seq result: {}".format(seq_label_result))

    print("triples: {}".format(triples))

    return json.dumps(triples, ensure_ascii=False)


def getTriples_v1(labels_ids, src, pred):
    # BIO_token_labels = ["[Padding]", "[category]", "[##WordPiece]", "[CLS]", "[SEP]", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "O"]  # id 0 --> [Paddding]

    print("labels_ids in getTriples_v1:\n{}".format(labels_ids))
    labels = [token_label_id2label[id] for id in labels_ids]
    print("labels in getTriples_v1:\n{}".format(labels))
    labels.pop(0)
    triples = []
    subs = []
    objs = []
    sub_indices = []
    obj_indices = []
    last_label = None
    last_i = 0
    bounds_map = {
        "B-SUB":"I-SUB", "B-OBJ":"I-OBJ"
    }
    label_index_map = {
        "B-SUB":sub_indices, "B-OBJ":obj_indices
    }
    for (i, label) in enumerate(labels):
        # TODO: "B-SUB", "B-OBJ"这类有点 hard-coding  如果后续有自定义标记的话，需要重写此处代码
        if last_label is None:
            if label in ("B-SUB", "B-OBJ"):
                last_label = label
                last_i = i
        else:
            if label != bounds_map[last_label]:
                label_index_map[last_label].append((last_i, i))
                last_label=None
            if label == "[SEP]":
                break
            if label in ("B-SUB", "B-OBJ"):
                last_label = label
                last_i = i

    # print("sub_indices:\n{}".format(sub_indices))
    # print("obj_indices:\n{}".format(obj_indices))

    for (start, end) in sub_indices:
        subs.append(src[start:end])
    

    for (start, end) in obj_indices:
        objs.append(src[start:end])

    for sub in subs:
        for obj in objs:
            triples.append({
                    "SUB":sub,
                    "predicate":pred,
                    "OBJ":obj
                    })
    return triples


def getSegments(document, depth=0):
    """
    It dosen't not work well when:
        1. document has quotation marks and split punctuation inside 
        2. Long is too long and without any split punctuation
    """

    """
    Negative examples:
        核心团队来自EMC/Pivotal，IBM，Oracle，Google， Microsoft等著名AI和大数据企业，拥有全球领先的AI和大数据处理技术，
    """

    # print("depth-{}:{}".format(depth, document))

    depth += 1

    # TODO: omits space etc

    segments = []
    doc_length = len(document)
    if doc_length <= max_raw_str_len:
        segments.append(document)
        return segments
    else:
        has_split_punc = False
        split_punctuation = ["。",".","！","!","；",";","？","?","，",","]
        for punc in split_punctuation:
            punc_index = index_safe(document, punc)
            if punc_index < 0:
                continue
            has_split_punc = True

            seg1, seg2 = [], []

            # avoid situation like this "哈哈。blablabla..." or "blablabla...。哈哈"
            if punc_index>2:
                if punc_index<doc_length-2:
                    seg1 = getSegments(document[:punc_index], depth)
                    seg2 = getSegments(document[punc_index+1:], depth)
                else:
                    seg1 = getSegments(document[:punc_index], depth)

            else:
                if punc_index<doc_length-2:
                    seg2 = getSegments(document[punc_index+1:], depth)
            break

            
        if not has_split_punc:
            seg1 = [document[:max_raw_str_len]]
            seg2 = getSegments(document[max_raw_str_len:], depth)
        
        segments.extend(seg1)
        segments.extend(seg2)

    return segments


def index_safe(src, sub):
    try:
        return src.index(sub)
    except Exception:
        return -1


class TokenManager:
    def __init__(self, segment):
        self.segment = segment
        self.token = tokenizer.tokenize(self.segment)
        self.token_not_UNK = tokenizer.tokenize_not_UNK(self.segment)
        self.predicate_ids = []
        self.input_ids = tokenizer.convert_tokens_to_ids(self.token)
    def addPredicateId(self, predicate_id):
        self.predicate_ids.append(predicate_id)


def getTokenManagers(segments):
    tokenManagers = []
    for seg in segments:
        tokenManagers.append(TokenManager(seg))
    return tokenManagers


@app.route('/holmes/nlp/re', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    # 对不同数据进行切分，切分到小于max_raw_str_len
    res = []
    for doc in json_data["data"]:
        segments = getSegments(doc)

        tokenManagers = getTokenManagers(segments)

        print("Total {} segments:\n{}".format(len(segments), segments))

        features = cls_tokenizer(tokenManagers, max_seq_length, tokenizer)

        result = sess.run(
            fetches,
            feed_dict={
                    'input_ids:0': features['input_ids'],
                    'input_mask:0': features['input_mask'],
                    'segment_ids:0': features['segment_ids']
                    }
        )

        tokenManagersForSeq = []

        for (prediction, tokenM) in zip(result, tokenManagers):
            prediction = prediction.tolist()
            for idx, class_probability in enumerate(prediction):
                if class_probability > 0.5:
                    tokenM.predicate_ids.append(idx)
            if len(tokenM.predicate_ids)>0:
                tokenManagersForSeq.append(tokenM)

        triples = []
        if len(tokenManagersForSeq)>0:
            features_seq = seq_tokenizer(tokenManagersForSeq, max_seq_length, tokenizer)

            print("\nfeatures_seq:\n{}\n\n".format(features_seq))

            seq_result = sess_seq.run(
                # TODO: 是否可以只跑fetches_seq[2]
                fetches_seq[2],
                feed_dict={
                        'input_ids:0': features_seq['input_ids'],
                        'input_mask:0': features_seq['input_mask'],
                        'segment_ids:0': features_seq['segment_ids']
                        }
            )

            seq_label_result = seq_result

            # for token_label_prediction in seq_label_result:
            #     token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction)
            #     print("raw out:\n{}".format(token_label_prediction))
            #     print("label out:\n{}".format(token_label_output_line))
            #     print("\n")

            res_index = 0
            for tokenM in tokenManagersForSeq:
                for predicate_id in tokenM.predicate_ids:
                    pred = label_list[predicate_id]
                    trps = getTriples(seq_label_result[res_index], tokenM.token_not_UNK ,pred)
                    triples.extend(trps)
                    res_index += 1

            # print("triples: {}".format(triples))

        res.append(triples)

    return json.dumps(res, ensure_ascii=False)



def getTriples(labels_ids, src, pred):
    # BIO_token_labels = ["[Padding]", "[category]", "[##WordPiece]", "[CLS]", "[SEP]", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "O"]  # id 0 --> [Paddding]

    print("labels_ids in getTriples_v1:\n{}".format(labels_ids))
    labels = [token_label_id2label[id] for id in labels_ids]
    print("labels in getTriples_v1:\n{}".format(labels))
    labels.pop(0)
    triples = []
    subs = []
    objs = []
    sub_indices = []
    obj_indices = []
    last_label = None
    last_i = 0
    bounds_map = {
        "B-SUB":"I-SUB", "B-OBJ":"I-OBJ"
    }
    label_index_map = {
        "B-SUB":sub_indices, "B-OBJ":obj_indices
    }
    for (i, label) in enumerate(labels):
        # TODO: "B-SUB", "B-OBJ"这类有点 hard-coding  如果后续有自定义标记的话，需要重写此处代码
        if last_label is None:
            if label in ("B-SUB", "B-OBJ"):
                last_label = label
                last_i = i
        else:
            if label != bounds_map[last_label]:
                label_index_map[last_label].append((last_i, i))
                last_label=None
            if label == "[SEP]":
                break
            if label in ("B-SUB", "B-OBJ"):
                last_label = label
                last_i = i

    # print("sub_indices:\n{}".format(sub_indices))
    # print("obj_indices:\n{}".format(obj_indices))

    for (start, end) in sub_indices:
        subs.append(src[start:end])
    

    for (start, end) in obj_indices:
        objs.append(src[start:end])

    for sub in subs:
        for obj in objs:
            triples.append({
                    "SUB":''.join(sub),
                    "predicate":pred,
                    "OBJ":''.join(obj)
                    })
    return triples


def cls_tokenizer(examples, max_seq_length, tokenizer):

    input_ids = []
    input_mask = []
    segment_ids = []

    for (ex_index, example) in enumerate(examples):

        tf.logging.info("building example %d : %s" % (ex_index, example))

        feature = build_cls_predict_data(
            ex_index, example, max_seq_length, tokenizer)
        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)

    features = dict()
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    features["segment_ids"] = segment_ids
    return features


def seq_tokenizer(examples, max_seq_length, tokenizer):

    input_ids = []
    input_mask = []
    segment_ids = []

    for (ex_index, example) in enumerate(examples):
        input_features = build_seq_predict_data(ex_index, example, max_seq_length, tokenizer)
        for feature in input_features:
            input_ids.append(feature.input_ids)
            input_mask.append(feature.input_mask)
            segment_ids.append(feature.segment_ids)

    features = dict()
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    features["segment_ids"] = segment_ids

    return features


def build_cls_predict_data(ex_index, example, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    input_ids = []

    input_ids.extend(tokenizer.convert_tokens_to_ids(["[CLS]"]))
    input_ids.extend(example.input_ids)
    input_ids.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))

    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

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
        segment_ids=segment_ids)
    return feature


def build_seq_predict_data(ex_index, example, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    features = []

    for predicate_id in example.predicate_ids:

        input_ids = []
        segment_ids = []

        text_token_len = len(example.input_ids)

        input_ids.extend(tokenizer.convert_tokens_to_ids(["[CLS]"]))
        input_ids.extend(example.input_ids)
        input_ids.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))

        segment_ids.extend([0]*len(input_ids))

        # bert_tokenizer.convert_tokens_to_ids(["[SEP]"]) --->[102]
        bias = 1  # 1-100 dict index not used
        input_ids.extend([predicate_id + bias]*text_token_len)
        segment_ids.extend([1]*text_token_len)

        input_ids.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))  # 102
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
        features.append(feature)
    return features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids



def main():
    app.run(host='0.0.0.0')


if __name__ == '__main__':

    fetches, sess = create_session()
    fetches_seq, sess_seq = create_seq_session()

    main()


# TODO: 
# 1. 模型优化
# 2. 数据预处理(清洗与切分)优化
# 3. 剔除重复triple
# 4. 处理\n \tab
# 5. 在没有split punc情况下，发生 maximum recursion depth exceeded while calling a Python object