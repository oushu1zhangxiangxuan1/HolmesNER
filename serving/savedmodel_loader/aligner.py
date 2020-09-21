import est_cls
from loader_seq import get_token_labels
import loader 
import json
# from server import index_safe


max_seq_length = 128


def predict_align(data, tokenizer, sess, sess_seq, fetches, fetches_seq):

    predicate_labels = est_cls.get_labels()

    token_label_list = get_token_labels()
    token_label_map = {}
    for (i, label) in enumerate(token_label_list):
        token_label_map[i] = label

    max_token_len = 62

    tokens = tokenizer.tokenize(data)
    tokens_not_UNK = tokenizer.tokenize_not_UNK(data)

    tokenSegmenter = TokenSegmenter(tokens, tokens_not_UNK)

    rich_tokens, _ = tokenSegmenter.split(max_token_len)


    input_ids = []
    input_mask = []
    segment_ids = []

    for token in rich_tokens:
        ids, mask, segment = token.convert_cls(tokenizer, max_seq_length)
        input_ids.append(ids)
        input_mask.append(mask)
        segment_ids.append(segment)

    result = sess.run(
        fetches,
        feed_dict={
                'input_ids:0': input_ids,
                'input_mask:0': input_mask,
                'segment_ids:0': segment_ids
                }
    )


    input_ids = []
    input_mask = []
    segment_ids = []

    for (prediction, token) in zip(result, rich_tokens):
        prediction = prediction.tolist()
        predicate_ids = []
        for idx, class_probability in enumerate(prediction):
            if class_probability > 0.5:
                predicate_ids.append(idx)
        ids, mask, segment = token.convert_seq(tokenizer, max_seq_length, predicate_ids)
        input_ids.extend(ids)
        input_mask.extend(mask)
        segment_ids.extend(segment)

    triples = []
    if len(input_ids)>0:
        seq_result = sess_seq.run(
            fetches_seq[2],
            feed_dict={
                    'input_ids:0': input_ids,
                    'input_mask:0': input_mask,
                    'segment_ids:0': segment_ids
                    }
        )

        res_index = 0
        for token in rich_tokens:
            for predicate_id in token.predicate_ids:
                pred = predicate_labels[predicate_id]
                # TODO: 需要考虑entity的去重及互相覆盖
                trps = getTriples(seq_result[res_index], token.token_not_UNK, pred, token_label_map, token.base_raw_index)
                token.set_triples(trps)
                res_index += 1

    return rich_tokens


def getTriples(labels_ids, src, pred, token_label_map, base_raw_index):
    # BIO_token_labels = ["[Padding]", "[category]", "[##WordPiece]", "[CLS]", "[SEP]", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "O"]  # id 0 --> [Paddding]

    # print("labels_ids in getTriples_v1:\n{}".format(labels_ids))
    labels = [token_label_map[id] for id in labels_ids]
    # print("labels in getTriples_v1:\n{}".format(labels))
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
        content = ''.join(src[start:end])
        start = base_raw_index + len(''.join(src[:start]))
        end = base_raw_index + len(''.join(src[:end]))
        entity = EntityIndex(start, end, content)
        subs.append(entity)

    for (start, end) in obj_indices:
        content = ''.join(src[start:end])
        start = base_raw_index + len(''.join(src[:start]))
        end = base_raw_index + len(''.join(src[:end]))
        entity = EntityIndex(start, end, content)
        objs.append(entity)

    for sub in subs:
        for obj in objs:
            triples.append(Triple(sub,obj,pred))

    return triples


class EntityIndex:
    def __init__(self, start, end, content):
        self.start = start
        self.end = end
        self.content = content

class Triple:
    def __init__(self, sub, obj, predicate):
        self.sub = sub
        self.obj = obj
        self.predicate = predicate



class RichTokenManager:
    # def __init__(self, token, token_not_UNK, tokenizer):
    def __init__(self, token, token_not_UNK, base_raw_index):
        self.token = token
        self.token_not_UNK = token_not_UNK
        # self.tokenizer = tokenizer
        self.predicate_ids = []
        # self.input_ids = tokenizer.convert_tokens_to_ids(self.token)
        self.base_raw_index = base_raw_index
        self.triples = []

    # def default(self, obj):
    #     d = {}
    #     d.update(obj.__dict__)
    #     return d

    def set_triples(self, triples):
        self.triples = triples

    def convert_cls(self, tokenizer, max_seq_length):

        input_ids = []

        input_ids.extend(tokenizer.convert_tokens_to_ids(["[CLS]"]))
        input_ids.extend(tokenizer.convert_tokens_to_ids(self.token))
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

        return input_ids, input_mask, segment_ids

    def convert_seq(self, tokenizer, max_seq_length, predicate_ids):
        self.predicate_ids = predicate_ids

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        for predicate_id in self.predicate_ids:

            input_ids = []
            segment_ids = []

            text_token_len = len(self.token)

            input_ids.extend(tokenizer.convert_tokens_to_ids(["[CLS]"]))
            input_ids.extend(tokenizer.convert_tokens_to_ids(self.token))
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


            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)

        return input_ids_list, input_mask_list, segment_ids_list


class TokenSegmenter:
    def __init__(self, token, token_not_UNK):
        self.token = token
        self.token_not_UNK = token_not_UNK

    def split(self, max_token_len):
        return _split_token_(self.token, self.token_not_UNK, max_token_len, 0)


def _split_token_(tokens, tokens_not_UNK, max_token_len, base_raw_index):

    # import sys
 
    # sys.setrecursionlimit(55) 

    # print("\n\ntokens:\n{}\n".format(tokens))
    # print("\ntokens_not_UNK:\n{}".format(tokens_not_UNK))
    # print("base_raw_index:\n{}\n".format(base_raw_index))
    assert len(tokens) == len(tokens_not_UNK)

    rich_tokens = []
    token_len = len(tokens)

    if token_len == 0:
        return rich_tokens, base_raw_index

    # TODO: 遇到句号等强分隔符就得先分
    # NEG example: 蔡庆辉，1974年10月生于福建莆田，厦门大学副教授。蓼子朴组属于桔梗目、菊科，草本或亚灌木
    if not token_len > max_token_len:
        rich_token = RichTokenManager(tokens, tokens_not_UNK, base_raw_index)
        rich_tokens.append(rich_token)
        return rich_tokens, base_raw_index+len(''.join(tokens_not_UNK))
    else:
        split_punctuation = ["。",".","！","!","；",";","？","?","，",","]

        for punc in split_punctuation:
            punc_index = index_safe(tokens_not_UNK, punc)
            if punc_index < 0:
                continue
            seg1, seg2 = [], []
            rich_lefts, base_raw_index = _split_token_(tokens[:punc_index], tokens_not_UNK[:punc_index], max_token_len, base_raw_index)
            rich_punc = RichTokenManager([tokens[punc_index]], [tokens_not_UNK[punc_index]], base_raw_index)
            base_raw_index += 1
            rich_rights, base_raw_index = _split_token_(tokens[punc_index+1:], tokens_not_UNK[punc_index+1:], max_token_len, base_raw_index)
            rich_tokens.extend(rich_lefts)
            rich_tokens.append(rich_punc)
            rich_tokens.extend(rich_rights)
            return rich_tokens, base_raw_index

        # 也可用二分
        rich_lefts, base_raw_index  = _split_token_(tokens[:max_token_len], tokens_not_UNK[:max_token_len], max_token_len, base_raw_index)
        rich_rights, base_raw_index = _split_token_(tokens[max_token_len:], tokens_not_UNK[max_token_len:], max_token_len, base_raw_index)
        rich_tokens.extend(rich_lefts)
        rich_tokens.extend(rich_rights)
        return rich_tokens, base_raw_index


def index_safe(src, sub):
    try:
        return src.index(sub)
    except Exception:
        return -1


class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        d = {}
        d.update(obj.__dict__)
        return d


if __name__ == '__main__':
    tokenizer = loader.get_tokenizer()

    max_token_len = 62
    data = "北京偶数科技有限公司于2016年12月29日成立。法定代表人常雷,公司经营范围包括：技术开发、技术咨询、技术服务、技术推广、技术转让；数据处理（数据处理中的银行卡中心、PUE值在1.5以上的云计算数据中心除外）；基础软件服务；应用软件服务；计算机系统服务；软件咨询；软件开发；产品设计；销售自行开发后的产品等。"
    # tokens = tokenizer.tokenize(data)
    # tokens_not_UNK = tokenizer.tokenize_not_UNK(data)

    # tks, ind = _split_token_(tokens, tokens_not_UNK, max_token_len, 0)
    # import json
    # print(json.dumps(tks, cls=MyJSONEncoder, ensure_ascii=False))

    triples = predict_align(data, tokenizer)
    print(json.dumps(triples, cls=MyJSONEncoder, ensure_ascii=False))