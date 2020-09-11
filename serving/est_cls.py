# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from bert import tokenization
# from bert import optimization
from bert import modeling

import collections
# import csv
import os
import sys
import tensorflow as tf


sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../bert")))

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * len(label_list),
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = example.text_a.split(" ")
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

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

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

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

    label_list = example.label.split(" ")
    label_ids = _predicate_label_to_id(label_list, label_map)

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" %
                        " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True)
    return feature


def build_single_predict_data(ex_index, example, label_list, max_seq_length, tokenizer):
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


def _predicate_label_to_id(predicate_label, predicate_label_map):
    predicate_label_map_length = len(predicate_label_map)
    predicate_label_ids = [0] * predicate_label_map_length
    for label in predicate_label:
        predicate_label_ids[predicate_label_map[label]] = 1
    return predicate_label_ids


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Converting example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def string_tokenizer(examples, label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a TFRecord file."""

    input_ids = []
    input_mask = []
    segment_ids = []

    for (ex_index, example) in enumerate(examples):

        tf.logging.info("building example %d : %s" % (ex_index, example))

        feature = build_single_predict_data(
            ex_index, example, label_list, max_seq_length, tokenizer)
        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)

        # def create_int_feature(values):
        #     f = tf.train.Feature(
        #         int64_list=tf.train.Int64List(value=list(values)))
        #     return f

        # features = collections.OrderedDict()
        # features["input_ids"] = create_int_feature(feature.input_ids)
        # features["input_mask"] = create_int_feature(feature.input_mask)
        # features["segment_ids"] = create_int_feature(feature.segment_ids)
        # features["label_ids"] = create_int_feature(feature.label_ids)
        # features["is_real_example"] = create_int_feature(
        #     [int(feature.is_real_example)])

    features = dict()
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    features["segment_ids"] = segment_ids
    # features["label_ids"] = feature.label_ids
    # features["is_real_example"] = [int(feature.is_real_example)]

    # print("\n\n\nfeatures:\n{}\n\n\n".format(features))

    return tf.data.Dataset.from_tensor_slices(features)


def file_based_input_fn_builder(input_file, seq_length, label_length,
                                is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([label_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def string_based_input_fn_builder(data, seq_length, label_list, tokenizer):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    # import tensorflow.contrib.eager as tfe
    # tfe.enable_eager_execution()

    # dataset = string_tokenizer(
    #     examples=data,
    #     label_list=label_list,
    #     max_seq_length=seq_length,
    #     tokenizer=tokenizer)

    # print(dataset)
    # print(dataset["input_ids"])
    # print(dataset["input_ids"].shape.ndims)

    # for one_element in tfe.Iterator(dataset):
    #     print(one_element)

    # return

    def input_fn(params):
        """The actual input function."""
        dataset = string_tokenizer(
            examples=data,
            label_list=label_list,
            max_seq_length=seq_length,
            tokenizer=tokenizer)
        batch_size = params["batch_size"]

        dataset = dataset.batch(batch_size=batch_size)

        return dataset

    return input_fn


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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, num_labels):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    # with tf.variable_scope("loss"):
    #     if is_training:
    #         # I.e., 0.1 dropout
    #         output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits_wx = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits_wx, output_bias)
    probabilities = tf.sigmoid(logits)

    return logits, probabilities


# def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
#                      num_train_steps, num_warmup_steps, use_tpu,
#                      use_one_hot_embeddings):
def model_fn_builder(bert_config, num_labels, init_checkpoint):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        # label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            num_labels)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={"probabilities": probabilities},
            scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def import_test():
    return "import success"


def create_estimator():

    tf.logging.set_verbosity(tf.logging.INFO)

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    label_list = get_labels()

    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    run_config = tf.contrib.tpu.RunConfig(
        # cluster=tpu_cluster_resolver,
        # master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        # tpu_config=tf.contrib.tpu.TPUConfig(
        #     iterations_per_loop=FLAGS.iterations_per_loop,
        #     num_shards=FLAGS.num_tpu_cores,
        #     per_host_input_for_training=is_per_host))
    )

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        # learning_rate=FLAGS.learning_rate,
        # num_train_steps=num_train_steps,
        # num_warmup_steps=num_warmup_steps,
        # use_tpu=FLAGS.use_tpu,
        # use_one_hot_embeddings=FLAGS.use_tpu
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        # eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    return estimator


def get_labels():
    return ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期',
            '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自',
            '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站',
            '邮政编码', '面积', '首都']


def main():
    # return
    estimator = create_estimator()

    predict_test_data = [
        "《中国风水十讲》是2007年华夏出版社出版的图书，作者是杨文衡",
        "你是最爱词:许常德李素珍/曲:刘天健你的故事写到你离去后为止",
        "《苏州商会档案丛编第二辑》是2012年华中师范大学出版社出版的图书，作者是马敏、祖苏、肖芃"
    ]
    num_actual_predict_examples = len(predict_test_data)

    label_list = get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    predict_input_fn = string_based_input_fn_builder(
        data=predict_test_data,
        seq_length=FLAGS.max_seq_length,
        label_list=label_list,
        tokenizer=tokenizer)

    result = estimator.predict(input_fn=predict_input_fn)

    num_written_lines = 0
    tf.logging.info("***** Predict results *****")
    for (i, prediction) in enumerate(result):
        print("\n\n prediction:\n{}".format(prediction))
        # continue
        # probabilities = prediction["probabilities"]
        # if i >= num_actual_predict_examples:
        #     break
        # output_line_score_value = " ".join(
        #     str(class_probability)
        #     for class_probability in probabilities) + "\n"
        # predicate_predict = []
        # for idx, class_probability in enumerate(probabilities):
        #     if class_probability > 0.5:
        #         predicate_predict.append(label_list[idx])
        # output_line_predicate_predict = " ".join(
        #     predicate_predict) + "\n"
        # predicate_predict_writer.write(
        #     output_line_predicate_predict)
        # score_value_writer.write(output_line_score_value)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


class Config:
    def __init__(self,
                 data_dir="bin/predicate_classifiction/classification_data",
                 vocab_file="pretrained_model/chinese_L-12_H-768_A-12/vocab.txt",
                 bert_config_file="pretrained_model/chinese_L-12_H-768_A-12/bert_config.json",
                 init_checkpoint="output/predicate_classification_model/epochs6/model.ckpt-487",
                 max_seq_length=128,
                 output_dir="./output/test_predict",
                 do_lower_case=False,
                 save_checkpoints_steps=0,
                 train_batch_size=0,
                 predict_batch_size=20,
                 ):
        self.data_dir = data_dir
        self.vocab_file = vocab_file
        self.bert_config_file = bert_config_file
        self.init_checkpoint = init_checkpoint
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.do_lower_case = do_lower_case
        self.save_checkpoints_steps = save_checkpoints_steps
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size

# flags = {
#     "data_dir":"bin/predicate_classifiction/classification_data",
#     "vocab_file":"pretrained_model/chinese_L-12_H-768_A-12/vocab.txt",
#     "bert_config_file":"pretrained_model/chinese_L-12_H-768_A-12/bert_config.json",
#     "init_checkpoint":"output/predicate_classification_model/epochs6/model.ckpt-487",
#     "max_seq_length":128,
#     "output_dir":"./output/test_predict",
#     "do_lower_case":False
# }


FLAGS = Config()


def get_tokenizer():
    return tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


def get_max_seq_length():
    return FLAGS.max_seq_length


if __name__ == "__main__":
    main()
