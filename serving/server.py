import json
from flask import Flask
from flask import request
import tensorflow as tf
from bert import modeling
from bert import optimization
# from bert import tokenization

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    return json_data["data"]


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids)

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

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits_wx = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits_wx, output_bias)
        probabilities = tf.sigmoid(logits)
        label_ids = tf.cast(labels, tf.float32)
        per_example_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_ids), axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, logits, probabilities


# def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
#                      num_train_steps, num_warmup_steps,
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
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

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


def build_estimator(ckpt_dir: str):

    config = Config("")

    bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)

    label_list = config.get_labels()

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=config.init_checkpoint)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    return


def main():
    app.run()


class Config:

    def __init__(self, config_file):
        self.bert_config_file = ""
        self.init_checkpoint = ""

        
    def get_labels(self):
        return ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期',
                '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自',
                '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站',
                '邮政编码', '面积', '首都']


if __name__ == '__main__':
    main()
