import tensorflow as tf
from kashgari.embeddings import BERTEmbedding
from bert import tokenization
from est_cls import InputFeatures


vocab_file = "/home/johnsaxon/github.com/Entity-Relation-Extraction/pretrained_model/chinese_L-12_H-768_A-12/vocab.txt"


def main():

    examples = [
        "《中国风水十讲》是2007年华夏出版社出版的图书，作者是杨文衡",
        "你是最爱词:许常德李素珍/曲:刘天健你的故事写到你离去后为止",
        "《苏州商会档案丛编第二辑》是2012年华中师范大学出版社出版的图书，作者是马敏、祖苏、肖芃"
    ]

    sess = tf.compat.v1.Session()

    model_path = "/home/johnsaxon/github.com/oushu1zhangxiangxuan1/HolmesNER/serving/savedmodel_loader/models/ner/m1" 
    # tf.saved_model.loader.load(
    tf.compat.v1.saved_model.loader.load(
        sess,
        [tf.saved_model.SERVING],
        model_path
    )

    prediction = sess.graph.get_tensor_by_name("layer_crf/cond/Merge:0")

    bert_embed = BERTEmbedding(
        "/home/johnsaxon/github.com/Entity-Relation-Extraction/pretrained_model/chinese_L-12_H-768_A-12", task=kashgari.LABELING, sequence_length=100
    )

    x0, x1 = bert_embed.process_x_dataset(examples)

    print(x0, x1)

    predictions_result = sess.run(
        prediction,
        feed_dict={
            'Input-Segment_1:0': x0,
            'Input-Token_1:0': x1
        }
    )
    sess.close()

    print(predictions_result)

    # for token_label_prediction in token_label_predictions_result:
    #     token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction)
    #     print(token_label_output_line)
    #     print("\n")


def create_ner_session():

    g = tf.Graph()

    sess = tf.compat.v1.Session(graph=g)

    model_path = "/home/johnsaxon/github.com/oushu1zhangxiangxuan1/HolmesNER/serving/savedmodel_loader/models/ner/m1" 
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
    main()
