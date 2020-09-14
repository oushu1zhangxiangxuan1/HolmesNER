import json
from flask import Flask
from flask import request
# import tensorflow as tf
# from bert import modeling
# from bert import optimization
# from bert import tokenization
import est_cls
from loader import create_session
# from loader_seq import create_seq_session
from loader import string_tokenizer
# from loader_seq import seq_tokenizer
import loader
# import numpy as np

app = Flask(__name__)

fetches, sess = create_session()

# fetches_seq, sess_seq = create_seq_session()

label_list = est_cls.get_labels()

tokenizer = loader.get_tokenizer()

max_seq_length = 128


@app.route('/')
def home():
    return "Welcome to Relation Extraction System"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    return json_data["data"]


@app.route('/predict/<data>', methods=['GET'])
def predict_single(data):
    response = []

    predict_test_data = [
        "《中国风水十讲》是2007年华夏出版社出版的图书，作者是杨文衡",
        "你是最爱词:许常德李素珍/曲:刘天健你的故事写到你离去后为止",
        "《苏州商会档案丛编第二辑》是2012年华中师范大学出版社出版的图书，作者是马敏、祖苏、肖芃"
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
        # for idx, class_probability in enumerate(prediction):
        #     predicate_predict = []
        #     if class_probability > 0.5:
        #         print(label_list[idx])
        #         print(type(label_list[idx]))
        #         # print(label_list[idx].encode('utf-8').decode('unicode-escape'))
        #         # print(label_list[idx].decode('utf-8'))
        #         # predicate_predict.append(label_list[idx].encode('utf-8'), 'utf-8'))
        #         predicate_predict.append(label_list[idx])
        predicate_predict = []
        predicate_predict.append(label_list[prediction.index(max(prediction))])
        # max_list = np.where(prediction == np.max(prediction))
        # for i in max_list:
        #     predicate_predict.append(label_list[i])
        res = {"relations": predicate_predict}
        response.append(res)

    # features_seq = seq_tokenizer(predict_test_data, label_list, max_seq_length, tokenizer)
    # seq_result = sess_seq.run(
    #     fetches_seq,
    #     feed_dict={
    #             'input_ids:0': features_seq['input_ids'],
    #             'input_mask:0': features_seq['input_mask'],
    #             'segment_ids:0': features_seq['segment_ids']
    #             }
    # )

    # print("seq result: {}".format(seq_result))

    return json.dumps(response)


def main():
    app.run(host='0.0.0.0')


if __name__ == '__main__':
    main()
