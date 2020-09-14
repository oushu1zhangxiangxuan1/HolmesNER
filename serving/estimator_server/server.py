import json
from flask import Flask
from flask import request
# import tensorflow as tf
# from bert import modeling
# from bert import optimization
# from bert import tokenization
from est_cls import import_test
from est_cls import create_estimator
import est_cls

app = Flask(__name__)
estimator = create_estimator()

label_list = est_cls.get_labels()

tokenizer = est_cls.get_tokenizer()

max_seq_length = est_cls.get_max_seq_length()


@app.route('/')
def home():
    return "Welcome to Relation Extraction System"


@app.route('/test')
def test():
    return import_test()


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
    num_actual_predict_examples = len(predict_test_data)

    predict_input_fn = est_cls.string_based_input_fn_builder(
        data=predict_test_data,
        seq_length=max_seq_length,
        label_list=label_list,
        tokenizer=tokenizer)

    result = estimator.predict(input_fn=predict_input_fn)

    for (i, prediction) in enumerate(result):
        print("\n\n prediction:\n{}".format(prediction))
        probabilities = prediction["probabilities"].tolist()
        if i >= num_actual_predict_examples:
            break
        predicate_predict = []
        for idx, class_probability in enumerate(probabilities):
            if class_probability > 0.5:
                print(label_list[idx])
                print(type(label_list[idx]))
                # print(label_list[idx].encode('utf-8').decode('unicode-escape'))
                # print(label_list[idx].decode('utf-8'))
                # predicate_predict.append(label_list[idx].encode('utf-8'), 'utf-8'))
                predicate_predict.append(label_list[idx])
            res = {"relations": predicate_predict}
            response.append(res)

    # return data
    return json.dumps(response)


def main():
    app.run(host='0.0.0.0')


if __name__ == '__main__':
    main()
