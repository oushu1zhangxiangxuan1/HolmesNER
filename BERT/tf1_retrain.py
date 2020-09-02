# import setuptools
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings import BERTEmbedding
import kashgari
from kashgari.tasks.labeling import BiLSTM_CRF_Model

"""
pip install tensorflow==1.15.3
pip install 'kashgari>=1.0.0,<2.0.0'
"""

"""
https://eliyar.biz/nlp_chinese_bert_ner/
"""


def main():

    train_x, train_y = ChineseDailyNerCorpus.load_data("train")
    valid_x, valid_y = ChineseDailyNerCorpus.load_data("validate")
    test_x, test_y = ChineseDailyNerCorpus.load_data("test")

    print(f"train data count: {len(train_x)}")
    print(f"validate data count: {len(valid_x)}")
    print(f"test data count: {len(test_x)}")

    bert_embed = BERTEmbedding(
        "models/chinese_L-12_H-768_A-12", task=kashgari.LABELING, sequence_length=100
    )
    model = BiLSTM_CRF_Model(bert_embed)
    model.fit(
        train_x,
        train_y,
        x_validate=valid_x,
        y_validate=valid_y,
        epochs=1,
        batch_size=512,
    )
    model.save("models/ner.h5")
    model.evaluate(test_x, test_y)
    predictions = model.predict_classes(test_x)
    print(predictions)


if "__main__" == __name__:
    main()
