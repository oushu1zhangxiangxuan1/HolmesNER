import tensorflow as tf
from tensorflow.keras.backend import set_session

config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
# config.gpu_options.allow_soft_placement=True
g = tf.get_default_graph()
sess = tf.Session(config=config, graph=g)
set_session(sess) 

from kashgari.utils import load_model

def main():
    from kashgari.corpus import ChineseDailyNerCorpus
    model = load_model('/home/johnsaxon/HolmesNER/BERT/ner.h5')

    print(dir(model))

    # test_x, test_y = ChineseDailyNerCorpus.load_data("test")
    # print("\n test_x:\n{}\n\n".format(test_x[0:5]))

    # metrics = model.evaluate(test_x[0:5], test_y[0:5])
    # print("\n\n")
    # print(metrics)
    # print("\n\n")

    print("\n=================predicton==============\n")    
    # test_x = test_x[0:5]
    test_x  = [
        ''.join(['第', '一', '次', '明', '确', '提', '出', '把', '自', '己', '建', '设', '成', '“', '国', '家', '科', '学', '思', '想', '库', '”', '的', '设', '想', '，', '路', '甬', '祥', '指', '出', '，', '面', '向', '新', '世', '纪', '，', '中', '科', '院', '学', '部', '要', '建', '设', '成', '最', '有', '影', '响', '的', '国', '家', '宏', '观', '决', '策', '科', '技', '咨', '询', '系', '统', '，', '要', '充', '分', '发', '挥', '院', '士', '群', '体', '的', '优', '势', '，', '加', '强', '科', '技', '战', '略', '研', '究', '，', '重', '点', '做', '好', '对', '国', '家', '宏', '观', '科', '技', '政', '策', '、', '科', '技', '发', '展', '计', '划', '、', '学', '科', '发', '展', '战', '略', '的', '制', '定', '以', '及', '经', '济', '建', '设', '、', '社', '会', '发', '展', '中', '重', '大', '科', '技', '问', '题', '的', '咨', '询', '工', '作', '。'])
    ]

    predictions = model.predict(test_x)
    print(predictions)
    print("\n\n")

    print("\n=================predicton entities==============\n")
    predictions = model.predict_entities(test_x)
    print(predictions)


def create_ner_model():
    global g, sess
    model_path = '/home/johnsaxon/HolmesNER/BERT/ner.h5'
    model = load_model(model_path)
    model.tf_model._make_predict_function()
    test = [['第', '一', '次', '明', '确', '提', '出', '把', '自', '己', '建', '设', '成', '“', '国', '家', '科', '学', '思', '想', '库', '”', '的', '设', '想', '，', '路', '甬', '祥', '指', '出', '，', '面', '向', '新', '世', '纪', '，', '中', '科', '院', '学', '部', '要', '建', '设', '成', '最', '有', '影', '响', '的', '国', '家', '宏', '观', '决', '策', '科', '技', '咨', '询', '系', '统', '，', '要', '充', '分', '发', '挥', '院', '士', '群', '体', '的', '优', '势', '，', '加', '强', '科', '技', '战', '略', '研', '究', '，', '重', '点', '做', '好', '对', '国', '家', '宏', '观', '科', '技', '政', '策', '、', '科', '技', '发', '展', '计', '划', '、', '学', '科', '发', '展', '战', '略', '的', '制', '定', '以', '及', '经', '济', '建', '设', '、', '社', '会', '发', '展', '中', '重', '大', '科', '技', '问', '题', '的', '咨', '询', '工', '作', '。']]
    model.predict(test)
    return model, g, sess


class NER_Model():
    def __init__(self, model_path):
        self.model = load_model(model_path)


if __name__ == '__main__':
    main()