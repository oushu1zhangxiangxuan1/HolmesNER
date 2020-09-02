from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.tasks.labeling import BiLSTM_CRF_Model

weight_file = "models/ner.h5/model_weights.h5"
arch_file = "models/ner.h5/model_info.json"
model_dir = "models/ner.h5"

model_json = None
with open(arch_file, 'r') as f:
    model_json = f.read()


# model = load_model(model_dir, custom_objects={"BiLSTM_CRF_Model": BiLSTM_CRF_Model})

model = model_from_json(model_json, custom_objects={
                        "BiLSTM_CRF_Model": BiLSTM_CRF_Model})
model.load_weights(weight_file)

"""
WARNING:root:Sequence length will auto set at 95% of sequence length
Traceback (most recent call last):
  File "keras_load.py", line 15, in <module>
    model.load_weights(weight_file)
AttributeError: 'BiLSTM_CRF_Model' object has no attribute 'load_weights'
"""


print(model.__doc__)
print(dir(model))
# WARNING:root:Sequence length will auto set at 95% of sequence length
# Bidirectional LSTM CRF Sequence Labeling Model
# ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__',
# '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__task__', '__weakref__', 'build_model', 'build_model_arc', 'build_multi_gpu_model', 'build_tpu_model', 'compile_model', 'embedding', 'evaluate', 'fit',
#  'fit_without_generator', 'get_data_generator', 'get_default_hyper_parameters', 'hyper_parameters', 'info', 'label2idx', 'model_info', 'pre_processor', 'predict', 'predict_entities', 'processor', 'save', 'task', 'tf_model', 'token2idx']


test_x, test_y = ChineseDailyNerCorpus.load_data("test")
print("\n test_x:\n{}\n\n".format(test_x[0:5]))
# predictions = model.predict(test_x[0:5])
predictions = model.predict_entities(test_x[0:5])
print(predictions)
