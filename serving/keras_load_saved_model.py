# import tensorflow as tf

from tensorflow.python.keras.saving.saved_model.load import KerasObjectLoader
from tensorflow.python.saved_model.load import load_internal
from tensorflow.python.keras.saving.saved_model.load import RevivedModel
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.saved_model import loader_impl

model_path = 'output/saved_model/cls/1599723701'

loader_impl.parse_saved_model(model_path)
model = load_internal(model_path,
                      tags=['serve'], loader_cls=KerasObjectLoader)

if not isinstance(model, RevivedModel):
    raise RuntimeError("Can not load model")

if model._training_config is None:
    raise RuntimeError("Model _training_config is None")

model.compile(
    **saving_utils.compile_args_from_training_config(model._training_config))

test_data = [[], [], [], []]

model.predict(test_data)
