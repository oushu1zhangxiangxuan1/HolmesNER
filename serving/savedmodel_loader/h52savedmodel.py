
from kashgari.utils import load_model
from kashgari.corpus import ChineseDailyNerCorpus
import tensorflow as tf

model = load_model('/home/johnsaxon/HolmesNER/BERT/ner.h5')
model_export_dir='models/ner/m1'

print(dir(model))

name_to_inputs = {i.name.split(":")[0]:i for i in model.tf_model.inputs}
name_to_outputs = {i.name:i for i in model.tf_model.outputs}



print(name_to_inputs)
print(name_to_outputs)
tf.saved_model.simple_save(tf.keras.backend.get_session(),
                           model_export_dir,
                           inputs=name_to_inputs,
                           outputs=name_to_outputs)



# print()

# test_x, test_y = ChineseDailyNerCorpus.load_data("test")
# print("\n test_x:\n{}\n\n".format(test_x[0:5]))

# metrics = model.evaluate(test_x[0:5], test_y[0:5])
# print("\n\n")
# print(metrics)
# print("\n\n")

# print("\n=================predicton==============\n")
# predictions = model.predict(test_x[0:5])
# print(predictions)
# print("\n\n")

# print("\n=================predicton entities==============\n")
# predictions = model.predict_entities(test_x[0:5])
# print(predictions)
