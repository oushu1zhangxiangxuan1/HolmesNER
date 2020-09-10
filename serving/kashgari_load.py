
from kashgari.utils import load_model
from kashgari.corpus import ChineseDailyNerCorpus

model = load_model('models/ner.h5')

print(dir(model))

test_x, test_y = ChineseDailyNerCorpus.load_data("test")
print("\n test_x:\n{}\n\n".format(test_x[0:5]))

metrics = model.evaluate(test_x[0:5], test_y[0:5])
print("\n\n")
print(metrics)
print("\n\n")

print("\n=================predicton==============\n")
predictions = model.predict(test_x[0:5])
print(predictions)
print("\n\n")

print("\n=================predicton entities==============\n")
predictions = model.predict_entities(test_x[0:5])
print(predictions)
