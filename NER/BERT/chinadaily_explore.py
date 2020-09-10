
from kashgari.corpus import ChineseDailyNerCorpus

train_x, train_y = ChineseDailyNerCorpus.load_data("train")
valid_x, valid_y = ChineseDailyNerCorpus.load_data("validate")
test_x, test_y = ChineseDailyNerCorpus.load_data("test")


print("train len: {}".format(len(train_x)))
print("valid len: {}".format(len(valid_x)))

print("test_x len: {}".format(len(test_x)))
print("test_y len: {}".format(len(test_y)))

print("\n\n")

print("test_x[0]: {}".format(test_x[0]))
print("test_y[0]: {}".format(test_y[0]))

print("\n\n")

print("test_x[1]: {}".format(test_x[1]))
print("test_y[1]: {}".format(test_y[1]))

# for i in range(len(test_x)):
#     print("test_x[{}]: {}".format(i, test_x[i]))
#     print("test_y[{}]: {}".format(i, test_y[i]))
#     continue