import pickle

word2id = open('../train_test_data/word2id_bio.pkl','rb')
word2id = pickle.load(word2id)
print(len(word2id))
for key, value in word2id.items():
    print(key, value)
