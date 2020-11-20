import sys
import pickle
import os
import random
import numpy as np

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

tag2label = {
             "B-SYMPTOM": 0,"B-DISEASE": 1,"B-TREATMENT": 2, "B-CHECK":3,
             "I-SYMPTOM": 4,"I-DISEASE": 5,"I-TREATMENT": 6, "I-CHECK":7,
             "O": 8,
             }

# def read_corpus(corpus_path):
#     """
#     read corpus and return the list of samples
#     :param corpus_path:
#     :return: raw_data
#     """
#     data = []
#     with open(corpus_path, encoding='utf-8') as fr:
#         lines = fr.readlines()
#     sent_, tag_ = [], []
#     for line in lines:
#         if line != '\n':
#             if len(line.strip().split('\t')) != 0:
#                 fields = line.strip().split('\t')
#                 char = fields[0]
#                 label = fields[-1]
#                 sent_.append(char)
#                 tag_.append(label)
#         elif len(sent_) != 0 and len(tag_) != 0:
#             data.append((sent_, tag_))
#             sent_, tag_ = [], []
#     return data

def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
        #print(lines)
    #句子---标签
    sent_, tag_ = [], []
    i = 0
    for line in lines:
        i = i+1
        #print(i)
        if line != '\n':
            #Python split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串.
            # str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
            # 指定的字符（默认为空格）,num -- 分割次数
            # strip() 方法用于移除字符串头尾.
            [char, label] = line.strip().split()
            #print('[char, label]',[char, label])#[char, label] ['的', 'O']
            sent_.append(char)
            #print(sent_)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
        #print(line)
    return data


#data = read_corpus("../train_test_data/train_data.txt")
#print(data)

def vocab_build(vocab_path, corpus_path, min_count,encoding='utf-8'):
    """
    BUG: I forget to transform all the English characters from full-width into half-width... 
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    # 删除词频出现较少的单词
    for word in low_freq_words:
        del word2id[word]

    # 剔除词频出现较少的单词，并且重新编号
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    # 将word2id存储进入文件中
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """
    将一个句子进行编号
    :param sent: 表示一个句子
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
    读取之前存入文件中的word2id词典
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))  #[len(vocab),300]
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def pad_sequences(sequences, pad_mark=0):
    """
    补齐，将数据sequences内的word变成长度相同
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        #print(label_)

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def get_entity(tag_seq, char_seq, keys):
    """
    返回实体类别
    :param tag_seq:
    :param char_seq:
    :param keys: key为list，表示返回需要的类别名称
    :return:
    """
    # entity = get_entity_one_(tag_seq, char_seq)
    # return entity
    entity = []
    for key in keys:
        entity.append(get_entity_key(tag_seq, char_seq, key))
    return entity

#找出实体
def get_entity_key(tag_seq, char_seq, key):
    entities = []
    entity = ''
    for (char, tag) in zip(char_seq, tag_seq):
        if tag == 'B-' + key or tag == 'I-' + key or tag == 'E-' + key:
            entity += char
        else:
            if len(entity) != 0:
                entities.append(entity)
                entity = ''
    if len(entity) != 0:
        entities.append(entity)
    return entities


if __name__ == '__main__':
    #'../train_test_data/train_bio_word.txt'
    #vocab_build('../train_test_data/word2id_bio.pkl', '../train_test_data/train_data.txt', 3)
    with open('../train_test_data/word2id_bio.pkl', 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    embedding_mat = np.zeros((len(word2id), 200), dtype=float)  # [len(vocab),300]
    #print(embedding_mat)
    with open("charWord2vec_model.pkl",'rb') as pre_word2vec:
        pre_embedding = pickle.load(pre_word2vec)
        #gensim_dict = Dictionary()
        i = 0
        for key_, value_  in word2id.items():
            #print(key_,value_)

            for key, value in pre_embedding.wv.vocab.items():
              if key_ == key:
                 i=i+1
                 embedding_mat[value_] = pre_embedding.wv[key]
                 #print(i,'value_',value_)

                #print(embedding_mat[value_])
                #print(key,":",pre_embedding.wv[key])
    #print(embedding_mat)
    np.save("word2vector/embedding_mat.npy",embedding_mat)
         #index = embedding_mat.shape[0]
    # #raw_data = read_corpus('data_path/train_data.txt')
    # #print(len(raw_data))
    #
    # #print(embedding_mat)
    #
    fr.close()
    pre_word2vec.close()


