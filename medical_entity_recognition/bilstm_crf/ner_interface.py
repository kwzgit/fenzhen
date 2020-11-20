#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import time
import numpy as np
import tensorflow as tf
from config import NER_MODEL_PATH
from medical_entity_recognition.bilstm_crf.model import BiLSTM_CRF
from medical_entity_recognition.bilstm_crf.utils import str2bool, get_logger, get_entity_keys
from medical_entity_recognition.bilstm_crf.data import read_corpus, read_dictionary, tag2label, random_embedding
import ner_model


# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--data_dir', type=str, default='../train_test_data', help='raw_data dir source')
parser.add_argument('--dictionary', type=str, default='word2id_bio.pkl', help='dictionary source')
#../train_test_data/train_data.txt
parser.add_argument('--train_data', type=str, default='train_data.txt', help='train raw_data source')
parser.add_argument('--test_data', type=str, default='test_data.txt', help='test raw_data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=200, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')  # 解决梯度爆炸的影响
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='embedding_mat.npy',help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=200, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training raw_data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1586501733', help='model for test and demo')#random_char_300,1524919794
parser.add_argument('--embedding_dir', type=str, default='word2vector', help='embedding files dir')
args = parser.parse_args()
# get char embeddings
word2id = read_dictionary(os.path.join(NER_MODEL_PATH, args.data_dir, args.dictionary))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = os.path.join(NER_MODEL_PATH, args.embedding_dir, args.pretrain_embedding)
    embeddings = np.array(np.load(embedding_path), dtype='float32')

# read corpus and get training raw_data
# 读取数据
if args.mode != 'demo':
    train_path = os.path.join('.', args.data_dir, args.train_data)
    test_path = os.path.join('.', args.data_dir, args.test_data)
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)
#paths setting
#生成输出目录
#bio_model/1524919794/
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join(NER_MODEL_PATH, "bio_model", timestamp)
if not os.path.exists(output_path):
    os.makedirs(output_path)
#bio_model/1524919794/summaries
summary_path = os.path.join(output_path, "summaries")
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

model_path = os.path.join(output_path, "checkpoints/")

if not os.path.exists(model_path):
    os.makedirs(model_path)

ckpt_prefix = os.path.join(model_path, "model")
result_path = os.path.join(output_path, "results")

if not os.path.exists(result_path):
    os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
get_logger(log_path).info(str(args))

class NERModel(ner_model.NERModelInterface):
    """
    """
    def __init__(self):
        """
        """
        self._build()

    def _build(self):
        """
        """
        ckpt_file = tf.train.latest_checkpoint(model_path)
        self.model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim,
                           embeddings=embeddings,
                           dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                           tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                           model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                           CRF=args.CRF, update_embedding=args.update_embedding)
        self.model.build_graph()
        saver = tf.train.Saver()
        self.sess = tf.Session()
        print('restore model from ', ckpt_file)
        saver.restore(self.sess, ckpt_file)

    def extract(self, sentence, tag_type='SYMPTOM'):
        """提取数据
        :param sentence: 句子
        :param tag_type:
        :return:
        """

        sent = list(sentence)
        sent_data = [(sent, ['O'] * len(sent))]
        tag = self.model.demo_one(self.sess, sent_data)
        result = get_entity_keys(tag, sent, [tag_type])
        return set(result[0])



# def predict(sentence, tag_type='SYMPTOM'):
#     """预测
#     :param sentence: 句子
#     :param tag_type: 标签类型
#     :return: list of string
#     """
#     ckpt_file = tf.train.latest_checkpoint(model_path)
#     model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim,
#                        embeddings=embeddings,
#                        dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
#                        tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
#                        model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
#                        CRF=args.CRF, update_embedding=args.update_embedding)
#     model.build_graph()
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         print('restore model from ', ckpt_file)
#         saver.restore(sess, ckpt_file)
#         sent = list(sentence)
#         sent_data = [(sent, ['O'] * len(sent))]
#         tag = model.demo_one(sess, sent_data)
#         result = get_entity_keys(tag, sent, [tag_type])
#         return result[0]



def main():
    """
    :return:
    """
    xx = '前列腺炎患者会出现膀胱刺激症，常伴有尿频、尿痛、尿灼热等症状，早上还会发现尿道口有许多粘液分泌物，有时还会出现排尿困难等症状'

    model = NERModel()
    for i in range(100):
        print(model.extract(xx))


if __name__ == '__main__':
    main()
