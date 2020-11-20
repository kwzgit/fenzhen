import tensorflow as tf
import numpy as np
import os
import argparse
import time
import random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity_keys
from data import read_corpus, read_dictionary, tag2label, random_embedding
import glob
import csv
# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--data_dir', type=str, default='../train_test_data', help='raw_data dir source')
parser.add_argument('--dictionary', type=str, default='word2id_bio.pkl', help='dictionary source')
#../train_test_data/train_data.txt
parser.add_argument('--train_data', type=str, default='train_data.txt', help='train raw_data source')
parser.add_argument('--test_data', type=str, default='test_data.txt', help='test raw_data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=81, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')#0.001
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')  # 解决梯度爆炸的影响
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='embedding_mat.npy',help='embedding_mat.npy,random,use pretrained char embedding or init it randomly')# 200
parser.add_argument('--embedding_dim', type=int, default=200, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training raw_data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1530832199', help='model for test and demo')
parser.add_argument('--embedding_dir', type=str, default='word2vector', help='embedding files dir')
args = parser.parse_args()

# get char embeddings
word2id = read_dictionary(os.path.join('.', args.data_dir, args.dictionary))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = os.path.join(os.path.curdir, args.embedding_dir, args.pretrain_embedding)
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
output_path = os.path.join('.', "bio_model", timestamp)
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

# training model
if args.mode == 'train':
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_prefix, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embedding)
    model.build_graph()

    # hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train raw_data: {0}\ndev raw_data: {1}".format(train_size, dev_size))
    # model.train(train_data, dev_data)

    # train model on the whole training raw_data
    print("train raw_data: {}".format(len(train_data)))
    model.train(train_data, test_data)  # we could use test_data as the dev_data to see the overfitting phenomena

# testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embedding)
    model.build_graph()
    print("test raw_data: {}".format(test_size))
    model.test(test_data)

elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim,
                       embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embedding)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)

        # 测试一份文件内容


        # with open('吞咽困难.csv', 'r') as fr:
        #     content = ''
        #     for line in fr:
        #         #content += line.strip()
        #         #print(line)
        #         content = line
        #         sent = list(content)
        #         sent_data = [(sent, ['O'] * len(sent))]
        #         print(sent_data)
        #         tag = model.demo_one(sess, sent_data)
        #         #print(tag)
        #         # for s, t in zip(sent, tag):
        #         #     print(s, t)
        # # body, chec, cure, dise, symp = get_entity_keys(tag, sent, ['BODY', 'CHECK', 'TREATMENT', 'DISEASE', 'SIGNS'])
        # # print('body:{}\nchec:{}\ncure:{}\ndise:{}\nsymp:{}\n'.format(body, chec, cure, dise, symp))
        #         symptom, disease, treatment, check = get_entity_keys(tag, sent,
        #                                                ['SYMPTOM', 'DISEASE', 'TREATMENT', 'CHECK'])
        #
        #         print(symptom)
        #         #print('SYMPTOM:{}\nDISEASE:{}\nTREATMENT:{}\nCHECK:{}\n'.format(symptom, disease, treatment, check))
        #
        #

        csv_list_train = glob.glob('/home/ywm/桌面/new_data_all/各个网站的所有数据/train_ER/*.csv')  #train_data
        #csv_list_train_ner = glob.glob('/home/ywm/桌面/new_data_all/各个网站的所有数据/train_ner_2000/*.csv') #train_ner_2000
        print(len(csv_list_train))

        for i in range(len(csv_list_train)):
            list_xywy = str(csv_list_train[i]).split("/")
            drop_csv = list_xywy[-1]
            disease_xywy = drop_csv[:-4]
            disease_xywy_drop = str(disease_xywy).replace(" ", "")
            print(disease_xywy_drop)

            fw_ = open("/home/ywm/桌面/new_data_all/各个网站的所有数据/miansi/" + disease_xywy_drop + ".csv", "w")
            fieldnames = ['question','symptom']
            fw = csv.DictWriter(fw_,fieldnames=fieldnames)
            fw.writeheader()
            with open("/home/ywm/桌面/new_data_all/各个网站的所有数据/train_ER/" + disease_xywy_drop + ".csv", "r") as fr_:
                content = ''
                fr = csv.reader(fr_)
                j = 0
                for line in fr:
                    content = str(line).replace('[','').replace(']','').strip("'")
                    #print(content)
                    sent = list(content)
                    #print(sent)
                    sent_data = [(sent, ['O'] * len(sent))]
                    #print(sent_data)
                    tag = model.demo_one(sess, sent_data)
                    #print(tag)
                    symptom, disease, treatment, check = get_entity_keys(tag, sent,['SYMPTOM', 'DISEASE', 'TREATMENT', 'CHECK'])
                    if len(symptom)>=2 and j<=1000:
                        j=j+1
                        fw.writerow({'question':line,'symptom':symptom})
                        print(symptom,disease,treatment,check)
                        print(j)
                    elif j>1000:
                        break

            fw_.close()
            fr_.close()

        # while(1):
        #     print('Please input your sentence:')
        #     demo_sent = input()
        #     if demo_sent == '' or demo_sent.isspace():
        #         print('See you next time!')
        #         break
        #     else:
        #         demo_sent = list(demo_sent.strip())
        #         # label全都是'O'
        #         demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        #         tag = model.demo_one(sess, demo_data)
        #         # entities = get_entity(tag, demo_sent)
        #         # print('ENTITY: {}\n'.format(entities))
        #         body, chec, cure, dise, symp = get_entity(tag, demo_sent)
        #         print('body:{}\nchec:{}\ncure:{}\ndise:{}\nsymp:{}\n'.format(body, chec, cure, dise, symp))
