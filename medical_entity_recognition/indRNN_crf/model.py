import numpy as np
import os
import time
import sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from medical_entity_recognition.indRNN_crf.data import pad_sequences, batch_yield, get_entity
from medical_entity_recognition.indRNN_crf.utils import get_logger
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from medical_entity_recognition.indRNN_crf.ind_rnn_cell import IndRNNCell
import tensorlayer as tl

class BiLSTM_CRF(object):
    def __init__(self,batch_size, epoch_num, hidden_dim, embeddings,
                 dropout_keep, optimizer, lr, clip_grad,
                 tag2label, vocab, shuffle,
                 model_path, summary_path, log_path, result_path,
                 CRF=True, update_embedding=True):

        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.dropout_keep_prob = dropout_keep
        self.optimizer = optimizer
        self.lr = lr
        self.clip_grad = clip_grad
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab  # word2id
        self.shuffle = shuffle
        self.model_path = model_path
        self.summary_path = summary_path
        self.logger = get_logger(log_path)
        self.result_path = result_path
        self.CRF = CRF
        self.update_embedding = update_embedding

        # self.filter_sizes = filter_sizes
        # self.num_filters = num_filters
        #self.sequence_length = sequence_length


    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        #在此处添加卷积层
        #self.convolution_max__layer_op()
        #在此处添加池化层
        #添加全连接层  输出维度为300，接到biLSTM
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    # Placeholders for input, output and dropout
    # 添加占位符
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    # 找到每个word对应的编码
    def lookup_layer_op(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope("words"):
                _word_embeddings = tf.Variable(self.embeddings,
                                               dtype=tf.float32,
                                               trainable=self.update_embedding,
                                               name="_word_embeddings")
                word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                         ids=self.word_ids,
                                                         name="word_embeddings")
                word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)
            self.word_embeddings = tf.layers.batch_normalization(word_embeddings, training=True, momentum=0)

#tf.layers.batch_normalization(word_embeddings, training=True, momentum=0)
    def biLSTM_layer_op(self):
        #TIME_STEPS = 784
        #NUM_UNITS = 128
        NUM_LAYERS = 4
        RECURRENT_MAX = pow(2, 1 / 110)
        #NUM_CLASSES = 10
        # Parameters taken from https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne
        #CLIP_GRADIENTS = True
        LAST_LAYER_LOWER_BOUND = pow(0.5, 1 / 110)
        #BATCH_SIZE_TRAIN = 32

        with tf.variable_scope("bi-lstm"):
        #     cell_fw = LSTMCell(self.hidden_dim)
        #     cell_bw = LSTMCell(self.hidden_dim)
        #
        #
        #     (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
        #         cell_fw=cell_fw,
        #         cell_bw=cell_bw,
        #         inputs=self.word_embeddings,
        #         sequence_length=self.sequence_lengths,
        #         dtype=tf.float32)
        #     output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        #     output = tf.nn.dropout(output, self.dropout_pl)
        #     print(output) #Tensor("bi-lstm/dropout/mul:0", shape=(?, ?, 600), dtype=float32)
#----------------------indrnn-------------------------------------------------------------------------
            layer_input = self.word_embeddings  #inputs
            output = None
            input_init = tf.random_uniform_initializer(-0.001, 0.001)

            for layer in range(1, NUM_LAYERS + 1):  # 1---7
                # Init only the last layer's recurrent weights around 1
                recurrent_init_lower = 0 if layer < NUM_LAYERS else LAST_LAYER_LOWER_BOUND
                recurrent_init = tf.random_uniform_initializer(recurrent_init_lower,
                                                               RECURRENT_MAX)
                # Build the layer
                cell = IndRNNCell(self.hidden_dim,
                                  recurrent_max_abs=RECURRENT_MAX,
                                  input_kernel_initializer=input_init,
                                  recurrent_kernel_initializer=recurrent_init)

                output, _ = tf.nn.dynamic_rnn(cell, layer_input,
                                            dtype=tf.float32,
                                            scope="rnn%d" % layer)

                #print(output)
                #if layer%2 == 0:
                output = tf.layers.batch_normalization(output,training=True,momentum=0)
                #print(output)

                layer_input = output
            #print("output",output) #output Tensor("bi-lstm/batch_normalization_1/batchnorm/add_1:0", shape=(?, ?, 300), dtype=float32)
            output = tf.nn.dropout(output, self.dropout_pl)
#----------------------------------Bi-IndRnn------------------------------------------------------------------
            #recurrent_init_lower = LAST_LAYER_LOWER_BOUND
            #recurrent_init = tf.random_uniform_initializer(recurrent_init_lower,RECURRENT_MAX)
            #
            # cell_fw = IndRNNCell(self.hidden_dim,
            #                  recurrent_max_abs=RECURRENT_MAX,
            #                  input_kernel_initializer=input_init,
            #                  recurrent_kernel_initializer=recurrent_init)
            # cell_bw = IndRNNCell(self.hidden_dim,
            #                  recurrent_max_abs=RECURRENT_MAX,
            #                  input_kernel_initializer=input_init,
            #                  recurrent_kernel_initializer=recurrent_init)
            #
            # # #cell_fw_1_dr = tf.nn.rnn_cell.DropoutWrapper(cell_fw_1,output_keep_prob=0.5)
            # #
            # # # cell_fw_2 = IndRNNCell(self.hidden_dim,
            # # #                    recurrent_max_abs=RECURRENT_MAX,
            # # #                    input_kernel_initializer=input_init,
            # # #                    recurrent_kernel_initializer=recurrent_init)
            # # # #cell_fw_2_dr = tf.nn.rnn_cell.DropoutWrapper(cell_fw_2,output_keep_prob=0.5)
            # # # cell_fw_3 = IndRNNCell(self.hidden_dim,
            # # #                    recurrent_max_abs=RECURRENT_MAX,
            # # #                    input_kernel_initializer=input_init,
            # # #                    recurrent_kernel_initializer=recurrent_init)
            # # # cell_fw_4 = IndRNNCell(self.hidden_dim,
            # # #                    recurrent_max_abs=RECURRENT_MAX,
            # # #                    input_kernel_initializer=input_init,
            # # #                    recurrent_kernel_initializer=recurrent_init)
            # # #
            # # # cell_bw_1 = IndRNNCell(self.hidden_dim,
            # # #                   recurrent_max_abs=RECURRENT_MAX,
            # # #                   input_kernel_initializer=input_init,
            # # #                 recurrent_kernel_initializer=recurrent_init)
            # # # #cell_bw_1_dr = tf.nn.rnn_cell.DropoutWrapper(cell_bw_1,output_keep_prob=0.5)
            # # # cell_bw_2 = IndRNNCell(self.hidden_dim,
            # # #                    recurrent_max_abs=RECURRENT_MAX,
            # # #                    input_kernel_initializer=input_init,
            # # #                    recurrent_kernel_initializer=recurrent_init)
            # # # #cell_bw_2_dr = tf.nn.rnn_cell.DropoutWrapper(cell_bw_2,output_keep_prob=0.5)
            # # # cell_bw_3 = IndRNNCell(self.hidden_dim,
            # # #                    recurrent_max_abs=RECURRENT_MAX,
            # # #                    input_kernel_initializer=input_init,
            # # #                    recurrent_kernel_initializer=recurrent_init)
            # # # cell_bw = IndRNNCell(self.hidden_dim,
            # # #                    recurrent_max_abs=RECURRENT_MAX,
            # # #                    input_kernel_initializer=input_init,
            # # #                    recurrent_kernel_initializer=recurrent_init)
            # #
            # # cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw_1,cell_fw_2,cell_fw_3,cell_fw_4])
            # # cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw_1,cell_bw_2,cell_bw_3,cell_bw_4])
            #
            # (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
            #      cell_fw=cell_fw,
            #      cell_bw=cell_bw,
            #      inputs=self.word_embeddings,
            #      sequence_length=self.sequence_lengths,
            #      dtype=tf.float32,
            #      scope="birnn1")
            #
            # #output_fw_seq = tf.layers.batch_normalization(output_fw_seq, training=True, momentum=0)
            # #output_bw_seq = tf.layers.batch_normalization(output_bw_seq, training=True, momentum=0)
            # output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            # #output = tf.nn.dropout(output, self.dropout_pl)
            # output = tf.layers.batch_normalization(output, training=True, momentum=0)
            # #
            # cell_fw_ = IndRNNCell(self.hidden_dim,
            #                  recurrent_max_abs=RECURRENT_MAX,
            #                  input_kernel_initializer=input_init,
            #                  recurrent_kernel_initializer=recurrent_init)
            # cell_bw_ = IndRNNCell(self.hidden_dim,
            #                  recurrent_max_abs=RECURRENT_MAX,
            #                  input_kernel_initializer=input_init,
            #                  recurrent_kernel_initializer=recurrent_init)
            #
            # (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
            #    cell_fw=cell_fw_,
            #    cell_bw=cell_bw_,
            #    inputs=output,
            #    sequence_length=self.sequence_lengths,
            #    dtype=tf.float32,
            #    scope="birnn2")
            # output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            # output = tf.nn.dropout(output, self.dropout_pl)
            # output = tf.layers.batch_normalization(output, training=True, momentum=0)
#-------------------------------------------------------------------------------------------
        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            #print(s)#Tensor("proj/Shape:0", shape=(3,), dtype=int32)
            output = tf.reshape(output, [-1, self.hidden_dim])
            #print(output)#Tensor("proj/Reshape:0", shape=(?, 300), dtype=float32)
            pred = tf.matmul(output, W) + b
            #print(pred)#Tensor("proj/add:0", shape=(?, 9), dtype=float32)
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])


    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(
                inputs=self.logits, tag_indices=self.labels, sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)
            #print(self.loss)  #Tensor("Neg:0", shape=(), dtype=float32)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)


    def trainstep_op(self):
        with tf.variable_scope("train_step"):

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            #------Adam更改
            if self.optimizer == 'Adam':
                LEARNING_RATE_INIT = 0.001
                LEARNING_RATE_DECAY_STEPS = 100
                learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, self.global_step,
                                                           LEARNING_RATE_DECAY_STEPS, 0.98,
                                                           staircase=True)
                optim = tf.train.AdamOptimizer(learning_rate)#self.lr_pl
             #————————————————————
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)

            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]


            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            # self.evaluate(label_list, seq_len_list, test)
            self.evaluate_(label_list, test)

    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """
        每次训练的一个epoch，在每个epoch里面有多个batch_size
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        # self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)
        self.evaluate_(label_list_dev, dev)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

#
# #------------------------------------------------------------------------
#         for _ in conlleval(model_predict, label_path, metric_path):
# #------------------------------------------------------------------------
#
#             self.logger.info(_)

    # 使用sklearn进行模型评估
    def evaluate_(self, label_list, data):
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag
            #print(label2tag)
        # label_pre = []
        # label_true = []
        # for label, (sent, tag) in zip(label_list, data):
        #     label_pre.extend([label2tag[label_] for label_ in label])
        #     label_true.extend([tag_ for tag_ in tag])
        #
        # lb = LabelBinarizer()
        # y_true_combined = lb.fit_transform(label_true)
        # y_pred_combined = lb.transform(label_pre)
        #
        # # tagset = set(lb.classes_) - {'O'}
        # tagset = set(lb.classes_)
        # tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        # class_indices = {
        #     cls: idx for idx, cls in enumerate(lb.classes_)
        # }
        #
        # print(classification_report(
        #     y_true_combined,
        #     y_pred_combined,
        #     labels=[class_indices[cls] for cls in tagset],
        #     target_names=tagset
        # ))

        contents = []
        y_test = []
        y_pred = []
        for content, label in data:
            #print(content)#['手', '指', '头', '肿', '，', '摸', '起', '来', '软', '软', '的',
            contents.extend([con for con in content])
            y_test.extend([y for y in label])
        for label in label_list:
            y_pred.extend([label2tag[y] for y in label])
            #print(label)
#---------------------------------------------------------------------
        #entity = ['BODY', 'CHECK', 'DISEASE', 'SIGNS', 'TREATMENT']
        entity = ['SYMPTOM', 'DISEASE', 'TREATMENT', 'CHECK']
        entities_pred = get_entity(y_pred, contents, entity)
        #print(entities_pred)#['超', '超', '：谷草转氨酶、乳酸脱氢酶、肌酸激酶、超',
        entities_true = get_entity(y_test, contents, entity)

        pre_all = 0
        rec_all = 0
        f1_all = 0
        for i, (pred, true) in enumerate(zip(entities_pred, entities_true)):
            # print("entities_pred",entities_pred)
            # print("entities_true",entities_true)
            # print(i)
            #print(entity[i])
            pre = 0
            rec = 0
            for p in pred:
                if p in true:
                    pre += 1
            for t in true:
                if t in pred:
                    rec += 1
            #初始值设定为0.1
            precision = 0.1
            recall = 0.1
            if len(pred) != 0:
                #实际上非常简单，精确率是针对我们预测结果而言的，它表示的是预测为正的样本中有多少是真正的正样本。那么预测为正就有两种可能了，
                # 一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP) P=TP/(TP+FP)
                precision = pre * 1.0 / len(pred)
            if len(true) != 0:
                #而召回率是针对我们原来的样本而言的，它表示的是样本中的正例有多少被预测正确了。那也有两种可能，一种是把原来的正类预测成正类(TP)，
                # 另一种就是把原来的正类预测为负类(FN)。R=TP/(TP+FN)
                #准确率(accuracy) = 预测对的/所有 = (TP+TN)/(TP+FN+FP+TN) = 70%TN: 将负类预测为负类数  30
                recall = rec * 1.0 / len(true)
            if precision == 0 and recall == 0:
                recall = 0.1
                precision = 0.1
            #F值  = 正确率 * 召回率 * 2 / (正确率 + 召回率) （F 值即为正确率和召回率的调和平均值）
            f1 = 2 * precision * recall / (precision + recall)
            pre_all += precision
            rec_all += recall
            f1_all += f1

            print('{:10s}: precision:{:.4f}, recall:{:.4f}, f1-score:{:.4f}'.format(entity[i], precision, recall, f1))

        print('{:10s}: precision:{:.4f}, recall:{:.4f}, f1-score:{:.4f}'.format(
            'average', pre_all / len(entity), rec_all / len(entity), f1_all / len(entity)))

