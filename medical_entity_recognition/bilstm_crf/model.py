import time
import sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from medical_entity_recognition.bilstm_crf.data import pad_sequences, batch_yield, get_entity
from medical_entity_recognition.bilstm_crf.utils import get_logger


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
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(
                inputs=self.logits, tag_indices=self.labels, sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

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
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
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
            label2tag[label] = tag #if label != 0 else label
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

#     def evaluate(self, label_list, seq_len_list, data, epoch=None):
#         """
#
#         :param label_list:
#         :param seq_len_list:
#         :param data:
#         :param epoch:
#         :return:
#         """
#         label2tag = {}
#         for tag, label in self.tag2label.items():
#             label2tag[label] = tag if label != 0 else label
#
#         model_predict = []
#         for label_, (sent, tag) in zip(label_list, data):
#             tag_ = [label2tag[label__] for label__ in label_]
#             sent_res = []
#             if len(label_) != len(sent):
#                 print(sent)
#                 print(len(label_))
#                 print(tag)
#             for i in range(len(sent)):
#                 # tag是真实值，tag_是预测值
#                 sent_res.append([sent[i], tag[i], tag_[i]])
#             model_predict.append(sent_res)
#         epoch_num = str(epoch+1) if epoch != None else 'test'
#         label_path = os.path.join(self.result_path, 'label_' + epoch_num)
#         metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
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

