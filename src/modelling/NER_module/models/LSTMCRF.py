import tensorflow as tf
import numpy as np
import time
import math
import os
import json

from src.modelling.NER_module.utils.padding import refine_seq_and_get_lengths

""" This class creates a model to learn to output Labels for a given sequence
    using LSTM or GRU recurrent layers"""


## TODO : handel variable size sequence,done(check this)

## TODO: code is really dirty need to fix it

class LSTMCRF:
    # def __del__(self):
    #     self.sess.close()

    def __init__(self, dict, dict2, dict_rev, dict_rev2, path, logs_dir=None,
                 pretrained_w2v=None,
                 learning_rate=0.005,
                 batch_size=64, embedding_size=100, iter_nums=60, timesteps=60, num_hiddens=128, gru=False):
        # self.x = x
        # self.y = y
        tf.reset_default_graph()
        if not os.path.exists(path):
            os.makedirs(path)
            # print('ClassifierModel is being created at :', path + "/conf.json")
            conf = {}
            with open(path + "/conf.json", 'w', encoding='utf-8') as f:
                # global json_loaded_data
                # conf['iteration'] = 0
                conf['embedding_size'] = embedding_size
                conf['hidden_num'] = num_hiddens
                conf['time_steps'] = timesteps
                conf['learning_rate'] = learning_rate
                conf['gru'] = gru
                conf['logs_dir'] = logs_dir
                conf['batch_size'] = batch_size
                # print(conf)
                json_string = json.dumps(conf)
                # print(json_string)
                f.write(json_string)
                f.flush()
                json_loaded_data = json.loads(json_string)
            load = False
        else:
            with open(path + "/conf.json", 'r') as f:
                # global json_loaded_data

                stri = f.read()
                # print('content loaded :', stri)
                json_loaded_data = json.loads(stri)

                # conf['iteration'] = 0
                embedding_size = json_loaded_data['embedding_size']
                num_hiddens = json_loaded_data['hidden_num']
                timesteps = json_loaded_data['time_steps']
                learning_rate = json_loaded_data['learning_rate']
                gru = json_loaded_data['gru']
                logs_dir = json_loaded_data['logs_dir']
                batch_size = json_loaded_data['batch_size']
                # print(json_loaded_data)
            # load = False
            load = True

        self.dict = dict
        self.dict_rev = dict_rev
        self.dict2 = dict2
        self.dict_rev2 = dict_rev2

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.path = path
        self.logs_dir = logs_dir
        self.pretrained_w2v = pretrained_w2v
        # self.dropout = dropout
        self.embedding_size = embedding_size
        # self.iter_nums = iter_nums
        self.timesteps = timesteps
        self.num_hiddens = num_hiddens
        self.gru = gru
        # self.load_data()
        self.vocab_size2 = len(self.dict2)

        self.vocab_size = len(self.dict)
        if self.pretrained_w2v is not None:
            self.data = tf.placeholder(tf.float32, [None, self.timesteps, self.embedding_size])
        else:
            self.data = tf.placeholder(tf.int32, [None, self.timesteps])
        self.drop_out = tf.placeholder(tf.float32)
        self.target = tf.placeholder(tf.float32, [None, self.timesteps, self.vocab_size2])
        self.length = tf.placeholder(tf.int32, [None])
        if self.pretrained_w2v is not None:
            self.inputs_embedded = self.data
        else:
            embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
            ## TODO: edit it and do the one hot, so zeros for padded ons
            self.inputs_embedded = tf.nn.embedding_lookup(embeddings, self.data)

        # used = tf.sign(tf.reduce_max(tf.abs(self.inputs_embedded), axis=2))
        # self.length = tf.reduce_sum(used, axis=1)
        # self.length = tf.cast(self.length, tf.int32)

        self.logits = self.logits_op()

        self.loss = self.cost(self.logits, len(self.dict2))

        self.train_op = self.optimize(self.loss)

        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.logs_dir is not None:
            self.train_writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
            self.scalar = tf.placeholder(tf.float32)
            tf.summary.scalar("iter_loss", self.scalar)
            self.merge = tf.summary.merge_all()

    def save_model(self):
        save_path = self.saver.save(self.sess, self.path + "/model.ckpt")
        # print("ClassifierModel saved in path: %s" % save_path)

    def load_model(self):
        self.saver.restore(self.sess, self.path + "/model.ckpt")
        # print("ClassifierModel restored from", self.path + "/model.ckpt", ".")

    def logits_op(self):
        # Recurrent network.

        if self.gru:
            network = tf.nn.rnn_cell.GRUCell(self.num_hiddens)
            network_bw = tf.nn.rnn_cell.GRUCell(self.num_hiddens)
        else:
            network = tf.nn.rnn_cell.LSTMCell(self.num_hiddens)
            network_bw = tf.nn.rnn_cell.LSTMCell(self.num_hiddens)

        network = tf.nn.rnn_cell.DropoutWrapper(
            network, output_keep_prob=self.drop_out)
        # network = tf.nn.rnn_cell.MultiRNNCell([network] * self._num_layers)
        # output, _ = tf.nn.dynamic_rnn(network, self.inputs_embedded, dtype=tf.float32)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            network, network_bw,
            self.inputs_embedded,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.nn.dropout(output, self.drop_out)

        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        ## have to use num_hidden * 2
        weight, bias = self._weight_and_bias(self.num_hiddens * 2, self.vocab_size2)

        output = tf.reshape(output, [-1, 2 * self.num_hiddens])
        logits = tf.matmul(output, weight) + bias
        logits = tf.reshape(logits, [-1, max_length, num_classes])

        return logits

    def cost(self, logits, num_classes):

        # Compute cross entropy for each frame.
        reshaped = tf.reshape(logits, [-1, self.timesteps, num_classes])
        real_label = tf.argmax(self.target, 2)
        # reshaped_target = tf.reshape()
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            reshaped, real_label, self.length)
        self.trans_params = trans_params  # need to evaluate it for decoding
        return tf.reduce_mean(-log_likelihood)

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(loss)

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    """ input must be a list of sequences with size of each sequence timesteps,
    even if passing one sequence it must be in a list"""

    # TODO: what if input larger than timesteps ?
    def get_label(self, seq):
        seq, lens = refine_seq_and_get_lengths(seq, self.timesteps)
        # print(seq)
        number_of_batches = math.ceil(len(seq) / self.batch_size)
        output = np.zeros(shape=(len(seq), self.timesteps))
        for i in range(number_of_batches):
            bs = min(len(seq) - i * self.batch_size, self.batch_size)
            if self.pretrained_w2v is not None:
                xbatch = np.zeros(shape=(bs, self.timesteps, self.embedding_size))
            else:
                xbatch = np.zeros(shape=(bs, self.timesteps))
            # print('len dict2', len(self.dict2))
            # ybatch = np.zeros(shape=(bs, self.timesteps, len(self.dict2)))
            for j in range(bs):
                index = i * self.batch_size + j
                if self.pretrained_w2v is None:
                    xbatch[j] = seq[index]
                for k in range(self.timesteps):
                    # idx = int(y[index][k])
                    # ybatch[j][k][idx] = 1

                    if self.pretrained_w2v is not None:
                        if seq[index][k] == 0:
                            xbatch[j][k] = np.zeros(shape=self.embedding_size)
                        elif self.dict[seq[index][k]] in self.pretrained_w2v:
                            xbatch[j][k] = self.pretrained_w2v[self.dict[seq[index][k]]]
                        else:
                            xbatch[j][k] = self.pretrained_w2v['ناموجود']

            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params],
                feed_dict={self.data: xbatch,
                             self.length: lens[i * self.batch_size:i * self.batch_size + bs],
                             self.drop_out: 1.0})

            viterbi_sequences = []
            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, lens):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
                while len(viterbi_seq) < self.timesteps:
                    viterbi_seq.append(0)
                viterbi_sequences += [viterbi_seq]
            res = viterbi_sequences


            output[i * self.batch_size: i * self.batch_size + bs, :] = res
        return output

    """This method initializes computation graph and starts training procedure"""

    def train(self, x, y, iteration):

        # global dict, dict2, train_op

        x, lens = refine_seq_and_get_lengths(x)
        number_of_batches = math.ceil(len(x) / self.batch_size)
        # number_of_batches = 10
        for iter in range(iteration):
            iter_loss = 0
            iter_error = 0
            start_time = time.time()
            # TODO: fix this -1
            for i in range(number_of_batches):
                bs = min(len(x) - i * self.batch_size, self.batch_size)
                if self.pretrained_w2v is not None:
                    xbatch = np.zeros(shape=(bs, self.timesteps, self.embedding_size))
                else:
                    xbatch = np.zeros(shape=(bs, self.timesteps))
                # print('len dict2', len(self.dict2))
                ybatch = np.zeros(shape=(bs, self.timesteps, len(self.dict2)))

                for j in range(bs):
                    index = i * self.batch_size + j
                    if self.pretrained_w2v is None:
                        xbatch[j] = x[index]
                    for k in range(self.timesteps):
                        if y[index][k] != 0:
                            idx = int(y[index][k]) - 1
                            if idx <= len(self.dict2):
                                ybatch[j][k][idx] = 1

                        if self.pretrained_w2v is not None:
                            if x[index][k] == 0:
                                xbatch[j][k] = np.zeros(shape=(self.embedding_size))
                            elif self.dict[x[index][k]] in self.pretrained_w2v:
                                xbatch[j][k] = self.pretrained_w2v[self.dict[x[index][k]]]
                            else:
                                xbatch[j][k] = self.pretrained_w2v['ناموجود']

                _, loss = self.sess.run([self.train_op, self.loss],
                                        feed_dict={self.data: xbatch, self.target: ybatch,
                                                   self.length: lens[i * self.batch_size:i * self.batch_size + bs],
                                                   self.drop_out: 0.5})
                # lens = [len([t for t in tt if t != 0]) for tt in x]
                # print("length1", l)
                # print("length2", lens)
                iter_loss += loss
                # iter_error += error
                # print(loss)

            print('Average loss at iteration', iter, ':', iter_loss / number_of_batches, '(',
                  time.time() - start_time,
                  's)')
            if self.logs_dir is not None:
                summary = self.sess.run(self.merge, {self.scalar: iter_loss})
                self.train_writer.add_summary(summary, iter)
            self.save_model()
