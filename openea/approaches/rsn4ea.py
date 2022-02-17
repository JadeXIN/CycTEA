import math
import multiprocessing as mp
import random
import time
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

from openea.models.trans.transe import BasicModel
from openea.modules.utils.util import load_session
from openea.modules.finding.evaluation import early_stop
from openea.approaches.bootea import bootstrapping
from openea.modules.finding.similarity import sim


class BasicReader(object):
    def read(self, data_path='data/dbp_wd_15k_V1/mapping/0_3/'):
        # add shortcuts
        kgs = self.kgs

        kg1 = pd.DataFrame(kgs.kg1.relation_triples_list, columns=['h_id', 'r_id', 't_id'])
        kg2 = pd.DataFrame(kgs.kg2.relation_triples_list, columns=['h_id', 'r_id', 't_id'])

        kb = pd.concat([kg1, kg2], ignore_index=True)

        # self._eid_1 = pd.Series(eid_1)
        # self._eid_2 = pd.Series(eid_2)

        self._ent_num = kgs.entities_num
        self._rel_num = kgs.relations_num
        # self._ent_id = e_map
        # self._rel_id = r_map
        self._ent_mapping = pd.DataFrame(list(kgs.train_links), columns=['kb_1', 'kb_2'])
        self._rel_mapping = pd.DataFrame({}, columns=['kb_1', 'kb_2'])
        self._ent_testing = pd.DataFrame(list(kgs.test_links), columns=['kb_1', 'kb_2'])
        self._rel_testing = pd.DataFrame({}, columns=['kb_1', 'kb_2'])

        # add reverse edges
        rev_kb = kb[['t_id', 'r_id', 'h_id']].values
        rev_kb[:, 1] += self._rel_num
        rev_kb = pd.DataFrame(rev_kb, columns=['h_id', 'r_id', 't_id'])
        self._rel_num *= 2
        kb = pd.concat([kb, rev_kb], ignore_index=True)
        # print(kb)
        # print(kb[len(kb)//2:])

        self._kb = kb
        # we first tag the entities that have algined entities according to entity_mapping
        self.add_align_infor()
        # we then connect two KGs by creating new triples involving aligned entities.
        self.add_weight()

    def add_align_infor(self):
        kb = self._kb

        ent_mapping = self._ent_mapping
        rev_e_m = ent_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})
        rel_mapping = self._rel_mapping
        rev_r_m = rel_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})

        ent_mapping = pd.concat([ent_mapping, rev_e_m], ignore_index=True)
        rel_mapping = pd.concat([rel_mapping, rev_r_m], ignore_index=True)

        ent_mapping = pd.Series(ent_mapping.kb_2.values, index=ent_mapping.kb_1.values)
        rel_mapping = pd.Series(rel_mapping.kb_2.values, index=rel_mapping.kb_1.values)

        self._e_m = ent_mapping
        self._r_m = rel_mapping

        kb['ah_id'] = kb.h_id
        kb['ar_id'] = kb.r_id
        kb['at_id'] = kb.t_id

        h_mask = kb.h_id.isin(ent_mapping)
        r_mask = kb.r_id.isin(rel_mapping)
        t_mask = kb.t_id.isin(ent_mapping)

        kb['ah_id'][h_mask] = ent_mapping.loc[kb['ah_id'][h_mask].values]
        kb['ar_id'][r_mask] = rel_mapping.loc[kb['ar_id'][r_mask].values]
        kb['at_id'][t_mask] = ent_mapping.loc[kb['at_id'][t_mask].values]

        self._kb = kb

    def add_weight(self):
        kb = self._kb[['h_id', 'r_id', 't_id', 'ah_id', 'ar_id', 'at_id']]

        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0

        h_mask = ~(kb.h_id == kb.ah_id)
        r_mask = ~(kb.r_id == kb.ar_id)
        t_mask = ~(kb.t_id == kb.at_id)

        kb.loc[h_mask, 'w_h'] = 1
        kb.loc[r_mask, 'w_r'] = 1
        kb.loc[t_mask, 'w_t'] = 1

        akb = kb[['ah_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']]
        akb = akb.rename(columns={'ah_id': 'h_id', 'ar_id': 'r_id', 'at_id': 't_id'})

        ahkb = kb[h_mask][['ah_id', 'r_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ah_id': 'h_id'})
        arkb = kb[r_mask][['h_id', 'ar_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ar_id': 'r_id'})
        atkb = kb[t_mask][['h_id', 'r_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(columns={'at_id': 't_id'})
        ahrkb = kb[h_mask & r_mask][['ah_id', 'ar_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id', 'ar_id': 'r_id'})
        ahtkb = kb[h_mask & t_mask][['ah_id', 'r_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id', 'at_id': 't_id'})
        artkb = kb[r_mask & t_mask][['h_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ar_id': 'r_id', 'at_id': 't_id'})
        ahrtkb = kb[h_mask & r_mask & t_mask][['ah_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id',
                     'ar_id': 'r_id',
                     'at_id': 't_id'})

        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0

        kb = pd.concat(
            [akb, ahkb, arkb, atkb, ahrkb, ahtkb, artkb, ahrtkb, kb[['h_id', 'r_id', 't_id', 'w_h', 'w_r', 'w_t']]],
            ignore_index=True).drop_duplicates()

        self._kb = kb.reset_index(drop=True)


# sampler
class BasicSampler(object):
    def sample_paths(self, repeat_times=2, save_path=None):
        opts = self._options

        kb = self._kb.copy()

        kb = kb[['h_id', 'r_id', 't_id']]

        # sampling triples with the h_id-(r_id,t_id) form.

        rtlist = np.unique(kb[['r_id', 't_id']].values, axis=0)

        rtdf = pd.DataFrame(rtlist, columns=['r_id', 't_id'])

        rtdf = rtdf.reset_index().rename({'index': 'tail_id'}, axis='columns')

        rtkb = kb.merge(
            rtdf, left_on=['r_id', 't_id'], right_on=['r_id', 't_id'])

        htail = np.unique(rtkb[['h_id', 'tail_id']].values, axis=0)

        htailmat = csr_matrix((np.ones(len(htail)), (htail[:, 0], htail[:, 1])),
                              shape=(self._ent_num, rtlist.shape[0]))

        # calulate corss-KG bias at first
        em = pd.concat(
            [self._ent_mapping.kb_1, self._ent_mapping.kb_2]).values

        rtkb['across'] = rtkb.t_id.isin(em)
        rtkb.loc[rtkb.across, 'across'] = opts.beta
        rtkb.loc[rtkb.across == 0, 'across'] = 1 - opts.beta

        rtailkb = rtkb[['h_id', 't_id', 'tail_id', 'across']]

        def gen_tail_dict(x):
            return x.tail_id.values, x.across.values / x.across.sum()

        rtailkb = rtailkb.groupby('h_id').apply(gen_tail_dict)

        rtailkb = pd.DataFrame({'tails': rtailkb})

        # start sampling

        hrt = np.repeat(kb.values, repeat_times, axis=0)

        # for starting triples
        def perform_random(x):
            return np.random.choice(x.tails[0], 1, p=x.tails[1].astype(np.float))

        # else
        def perform_random2(x):
            # calculate depth bias
            pre_c = htailmat[np.repeat(x.pre, x.tails[0].shape[0]), x.tails[0]]
            pre_c[pre_c == 0] = opts.alpha
            pre_c[pre_c == 1] = 1 - opts.alpha
            p = x.tails[1].astype(np.float).reshape(
                [-1, ]) * pre_c.A.reshape([-1, ])
            p = p / p.sum()
            return np.random.choice(x.tails[0], 1, p=p)

        # print(rtailkb.loc[hrt[:, 2]])
        rt_x = rtailkb.loc[hrt[:, 2]].apply(perform_random, axis=1)
        rt_x = rtlist[np.concatenate(rt_x.values)]

        rts = [hrt, rt_x]
        print('hrt', 'rt_x', len(hrt), len(rt_x))
        c_length = 5
        while c_length < opts.max_length:
            curr = rtailkb.loc[rt_x[:, 1]]
            print(len(curr), len(hrt[:, 0]))
            curr.loc[:, 'pre'] = hrt[:, 0]

            rt_x = curr.apply(perform_random2, axis=1)
            rt_x = rtlist[np.concatenate(rt_x.values)]

            rts.append(rt_x)
            c_length += 2

        data = np.concatenate(rts, axis=1)
        data = pd.DataFrame(data)

        self._train_data = data

        if save_path is not None:
            print("save paths to:", save_path)
            data.to_csv(save_path)


class RSN4EA(BasicReader, BasicSampler, BasicModel):
    def __init__(self):
        super().__init__()
        self.new_sup_links = list()
        self.labeled_align = set()
        self.ref_ent1 = None
        self.ref_ent2 = None

    def init(self):
        self._options = opts = self.args
        opts.data_path = opts.training_data

        self.read(data_path=self._options.data_path)

        # sequence_datapath = '%spaths_%.1f_%.1f' % (
        #     self._options.data_path, self._options.alpha, self._options.beta)

        sequence_datapath = '%sent_paths_%.1f_%.1f_len%s' % (os.path.join(self._options.data_path,
                                                                          self.args.dataset_division),
                                                             self._options.alpha,
                                                             self._options.beta,
                                                             self._options.max_length)

        if not os.path.exists(sequence_datapath):
            self.sample_paths(save_path=sequence_datapath)
        else:
            print('load existing training sequences')
            self._train_data = pd.read_csv(sequence_datapath, index_col=0)

        self._define_variables()
        self._define_embed_graph()
        self._generate_semi_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2

    def _define_variables(self):
        options = self._options
        hidden_size = options.hidden_size

        self._entity_embedding = tf.get_variable(
            'entity_embedding',
            [self._ent_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self._relation_embedding = tf.get_variable(
            'relation_embedding',
            [self._rel_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )

        self.ent_embeds, self.rel_embeds = self._entity_embedding, self._relation_embedding

        self._rel_w = tf.get_variable(
            "relation_softmax_w",
            [self._rel_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self._rel_b = tf.get_variable(
            "relation_softmax_b",
            [self._rel_num],
            initializer=tf.constant_initializer(0)
        )
        self._ent_w = tf.get_variable(
            "entity_softmax_w",
            [self._ent_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self._ent_b = tf.get_variable(
            "entity_softmax_b",
            [self._ent_num],
            initializer=tf.constant_initializer(0)
        )

        self.entity_w, self._entity_b = self._ent_w, self._ent_b

        self._lr = tf.Variable(options.learning_rate, trainable=False)

        self._optimizer = tf.train.AdamOptimizer(options.learning_rate)  # , beta2=0.98, epsilon=1e-9

    def bn(self, inputs, is_train=True, reuse=True):
        return tf.contrib.layers.batch_norm(inputs,
                                            center=True,
                                            scale=True,
                                            is_training=is_train,
                                            reuse=reuse,
                                            scope='bn',
                                            )

    def lstm_cell(self, drop=True, keep_prob=0.5, num_layers=2, hidden_size=None):
        if not hidden_size:
            hidden_size = self._options.hidden_size

        def basic_lstm_cell():
            return tf.contrib.rnn.LSTMCell(
                num_units=hidden_size,
                initializer=tf.orthogonal_initializer,
                forget_bias=1,
                reuse=tf.get_variable_scope().reuse,
                activation=tf.identity
            )

        def drop_cell():
            return tf.contrib.rnn.DropoutWrapper(
                basic_lstm_cell(),
                output_keep_prob=keep_prob
            )

        if drop:
            gen_cell = drop_cell
        else:
            gen_cell = basic_lstm_cell

        if num_layers == 0:
            return gen_cell()

        cell = tf.contrib.rnn.MultiRNNCell(
            [gen_cell() for _ in range(num_layers)],
            state_is_tuple=True,
        )
        return cell

    def sampled_loss(self, inputs, labels, w, b, weight=1, is_entity=False):
        num_sampled = min(self._options.num_samples, w.shape[0] // 3)

        labels = tf.reshape(labels, [-1, 1])

        losses = tf.nn.nce_loss(
            weights=w,
            biases=b,
            labels=labels,
            inputs=tf.reshape(inputs, [-1, w.get_shape().as_list()[1]]),
            num_sampled=num_sampled,
            num_classes=w.shape[0],
            partition_strategy='div',
        )
        return losses * weight

    def logits(self, inputs, w, b):
        return tf.nn.bias_add(tf.matmul(inputs, tf.transpose(w)), b)

    # shuffle data
    def sample(self, data):
        choices = np.random.choice(len(data), size=len(data), replace=False)
        return data.iloc[choices]

    # build an RSN of length l
    def build_sub_graph(self, length=15, reuse=False):
        options = self._options
        hidden_size = options.hidden_size
        batch_size = options.batch_size

        seq = tf.placeholder(
            tf.int32, [batch_size, length], name='seq' + str(length))

        e_em, r_em = self._entity_embedding, self._relation_embedding

        # seperately read, and then recover the order
        ent = seq[:, :-1:2]
        rel = seq[:, 1::2]

        ent_em = tf.nn.embedding_lookup(e_em, ent)
        rel_em = tf.nn.embedding_lookup(r_em, rel)

        em_seq = []
        for i in range(length - 1):
            if i % 2 == 0:
                em_seq.append(ent_em[:, i // 2])
            else:
                em_seq.append(rel_em[:, i // 2])

        # seperately bn
        with tf.variable_scope('input_bn'):
            if not reuse:
                bn_em_seq = [tf.reshape(self.bn(em_seq[i], reuse=(
                        i is not 0)), [-1, 1, hidden_size]) for i in range(length - 1)]
            else:
                bn_em_seq = [tf.reshape(
                    self.bn(em_seq[i], reuse=True), [-1, 1, hidden_size]) for i in range(length - 1)]

        bn_em_seq = tf.concat(bn_em_seq, axis=1)

        ent_bn_em = bn_em_seq[:, ::2]

        with tf.variable_scope('rnn', reuse=reuse):

            cell = self.lstm_cell(True, options.keep_prob, options.num_layers)

            outputs, state = tf.nn.dynamic_rnn(cell, bn_em_seq, dtype=tf.float32)

        # with tf.variable_scope('transformer', reuse=reuse):
        #     outputs = transformer_model(input_tensor=bn_em_seq,
        #                                 hidden_size=hidden_size,
        #                                 intermediate_size=hidden_size*4,
        #                                 num_attention_heads=8)

        rel_outputs = outputs[:, 1::2, :]
        outputs = [outputs[:, i, :] for i in range(length - 1)]

        ent_outputs = outputs[::2]

        # RSN
        res_rel_outputs = tf.contrib.layers.fully_connected(rel_outputs, hidden_size, biases_initializer=None,
                                                            activation_fn=None) + \
                          tf.contrib.layers.fully_connected(
                              ent_bn_em, hidden_size, biases_initializer=None, activation_fn=None)

        # recover the order
        res_rel_outputs = [res_rel_outputs[:, i, :] for i in range((length - 1) // 2)]
        outputs = []
        for i in range(length - 1):
            if i % 2 == 0:
                outputs.append(ent_outputs[i // 2])
            else:
                outputs.append(res_rel_outputs[i // 2])

        # output bn
        with tf.variable_scope('output_bn'):
            if reuse:
                bn_outputs = [tf.reshape(
                    self.bn(outputs[i], reuse=True), [-1, 1, hidden_size]) for i in range(length - 1)]
            else:
                bn_outputs = [tf.reshape(self.bn(outputs[i], reuse=(
                        i is not 0)), [-1, 1, hidden_size]) for i in range(length - 1)]

        def cal_loss(bn_outputs, seq):
            losses = []

            masks = np.random.choice([0., 1.0], size=batch_size, p=[0.5, 0.5])
            weight = tf.random_shuffle(tf.cast(masks, tf.float32))
            for i, output in enumerate(bn_outputs):
                if i % 2 == 0:
                    losses.append(self.sampled_loss(
                        output, seq[:, i + 1], self._rel_w, self._rel_b, weight=weight, is_entity=i))
                else:
                    losses.append(self.sampled_loss(
                        output, seq[:, i + 1], self._ent_w, self._ent_b, weight=weight, is_entity=i))
            losses = tf.stack(losses, axis=1)
            return losses

        seq_loss = cal_loss(bn_outputs, seq)

        losses = tf.reduce_sum(seq_loss) / batch_size

        return losses, seq

    # build the main graph
    def _define_embed_graph(self):
        options = self._options

        loss, seq = self.build_sub_graph(length=options.max_length, reuse=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 2.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = self._optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step()
            )

        self._seq, self._loss, self._train_op = seq, loss, train_op

    # training procedure
    def seq_train(self, data, choices=None, epoch=None):
        opts = self._options

        choices = np.random.choice(len(data), size=len(data), replace=True)
        batch_size = opts.batch_size

        num_batch = len(data) // batch_size

        fetches = {
            'loss': self._loss,
            'train_op': self._train_op
        }

        losses = 0
        for i in range(num_batch):
            one_batch_choices = choices[i * batch_size: (i + 1) * batch_size]
            one_batch_data = data.iloc[one_batch_choices]

            feed_dict = {}
            seq = one_batch_data.values[:, :opts.max_length]
            feed_dict[self._seq] = seq

            vals = self.session.run(fetches, feed_dict)

            del one_batch_data

            loss = vals['loss']
            losses += loss
        self._last_mean_loss = losses / num_batch

        return self._last_mean_loss

    # bootstrapping for RSN
    def co_teaching(self):
        ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, ref_ent1)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, ref_ent2)
        embeds1 = embeds1.eval(session=self.session)
        embeds2 = embeds2.eval(session=self.session)
        sim_mat = sim(embeds1, embeds2,
                      metric=self.args.eval_metric,
                      normalize=self.args.eval_norm,
                      csls_k=self.args.csls)
        sim_mat = self.sim_norm(sim_mat)
        min_val = np.min(sim_mat)
        max_val = np.max(sim_mat)
        avg_val = np.mean(sim_mat)
        sim_th = max(self.args.sim_th, (max_val + avg_val) / 2)
        print("min, max, avg sim and sim_th:", min_val, max_val, avg_val, sim_th)

        sim_mat1 = sim_mat
        sim_mat2 = sim_mat

        # aligner1
        print("============= bootstrapping via aligner1 (graph matching) ============= ")
        new_align1, _, _ = bootstrapping(sim_mat1, ref_ent1, ref_ent2, self.labeled_align, sim_th, self.args.k)
        # aligner2
        print("============= bootstrapping via aligner2 (stable matching) ============= ")
        new_align2, _, _ = bootstrapping_stable_matching(sim_mat2, ref_ent1, ref_ent2, self.labeled_align, sim_th,
                                                         self.args.k)

        agreement_align = new_align1 & new_align2
        print("============= agreement alignment of aligner1 and aligner2 ============= ")
        check_new_alignment(agreement_align, context="agreement_alignment")

        left_ents1_aligner1 = set([i for i in range(len(ref_ent1))]) - set([i for i, j in new_align1])
        left_ents2_aligner1 = set([i for i in range(len(ref_ent2))]) - set([j for i, j in new_align1])
        left_ents1_aligner2 = set([i for i in range(len(ref_ent1))]) - set([i for i, j in new_align2])
        left_ents2_aligner2 = set([i for i in range(len(ref_ent2))]) - set([j for i, j in new_align2])

        left_ents1 = set([i for i in range(len(ref_ent1))]) - set([i for i, j in agreement_align])
        left_ents2 = set([i for i in range(len(ref_ent2))]) - set([j for i, j in agreement_align])

        ensemble_sim_mat = self.sim_norm(sim_mat1 + sim_mat2)
        print("============= re stable alignment  ============= ")
        re_matching = re_stable_matching(left_ents1, left_ents2, ensemble_sim_mat, sim_th + 0.1, self.args.k)

        new_alignment = agreement_align | re_matching
        self.labeled_align = add_new_alignment(self.labeled_align, new_alignment, ensemble_sim_mat)
        check_new_alignment(self.labeled_align, context="final new alignment")

        self.new_sup_links = [(ref_ent1[i], ref_ent2[j]) for i, j in self.labeled_align]

    @staticmethod
    def sim_norm(csls_sim_mat):
        min_val = np.min(csls_sim_mat)
        max_val = np.max(csls_sim_mat)
        val_range = max_val - min_val
        return (csls_sim_mat - min_val) / val_range

    def self_training(self):
        ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, ref_ent1)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, ref_ent2)
        embeds1 = embeds1.eval(session=self.session)
        embeds2 = embeds2.eval(session=self.session)
        sim_mat = sim(embeds1, embeds2,
                      metric=self.args.eval_metric,
                      normalize=self.args.eval_norm,
                      csls_k=self.args.csls)
        sim_mat = self.sim_norm(sim_mat)
        min_val = np.min(sim_mat)
        max_val = np.max(sim_mat)
        avg_val = np.mean(sim_mat)
        sim_th = max(self.args.sim_th, (max_val + avg_val) / 2)
        print("min, max, avg sim and sim_th:", min_val, max_val, avg_val, sim_th)
        # self.labeled_align, new_sup_ent1, new_sup_ent2 = bootstrapping(sim_mat, ref_ent1, ref_ent2,
        #                                                                self.labeled_align, sim_th,
        #                                                                self.args.k)
        self.labeled_align, new_sup_ent1, new_sup_ent2 = bootstrapping_stable_matching(sim_mat, ref_ent1, ref_ent2,
                                                                                       self.labeled_align, sim_th,
                                                                                       self.args.k)
        self.new_sup_links = [(new_sup_ent1[i], new_sup_ent2[i]) for i in range(len(new_sup_ent1))]

    def csls_norm(self, csls_sim_mat):
        min_val = np.min(csls_sim_mat)
        max_val = np.max(csls_sim_mat)
        val_range = max_val - min_val
        return (csls_sim_mat - min_val) / val_range

    def eval_ref_sim_mat(self):
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent1)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent2)
        embeds1 = embeds1.eval(session=self.session)
        embeds2 = embeds2.eval(session=self.session)
        sim_mat = sim(embeds1, embeds2,
                      metric=self.args.eval_metric,
                      normalize=self.args.eval_norm,
                      csls_k=self.args.csls)
        sim_mat = self.csls_norm(sim_mat)
        return sim_mat

    def bootstrapping(self):
        ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, ref_ent1)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, ref_ent2)
        embeds1 = embeds1.eval(session=self.session)
        embeds2 = embeds2.eval(session=self.session)
        sim_mat = sim(embeds1, embeds2,
                      metric=self.args.eval_metric,
                      normalize=self.args.eval_norm,
                      csls_k=self.args.csls)
        sim_mat = self.csls_norm(sim_mat)
        self.labeled_align, new_sup_ent1, new_sup_ent2 = bootstrapping(sim_mat, ref_ent1, ref_ent2,
                                                                       self.labeled_align, self.args.sim_th,
                                                                       self.args.k)
        self.new_sup_links = [(new_sup_ent1[i], new_sup_ent2[i]) for i in range(len(new_sup_ent1))]

    def compute_pos_loss(self, pos_links):
        index1 = pos_links[:, 0]
        index2 = pos_links[:, 1]
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, tf.cast(index1, tf.int32))
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, tf.cast(index2, tf.int32))
        embeds1 = tf.nn.l2_normalize(embeds1, 1)
        embeds2 = tf.nn.l2_normalize(embeds2, 1)
        pos_loss = tf.reduce_sum(tf.reduce_sum(tf.square(embeds1 - embeds2), 1))
        return pos_loss

    def _generate_semi_graph(self):
        self.new_pos_links = tf.placeholder(tf.int32, shape=[None, 2], name="semi_align")
        self.semi_loss = self.compute_pos_loss(self.new_pos_links)
        self.semi_optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.semi_loss)

    def semi_train(self):
        # if len(self.new_sup_links) > 0:
        #     fetches = {"loss": self.semi_loss, "optimizer": self.semi_optimizer}
        #     batch_size = min(len(self.new_sup_links), self.args.batch_size)
        #     steps = len(self.new_sup_links) // batch_size + 1
        #     loss = 0.0
        #     for i in range(steps):
        #         batch_links = random.sample(self.new_sup_links, batch_size)
        #         feed_dict = {self.new_pos_links: batch_links}
        #         vals = self.session.run(fetches=fetches, feed_dict=feed_dict)
        #         loss += vals['loss'] / len(batch_links)
        #     print('bp_loss: %.3f' % (loss / steps))
        if len(self.new_sup_links) > 0:
            fetches = {"loss": self.semi_loss, "optimizer": self.semi_optimizer}
            feed_dict = {self.new_pos_links: self.new_sup_links}
            vals = self.session.run(fetches=fetches, feed_dict=feed_dict)
            loss = vals['loss'] / len(self.new_sup_links)
            print('bp_loss: %.3f' % loss)

    def launch_rsn4ea_training_k_repo(self, iter, iter_nums, train_data):
        for i in range(1, iter_nums + 1):
            time_i = time.time()
            epoch = (iter - 1) * iter_nums + i
            last_mean_loss = self.seq_train(train_data)
            print('epoch %i, avg. batch_loss: %f,  cost time: %.4f s' % (epoch, last_mean_loss, time.time() - time_i))

    def run(self):
        t = time.time()
        train_data = self._train_data
        for i in range(1, self.args.max_epoch + 1):
            time_i = time.time()
            last_mean_loss = self.seq_train(train_data)
            print('epoch %i, avg. batch_loss: %f,  cost time: %.4f s' % (i, last_mean_loss, time.time() - time_i))
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)[0]
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop and i > self.args.min_epoch:
                    break
                if self.args.sim_th > 0.0:
                    self.bootstrapping()
                    self.semi_train()
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
