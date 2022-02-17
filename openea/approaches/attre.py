import tensorflow as tf
import math
import multiprocessing as mp
import random
import time

from openea.models.trans.transe import TransE
import openea.modules.train.batch as bat
from openea.modules.finding.evaluation import early_stop
from openea.modules.utils.util import task_divide
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.losses import margin_loss
from openea.models.basic_model import BasicModel
from openea.modules.base.losses import get_loss_func
from openea.modules.load.kg import KG
from openea.approaches.bootea import generate_supervised_triples, generate_pos_batch


def formatting_attr_triples(kgs, literal_len):
    """
    Formatting attribute triples from kgs for AttrE.
    :param kgs: modules.load.kgs
    :param literal_len: [optional] Literal truncation length, taking the first literal_len characters.
    :return: attribute_triples_list1_new, attribute_triples_list2_new, char_list size
    """

    def clean_attribute_triples(triples):
        triples_new = []
        for (e, a, v) in triples:
            v = v.split('(')[0].rstrip(' ')
            v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '') \
                .replace('_', ' ').replace('-', ' ').split('"')[0]
            triples_new.append((e, a, v))
        return triples_new

    attribute_triples_list1 = clean_attribute_triples(kgs.kg1.local_attribute_triples_list)
    attribute_triples_list2 = clean_attribute_triples(kgs.kg2.local_attribute_triples_list)

    value_list = list(set([v for (_, _, v) in attribute_triples_list1 + attribute_triples_list2]))
    char_set = set()
    ch_num = {}
    for literal in value_list:
        for ch in literal:
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n

    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            char_set.add(ch_num[i][0])
    char_list = list(char_set)
    char_id_dict = {}
    for i in range(len(char_list)):
        char_id_dict[char_list[i]] = i + 1

    value_char_ids_dict = {}
    for value in value_list:
        char_id_list = [0 for _ in range(literal_len)]
        for i in range(min(len(value), literal_len)):
            if value[i] in char_set:
                char_id_list[i] = char_id_dict[value[i]]
        value_char_ids_dict[value] = char_id_list

    attribute_triples_list1_new, attribute_triples_list2_new = list(), list()
    value_id_char_ids = list()
    value_id_cnt = 0
    for (e_id, a_id, v) in attribute_triples_list1:
        attribute_triples_list1_new.append((e_id, a_id, value_id_cnt))
        value_id_char_ids.append(value_char_ids_dict[v])
        value_id_cnt += 1

    for (e_id, a_id, v) in attribute_triples_list2:
        attribute_triples_list2_new.append((e_id, a_id, value_id_cnt))
        value_id_char_ids.append(value_char_ids_dict[v])
        value_id_cnt += 1
    return attribute_triples_list1_new, attribute_triples_list2_new, value_id_char_ids, len(char_list) + 1


def add_compositional_func(character_vectors):
    value_vector_list = tf.reduce_mean(character_vectors, axis=1)
    value_vector_list = tf.nn.l2_normalize(value_vector_list, 1)
    return value_vector_list


def n_gram_compositional_func(character_vectors, value_lens, batch_size, embed_size):
    pos_c_e_in_lstm = tf.unstack(character_vectors, num=value_lens, axis=1)
    pos_c_e_lstm = calculate_ngram_weight(pos_c_e_in_lstm, batch_size, embed_size)
    return pos_c_e_lstm


def calculate_ngram_weight(unstacked_tensor, batch_size, embed_size):
    stacked_tensor = tf.stack(unstacked_tensor, axis=1)
    stacked_tensor = tf.reverse(stacked_tensor, [1])
    index = tf.constant(len(unstacked_tensor))
    expected_result = tf.zeros([batch_size, embed_size])

    def condition(index, summation):
        return tf.greater(index, 0)

    def body(index, summation):
        precessed = tf.slice(stacked_tensor, [0, index - 1, 0], [-1, -1, -1])
        summand = tf.reduce_mean(precessed, 1)
        return tf.subtract(index, 1), tf.add(summation, summand)

    result = tf.while_loop(condition, body, [index, expected_result])
    return result[1]


def generate_supervised_attr_triples(av_dict1, av_dict2, ents1, ents2):
    assert len(ents1) == len(ents2)
    newly_attr_triples1, newly_attr_triples2 = list(), list()
    for i in range(len(ents1)):
        newly_attr_triples1.extend(generate_newly_attr_triples(ents1[i], ents2[i], av_dict1))
        newly_attr_triples2.extend(generate_newly_attr_triples(ents2[i], ents1[i], av_dict2))
    print("newly triples: {}, {}".format(len(newly_attr_triples1), len(newly_attr_triples2)))
    return newly_attr_triples1, newly_attr_triples2


def generate_newly_attr_triples(ent1, ent2, av_dict1):
    newly_triples = list()
    for a, v in av_dict1.get(ent1, set()):
        newly_triples.append((ent2, a, v))
    return newly_triples


class AttrE(BasicModel):

    def __init__(self):
        super().__init__()
        self.ref_ent1 = None
        self.ref_ent2 = None
        self.new_sup_links = None
        self.semi_loss = None
        self.semi_optimizer = None
        self.labeled_align = set()
        self.labeled_align_1 = set()

    def init(self):
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self.attribute_triples_list1, self.attribute_triples_list2, self.value_id_char_ids, self.char_list_size = \
            formatting_attr_triples(self.kgs, self.args.literal_len)
        self._define_variables()
        self._define_embed_graph()
        self._define_semi_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def _define_variables(self):
        with tf.variable_scope('relation' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds_se',
                                              self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm, dtype=tf.float32)
        with tf.variable_scope('character' + 'embeddings'):
            self.ent_embeds_ce = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds_ce',
                                                 self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
            self.attr_embeds = init_embeddings([self.kgs.attributes_num, self.args.dim], 'attr_embeds',
                                               self.args.init, self.args.attr_l2_norm, dtype=tf.float32)
            self.char_embeds = init_embeddings([self.char_list_size, self.args.dim], 'char_embeds',
                                               self.args.init, self.args.char_l2_norm, dtype=tf.float32)
            self.value_id_char_ids = tf.constant(self.value_id_char_ids)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])

            self.pos_es = tf.placeholder(tf.int32, shape=[None])
            self.pos_as = tf.placeholder(tf.int32, shape=[None])
            self.pos_vs = tf.placeholder(tf.int32, shape=[None])
            self.neg_es = tf.placeholder(tf.int32, shape=[None])
            self.neg_as = tf.placeholder(tf.int32, shape=[None])
            self.neg_vs = tf.placeholder(tf.int32, shape=[None])

            self.joint_ents = tf.placeholder(tf.int32, shape=[None])

        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)

            pes = tf.nn.embedding_lookup(self.ent_embeds_ce, self.pos_es)
            pas = tf.nn.embedding_lookup(self.attr_embeds, self.pos_as)
            pvs = tf.nn.embedding_lookup(self.char_embeds, tf.nn.embedding_lookup(self.value_id_char_ids, self.pos_vs))
            nes = tf.nn.embedding_lookup(self.ent_embeds_ce, self.neg_es)
            nas = tf.nn.embedding_lookup(self.attr_embeds, self.neg_as)
            nvs = tf.nn.embedding_lookup(self.char_embeds, tf.nn.embedding_lookup(self.value_id_char_ids, self.neg_vs))

            pvs = n_gram_compositional_func(pvs, self.args.literal_len, self.args.batch_size, self.args.dim)
            nvs = n_gram_compositional_func(nvs, self.args.literal_len,
                                            self.args.batch_size * self.args.neg_triple_num, self.args.dim)

            ents_se = tf.nn.embedding_lookup(self.ent_embeds, self.joint_ents)
            ents_ce = tf.nn.embedding_lookup(self.ent_embeds_ce, self.joint_ents)

        with tf.name_scope('triple_loss'):
            self.triple_loss = get_loss_func(phs, prs, pts, nhs, nrs, nts, self.args)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

            self.triple_loss_ce = get_loss_func(pes, pas, pvs, nes, nas, nvs, self.args)
            self.triple_optimizer_ce = generate_optimizer(self.triple_loss_ce, self.args.learning_rate,
                                                          opt=self.args.optimizer)

            cos_sim = tf.reduce_sum(tf.multiply(ents_se, ents_ce), 1, keep_dims=True)
            self.joint_loss = tf.reduce_sum(1 - cos_sim)
            self.optimizer_joint = generate_optimizer(self.joint_loss, self.args.learning_rate, opt=self.args.optimizer)

    def _define_semi_graph(self):
        self.new_h = tf.placeholder(tf.int32, shape=[None])
        self.new_r = tf.placeholder(tf.int32, shape=[None])
        self.new_t = tf.placeholder(tf.int32, shape=[None])
        self.new_e = tf.placeholder(tf.int32, shape=[None])
        self.new_a = tf.placeholder(tf.int32, shape=[None])
        self.new_v = tf.placeholder(tf.int32, shape=[None])
        self.joint_ents = tf.placeholder(tf.int32, shape=[None])

        phs = tf.nn.embedding_lookup(self.ent_embeds, self.new_h)
        prs = tf.nn.embedding_lookup(self.rel_embeds, self.new_r)
        pts = tf.nn.embedding_lookup(self.ent_embeds, self.new_t)
        pes = tf.nn.embedding_lookup(self.ent_embeds_ce, self.new_e)
        pas = tf.nn.embedding_lookup(self.attr_embeds, self.new_v)
        pvs = tf.nn.embedding_lookup(self.char_embeds, tf.nn.embedding_lookup(self.value_id_char_ids, self.pos_vs))
        pvs = n_gram_compositional_func(pvs, self.args.literal_len, self.args.batch_size, self.args.dim)
        ents_se = tf.nn.embedding_lookup(self.ent_embeds, self.joint_ents)
        ents_ce = tf.nn.embedding_lookup(self.ent_embeds_ce, self.joint_ents)

        self.alignment_loss = - tf.reduce_sum(tf.log(tf.sigmoid(-tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1))))
        self.alignment_optimizer = generate_optimizer(self.alignment_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)
        self.alignment_ce_loss = - tf.reduce_sum(tf.log(tf.sigmoid(-tf.reduce_sum(tf.pow(pes + pas - pvs, 2), 1))))
        self.alignment_ce_optimizer = generate_optimizer(self.alignment_ce_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)
        cos_sim = tf.reduce_sum(tf.multiply(ents_se, ents_ce), 1, keep_dims=True)
        self.joint_loss = tf.reduce_sum(1 - cos_sim)
        self.optimizer_joint = generate_optimizer(self.joint_loss, self.args.learning_rate, opt=self.args.optimizer)

    def eval_ref_sim_mat(self):
        # refs1_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent1)
        # refs2_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent2)
        # refs1_embeddings = tf.nn.l2_normalize(refs1_embeddings, 1).eval(session=self.session)
        # refs2_embeddings = tf.nn.l2_normalize(refs2_embeddings, 1).eval(session=self.session)
        # return np.matmul(refs1_embeddings, refs2_embeddings.T)
        refs1_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent1)
        refs2_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent2)
        refs1_embeddings = tf.nn.l2_normalize(refs1_embeddings, 1)
        refs2_embeddings = tf.nn.l2_normalize(refs2_embeddings, 1)
        mat_op = tf.matmul(refs1_embeddings, refs2_embeddings, transpose_b=True)
        sim_mat = self.session.run(mat_op)
        return sim_mat

    def launch_triple_training_1epo_ce(self, epoch, triple_steps, steps_tasks, batch_queue):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_attribute_triple_batch_queue,
                       args=(self.attribute_triples_list1, self.attribute_triples_list2,
                             set(self.attribute_triples_list1), set(self.attribute_triples_list2),
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, None, None, self.args.neg_triple_num, True)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.triple_loss_ce, self.triple_optimizer_ce],
                                             feed_dict={self.pos_es: [x[0] for x in batch_pos],
                                                        self.pos_as: [x[1] for x in batch_pos],
                                                        self.pos_vs: [x[2] for x in batch_pos],
                                                        self.neg_es: [x[0] for x in batch_neg],
                                                        self.neg_as: [x[1] for x in batch_neg],
                                                        self.neg_vs: [x[2] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.attribute_triples_list1)
        random.shuffle(self.attribute_triples_list2)
        print(
            'epoch {}, CE, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_joint_training_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.batch_size))
        for i in range(steps):
            batch_ents = list(entities)
            batch_loss, _ = self.session.run(fetches=[self.joint_loss, self.optimizer_joint],
                                             feed_dict={self.joint_ents: batch_ents})
            trained_samples_num += len(batch_ents)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        print('epoch {}, joint learning loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_attre_training_k_repo(self, iter, iter_nums, relation_triple_steps, relation_step_tasks,
                                     relation_batch_queue, attribute_triple_steps, attribute_step_tasks,
                                     attribute_batch_queue, entity_list):
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i

            self.launch_triple_training_1epo(epoch, relation_triple_steps, relation_step_tasks, relation_batch_queue,
                                             None, None)
            self.launch_triple_training_1epo_ce(epoch, attribute_triple_steps, attribute_step_tasks,
                                                attribute_batch_queue)
            self.launch_joint_training_1epo(epoch, entity_list)

    def semi_train(self):
        attre_seed1 = [i[0] for i in self.new_sup_links]
        attre_seed2 = [i[1] for i in self.new_sup_links]
        self.train_alignment(self.kgs.kg1, self.kgs.kg2, attre_seed1, attre_seed2, 1)

    def train_alignment(self, kg1: KG, kg2: KG, entities1, entities2, training_epochs):
        if entities1 is None or len(entities1) == 0:
            return
        newly_tris1, newly_tris2 = generate_supervised_triples(kg1.rt_dict, kg1.hr_dict, kg2.rt_dict, kg2.hr_dict,
                                                               entities1, entities2)
        steps = math.ceil(((len(newly_tris1) + len(newly_tris2)) / self.args.batch_size))
        if steps == 0:
            steps = 1
        for i in range(training_epochs):
            t1 = time.time()
            alignment_loss = 0
            for step in range(steps):
                newly_batch1, newly_batch2 = generate_pos_batch(newly_tris1, newly_tris2, step, self.args.batch_size)
                newly_batch1.extend(newly_batch2)
                alignment_fetches = {"loss": self.alignment_loss, "train_op": self.alignment_optimizer}
                alignment_feed_dict = {self.new_h: [tr[0] for tr in newly_batch1],
                                       self.new_r: [tr[1] for tr in newly_batch1],
                                       self.new_t: [tr[2] for tr in newly_batch1]}
                alignment_vals = self.session.run(fetches=alignment_fetches, feed_dict=alignment_feed_dict)
                alignment_loss += alignment_vals["loss"]
            alignment_loss /= (len(newly_tris1) + len(newly_tris2))
            print("alignment_loss = {:.3f}, time = {:.3f} s".format(alignment_loss, time.time() - t1))

        newly_tris1_ce, newly_tris2_ce = generate_supervised_attr_triples(kg1.av_dict, kg2.av_dict, entities1, entities2)
        steps = math.ceil(((len(newly_tris1_ce) + len(newly_tris2_ce)) / self.args.batch_size))
        if steps == 0:
            steps = 1
        for i in range(training_epochs):
            t1 = time.time()
            alignment_ce_loss = 0
            for step in range(steps):
                newly_attr_batch1, newly_attr_batch2 = generate_pos_batch(newly_tris1_ce, newly_tris2_ce, step, self.args.batch_size)
                newly_attr_batch1.extend(newly_attr_batch2)
                alignment_attr_fetches = {"loss": self.alignment_ce_loss, "train_op": self.alignment_ce_optimizer}
                alignment_attr_feed_dict = {self.new_e: [tr[0] for tr in newly_attr_batch1],
                                            self.new_a: [tr[1] for tr in newly_attr_batch1],
                                            self.new_v: [tr[2] for tr in newly_attr_batch1]}
                alignment_ce_vals = self.session.run(fetches=alignment_attr_fetches, feed_dict=alignment_attr_feed_dict)
                alignment_ce_loss += alignment_ce_vals["loss"]
            alignment_ce_loss /= (len(newly_tris1_ce) + len(newly_tris2_ce))
            print("alignment_ce_loss = {:.3f}, time = {:.3f} s".format(alignment_ce_loss, time.time() - t1))

        entity_list = list(entities1 + entities2)
        self.launch_joint_training_1epo(1, entity_list)

    def run(self):
        t = time.time()
        relation_triples_num = len(self.kgs.kg1.relation_triples_list) + len(self.kgs.kg2.relation_triples_list)
        attribute_triples_num = len(self.attribute_triples_list1) + len(self.attribute_triples_list2)
        relation_triple_steps = int(math.ceil(relation_triples_num / self.args.batch_size))
        attribute_triple_steps = int(math.ceil(attribute_triples_num / self.args.batch_size))
        relation_step_tasks = task_divide(list(range(relation_triple_steps)), self.args.batch_threads_num)
        attribute_step_tasks = task_divide(list(range(attribute_triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        relation_batch_queue = manager.Queue()
        attribute_batch_queue = manager.Queue()
        entity_list = list(self.kgs.kg1.entities_list + self.kgs.kg2.entities_list)
        for i in range(1, self.args.max_epoch + 1):
            self.launch_triple_training_1epo(i, relation_triple_steps, relation_step_tasks, relation_batch_queue, None,
                                             None)
            self.launch_triple_training_1epo_ce(i, attribute_triple_steps, attribute_step_tasks, attribute_batch_queue)
            self.launch_joint_training_1epo(i, entity_list)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)[0]
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
