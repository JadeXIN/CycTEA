import gc
import math
import multiprocessing as mp
import random
import time
import numpy as np
import tensorflow as tf
import sys

from openea.modules.finding.evaluation import early_stop
import openea.modules.train.batch as bat
from openea.approaches.aligne import AlignE
from openea.modules.utils.util import task_divide
from openea.modules.bootstrapping.alignment_finder import find_potential_alignment_mwgm, check_new_alignment, \
    filter_sim_mat, search_nearest_k
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.load.kg import KG
from openea.modules.utils.util import load_session
# from sklearn.externals import joblib
import joblib
from openea.modules.finding.similarity import sim
# from openea.approaches.rlea import cluster_bootstrapping
from openea.models.basic_model import BasicModel
from openea.approaches.bootea import BootEA, update_labeled_alignment_x, update_labeled_alignment_y
from openea.approaches.alinet import AliNet
from openea.approaches.rsn4ea import RSN4EA
from openea.approaches.attre import AttrE
from openea.approaches.bootea import bootstrapping
from openea.modules.args.args_hander import check_args, load_args
from openea.modules.finding.evaluation import valid, test, early_stop
import openea.modules.load.read as rd
import multiprocessing
from openea.modules.finding.alignment import calculate_rank

# tf.compat.v1.random.set_random_seed(1234)


def galeshapley(suitor_pref_dict, reviewer_pref_dict, max_iteration):
    """ The Gale-Shapley algorithm. This is known to provide a unique, stable
    suitor-optimal matching. The algorithm is as follows:

    (1) Assign all suitors and reviewers to be unmatched.

    (2) Take any unmatched suitor, s, and their most preferred reviewer, r.
            - If r is unmatched, match s to r.
            - Else, if r is matched, consider their current partner, r_partner.
                - If r prefers s to r_partner, unmatch r_partner from r and
                  match s to r.
                - Else, leave s unmatched and remove r from their preference
                  list.
    (3) Go to (2) until all suitors are matched, then end.

    Parameters
    ----------
    suitor_pref_dict : dict
        A dictionary with suitors as keys and their respective preference lists
        as values
    reviewer_pref_dict : dict
        A dictionary with reviewers as keys and their respective preference
        lists as values
    max_iteration : int
        An integer as the maximum iterations

    Returns
    -------
    matching : dict
        The suitor-optimal (stable) matching with suitors as keys and the
        reviewer they are matched with as values
    """
    suitors = list(suitor_pref_dict.keys())
    matching = dict()
    rev_matching = dict()

    for i in range(max_iteration):
        if len(suitors) <= 0:
            break
        for s in suitors:
            if len(suitor_pref_dict[s]) == 0:
                continue
            r = suitor_pref_dict[s][0]
            if r not in matching.values():
                matching[s] = r
                rev_matching[r] = s
            else:
                r_partner = rev_matching.get(r)
                if reviewer_pref_dict[r].index(s) < reviewer_pref_dict[r].index(r_partner):
                    del matching[r_partner]
                    matching[s] = r
                    rev_matching[r] = s
                else:
                    suitor_pref_dict[s].remove(r)
        suitors = list(set(suitor_pref_dict.keys()) - set(matching.keys()))
    return matching


def exchange_xy(xy):
    return set([(y, x) for x, y in xy])


def stable_matching(sim_mat, sim_th, k, cut=100):
    t = time.time()

    kg1_candidates, kg2_candidates = dict(), dict()

    potential_aligned_pairs = filter_sim_mat(sim_mat, sim_th)
    if len(potential_aligned_pairs) == 0:
        return None
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim threshold")
    if k <= 0:
        return potential_aligned_pairs
    nearest_k_neighbors1 = search_nearest_k(sim_mat, k)
    nearest_k_neighbors2 = search_nearest_k(sim_mat.T, k)
    nearest_k_neighbors = nearest_k_neighbors1 | exchange_xy(nearest_k_neighbors2)
    potential_aligned_pairs &= nearest_k_neighbors
    if len(potential_aligned_pairs) == 0:
        return None
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim and nearest k")

    i_candidate = dict()
    i_candidate_sim = dict()
    j_candidate = dict()
    j_candidate_sim = dict()

    for i, j in potential_aligned_pairs:
        i_candidate_list = i_candidate.get(i, list())
        i_candidate_list.append(j)
        i_candidate[i] = i_candidate_list

        i_candidate_sim_list = i_candidate_sim.get(i, list())
        i_candidate_sim_list.append(sim_mat[i, j])
        i_candidate_sim[i] = i_candidate_sim_list

        j_candidate_list = j_candidate.get(j, list())
        j_candidate_list.append(i)
        j_candidate[j] = j_candidate_list

        j_candidate_sim_list = j_candidate_sim.get(j, list())
        j_candidate_sim_list.append(sim_mat[i, j])
        j_candidate_sim[j] = j_candidate_sim_list

    prefix1 = "x_"
    prefix2 = "y_"

    for i, i_candidate_list in i_candidate.items():
        i_candidate_sim_list = np.array(i_candidate_sim.get(i))
        sorts = np.argsort(-i_candidate_sim_list)
        i_sorted_candidate_list = np.array(i_candidate_list)[sorts].tolist()
        x_i = prefix1 + str(i)
        kg1_candidates[x_i] = [prefix2 + str(y) for y in i_sorted_candidate_list]
    for j, j_candidate_list in j_candidate.items():
        j_candidate_sim_list = np.array(j_candidate_sim.get(j))
        sorts = np.argsort(-j_candidate_sim_list)
        j_sorted_candidate_list = np.array(j_candidate_list)[sorts].tolist()
        y_j = prefix2 + str(j)
        kg2_candidates[y_j] = [prefix1 + str(x) for x in j_sorted_candidate_list]

    print("generating candidate lists costs time {:.1f} s ".format(time.time() - t),
          len(kg1_candidates),
          len(kg2_candidates))
    t = time.time()
    matching = galeshapley(kg1_candidates, kg2_candidates, cut)
    new_alignment = set()
    n = 0
    for i, j in matching.items():
        x = int(i.split('_')[-1])
        y = int(j.split('_')[-1])
        new_alignment.add((x, y))
        if x == y:
            n += 1
    cost = time.time() - t
    print("stable matching = {}, precision = {:.3f}%, time = {:.3f} s ".format(len(matching),
                                                                               n / len(matching) * 100, cost))
    return new_alignment


def re_stable_matching(left_ents1, left_ents2, sim_mat, sim_th, k, cut=100):
    t = time.time()

    kg1_candidates, kg2_candidates = dict(), dict()

    potential_aligned_pairs = filter_sim_mat(sim_mat, sim_th)
    if len(potential_aligned_pairs) == 0:
        return set()
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim threshold")
    if k <= 0:
        return potential_aligned_pairs
    nearest_k_neighbors1 = search_nearest_k(sim_mat, k)
    nearest_k_neighbors2 = search_nearest_k(sim_mat.T, k)
    nearest_k_neighbors = nearest_k_neighbors1 | exchange_xy(nearest_k_neighbors2)
    potential_aligned_pairs &= nearest_k_neighbors
    if len(potential_aligned_pairs) == 0:
        return set()
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim and nearest k")

    left_aligned_pairs = set()
    for i, j in potential_aligned_pairs:
        if i in left_ents1 and j in left_ents2:
            left_aligned_pairs.add((i, j))
    print("left entities for re-matching:", len(left_aligned_pairs))
    potential_aligned_pairs = left_aligned_pairs

    i_candidate = dict()
    i_candidate_sim = dict()
    j_candidate = dict()
    j_candidate_sim = dict()

    for i, j in potential_aligned_pairs:
        i_candidate_list = i_candidate.get(i, list())
        i_candidate_list.append(j)
        i_candidate[i] = i_candidate_list

        i_candidate_sim_list = i_candidate_sim.get(i, list())
        i_candidate_sim_list.append(sim_mat[i, j])
        i_candidate_sim[i] = i_candidate_sim_list

        j_candidate_list = j_candidate.get(j, list())
        j_candidate_list.append(i)
        j_candidate[j] = j_candidate_list

        j_candidate_sim_list = j_candidate_sim.get(j, list())
        j_candidate_sim_list.append(sim_mat[i, j])
        j_candidate_sim[j] = j_candidate_sim_list

    prefix1 = "x_"
    prefix2 = "y_"

    for i, i_candidate_list in i_candidate.items():
        i_candidate_sim_list = np.array(i_candidate_sim.get(i))
        sorts = np.argsort(-i_candidate_sim_list)
        i_sorted_candidate_list = np.array(i_candidate_list)[sorts].tolist()
        x_i = prefix1 + str(i)
        kg1_candidates[x_i] = [prefix2 + str(y) for y in i_sorted_candidate_list]
    for j, j_candidate_list in j_candidate.items():
        j_candidate_sim_list = np.array(j_candidate_sim.get(j))
        sorts = np.argsort(-j_candidate_sim_list)
        j_sorted_candidate_list = np.array(j_candidate_list)[sorts].tolist()
        y_j = prefix2 + str(j)
        kg2_candidates[y_j] = [prefix1 + str(x) for x in j_sorted_candidate_list]

    print("generating candidate lists costs time {:.1f} s ".format(time.time() - t),
          len(kg1_candidates),
          len(kg2_candidates))
    t = time.time()
    matching = galeshapley(kg1_candidates, kg2_candidates, cut)
    if len(matching) == 0:
        print("re-matching = {}, precision = {:.3f}%, time = {:.3f} s ".format(0, 0, time.time() - t))
        return set()
    else:
        new_alignment = set()
        n = 0
        for i, j in matching.items():
            x = int(i.split('_')[-1])
            y = int(j.split('_')[-1])
            new_alignment.add((x, y))
            if x == y:
                n += 1
        cost = time.time() - t
        print("re-matching = {}, precision = {:.3f}%, time = {:.3f} s ".format(len(matching),
                                                                               n / len(matching) * 100, cost))
        return new_alignment


def bootstrapping_stable_matching(sim_mat, unaligned_entities1, unaligned_entities2, labeled_alignment, sim_th, k):
    curr_labeled_alignment = stable_matching(sim_mat, sim_th, k)
    if curr_labeled_alignment is not None:
        labeled_alignment = update_labeled_alignment_x(labeled_alignment, curr_labeled_alignment, sim_mat)
        labeled_alignment = update_labeled_alignment_y(labeled_alignment, sim_mat)
        del curr_labeled_alignment
    if labeled_alignment is not None:
        newly_aligned_entities1 = [unaligned_entities1[pair[0]] for pair in labeled_alignment]
        newly_aligned_entities2 = [unaligned_entities2[pair[1]] for pair in labeled_alignment]
    else:
        newly_aligned_entities1, newly_aligned_entities2 = None, None
    del sim_mat
    gc.collect()
    return labeled_alignment, newly_aligned_entities1, newly_aligned_entities2


def add_new_alignment(labeled_alignment, curr_labeled_alignment, sim_mat):
    if curr_labeled_alignment is not None:
        labeled_alignment = update_labeled_alignment_x(labeled_alignment, curr_labeled_alignment, sim_mat)
        labeled_alignment = update_labeled_alignment_y(labeled_alignment, sim_mat)
    return labeled_alignment


def check_conflict(labeled_align1, labeled_align2):
    head2tails_map = dict()
    tail2heads_map = dict()
    for idx in labeled_align1 | labeled_align2:
        if idx[0] not in head2tails_map:
            head2tails_map[idx[0]] = set()
        head2tails_map[idx[0]].add(idx[1])
        if idx[1] not in tail2heads_map:
            tail2heads_map[idx[1]] = set()
        tail2heads_map[idx[1]].add(idx[0])
    conflict_num = 0
    conflict_12 = {}
    conflict_21 = {}
    for k in head2tails_map:
        if len(head2tails_map[k]) > 1:
            conflict_num += 1
            conflict_12[k] = head2tails_map[k]
    for k in tail2heads_map:
        if len(tail2heads_map[k]) > 1:
            conflict_num += 1
            conflict_21[k] = tail2heads_map[k]
    print("### Observation ### conflict num %d/%d/%d" % (conflict_num, len(labeled_align1), len(labeled_align2)))
    return head2tails_map, tail2heads_map, conflict_12, conflict_21


def get_rank(sim_mat, num):
    rank = 0
    for i in sim_mat:
        if i < num:
            rank += 1
    return rank


def update_labeled_align(conflict12, labeled_align, label_dict1, label_dict2, ref_mat1):
    conflict12_copy = dict()
    for key in conflict12:
        # print((key, conflict12[key]))
        # print(label_dict1.get(key, -1))
        labeled_align.discard((key, label_dict1[key]))
        if ref_mat1[key, label_dict1[key]] < ref_mat1[key, label_dict2[key]]:
            labeled_align.add((key, label_dict2[key]))
        else:
            conflict12_copy[key] = conflict12[key]
    return conflict12_copy, labeled_align


def conflict_resolution(label_dict1, label_dict2, label_dict3, ref_mat1, ref_mat2, ref_mat3, diff2, conflict12,
                        conflict21, remove=False):
    conflict = 0
    conflict_gold = 0
    real_conflict = 0
    correct = 0
    for key in conflict12:
        diff2.discard((key, label_dict2[key]))
        conflict += 1
        if key == label_dict1[key] or key == label_dict2[key] or key == label_dict3.get(key, None):
            conflict_gold += 1
        if remove:
            continue
        if ref_mat1[key, label_dict1[key]] > ref_mat1[key, label_dict2[key]] and ref_mat2[key, label_dict2[key]] > \
                ref_mat2[key, label_dict1[key]]:
            real_conflict += 1
            ans3 = label_dict3.get(key, None)
            if ans3 is None:
                rank1 = get_rank(ref_mat1[key, :], ref_mat1[key, label_dict1[key]])
                rank2 = get_rank(ref_mat2[key, :], ref_mat2[key, label_dict2[key]])
                if rank1 >= rank2:
                    diff2.add((key, label_dict1[key]))
                else:
                    diff2.add((key, label_dict2[key]))
            elif ans3 == label_dict1[key]:
                diff2.add((key, label_dict1[key]))
            elif ans3 == label_dict2[key]:
                diff2.add((key, label_dict2[key]))
            else:
                rank1 = get_rank(ref_mat1[key, :], ref_mat1[key, label_dict1[key]])
                rank2 = get_rank(ref_mat2[key, :], ref_mat2[key, label_dict2[key]])
                rank3 = get_rank(ref_mat3[key, :], ref_mat3[key, label_dict3[key]])
                if rank1 == max([rank1, rank2, rank3]) and rank1 != rank2 and rank1 != rank3 and rank2 != rank3:
                    diff2.add((key, label_dict1[key]))
                elif rank2 == max([rank1, rank2, rank3]) and rank1 != rank2 and rank1 != rank3 and rank2 != rank3:
                    diff2.add((key, label_dict2[key]))
                elif rank3 == max([rank1, rank2, rank3]) and rank1 != rank2 and rank1 != rank3 and rank2 != rank3:
                    diff2.add((key, label_dict3[key]))
                else:
                    diff2.add((key, label_dict1[key]))
        else:
            diff2.add((key, label_dict2[key]))
        x, y = diff2[-1][0], diff2[-1][1]
        if x == y:
            correct += 1
    print('num of conflict: ' + str(conflict))
    print('num of real conflict: ' + str(real_conflict))
    print('num of ground truth exist in conflict: ' + str(conflict_gold))
    print("right alignment: {}/{}={:.3f}".format(correct, len(conflict12), correct / len(conflict12)))
    diff2_dict = dict(diff2)
    diff2_dict_inv = dict()
    for i, j in diff2_dict.items():
        i_set = diff2_dict_inv.get(j, set())
        i_set.add(i)
        diff2_dict_inv[j] = i_set

    for key in conflict21:
        i_set = conflict21[key]
        if diff2_dict_inv.get(key):
            diff2_dict_inv[key] = diff2_dict_inv[key] | i_set
        else:
            diff2_dict_inv[key] = i_set
    update_diff2 = set()
    for key, i_set in diff2_dict_inv.items():
        if len(i_set) > 1:
            max_i = -1
            max_sim = -10
            for i in i_set:
                if ref_mat2[i, key] > max_sim:
                    max_sim = ref_mat2[i, key]
                    max_i = i
            update_diff2.add((max_i, key))
        else:
            update_diff2.add((i_set.pop(), key))
    return update_diff2


def edit_conflict(label_dict1, label_dict2, ref_mat1, ref_mat2, diff1, diff2, conflict12, conflict21, remove=False):
    n1, n2, n3, n4 = 0, 0, 0, 0
    conflict = 0
    conflict_gold = 0
    for key in conflict12:
        diff1.discard((key, label_dict1[key]))
        diff2.discard((key, label_dict2[key]))
        conflict += 1
        if key == label_dict1[key] or key == label_dict2[key]:
            conflict_gold += 1
        if remove:
            continue
        else:
            if ref_mat1[key, label_dict1[key]] > ref_mat1[key, label_dict2[key]]:
                diff1.add((key, label_dict1[key]))
            else:
                diff1.add((key, label_dict2[key]))
            if ref_mat2[key, label_dict1[key]] > ref_mat2[key, label_dict2[key]]:
                diff2.add((key, label_dict1[key]))
            else:
                diff2.add((key, label_dict2[key]))
    diff1_dict, diff2_dict = dict(diff1), dict(diff2)
    diff1_dict_inv, diff2_dict_inv = dict(), dict()
    for i, j in diff1_dict.items():
        i_set = diff1_dict_inv.get(j, set())
        i_set.add(i)
        diff1_dict_inv[j] = i_set
    for i, j in diff2_dict.items():
        i_set = diff2_dict_inv.get(j, set())
        i_set.add(i)
        diff2_dict_inv[j] = i_set

    for key in conflict21:
        i_set = conflict21[key]
        if diff1_dict_inv.get(key):
            diff1_dict_inv[key] = diff1_dict_inv[key] | i_set
        else:
            diff1_dict_inv[key] = i_set
        if diff2_dict_inv.get(key):
            diff2_dict_inv[key] = diff2_dict_inv[key] | i_set
        else:
            diff2_dict_inv[key] = i_set
    update_diff1 = set()
    update_diff2 = set()
    for key, i_set in diff1_dict_inv.items():
        if len(i_set) > 1:
            max_i = -1
            max_sim = -10
            for i in i_set:
                if ref_mat1[i, key] > max_sim:
                    max_sim = ref_mat1[i, key]
                    max_i = i
            update_diff1.add((max_i, key))
        else:
            update_diff1.add((i_set.pop(), key))
    for key, i_set in diff2_dict_inv.items():
        if len(i_set) > 1:
            max_i = -1
            max_sim = -10
            for i in i_set:
                if ref_mat2[i, key] > max_sim:
                    max_sim = ref_mat2[i, key]
                    max_i = i
            update_diff2.add((max_i, key))
        else:
            update_diff2.add((i_set.pop(), key))
    return update_diff1, update_diff2


def conflict_rematch(sim_mat, sim_th, k, conflict, cut=100):
    t = time.time()
    kg1_candidates, kg2_candidates = dict(), dict()
    print("total conflict numbers: " + str(sim_mat.shape[0]))
    potential_aligned_pairs_index = filter_sim_mat(sim_mat, sim_th)
    potential_aligned_pairs = set([(conflict[i[0]], i[1]) for i in potential_aligned_pairs_index])
    if len(potential_aligned_pairs) == 0:
        return set()
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim threshold")
    if k <= 0:
        return potential_aligned_pairs
    nearest_k_neighbors_index = search_nearest_k(sim_mat, k)
    nearest_k_neighbors = set([(conflict[i[0]], i[1]) for i in nearest_k_neighbors_index])
    potential_aligned_pairs_index &= nearest_k_neighbors_index
    potential_aligned_pairs &= nearest_k_neighbors
    if len(potential_aligned_pairs) == 0:
        return set()
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim and nearest k")
    i_candidate = dict()
    i_candidate_sim = dict()
    j_candidate = dict()
    j_candidate_sim = dict()

    for index, (i, j) in enumerate(potential_aligned_pairs):
        i_candidate_list = i_candidate.get(i, list())
        i_candidate_list.append(j)
        i_candidate[i] = i_candidate_list

        i_candidate_sim_list = i_candidate_sim.get(i, list())
        i_candidate_sim_list.append(sim_mat[list(potential_aligned_pairs_index)[index][0], list(potential_aligned_pairs_index)[index][1]])
        i_candidate_sim[i] = i_candidate_sim_list

        j_candidate_list = j_candidate.get(j, list())
        j_candidate_list.append(i)
        j_candidate[j] = j_candidate_list

        j_candidate_sim_list = j_candidate_sim.get(j, list())
        j_candidate_sim_list.append(sim_mat[list(potential_aligned_pairs_index)[index][0], list(potential_aligned_pairs_index)[index][1]])
        j_candidate_sim[j] = j_candidate_sim_list

    prefix1 = "x_"
    prefix2 = "y_"

    for i, i_candidate_list in i_candidate.items():
        i_candidate_sim_list = np.array(i_candidate_sim.get(i))
        sorts = np.argsort(-i_candidate_sim_list)
        i_sorted_candidate_list = np.array(i_candidate_list)[sorts].tolist()
        x_i = prefix1 + str(i)
        kg1_candidates[x_i] = [prefix2 + str(y) for y in i_sorted_candidate_list]
    for j, j_candidate_list in j_candidate.items():
        j_candidate_sim_list = np.array(j_candidate_sim.get(j))
        sorts = np.argsort(-j_candidate_sim_list)
        j_sorted_candidate_list = np.array(j_candidate_list)[sorts].tolist()
        y_j = prefix2 + str(j)
        kg2_candidates[y_j] = [prefix1 + str(x) for x in j_sorted_candidate_list]

    print("generating candidate lists costs time {:.1f} s ".format(time.time() - t),
          len(kg1_candidates),
          len(kg2_candidates))
    matching = galeshapley(kg1_candidates, kg2_candidates, cut)
    new_alignment = set()
    n = 0
    for i, j in matching.items():
        x = int(i.split('_')[-1])
        y = int(j.split('_')[-1])
        new_alignment.add((x, y))
        if x == y:
            n += 1
    cost = time.time() - t
    print("conflict re-matching = {}, precision = {:.3f}%, time = {:.3f} s ".format(len(matching),
                                                                           n / len(matching) * 100, cost))
    return new_alignment


def inverse_dict(dict):
    dict_inv = dict()
    for i, j in dict.items():
        i_set = dict_inv.get(j, set())
        i_set.add(i)
        dict_inv[j] = i_set
    return dict_inv


def greedy_alignment(sim_mat, top_k, nums_threads, metric, normalize, csls_k, accurate):
    t = time.time()
    num = sim_mat.shape[0]
    if nums_threads > 1:
        hits = [0] * len(top_k)
        mr, mrr = 0, 0
        alignment_rest = set()
        rests = list()
        search_tasks = task_divide(np.array(range(num)), nums_threads)
        pool = multiprocessing.Pool(processes=len(search_tasks))
        for task in search_tasks:
            mat = sim_mat[task, :]
            rests.append(pool.apply_async(calculate_rank, (task, mat, top_k, accurate, num)))
        pool.close()
        pool.join()
        for rest in rests:
            sub_mr, sub_mrr, sub_hits, sub_hits1_rest = rest.get()
            mr += sub_mr
            mrr += sub_mrr
            hits += np.array(sub_hits)
            alignment_rest |= sub_hits1_rest
    else:
        mr, mrr, hits, alignment_rest = calculate_rank(list(range(num)), sim_mat, top_k, accurate, num)
    assert len(alignment_rest) == num
    hits = np.array(hits) / num * 100
    for i in range(len(hits)):
        hits[i] = round(hits[i], 3)
    cost = time.time() - t
    if accurate:
        if csls_k > 0:
            print("accurate results with csls: csls={}, hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(csls_k, top_k, hits, mr, mrr, cost))
        else:
            print("accurate results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(top_k, hits, mr, mrr, cost))
    else:
        if csls_k > 0:
            print("quick results with csls: csls={}, hits@{} = {}%, time = {:.3f} s ".format(csls_k, top_k, hits, cost))
        else:
            print("quick results: hits@{} = {}%, time = {:.3f} s ".format(top_k, hits, cost))
    hits1 = hits[0]  # todo: return hits1
    del sim_mat
    gc.collect()
    return alignment_rest, hits, mr, mrr


def sim_norm(csls_sim_mat):
    min_val = np.min(csls_sim_mat)
    max_val = np.max(csls_sim_mat)
    val_range = max_val - min_val
    return (csls_sim_mat - min_val) / val_range


# "From Diversity-based Prediction to Better Ontology & Schema Matching." WWW2016
def my_mcd_matrix(sim_matrix):
    n, m = sim_matrix.shape[0], sim_matrix.shape[1]
    row_sum = np.sum(sim_matrix, axis=1, keepdims=True)
    col_sum = np.sum(sim_matrix, axis=0, keepdims=True)
    row_mean = np.mean(sim_matrix, axis=1, keepdims=True)
    col_mean = np.mean(sim_matrix, axis=0, keepdims=True)
    mu_mat = (- sim_matrix + row_sum + col_sum) / (n + m - 1)
    mat = (sim_matrix - mu_mat) / ((row_mean + col_mean) / 2)
    return sim_norm(mat)


class Cycle(BasicModel):
    def __init__(self):
        super().__init__()
        self.loss = 0
        self.output = None
        self.optimizer = None
        self.model_init = None
        self.sess = None
        self.feeddict = None
        self.bootea = None
        self.alinet = None
        self.model3 = None
        self.ref_ent1 = None
        self.ref_ent2 = None

    def init(self):
        if '100K' in self.args.training_data:
            args_bootea = load_args('./args/bootea_args_100K.json')
            args_alinet = load_args('./args/alinet_args_100K.json')
            args_model3 = load_args('./args/' + self.args.model3 + '_args_100K.json')
        else:
            args_bootea = load_args('./args/bootea_args_15K.json')
            args_alinet = load_args('./args/alinet_args_15K.json')
            args_model3 = load_args('./args/' + self.args.model3 + '_args_15K.json')

        self.bootea = BootEA()
        args_bootea.training_data = args_bootea.training_data + sys.argv[2] + '/'
        args_bootea.dataset_division = sys.argv[3]
        self.bootea.set_args(args_bootea)
        self.bootea.set_kgs(self.kgs)
        self.bootea.init()

        self.alinet = AliNet()
        args_alinet.training_data = args_alinet.training_data + sys.argv[2] + '/'
        args_alinet.dataset_division = sys.argv[3]
        self.alinet.set_args(args_alinet)
        self.alinet.set_kgs(self.kgs)
        self.alinet.init()

        if self.args.model3 == 'rsn4ea':
            self.model3 = RSN4EA()
        args_model3.training_data = args_model3.training_data + sys.argv[2] + '/'
        args_model3.dataset_division = sys.argv[3]
        self.model3.set_args(args_model3)
        self.model3.set_kgs(self.kgs)
        self.model3.init()

        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2


    def test(self, save=False):
        if self.args.ensemble:
            self.bootea.test(save)
            self.alinet.test(save)
            self.model3.test(save)
            print("========== ensemble results==========")
            self.ensemble_test(save)
        else:
            self.bootea.test(save)
            self.alinet.test(save)
            self.model3.test(save)

    def ensemble_valid(self, stop_metric, weights):
        model1_embeds1, model1_embeds2, model1_mapping = self.bootea._eval_valid_embeddings()
        model2_embeds1, model2_embeds2, model2_mapping = self.alinet._eval_valid_embeddings()
        model3_embeds1, model3_embeds2, model3_mapping = self.model3._eval_valid_embeddings()
        sim_mat1 = sim(model1_embeds1, model1_embeds2, metric=self.bootea.args.eval_metric, normalize=self.bootea.args.
                       eval_metric, csls_k=0)
        sim_mat2 = sim(model2_embeds1, model2_embeds2, metric=self.alinet.args.eval_metric, normalize=self.alinet.args.
                       eval_metric, csls_k=0)
        sim_mat3 = sim(model3_embeds1, model3_embeds2, metric=self.model3.args.eval_metric, normalize=self.model3.args.
                       eval_metric, csls_k=0)
        sim_mat = weights[0] * sim_mat1 + weights[1] * sim_mat2 + weights[2] * sim_mat3
        _, hits1_12, mr_12, mrr_12 = greedy_alignment(sim_mat, self.args.top_k, self.args.test_threads_num,
                                                      metric=self.args.eval_metric, normalize=self.args.eval_norm,
                                                      csls_k=0, accurate=False)

        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def ensemble_test(self, save=True):
        model1_embeds1, model1_embeds2, model1_mapping = self.bootea._eval_test_embeddings()
        model2_embeds1, model2_embeds2, model2_mapping = self.alinet._eval_test_embeddings()
        model3_embeds1, model3_embeds2, model3_mapping = self.model3._eval_test_embeddings()
        sim_mat1 = sim(model1_embeds1, model1_embeds2, metric=self.bootea.args.eval_metric, normalize=self.bootea.args.
                       eval_metric, csls_k=0)
        sim_mat2 = sim(model2_embeds1, model2_embeds2, metric=self.alinet.args.eval_metric, normalize=self.alinet.args.
                       eval_metric, csls_k=0)
        sim_mat3 = sim(model3_embeds1, model3_embeds2, metric=self.model3.args.eval_metric, normalize=self.model3.args.
                       eval_metric, csls_k=0)
        sim_mat = self.w1 * sim_mat1 + self.w2 * sim_mat2 + self.w3 * sim_mat3
        rest_12, _, _, _ = greedy_alignment(sim_mat, self.args.top_k, self.args.test_threads_num,
                                                      metric=self.args.eval_metric, normalize=self.args.eval_norm,
                                                      csls_k=0, accurate=True)
        rest_12, _, _, _ = greedy_alignment(sim_mat, self.args.top_k, self.args.test_threads_num,
                                                      metric=self.args.eval_metric, normalize=self.args.eval_norm,
                                                      csls_k=self.args.csls, accurate=True)

        if save:
            rd.save_results(self.out_folder, rest_12)
        pass

    def save(self):
        pass

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.bootea.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.bootea.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        alinet_neighbors1, alinet_neighbors2 = None, None
        labeled_align = set()
        labeled_align_1 = set()
        sub_num = self.args.sub_epoch
        iter_nums = self.args.max_epoch // sub_num
        alinet_steps = len(self.alinet.sup_ent2) // self.alinet.args.batch_size
        if alinet_steps == 0:
            alinet_steps = 1
        flag1, flag2 = 0, 0
        flag11, flag22 = 0, 0
        cycle_flag1, cycle_flag2 = 0, 0
        if self.args.model3 == 'rsn4ea':
            train_data = self.model3._train_data

        ground_pair = self.kgs.test_links + self.kgs.valid_links
        ground_dict = dict(ground_pair)
        jaccard_list = []
        seed_performance = {'boot1':[], 'boot2': [], 'alinet1':[], 'alinet2':[], 'model3_1':[], 'model3_2':[]}

        # define function of checking the accuracy of new alignment seeds
        def check_correctness(ent1_list, ent2_list, context="check alignment", all_pos=False):
            count = 0
            for idx in range(len(ent1_list)):
                ent1 = ent1_list[idx]
                ent2 = ent2_list[idx]
                if ground_dict.get(ent1) == ent2:
                    count += 1
            print("{}, right alignment: {}/{}={:.3f}".format(context, count, len(ent1_list), count / len(ent1_list)))
            if all_pos:
                precision = count / len(ent1_list)
                recall = count / all_pos
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                print("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))
                return precision, recall, f1

        # for i in range(1, iter_nums + 1):
        for i in range(1, 50 + 1):
            print("\niteration", i)
            # bootea training
            self.bootea.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                              neighbors2)
            # alinet training
            self.alinet.launch_alinet_training_k_repo(i, sub_num, alinet_steps, alinet_neighbors1, alinet_neighbors2)

            # rsn4ea training
            if self.args.model3 == 'rsn4ea':
                if i == 1:
                    self.model3.launch_rsn4ea_training_k_repo(i, 3 * sub_num, train_data)  # todo
                else:
                    self.model3.launch_rsn4ea_training_k_repo(i, 1, train_data)

            if i * sub_num >= self.args.start_valid:
                if self.args.ensemble:
                    flag = self.bootea.valid(self.bootea.args.stop_metric)[0]
                    self.bootea.flag1, self.bootea.flag2, self.bootea.early_stop = early_stop(self.bootea.flag1,
                                                                                              self.bootea.flag2, flag)
                    alinet_flag = self.alinet.valid(self.alinet.args.stop_metric)[0]
                    flag1, flag2, is_stop = early_stop(flag1, flag2, alinet_flag)

                    model3_flag = self.model3.valid(self.model3.args.stop_metric)[0]
                    flag11, flag22, is_stop_model3 = early_stop(flag11, flag22, model3_flag)

                    norm = flag + alinet_flag + model3_flag
                    self.w1, self.w2, self.w3 = flag/norm, alinet_flag/norm, model3_flag/norm
                    cycle_flag = self.ensemble_valid(self.args.stop_metric, [self.w1, self.w2, self.w3])[0]
                    cycle_flag1, cycle_flag2, cycle_is_stop = early_stop(cycle_flag1, cycle_flag2, cycle_flag)
                    if cycle_is_stop or i == iter_nums:
                        print("\n == ensemble training stop == \n")
                else:
                    flag = self.bootea.valid(self.bootea.args.stop_metric)[0]
                    self.bootea.flag1, self.bootea.flag2, self.bootea.early_stop = early_stop(self.bootea.flag1,
                                                                                              self.bootea.flag2, flag)
                    alinet_flag = self.alinet.valid(self.alinet.args.stop_metric)[0]
                    flag1, flag2, is_stop = early_stop(flag1, flag2, alinet_flag)

                    model3_flag = self.model3.valid(self.model3.args.stop_metric)[0]
                    flag11, flag22, is_stop_model3 = early_stop(flag11, flag22, model3_flag)

                    if (self.bootea.early_stop and is_stop and is_stop_model3) or i == iter_nums:
                        print("\n == training stop == \n")

            if i * sub_num >= self.alinet.args.start_augment * self.alinet.args.eval_freq:
                ref_mat1 = self.bootea.eval_ref_sim_mat()
                ref_mat1 = sim_norm(ref_mat1)
                ref_mat1 = my_mcd_matrix(ref_mat1)
                min_val1 = np.min(ref_mat1)
                max_val1 = np.max(ref_mat1)
                avg_val1 = np.mean(ref_mat1)
                sim_th1 = min(self.bootea.args.sim_th, (max_val1 + avg_val1)/2)
                labeled_align, entities1, entities2 = bootstrapping_stable_matching(ref_mat1, self.bootea.ref_ent1,
                                                self.bootea.ref_ent2, labeled_align, sim_th1, self.bootea.args.k)
                bootea_seed = list(zip(entities1, entities2))
                ref_mat2 = self.alinet.eval_ref_sim_mat()
                ref_mat2 = sim_norm(ref_mat2)
                ref_mat2 = my_mcd_matrix(ref_mat2)
                min_val2 = np.min(ref_mat2)
                max_val2 = np.max(ref_mat2)
                avg_val2 = np.mean(ref_mat2)
                sim_th2 = min(self.alinet.args.sim_th, (max_val2 + avg_val2) / 2)
                self.alinet.labeled_align, new_sup_ent1, new_sup_ent2 = bootstrapping_stable_matching(ref_mat2,
                    self.alinet.ref_ent1, self.alinet.ref_ent2, self.alinet.labeled_align, sim_th2, self.alinet.args.k)
                self.alinet.new_sup_links = [(new_sup_ent1[i], new_sup_ent2[i]) for i in range(len(new_sup_ent1))]

                ref_mat3 = self.model3.eval_ref_sim_mat()
                ref_mat3 = sim_norm(ref_mat3)
                ref_mat3 = my_mcd_matrix(ref_mat3)
                min_val3 = np.min(ref_mat3)
                max_val3 = np.max(ref_mat3)
                avg_val3 = np.mean(ref_mat3)
                sim_th3 = min(self.model3.args.sim_th, (max_val3 + avg_val3) / 2)
                self.model3.labeled_align, model3_entities1, model3_entities2 = bootstrapping_stable_matching(ref_mat3,
                                                                                              self.model3.ref_ent1,
                                                                                              self.model3.ref_ent2,
                                                                                              self.model3.labeled_align,
                                                                                              sim_th3,
                                                                                              self.model3.args.k)
                self.model3.new_sup_links = [(model3_entities1[i], model3_entities2[i]) for i in range(len(model3_entities1))]

                if len(entities1) > 0:
                    p1, r1, f1 = check_correctness(entities1, entities2, context='seed of AlignE', all_pos=len(self.ref_ent1))
                if len(new_sup_ent1) > 0:
                    p2, r2, f2 = check_correctness(new_sup_ent1, new_sup_ent2, context='seed of AliNet', all_pos=len(self.ref_ent1))
                if len(model3_entities1) > 0:
                    p3, r3, f3 = check_correctness(model3_entities1, model3_entities2, context='seed of model3',
                                                   all_pos=len(self.ref_ent1))

                if len(entities1) > 0 and len(new_sup_ent1) > 0 and len(model3_entities1) > 0:
                    seed_performance['boot1'].append((p1, r1, f1))
                    seed_performance['alinet1'].append((p2, r2, f2))
                    seed_performance['model3_1'].append((p3, r3, f3))

                a = set(bootea_seed) & set(self.alinet.new_sup_links) & set(self.model3.new_sup_links)
                b = set(bootea_seed) | set(self.alinet.new_sup_links) | set(self.model3.new_sup_links)
                jcard = len(a) / len(b)
                print('jcard similarity of new seed of 3 models: ' + str(jcard))
                c = (set(bootea_seed) & set(self.alinet.new_sup_links)) | (set(self.alinet.new_sup_links) &
                                                                           set(self.model3.new_sup_links)) | (
                                set(bootea_seed) & set(self.model3.new_sup_links))
                correct1 = 0
                for ii in a:
                    if ground_dict.get(ii[0]) == ii[1]:
                        correct1 += 1
                if len(a) == 0:
                    print('joint set has correct proportion: ' + str(0.0))
                else:
                    print('joint set has correct proportion: ' + str(correct1 / len(a)))
                correct2 = 0
                for ii in b:
                    if ground_dict.get(ii[0]) == ii[1]:
                        correct2 += 1
                print('union set has correct proportion: ' + str(correct2 / len(b)))
                jaccard_list.append((jcard, correct1, correct2))

                if i * sub_num % 2 == 0 and self.args.exchange:
                    if i * sub_num < self.args.start_valid:
                        flag = self.bootea.valid(self.bootea.args.stop_metric)[0]
                        alinet_flag = self.alinet.valid(self.alinet.args.stop_metric)[0]
                        model3_flag = self.model3.valid(self.model3.args.stop_metric)[0]
                        norm = flag + alinet_flag + model3_flag
                        self.w1, self.w2, self.w3 = flag / norm, alinet_flag / norm, model3_flag / norm

                    # order arrangement
                    jac12 = len(set(bootea_seed) & set(self.alinet.new_sup_links))/len(set(bootea_seed) |
                                                                                       set(self.alinet.new_sup_links))
                    jac23 = len(set(self.alinet.new_sup_links) & set(self.model3.new_sup_links))/len(set(
                        self.alinet.new_sup_links) | set(self.model3.new_sup_links))
                    jac31 = len(set(bootea_seed) & set(self.model3.new_sup_links))/len(set(bootea_seed) |
                                                                                       set(self.model3.new_sup_links))
                    order123 = (jac12 + math.exp(self.w1 - self.w2)) + (jac23 + math.exp(self.w2 - self.w3)) +\
                               (jac31 + math.exp(self.w3 - self.w1))
                    order132 = (jac31 + math.exp(self.w1 - self.w3)) + (jac23 + math.exp(self.w3 - self.w2)) +\
                               (jac12 + math.exp(self.w2 - self.w1))
                    if order123 > order132:
                        order = order123
                        print('======= order in iteration {:.3f}% is 1->2->3 ======'.format(i))
                    else:
                        order = order132
                        print('======= order in iteration {:.3f}% is 1->3->2 ======'.format(i))

                label_dict1, label_dict2, label_dict3 = dict(labeled_align), dict(self.alinet.labeled_align), \
                                                        dict(self.model3.labeled_align)
                # label_dict1_inv, label_dict2_inv, label_dict3_inv = inverse_dict(label_dict1), \
                #                                                 inverse_dict(label_dict2), inverse_dict(label_dict3)

                
                if self.args.exchange:
                    if order == order123:
                        # orer 123
                        bootea_seed1, bootea_seed2 = model3_entities1, model3_entities2
                        self.alinet.new_sup_links = list(zip(entities1, entities2))
                        self.model3.new_sup_links = list(zip(new_sup_ent1, new_sup_ent2))
                    else:
                        # order op1: 132
                        bootea_seed1, bootea_seed2 = new_sup_ent1, new_sup_ent2
                        self.model3.new_sup_links = list(zip(entities1, entities2))
                        self.alinet.new_sup_links = list(zip(model3_entities1, model3_entities2))
                elif self.args.conflict_resolve:
                    head2tails_map, tail2heads_map, conflict12, conflict21 = check_conflict(labeled_align,
                                                                                            self.alinet.labeled_align)
                    aa = set(labeled_align) & set(self.alinet.labeled_align)
                    diff2 = set(labeled_align) - set(aa)
                    diff2 = conflict_resolution(label_dict1, label_dict2, label_dict3, ref_mat1, ref_mat2,
                                                       ref_mat3, diff2, conflict12, conflict21)
                    self.alinet.new_sup_links = [(self.ref_ent1[pair[0]], self.ref_ent2[pair[1]]) for pair in (aa | diff2)]

                    head2tails_map, tail2heads_map, conflict23, conflict32 = check_conflict(self.alinet.labeled_align,
                                                                                            self.model3.labeled_align)
                    aa = set(self.alinet.labeled_align) & set(self.model3.labeled_align)
                    diff3 = set(self.alinet.labeled_align) - set(aa)
                    diff3 = conflict_resolution(label_dict2, label_dict3, label_dict1, ref_mat2, ref_mat3,
                                                ref_mat1, diff3, conflict23, conflict32)
                    self.model3.new_sup_links = [(self.ref_ent1[pair[0]], self.ref_ent2[pair[1]]) for pair in
                                                 (aa | diff3)]

                    head2tails_map, tail2heads_map, conflict31, conflict13 = check_conflict(self.model3.labeled_align,
                                                                                            labeled_align)
                    aa = set(self.model3.labeled_align) & set(labeled_align)
                    diff1 = set(self.model3.labeled_align) - set(aa)
                    diff1 = conflict_resolution(label_dict3, label_dict1, label_dict2, ref_mat3, ref_mat1,
                                                ref_mat2, diff1, conflict31, conflict13)
                    bootea_seed1, bootea_seed2 = zip(*[(self.ref_ent1[pair[0]], self.ref_ent2[pair[1]]) for pair in
                                                 (aa | diff1)])
                elif self.args.rematching:
                    # 1-2
                    agreement_align12 = labeled_align & self.alinet.labeled_align
                    left_ents1 = set([i for i in range(len(self.ref_ent1))]) - set([i for i, j in agreement_align12])
                    left_ents2 = set([i for i in range(len(self.ref_ent2))]) - set([i for i, j in agreement_align12])
                    if self.args.ensemble:
                        ensemble_sim_mat12 = self.sim_norm(ref_mat1 + ref_mat2)
                    else:
                        alpha = flag/(flag + alinet_flag)
                        ensemble_sim_mat12 = self.sim_norm(alpha * ref_mat1 + (1 - alpha) * ref_mat2)
                    print("============= re stable alignment of model 1-2 ============= ")
                    re_matching12 = re_stable_matching(left_ents1, left_ents2, ensemble_sim_mat12, (sim_th1 + sim_th2)/2
                                                       + 0.1, self.args.k)
                    new_alignment12 = agreement_align12 | re_matching12
                    labeled_align = add_new_alignment(labeled_align, new_alignment12,
                                                                  ensemble_sim_mat12)
                    check_new_alignment(labeled_align, context='final new alignment of 1-2')
                    self.alinet.new_sup_links = [(self.ref_ent1[i], self.ref_ent2[j]) for i, j in labeled_align]
                    # 2-3
                    agreement_align23 = self.alinet.labeled_align & self.model3.labeled_align
                    left_ents1 = set([i for i in range(len(self.ref_ent1))]) - set([i for i, j in agreement_align23])
                    left_ents2 = set([i for i in range(len(self.ref_ent2))]) - set([i for i, j in agreement_align23])
                    if self.args.ensemble:
                        ensemble_sim_mat23 = sim_norm(ref_mat2 + ref_mat3)
                    else:
                        alpha = alinet_flag / (alinet_flag + model3_flag)
                        ensemble_sim_mat23 = sim_norm(alpha * ref_mat2 + (1 - alpha) * ref_mat3)
                    print("============= re stable alignment of model 2-3 ============= ")
                    re_matching23 = re_stable_matching(left_ents1, left_ents2, ensemble_sim_mat23,
                                                       (sim_th2 + sim_th3) / 2
                                                       + 0.1, self.args.k)
                    new_alignment23 = agreement_align23 | re_matching23
                    self.alinet.labeled_align = add_new_alignment(self.alinet.labeled_align, new_alignment23,
                                                                  ensemble_sim_mat23)
                    check_new_alignment(self.alinet.labeled_align, context='final new alignment of 2-3')
                    self.model3.new_sup_links = [(self.ref_ent1[i], self.ref_ent2[j]) for i, j in
                                                 self.alinet.labeled_align]
                    # 3-1
                    agreement_align31 = self.model3.labeled_align & labeled_align
                    left_ents1 = set([i for i in range(len(self.ref_ent1))]) - set([i for i, j in agreement_align31])
                    left_ents2 = set([i for i in range(len(self.ref_ent2))]) - set([i for i, j in agreement_align31])
                    if self.args.ensemble:
                        ensemble_sim_mat31 = sim_norm(ref_mat3 + ref_mat1)
                    else:
                        alpha = model3_flag / (model3_flag + flag)
                        ensemble_sim_mat31 = sim_norm(alpha * ref_mat3 + (1 - alpha) * ref_mat1)
                    print("============= re stable alignment of model 3-1 ============= ")
                    re_matching31 = re_stable_matching(left_ents1, left_ents2, ensemble_sim_mat31,
                                                       (sim_th3 + sim_th1) / 2
                                                       + 0.1, self.args.k)
                    new_alignment31 = agreement_align31 | re_matching31
                    self.model3.labeled_align = add_new_alignment(self.model3.labeled_align, new_alignment31, ensemble_sim_mat31)
                    check_new_alignment(self.model3.labeled_align, context='final new alignment of 3-1')
                    bootea_seed1, bootea_seed2 = zip(*[(self.ref_ent1[i], self.ref_ent2[j]) for i, j in self.model3.labeled_align])
                elif self.args.conflict_rematch:
                    # 1-2
                    head2tails_map, tail2heads_map, conflict12, conflict21 = check_conflict(labeled_align,
                                                                                            self.alinet.labeled_align)
                    conflict12, labeled_align = update_labeled_align(conflict12, labeled_align, label_dict1, label_dict2, ref_mat1)
                    left_ents12 = set(conflict12.keys())
                    right_ents12 = set(self.ref_ent2) - set([i[1] for i in list(labeled_align)])
                    ensemble_sim_mat12_whole = sim_norm(ref_mat1 + ref_mat2)
                    print("============= rematching of conflict of model 1-2 ============= ")
                    new_alignment12 = re_stable_matching(left_ents12, right_ents12, ensemble_sim_mat12_whole, (sim_th1 + sim_th2) / 2 + 0.1, self.args.k)
                    labeled_align = add_new_alignment(labeled_align, new_alignment12,
                                                                  ensemble_sim_mat12_whole)
                    check_new_alignment(labeled_align, context='final new alignment of 1-2')
                    self.alinet.new_sup_links = [(self.ref_ent1[i], self.ref_ent2[j]) for i, j in
                                                 labeled_align]
                    # 2-3
                    head2tails_map, tail2heads_map, conflict23, conflict32 = check_conflict(self.alinet.labeled_align,
                                                                                            self.model3.labeled_align)
                    conflict23, self.alinet.labeled_align = update_labeled_align(conflict23, self.alinet.labeled_align,
                                                                                 label_dict2, label_dict3, ref_mat2)
                    left_ents23 = set(conflict23.keys())
                    right_ents23 = set(self.ref_ent2) - set([i[1] for i in list(self.alinet.labeled_align)])
                    ensemble_sim_mat23_whole = sim_norm(ref_mat2 + ref_mat3)
                    print("============= rematching of conflict of model 2-3 ============= ")
                    new_alignment23 = re_stable_matching(left_ents23, right_ents23, ensemble_sim_mat23_whole,
                                                         (sim_th2 + sim_th3) / 2 + 0.1, self.args.k)
                    self.alinet.labeled_align = add_new_alignment(self.alinet.labeled_align, new_alignment23,
                                                      ensemble_sim_mat23_whole)
                    check_new_alignment(self.alinet.labeled_align, context='final new alignment of 2-3')
                    self.model3.new_sup_links = [(self.ref_ent1[i], self.ref_ent2[j]) for i, j in
                                                 self.alinet.labeled_align]
                    # 3-1
                    head2tails_map, tail2heads_map, conflict31, conflict13 = check_conflict(self.model3.labeled_align,
                                                                                            labeled_align)
                    conflict31, self.model3.labeled_align = update_labeled_align(conflict31, self.model3.labeled_align,
                                                                                 label_dict3, label_dict1, ref_mat3)
                    left_ents31 = set(conflict31.keys())
                    right_ents31 = set(self.ref_ent2) - set([i[1] for i in list(self.model3.labeled_align)])
                    ensemble_sim_mat31_whole = sim_norm(ref_mat3 + ref_mat1)
                    print("============= rematching of conflict of model 3-1 ============= ")
                    new_alignment31 = re_stable_matching(left_ents31, right_ents31, ensemble_sim_mat31_whole,
                                                         (sim_th3 + sim_th1) / 2 + 0.1, self.args.k)
                    self.model3.labeled_align = add_new_alignment(self.model3.labeled_align, new_alignment31,
                                                      ensemble_sim_mat31_whole)
                    check_new_alignment(self.model3.labeled_align, context='final new alignment of 3-1')
                    bootea_seed1, bootea_seed2 = zip(*[(self.ref_ent1[i], self.ref_ent2[j]) for i, j in self.model3.labeled_align])

                p1, r1, f1 = check_correctness(bootea_seed1, bootea_seed2, context='seed of AlignE', all_pos=len(self.ref_ent1))
                p2, r2, f2 = check_correctness([i[0] for i in self.alinet.new_sup_links], [i[1] for i in self.alinet.new_sup_links],
                                  context='seed of AliNet', all_pos=len(self.ref_ent1))
                p3, r3, f3 = check_correctness([i[0] for i in self.model3.new_sup_links], [i[1] for i in self.model3.new_sup_links],
                                  context='seed of model3', all_pos=len(self.ref_ent1))
                seed_performance['boot2'].append((p1, r1, f1))
                seed_performance['alinet2'].append((p2, r2, f2))
                seed_performance['model3_2'].append((p3, r3, f3))

                del ref_mat1
                del ref_mat2
                del ref_mat3
                gc.collect()
                # semi-supervised training of 3 models
                self.bootea.train_alignment(self.kgs.kg1, self.kgs.kg2, bootea_seed1, bootea_seed2, 1)
                steps = math.ceil((len(self.alinet.new_sup_links) / self.alinet.args.batch_size))
                fetches = {"loss": self.alinet.semi_loss, "optimizer": self.alinet.semi_optimizer}
                for s in range(steps):
                    feed_dict = {self.alinet.new_pos_links: self.alinet.new_sup_links[s*self.alinet.args.batch_size:
                                              min((s+1)*self.alinet.args.batch_size, len(self.alinet.new_sup_links))]}
                    self.alinet.session.run(fetches=fetches, feed_dict=feed_dict)
                if self.args.model3 == 'rdgcn':
                    steps = math.ceil((len(self.model3.new_sup_links) / self.model3.args.batch_size))
                    fetches = {"loss": self.model3.semi_loss, "optimizer": self.model3.semi_optimizer}
                    for s in range(steps):
                        feed_dict = {self.model3.new_pos_links: self.model3.new_sup_links[s * self.model3.args.batch_size:
                                                min((s + 1) * self.model3.args.batch_size, len(self.model3.new_sup_links))]}
                        self.model3.session.run(fetches=fetches, feed_dict=feed_dict)
                if self.args.model3 == 'rsn4ea':
                    self.model3.semi_train()
                if self.args.model3 == 'attre':
                    self.model3.semi_train()

            # generate neighbours of 3 models
            t1 = time.time()
            assert 0.0 < self.bootea.args.truncated_epsilon < 1.0
            neighbors_num1 = int((1 - self.bootea.args.truncated_epsilon) * self.kgs.kg1.entities_num)
            neighbors_num2 = int((1 - self.bootea.args.truncated_epsilon) * self.kgs.kg2.entities_num)
            if neighbors1 is not None:
                del neighbors1, neighbors2
            gc.collect()
            neighbors1 = bat.generate_neighbours(self.bootea.eval_kg1_useful_ent_embeddings(),
                                                 self.kgs.useful_entities_list1,
                                                 neighbors_num1, self.bootea.args.batch_threads_num)
            neighbors2 = bat.generate_neighbours(self.bootea.eval_kg2_useful_ent_embeddings(),
                                                 self.kgs.useful_entities_list2,
                                                 neighbors_num2, self.bootea.args.batch_threads_num)
            ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
            print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
            if alinet_neighbors1 is not None:
                del alinet_neighbors1, alinet_neighbors2
            gc.collect()
            alinet_neighbors1, alinet_neighbors2 = self.alinet.find_neighbors()
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))







