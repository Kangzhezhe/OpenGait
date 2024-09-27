import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os

from time import strftime, localtime

def print_log(message):
    print('[{0}] [INFO]: {1}'.format(strftime('%Y-%m-%d %H:%M:%S', localtime()), message))

def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1) # [n c p]
        y = F.normalize(y, p=2, dim=1) # [n c p]
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y)
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist/num_bin

# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    dividend = acc.shape[1] - 1
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result

def evaluation_gallery(data, dataset, metric='euc'):
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)

    probe_seq_dict = [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']]
    gallery_seq_dict = [['nm-01', 'nm-02', 'nm-03', 'nm-04']]

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict):
        for gallery_seq in gallery_seq_dict:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x, metric)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)
    for i in range(1):
        print_log('===Rank-%d (Include identical-view cases)===' % (i + 1))
        print_log('NM: %.3f,\tBG: %.3f,\tCL: %.3f,\tMEAN: %.3f' % (
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i]),
            np.mean(acc[:, :, :, i])))
    for i in range(1):
        print_log('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
        print_log('NM: %.3f,\tBG: %.3f,\tCL: %.3f,\tMEAN: %.3f' % (
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i]),
            (de_diag(acc[0, :, :, i])+de_diag(acc[1, :, :, i])+de_diag(acc[2, :, :, i]))/3))
    np.set_printoptions(precision=2, floatmode='fixed')
    for i in range(1):
        print_log('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
        print_log('NM: {}'.format(de_diag(acc[0, :, :, i], True)))
        print_log('BG: {}'.format(de_diag(acc[1, :, :, i], True)))
        print_log('CL: {}'.format(de_diag(acc[2, :, :, i], True)))


def evaluation(data, dataset, metric='euc'):
    feature, _, seq_type, label = data
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]
    
    dist = cuda_dist(probe_x, gallery_x, metric)
    
    idx = dist.sort(1)[1].numpy()
    save_path = osp.join(
        "./work/result/result.csv")
    os.makedirs("./work/result/", exist_ok=True)
    with open(save_path, "w") as f:
        f.write("videoID, label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print_log("Your test result is saved to {}: {}".format(os.getcwd(), save_path))


def evaluation_re_ranking(data, dataset, metric='euc'):
    feature, _, seq_type, label = data
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]

    feat = np.concatenate([probe_x, gallery_x])
    dist = cuda_dist(feat, feat, metric).cpu().numpy()
    print_log("Starting Re-ranking")
    re_rank = re_ranking(dist, probe_x.shape[0], k1=6, k2=6, lambda_value=0.3)
    idx = np.argsort(re_rank, axis=1)

    save_path = osp.join(
        "./work/result/result.csv")
    os.makedirs("./work/result/", exist_ok=True)
    with open(save_path, "w") as f:
        f.write("videoID, label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print_log("CASIA-B result saved to {}/{}".format(os.getcwd(), save_path))

def re_ranking(original_dist, query_num, k1, k2, lambda_value):
    # Modified from https://github.com/michuanhaohao/reid-strong-baseline/blob/master/utils/re_ranking.py
    all_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                            :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
        original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist