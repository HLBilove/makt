import numpy as np
import torch
import math
from math import sqrt
from sklearn import metrics
from utils import model_id_type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transpose_data_model = {'makt'}


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False

def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)

def compute_rmse(prediction, target):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))

    rmse =sqrt(sum(squaredError) / len(squaredError))
    return rmse

def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, optimizer, data, label):
    net.train()
    q_data = data[0]
    qa_data = data[1]
    qa1_data = data[2]
    qa2_data = data[3]
    e_data = data[4]
    if_data = data[5]

    model_type = model_id_type(params.model)

    N = int(math.ceil(len(q_data) / params.batch_size))
    q_data = q_data.T
    qa_data = qa_data.T
    qa1_data = qa1_data.T
    qa2_data = qa2_data.T
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]
    qa1_data = qa1_data[:, shuffled_ind]
    qa2_data = qa2_data[:, shuffled_ind]


    e_data = e_data.T
    e_data = e_data[:, shuffled_ind]
    if_data = if_data.T
    if_data = if_data[:, shuffled_ind]

    pred_list = []
    target_list = []
    pred_a1_list = []
    target_a1_list = []
    pred_a2_list = []
    target_a2_list = []

    element_count = 0
    true_el = 0
    for idx in range(N):
        optimizer.zero_grad()
        q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        e_one_seq = e_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        if_one_seq = if_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]

        qa_one_seq = qa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        qa1_one_seq = qa1_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        qa2_one_seq = qa2_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        if model_type in transpose_data_model:
            input_q = np.transpose(q_one_seq[:, :]) 
            input_qa = np.transpose(qa_one_seq[:, :])
            input_qa1 = np.transpose(qa1_one_seq[:, :])
            input_qa2 = np.transpose(qa2_one_seq[:, :])
            target = np.transpose(qa_one_seq[:, :])
            target_a1 = np.transpose(qa1_one_seq[:, :])
            target_a2 = np.transpose(qa2_one_seq[:, :])
            input_e = np.transpose(e_one_seq[:, :])
            input_if = np.transpose(if_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])
            input_qa = (qa_one_seq[:, :])
            input_qa1 = (qa1_one_seq[:, :])
            input_qa2 = (qa2_one_seq[:, :])
            target = (qa_one_seq[:, :])
            target_a1 = (qa1_one_seq[:, :])
            target_a2 = (qa2_one_seq[:, :])
            input_e = (e_one_seq[:, :])
            input_if = (if_one_seq[:, :])
        target = (target - 1) / params.n_skill
        target_1 = np.floor(target)
        target_a1 = (target_a1 - 1) / params.n_skill
        target_2 = np.floor(target_a1)
        target_a2 = (target_a2 - 1) / params.n_skill
        target_3 = np.floor(target_a2)

        el = np.sum(target_1 >= -.9)
        element_count += el

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        input_qa1 = torch.from_numpy(input_qa1).long().to(device)
        input_qa2 = torch.from_numpy(input_qa2).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        target_a1 = torch.from_numpy(target_2).float().to(device)
        target_a2 = torch.from_numpy(target_3).float().to(device)

        input_e = torch.from_numpy(input_e).long().to(device)
        input_if = torch.from_numpy(input_if).long().to(device)
        loss, pred, true_ct, pred_a1, true_a1_ct, pred_a2, true_a2_ct = net(input_q, input_qa, 
                        input_qa1, input_qa2, target, target_a1, target_a2, input_e, input_if)

        pred = pred.detach().cpu().numpy()
        pred_a1 = pred_a1.detach().cpu().numpy()
        pred_a2 = pred_a2.detach().cpu().numpy()
        loss.backward()
        true_el += true_ct.cpu().numpy()

        if params.maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=params.maxgradnorm)

        optimizer.step()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))
        target_a1 = target_2.reshape((-1,))
        target_a2 = target_3.reshape((-1,))

        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

        nopadding_a1_index = np.flatnonzero(target_a1 >= -0.9)
        nopadding_a1_index = nopadding_a1_index.tolist()
        pred_a1_nopadding = pred_a1[nopadding_a1_index]
        target_a1_nopadding = target_a1[nopadding_a1_index]

        pred_a1_list.append(pred_a1_nopadding)
        target_a1_list.append(target_a1_nopadding)

        nopadding_a2_index = np.flatnonzero(target_a2 >= -0.9)
        nopadding_a2_index = nopadding_a2_index.tolist()
        pred_a2_nopadding = pred_a2[nopadding_a2_index]
        target_a2_nopadding = target_a2[nopadding_a2_index]

        pred_a2_list.append(pred_a2_nopadding)
        target_a2_list.append(target_a2_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    all_a1_pred = np.concatenate(pred_a1_list, axis=0)
    all_a1_target = np.concatenate(target_a1_list, axis=0)

    all_a2_pred = np.concatenate(pred_a2_list, axis=0)
    all_a2_target = np.concatenate(target_a2_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    acc = compute_accuracy(all_target, all_pred)
    rmse_a1 = compute_rmse(all_a1_target, all_a1_pred)
    # auc_a1 = compute_auc(all_a1_target, all_a1_pred)
    rmse_a2 = compute_rmse(all_a2_target, all_a2_pred)
    return loss, acc, auc, rmse_a1, rmse_a2


def test(net, params, optimizer, data, label):
    q_data = data[0]
    qa_data = data[1]
    qa1_data = data[2]
    qa2_data = data[3]
    e_data = data[4]
    if_data = data[5]

    model_type = model_id_type(params.model)
    net.eval()

    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T
    qa_data = qa_data.T
    qa1_data = qa1_data.T
    qa2_data = qa2_data.T
    e_data = e_data.T
    if_data = if_data.T

    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []
    pred_a1_list = []
    target_a1_list = []
    pred_a2_list = []
    target_a2_list = []

    count = 0
    true_el = 0
    element_count = 0
    for idx in range(N):
        q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        e_one_seq = e_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        if_one_seq = if_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]

        qa_one_seq = qa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        qa1_one_seq = qa1_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        qa2_one_seq = qa2_data[:, idx * params.batch_size:(idx+1) * params.batch_size]

        if model_type in transpose_data_model:
            input_q = np.transpose(q_one_seq[:, :])
            input_qa = np.transpose(qa_one_seq[:, :])
            input_qa1 = np.transpose(qa1_one_seq[:, :])
            input_qa2 = np.transpose(qa2_one_seq[:, :])
            target = np.transpose(qa_one_seq[:, :])
            target_a1 = np.transpose(qa1_one_seq[:, :])
            target_a2 = np.transpose(qa2_one_seq[:, :])
            input_e = np.transpose(e_one_seq[:, :])
            input_if = np.transpose(if_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])
            input_qa = (qa_one_seq[:, :])
            input_qa1 = (qa1_one_seq[:, :])
            input_qa2 = (qa2_one_seq[:, :])
            target = (qa_one_seq[:, :])
            target_a1 = (qa1_one_seq[:, :])
            target_a2 = (qa2_one_seq[:, :])
            input_e = (e_one_seq[:, :])
            input_if = (if_one_seq[:, :])
        target = (target - 1) / params.n_skill
        target_1 = np.floor(target)
        target_a1 = (target_a1 - 1) / params.n_skill
        target_2 = np.floor(target_a1)
        target_a2 = (target_a2 - 1) / params.n_skill
        target_3 = np.floor(target_a2)

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        input_qa1 = torch.from_numpy(input_qa1).long().to(device)
        input_qa2 = torch.from_numpy(input_qa2).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        target_a1 = torch.from_numpy(target_2).float().to(device)
        target_a2 = torch.from_numpy(target_3).float().to(device)
        input_e = torch.from_numpy(input_e).long().to(device)
        input_if = torch.from_numpy(input_if).long().to(device)

        with torch.no_grad():
            loss, pred, ct, pred_a1, ct_a1, pred_a2, ct_a2 = net(input_q, input_qa, input_qa1,
            input_qa2, target, target_a1, target_a2, input_e, input_if)

        pred = pred.cpu().numpy()
        pred_a1 = pred_a1.cpu().numpy()
        pred_a2 = pred_a2.cpu().numpy()
        true_el += ct.cpu().numpy()
        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            count += real_batch_size
        else:
            count += params.batch_size

        target = target_1.reshape((-1,))
        target_a1 = target_2.reshape((-1,))
        target_a2 = target_3.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

        nopadding_a1_index = np.flatnonzero(target_a1 >= -0.9)
        nopadding_a1_index = nopadding_a1_index.tolist()
        pred_a1_nopadding = pred_a1[nopadding_a1_index]
        target_a1_nopadding = target_a1[nopadding_a1_index]

        pred_a1_list.append(pred_a1_nopadding)
        target_a1_list.append(target_a1_nopadding)

        nopadding_a2_index = np.flatnonzero(target_a2 >= -0.9)
        nopadding_a2_index = nopadding_a2_index.tolist()
        pred_a2_nopadding = pred_a2[nopadding_a2_index]
        target_a2_nopadding = target_a2[nopadding_a2_index]

        pred_a2_list.append(pred_a2_nopadding)
        target_a2_list.append(target_a2_nopadding)

    assert count == seq_num, "Seq not matching"

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    all_a1_pred = np.concatenate(pred_a1_list, axis=0)
    all_a1_target = np.concatenate(target_a1_list, axis=0)

    all_a2_pred = np.concatenate(pred_a2_list, axis=0)
    all_a2_target = np.concatenate(target_a2_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    acc = compute_accuracy(all_target, all_pred)
    rmse_a1 = compute_rmse(all_a1_target, all_a1_pred)
    # auc_a1 = compute_auc(all_a1_target, all_a1_pred)
    rmse_a2 = compute_rmse(all_a2_target, all_a2_pred)
    if label == 'Test':
        return loss, acc, auc, rmse_a1, rmse_a2, all_target, all_pred
    else:
        return loss, acc, auc, rmse_a1, rmse_a2
