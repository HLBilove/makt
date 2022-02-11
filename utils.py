# Code reused from https://github.com/arghosh/AKT

import os
import torch
from makt import MAKT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass

def get_file_name_identifier(params):
    words = params.model.split('_')
    model_type = words[0]
    if model_type in {'makt', 'akt'}:
        file_name = [['_b', params.batch_size], ['_nb', params.n_block], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_do', params.dropout], ['_dm', params.d_model], ['_ts', params.train_set], ['_kq', params.kq_same], ['_l2', params.l2]]
    return file_name


def model_id_type(model_name):
    words = model_name.split('_')
    return words[0]

def load_model(params):
    words = params.model.split('_')
    model_type = words[0]

    if model_type in {'makt'}:
        model = MAKT(n_skill=params.n_skill, n_exercise=params.n_exercise, n_if=params.n_if, n_a1=params.n_a1, n_a2=params.n_a2,
                        dataset=params.dataset, n_blocks=params.n_block, d_model=params.d_model, dropout=params.dropout,
                        kq_same=params.kq_same, model_type=model_type, l2=params.l2).to(device)
    else:
        model = None
    return model