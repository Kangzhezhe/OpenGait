import os
import os.path as osp

from .data_set import DataSet, DataSet2


def load_data(dataset_path, resolution, dataset, cache=True):
    
    if cache==True:
        dataset_path = osp.join(dataset_path, 'train')
    else:
        dataset_path = osp.join(dataset_path, 'test')
    seq_dir = list()
    label = list()
    seq_type = list()
    view = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            if cache==False:
                _view = '000'
                _seq_dir = osp.join(label_path, _seq_type)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append(_view)
            else:
                for _view in sorted(list(os.listdir(seq_type_path))):
                    _seq_dir = osp.join(seq_type_path, _view)
                    seqs = os.listdir(_seq_dir)
                    if len(seqs) > 0:
                        seq_dir.append([_seq_dir])
                        label.append(_label)
                        seq_type.append(_seq_type)
                        view.append(_view)
    if cache==False:
        data_source = DataSet(seq_dir, label, seq_type, view, cache, resolution)
    else:
        data_source = DataSet2(seq_dir, label, seq_type, view, cache, resolution)

    return data_source
