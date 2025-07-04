import h5py
import numpy as np
import os
import torch

import matplotlib.pyplot as plt


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "./results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc


def plotting_trial_result(conf):
    file_path = conf.goal + "_result.h5"
    result_data = {}
    with h5py.File(file_path, 'r') as hf:
        result_data['acc'] = np.array(hf.get('rs_test_acc'))
        result_data['loss'] = np.array(hf.get('rs_train_loss'))

    for key, value in result_data.items():
        plt.figure(figsize=(10, 5))
        plt.plot(value, marker='o', linestyle='-', color='b', label=key)
        plt.title(conf.goal + '_' + key)
        plt.xlabel('Communication round')
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        result_name = conf.goal + '_' + key
        plt.savefig(result_name + '.png')

def calc_proto_distance(proto, device='cuda'):
    tensor_proto = torch.stack(list(proto.values()), dim=0)
    proto_scale = torch.mean(torch.norm(tensor_proto, dim=1, keepdim=True))
    tensor_proto = tensor_proto / torch.norm(tensor_proto, dim=1, keepdim=True)

    diff = tensor_proto.unsqueeze(1) - tensor_proto.unsqueeze(0)

    distances_squared = torch.sum(diff * diff, dim=-1)
    distances_squared = distances_squared + torch.eye(tensor_proto.size()[0], device=device) * 1e-6

    distances = torch.sqrt(distances_squared)
    eval_dist = torch.where(torch.eye(len(tensor_proto), device=device).bool(),
                            torch.tensor(float('inf'), device=device), distances)
    min_dist = torch.min(eval_dist[~torch.eye(tensor_proto.size()[0], dtype=bool)])
    max_dist = torch.max(eval_dist[~torch.eye(tensor_proto.size()[0], dtype=bool)])
    avg_dist = torch.mean(eval_dist[~torch.eye(tensor_proto.size()[0], dtype=bool)])

    print(f"Minimum distance = {min_dist:.4f}")
    print(f"Maximum distance = {max_dist:.4f}")
    print(f"Average distance = {avg_dist:.4f}")
    print(f"Average scale = {proto_scale:.4f}")