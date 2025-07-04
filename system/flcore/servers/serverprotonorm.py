import time
import os
import csv
import pickle
import torch
import numpy as np
from flcore.clients.clientprotonorm import clientProtoNorm
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from collections import defaultdict

import matplotlib.pyplot as plt


class ProtoNorm(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.args = args

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProtoNorm)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.num_classes = args.num_classes
        self.early_iter_log = []

    def train(self):
        save_proto_name_list = []

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            print("\nLocal Training.")
            for client in self.selected_clients:
                client.train()

            print("\nPrototype Aggregation.")
            if i == self.global_rounds:
                self.args.thomson_log = True
            else:
                self.args.thomson_log = False

            self.receive_protos(i)
            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        self.save_results()

    def receive_protos(self, round_idx):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            uploaded_protos.append(protos)

        global_protos = proto_aggregation(self.args, uploaded_protos)
        thomson_global_proto = defaultdict(list)

        if self.args.prototype_alignment:
            thomson_global_proto, stop_iter = optimize_proto(
                global_protos,
                conf=self.args,
                n_iterations=self.args.thomson_iteration_number,
                learning_rate=self.args.thomson_learning_rate,
                momentum=self.args.thomson_momentum,
                device=self.args.device,
                verbose=True
            )
            print(f"stop iteration: {stop_iter}")
        else:
            stop_iter = 0
            for i in range(len(global_protos)):
                if self.args.rescale_proto == 'simple':
                    thomson_global_proto[i] = global_protos[i] / torch.norm(global_protos[i])
                    thomson_global_proto[i] = global_protos[i] * self.args.constant_scale_factor
                else:
                    thomson_global_proto = global_protos

        self.early_iter_log.append(stop_iter)

        save_item(thomson_global_proto, self.role, 'global_protos', self.save_folder_name)


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(conf, local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])
    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


def optimize_proto(
        proto, conf, n_iterations=100, learning_rate=0.1, momentum=0.9, device='cuda',verbose=True):
    result_proto = defaultdict(list)
    label_list = list(proto.keys())   
    tensor_proto = torch.stack(list(proto.values()), dim=0)

    global_proto_scale = torch.norm(tensor_proto, dim=1, keepdim=True)

    # initial normalization for Thomson solver
    tensor_proto = tensor_proto / global_proto_scale

    velocity = torch.zeros_like(tensor_proto)
    velocity_log = torch.zeros_like(tensor_proto)
    tensor_norm_log = torch.zeros_like(torch.norm(tensor_proto, dim=1, keepdim=True))
    lr = learning_rate

    same_mag_forces = 0

    # Optimization loop
    for iter in range(n_iterations):
        # Compute pairwise differences
        diff = tensor_proto.unsqueeze(1) - tensor_proto.unsqueeze(0)

        # Compute squared distances
        distances_squared = torch.sum(diff * diff, dim=-1)
        distances_squared = distances_squared + torch.eye(tensor_proto.size()[0], device=device) * 1e-6

        # Compute force magnitudes
        force_magnitudes = 1.0 / (distances_squared + 1e-10)
        force_magnitudes.fill_diagonal_(0)

        # Compute total forces
        forces = torch.sum(force_magnitudes.unsqueeze(-1) * diff, dim=1)

        # Update velocities with momentum
        velocity = momentum * velocity + lr * forces

        # Update positions
        tensor_proto = tensor_proto + velocity
        velocity_log += velocity

        # Project back to unit sphere
        tensor_norm = torch.norm(tensor_proto, dim=1, keepdim=True)
        tensor_norm_log += tensor_norm
        tensor_proto = tensor_proto / tensor_norm

        # Adaptive learning rate
        if iter % 10 == 0:
            lr *= 0.95

        # early stopping #
        if iter == conf.thomson_early_stop:
            prev_iter = iter
            prev_forces_norm = torch.norm(forces, dim=1, keepdim=True)
            prev_forces_mean = torch.mean(prev_forces_norm)
            trimmed_prev_forces = float("{:.4f}".format(prev_forces_mean))

        elif iter > conf.thomson_early_stop:
            curr_iter = iter
            curr_forces_norm = torch.norm(forces, dim=1, keepdim=True)
            curr_forces_mean = torch.mean(curr_forces_norm)
            trimmed_curr_forces = float("{:.4f}".format(curr_forces_mean))

            diff_forces = trimmed_prev_forces - trimmed_curr_forces
            if (diff_forces == 0) & ((curr_iter - prev_iter) == 1):
                same_mag_forces += 1
                prev_iter = curr_iter
                trimmed_prev_forces = trimmed_curr_forces
                if same_mag_forces == 10:
                    print(f"No significant change in forces at iteration #{curr_iter}")
                    break
            else:
                same_mag_forces = 0
                prev_iter = curr_iter
                trimmed_prev_forces = trimmed_curr_forces

        if verbose and iter % 100 == 0:
            with torch.no_grad():
                distances = torch.sqrt(distances_squared)
                eval_dist = torch.where(torch.eye(len(tensor_proto), device=device).bool(), torch.tensor(float('inf'), device=device), distances)
                min_dist = torch.min(eval_dist[~torch.eye(tensor_proto.size()[0], dtype=bool)])
                max_dist = torch.max(eval_dist[~torch.eye(tensor_proto.size()[0], dtype=bool)])
                avg_dist = torch.mean(eval_dist[~torch.eye(tensor_proto.size()[0], dtype=bool)])
                print(f"Iteration {iter}: Minimum distance = {min_dist:.4f}")
                print(f"Iteration {iter}: Maximum distance = {max_dist:.4f}")
                print(f"Iteration {iter}: Average distance = {avg_dist:.4f}")

    for i in range(len(proto)):
       result_proto[label_list[i]] = tensor_proto[i, :]

    return result_proto, iter
