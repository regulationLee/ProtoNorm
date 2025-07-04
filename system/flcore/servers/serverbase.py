import torch
import os
import math
import numpy as np
import h5py
import csv
import copy
import time
import random
import shutil
from utils.data_utils import read_client_data
from flcore.clients.clientbase import load_item, save_item

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.round_cnt = 0
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.diverge = False
        self.role = 'Server'
        if args.save_folder_name == 'temp':
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/'
        elif 'temp' in args.save_folder_name:
            args.save_folder_name_full = args.save_folder_name
        else:
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        self.save_folder_name = args.save_folder_name_full

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.tmp_rs_test_acc = []
        self.tmp_rs_train_loss = []

        self.Budget = []

        self.rs_min_dist = [0]
        self.rs_max_dist = [0]
        self.rs_avg_dist = [0]

        self.rs_min_ang = [0]
        self.rs_max_ang = [0]
        self.rs_avg_ang = [0]

        self.max_norm_class_wise_dist = None
        self.min_norm_class_wise_dist = None

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate


    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.update_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def save_results(self):
        if (len(self.rs_test_acc)):
            if self.diverge:
                data_to_add = {
                    "exp_name": self.args.goal,
                    "accuracy": "nan",
                    "avg_time": "nan",
                }
                print("\nDiverge")
            else:
                file_path = self.goal + "_result.h5"
                print("File path: " + file_path)

                if self.tmp_rs_test_acc == []:
                    self.tmp_rs_test_acc = self.rs_test_acc

                if self.tmp_rs_train_loss == []:
                    self.tmp_rs_train_loss = self.rs_train_loss

                with h5py.File(file_path, 'w') as hf:
                    hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                    hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

                    hf.create_dataset('logit_test_acc', data=self.tmp_rs_test_acc)
                    hf.create_dataset('logit_train_loss', data=self.tmp_rs_train_loss)


                data_to_add = {
                    "exp_name": self.args.goal,
                    "accuracy": max(self.rs_test_acc),
                    "logit_accuracy": max(self.tmp_rs_test_acc),
                    "avg_time": sum(self.Budget[1:]) / len(self.Budget[1:]),
                }
                print(f"Best accuracy: {data_to_add['accuracy']:.4f}")
                print(f"Best logit accuracy: {data_to_add['logit_accuracy']:.4f}")
                print(f"Average Process time: {data_to_add['avg_time']:.4f}")

            result_file = "[results].csv"
            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(data_to_add.values())
                print(f"\nEXP: {self.args.goal} recorded.")

        if 'temp' in self.save_folder_name:
            try:
                shutil.rmtree(self.save_folder_name)
                print('Temp files deleted.')
            except:
                print('Already deleted.')

    def test_metrics(self):        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
            print(f'Client {c.id}: Loss: {cl*1.0/ns}')

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def eval_metrics(self):
        test_num_samples = []
        train_num_samples = []
        tot_correct = []
        tot_auc = []
        train_losses = []
        client_acc = []
        client_loss = []
        for c in self.clients:
            ct, test_ns, auc = c.test_metrics()
            cl, train_ns = c.train_metrics()
            tot_correct.append(ct * 1.0)

            tot_auc.append(auc * test_ns)
            test_num_samples.append(test_ns)

            train_num_samples.append(train_ns)
            train_losses.append(cl * 1.0)

            client_acc.append(ct * 1.0 / test_ns)
            client_loss.append(cl * 1.0 / train_ns)

        ids = [c.id for c in self.clients]

        print(f'Client Acc Max: {max(client_acc):.4f}, Min: {min(client_acc):.4f}')
        print(f'Client Loss Max: {max(client_loss):.4f}, Min: {min(client_loss):.4f}\n')

        return ids, test_num_samples, tot_correct, tot_auc, train_num_samples, train_losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        self.round_cnt += 1

        stats = self.eval_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats[5]) * 1.0 / sum(stats[4])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Averaged Train Loss: {:.4f}".format(train_loss))

        if math.isnan(train_loss):
            self.diverge = True

        if self.args.visualization:
            if self.round_cnt == 20 or self.round_cnt == self.global_rounds + 1:
                self.plotting_client_layer()

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def plotting_client_layer(self):
        algo = self.dataset + "_" + self.algorithm
        result_dir = "./results/"
        algo = algo + "_" + self.goal

        result_name = self.goal + '_[client_layer_norm].npy'
        result_path = os.path.join(result_dir, result_name)
        concat_weight = torch.zeros([self.args.num_clients, self.args.num_classes])

        for client in self.selected_clients:
            model = load_item(client.role, 'model', client.save_folder_name)
            if hasattr(model.base, 'fc'):
                fc_weight_norm = torch.norm(model.head.weight, dim=1).unsqueeze(0)
            elif hasattr(model.base, 'classifier'):
                fc_weight_norm = torch.norm(model.head[1].weight, dim=1).unsqueeze(0)
            else:
                fc_weight_norm = torch.norm(client.model.fc.weight, dim=1).unsqueeze(0)
            concat_weight[client.id, :] = fc_weight_norm

        numpy_client_weight = concat_weight.detach().cpu().numpy()
        print('save client weight')
        np.save(result_path, numpy_client_weight)

    def calc_proto_distance(self, global_proto, device='cuda'):
        class_wise_dist = np.zeros([self.args.num_classes, self.args.num_classes])
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        for k1 in global_proto.keys():
            for k2 in global_proto.keys():
                if k1 > k2:
                    dis = torch.norm(global_proto[k1] - global_proto[k2], p=2)
                    class_wise_dist[k1, k2] = dis.item()
                    class_wise_dist[k2, k1] = dis.item()
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)

        norm_class_wise_dist = class_wise_dist / np.linalg.norm(class_wise_dist, axis=1, keepdims=True)
        max_norm_class_wise_dist = np.max(norm_class_wise_dist, axis=1)
        self.max_norm_class_wise_dist = max_norm_class_wise_dist
        self.rs_max_dist.append(np.max(max_norm_class_wise_dist))

        np.fill_diagonal(norm_class_wise_dist, 1e9)
        min_norm_class_wise_dist = np.min(norm_class_wise_dist, axis=1)
        self.min_norm_class_wise_dist = min_norm_class_wise_dist
        self.rs_min_dist.append(np.min(min_norm_class_wise_dist))

        self.rs_avg_dist.append(0)

        tensor_proto = torch.stack(list(global_proto.values()), dim=0)
        proto_scale = torch.mean(torch.norm(tensor_proto, dim=1, keepdim=True))

        print(f"Minimum distance = {min_norm_class_wise_dist}")
        print(f"Maximum distance = {max_norm_class_wise_dist}")
        print(f"Average Prototype scale = {proto_scale:.4f}")

    def calculate_weights(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples

        for client in active_clients:
            weight = client.train_samples / tot_samples
            save_item(weight, client.role, 'weight', self.save_folder_name)

    def aggregate_parameters(self):
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()

        active_clients = random.sample(self.selected_clients,
                                       int((1 - self.client_drop_rate) * self.current_num_join_clients))
        for client in active_clients:
            model = load_item(client.role, 'model', self.save_folder_name)
            weight = load_item(client.role, 'weight', self.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), model.parameters()):
                server_param.data += client_param.data.clone() * weight

        save_item(global_model, self.role, 'global_model', self.save_folder_name)

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            global_model = load_item(self.role, 'global_model', self.save_folder_name)
            model = load_item(client.role, 'model', self.save_folder_name)
            start_time = time.time()

            client.set_parameters(global_model, model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)