import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import timeit
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score, auc
from utils import *
import sys
import os
import argparse
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*window+1,stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)
        self.trans = make_model(n_word, N=1, d_model=128, d_ff=512, h=8, dropout=0.1, MAX_LEN=seq_max)
        self.trans_out = nn.Linear(128,10)

        self.dropout = 0.6
        self.weigth = None
        self.compound_attn = nn.ParameterList([nn.Parameter(torch.randn(size=(2 * dim, 1))) for _ in range(layer_output)])
        # self.compound_attn = [nn.Parameter(torch.randn(size=(2 * dim, 1))) for _ in range(layer_output)]
        # for i in range(layer_output):
        #     self.register_parameter('compound_attn_{}'.format(i), self.compound_attn[i])

    def gat(self, xs, x_mask, A, layer):
        x_mask = x_mask.reshape(x_mask.size()[0], x_mask.size()[1], 1)
        for i in range(layer):
            h = torch.relu(self.W_gnn[i](xs))
            h = h*x_mask
            size = h.size()[0]
            N = h.size()[1]
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1, 2 * dim)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(A > 0, e, zero_vec)  # 保证softmax 不为 0
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout)
            h_prime = torch.matmul(attention, h)
            xs = xs+h_prime
        xs = xs*x_mask
        return torch.unsqueeze(torch.mean(xs, 1), 1)

    def attention_cnn(self, x, xs, layer):
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(torch.matmul(h, hs.permute([0, 2, 1])))
        ys = weights.permute([0, 2, 1]) * hs
        self.weight = weights
        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 1), 1)

    def forward(self, inputs):
        fingerprints, fingerprints_mask, adjacency, words, words_mask = inputs
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gat(fingerprint_vectors, fingerprints_mask, adjacency, layer_gnn)

        words_mask = words_mask.unsqueeze(-2)
        word_vectors_trans = self.trans(words, words_mask)  # [batch, length, feature_len]
        word_vectors = self.trans_out(F.relu(word_vectors_trans))  # [batch, length, feature_conv_len]
        protein_vector = self.attention_cnn(compound_vector, word_vectors, layer_cnn)
        cat_vector = torch.cat((compound_vector, protein_vector), 2)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        return torch.squeeze(interaction, 1)

    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            return predicted_interaction

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        total_num = 0
        for idx, batch in enumerate(train_loader):
            batch = tuple(i.to(device) for i in batch)
            loss = self.model(batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_loss = loss.to('cpu').data.numpy()
            num = len(batch)
            total_loss += mean_loss*num
            total_num += num
        return total_loss/total_num

class Tester(object):
    def __init__(self, model):
        self.model = model
    def test(self, test_loader):
        self.model.eval()
        T, Y, S = [], [], []
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                batch = tuple(i.to(device) for i in batch)

                predicted_interaction = self.model(batch, train=False)
                correct_interaction = batch[-1]

                correct_labels = correct_interaction.to('cpu').data.numpy()
                ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
                predicted_labels = list(map(lambda x: np.argmax(x), ys))
                predicted_scores = list(map(lambda x: x[1], ys))

                T.extend(correct_labels.tolist())
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)

        pr, re, _ = precision_recall_curve(T, Y)
        aupr = auc(re, pr)
        F1 = f1_score(T, Y)
        return AUC, precision, recall, aupr, F1

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model, filename)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser("dude main training")
    parser.add_argument("--dataset", type=str, default='dude', help='dataset name')
    parser.add_argument('--radius', type=int, default=2, help='the r-radius')
    parser.add_argument('--ngram', type=int, default=3, help='the n-gram')
    parser.add_argument('--dim', type=int, default=10, help='the dimension of embedding')
    parser.add_argument('--layer_gnn', type=int, default=3, help='the number of gnn')
    parser.add_argument('--window', type=int, default=11, help='the kerner size')
    parser.add_argument('--layer_cnn', type=int, default=3, help='the number of cnn')
    parser.add_argument('--layer_output', type=int, default=3, help='the number for layer of output')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='the decay of learning rate')
    parser.add_argument('--decay_interval', type=int, default=10, help='decay interval')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='the weight decay')
    parser.add_argument('--iteration', type=int, default=100, help='the iteration for training')
    parser.add_argument('--save_name_append', type=str, default='test', help='the append suffix name for training')
    parser.add_argument('--seed', type=int, default=1234, help='training seed')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--n_fold', type=int, default=3, help='n_fold')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET = args.dataset
    seed_num = args.seed
    setup_seed(seed_num)
    radius = args.radius
    ngram = args.ngram
    dim = args.dim
    layer_gnn = args.layer_gnn
    window = args.window
    layer_cnn = args.layer_gnn
    layer_output = args.layer_output
    lr = args.lr
    lr_decay = args.lr_decay
    decay_interval = args.decay_interval
    weight_decay = args.weight_decay
    iteration = args.iteration
    n_fold = args.n_fold
    batch_size = args.batch_size
    graph_max_map = {'drugbank':85, 'human':300, 'celegans':200, 'dude':100}
    seq_max_map = {'drugbank': 1200, 'human':2000, 'celegans':2000, 'dude':820}
    # graph_max_map = {'drugbank':90}
    # seq_max_map = {'drugbank': 1200}
    graph_max = graph_max_map[DATASET]
    seq_max = seq_max_map[DATASET]

    print('Training...')
    for fold_num in range(args.n_fold):
        dir_input = f'../dataset/{DATASET}/fold{fold_num}/'
        fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
        word_dict = load_pickle(dir_input + 'word_dict.pickle')
        n_fingerprint = len(fingerprint_dict)
        n_word = len(word_dict)
        compounds_cpu = np.load(dir_input + 'compounds' + '.npy', allow_pickle=True)
        adjacencies_cpu = np.load(dir_input + 'adjacencies' + '.npy', allow_pickle=True)
        proteins_cpu = np.load(dir_input + 'proteins' + '.npy', allow_pickle=True)
        interactions_cpu = np.load(dir_input + 'interactions' + '.npy', allow_pickle=True).squeeze()

        compounds_cpu_test = np.load(dir_input + 'compounds_test' + '.npy', allow_pickle=True)
        adjacencies_cpu_test = np.load(dir_input + 'adjacencies_test' + '.npy', allow_pickle=True)
        proteins_cpu_test = np.load(dir_input + 'proteins_test' + '.npy', allow_pickle=True)
        interactions_cpu_test = np.load(dir_input + 'interactions_test' + '.npy', allow_pickle=True).squeeze()

        print(len(proteins_cpu), len(adjacencies_cpu))
        print("max protein len:", max([i[0] for i in proteins_cpu]))
        print("max adj len:", max([len(i) for i in adjacencies_cpu]))
        print("interactions_cpu shape", interactions_cpu.shape)
        print("graph_max:", graph_max, "   seq_max:", seq_max)

        # padding
        compounds, compounds_mask, adjs, proteins, proteins_mask = [], [], [], [], []
        compounds_test, compounds_mask_test, adjs_test, proteins_test, proteins_mask_test = [], [], [], [], []
        for cmp in compounds_cpu:
            cmp_len = len(cmp)
            if (cmp_len > graph_max):
                res = cmp[:graph_max]
                mask = [1] * graph_max
            else:
                res = cmp.tolist() + [0] * (graph_max - cmp_len)
                mask = [1] * cmp_len + [0] * (graph_max - cmp_len)
            compounds.append(res)
            compounds_mask.append(mask)

        for adj in adjacencies_cpu:
            if (len(adj) > graph_max):
                adj = adj[:graph_max, :graph_max]
            pad = np.zeros([graph_max, graph_max], dtype=np.int32)
            pad[np.where(adj > 0)[0], np.where(adj > 0)[1]] = 1
            adjs.append(pad)

        for pro in proteins_cpu:
            pro_len = len(pro)
            if pro_len > seq_max:
                pro = pro[:seq_max]
                mask = [1] * seq_max
            else:
                pro = pro + [0] * (seq_max - pro_len)
                mask = [1] * pro_len + [0] * (seq_max - pro_len)
            proteins.append(pro)
            proteins_mask.append(mask)

        ################
        for cmp in compounds_cpu_test:
            cmp_len = len(cmp)
            if (cmp_len > graph_max):
                res = cmp[:graph_max]
                mask = [1] * graph_max
            else:
                res = cmp.tolist() + [0] * (graph_max - cmp_len)
                mask = [1] * cmp_len + [0] * (graph_max - cmp_len)
            compounds_test.append(res)
            compounds_mask_test.append(mask)

        for adj in adjacencies_cpu_test:
            if (len(adj) > graph_max):
                adj = adj[:graph_max, :graph_max]
            pad = np.zeros([graph_max, graph_max], dtype=np.int32)
            pad[np.where(adj > 0)[0], np.where(adj > 0)[1]] = 1
            adjs_test.append(pad)

        for pro in proteins_cpu_test:
            pro_len = len(pro)
            if pro_len > seq_max:
                pro = pro[:seq_max]
                mask = [1] * seq_max
            else:
                pro = pro + [0] * (seq_max - pro_len)
                mask = [1] * pro_len + [0] * (seq_max - pro_len)
            proteins_test.append(pro)
            proteins_mask_test.append(mask)

        compounds = np.array(compounds)
        compounds_mask = np.array(compounds_mask)
        adjs = np.array(adjs)
        proteins = np.array(proteins)
        proteins_mask = np.array(proteins_mask)

        compounds_test = np.array(compounds_test)
        compounds_mask_test = np.array(compounds_mask_test)
        adjs_test = np.array(adjs_test)
        proteins_test = np.array(proteins_test)
        proteins_mask_test = np.array(proteins_mask_test)

        # split to 5 fold cv  随机洗牌
        compounds = shuffle_dataset(compounds, seed_num)
        compounds_mask = shuffle_dataset(compounds_mask, seed_num)
        adjs = shuffle_dataset(adjs, seed_num)
        proteins = shuffle_dataset(proteins, seed_num)
        proteins_mask = shuffle_dataset(proteins_mask, seed_num)
        interactions = shuffle_dataset(interactions_cpu, seed_num)

        compounds_test = shuffle_dataset(compounds_test, seed_num)
        compounds_mask_test = shuffle_dataset(compounds_mask_test, seed_num)
        adjs_test = shuffle_dataset(adjs_test, seed_num)
        proteins_test = shuffle_dataset(proteins_test, seed_num)
        proteins_mask_test = shuffle_dataset(proteins_mask_test, seed_num)
        interactions_test = shuffle_dataset(interactions_cpu_test, seed_num)

        file_AUCs = f'../output/result/res-{args.dataset}{fold_num}-{args.save_name_append}.txt'
        file_model = f'../output/model/{args.dataset}{fold_num}-{args.save_name_append}.pt'
        if not os.path.exists(file_AUCs):
            os.makedirs('../output/result/', exist_ok=True)
        if not os.path.exists(file_model):
            os.makedirs('../output/model/', exist_ok=True)

        train_cmp = torch.from_numpy(compounds).long()
        tmp_val, tmp_test = split_dataset(compounds_test, 0.5)
        val_cmp = torch.from_numpy(tmp_val).long()
        test_cmp = torch.from_numpy(tmp_test).long()

        train_cmp_mask = torch.from_numpy(compounds_mask).float()
        tmp_val, tmp_test = split_dataset(compounds_mask_test, 0.5)
        val_cmp_mask = torch.from_numpy(tmp_val).float()
        test_cmp_mask = torch.from_numpy(tmp_test).float()

        train_adj = torch.from_numpy(adjs).float()
        tmp_val, tmp_test = split_dataset(adjs_test, 0.5)
        val_adj = torch.from_numpy(tmp_val).float()
        test_adj = torch.from_numpy(tmp_test).float()

        train_protein = torch.from_numpy(proteins).long()
        tmp_val, tmp_test = split_dataset(proteins_test, 0.5)
        val_protein = torch.from_numpy(tmp_val).long()
        test_protein = torch.from_numpy(tmp_test).long()

        train_protein_mask = torch.from_numpy(proteins_mask).float()
        tmp_val, tmp_test = split_dataset(proteins_mask_test, 0.5)
        val_protein_mask = torch.from_numpy(tmp_val).float()
        test_protein_mask = torch.from_numpy(tmp_test).float()

        train_interactions = torch.from_numpy(interactions).long()
        tmp_val, tmp_test = split_dataset(interactions_test, 0.5)
        val_interactions = torch.from_numpy(tmp_val).long()
        test_interactions = torch.from_numpy(tmp_test).long()

        train_data = TensorDataset(train_cmp, train_cmp_mask, train_adj, train_protein, train_protein_mask, train_interactions)
        train_sample = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sample, batch_size=batch_size)

        val_data = TensorDataset(val_cmp, val_cmp_mask, val_adj, val_protein, val_protein_mask, val_interactions)
        val_sample = RandomSampler(val_data)
        val_loader = DataLoader(val_data, sampler=val_sample, batch_size=batch_size * 2)

        test_data = TensorDataset(test_cmp, test_cmp_mask, test_adj, test_protein, test_protein_mask, test_interactions)
        test_sample = RandomSampler(test_data)
        test_loader = DataLoader(test_data, sampler=test_sample, batch_size=batch_size * 2)
        torch.manual_seed(seed_num)

        model = CompoundProteinInteractionPrediction()
        model = model.to(device)

        trainer = Trainer(model)
        tester = Tester(model)

        AUCs = ('Fold\tEpoch\tTime(min)\tLoss_train\tAUC_dev\tAUPR_dev\tPrecision_dev\tRecall_dev\tF1_dev')
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')
        print('Training...')
        print(AUCs)

        best_auc_val = 0.0
        start = timeit.default_timer()
        for epoch in range(1, args.iteration):
            if epoch % args.decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= args.lr_decay

            loss_train = trainer.train(train_loader)
            auc_val, pre_val, recall_val, aupr_val, f1_val = tester.test(val_loader)
            end = timeit.default_timer()
            time = end - start
            print('\t'.join(map(lambda x: str(round(x, 5)),[fold_num, epoch, time // 60, loss_train, auc_val, aupr_val, pre_val, recall_val, f1_val])))

            if auc_val > best_auc_val:
                best_auc_val = auc_val
                tester.save_model(model, file_model)
                test_auc, test_pre, test_recall, test_aupr, test_f1 = tester.test(test_loader)
                test_info = [fold_num, epoch, test_auc, test_aupr, test_pre, test_recall, test_f1]
                tester.save_AUCs(test_info, file_AUCs)
                print('Test-->', '\t'.join(map(lambda x: str(round(x, 5)), test_info)))