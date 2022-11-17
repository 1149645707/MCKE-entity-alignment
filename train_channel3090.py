import torch
import os
import numpy as np
from pathlib import Path
from collections import OrderedDict
from data_init import LoadData
from net import *
from torch.optim import Adagrad
from util import print_time_info, set_random_seed, get_hits,getResult
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def cosine_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / torch.clamp(a.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    b = b / torch.clamp(b.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    if b.shape[0] * b.shape[1] > 20000 * 128:      #20000 * 128
        return cosine_similarity_nbyn_batched(a, b)
    return torch.mm(a, b.t())


def cosine_similarity_nbyn_batched(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    batch_size = 512   #512
    data_num = b.shape[0]
    b = b.t()
    sim_matrix = []
    for i in range(0, data_num, batch_size):
        sim_matrix.append(torch.mm(a, b[:, i:i + batch_size]).cpu()) #.cpu()
    sim_matrix = torch.cat(sim_matrix, dim=1)
    return sim_matrix


def get_nearest_neighbor(sim, nega_sample_num=25):
    # Sim do not have to be a square matrix
    # Let us assume sim is a numpy array
    ranks = torch.argsort(sim, dim=1)
    ranks = ranks[:, 1:nega_sample_num + 1]
    return ranks



class AlignLoss(nn.Module):
    def __init__(self, margin, p=2, reduction='mean'):
        super(AlignLoss, self).__init__()
        self.p = p
        self.criterion = nn.TripletMarginLoss(margin, p=p, reduction=reduction)

    def forward(self, repre_sr, repre_tg):
        '''
        score shape: [batch_size, 2, embedding_dim]
        '''
        # distance = torch.abs(score).sum(dim=-1) * self.re_scale
        sr_true = repre_sr[:, 0, :]
        sr_nega = repre_sr[:, 1, :]
        tg_true = repre_tg[:, 0, :]
        tg_nega = repre_tg[:, 1, :]


        loss = self.criterion(torch.cat((sr_true, tg_true), dim=0), torch.cat((tg_true, sr_true), dim=0),
                              torch.cat((tg_nega, sr_nega), dim=0))
        return loss


def sort_and_keep_indices(matrix, device):
    batch_size = 512
    data_len = matrix.shape[0]
    sim_matrix = []
    indice_list = []
    for i in range(0, data_len, batch_size):
        batch = matrix[i:i + batch_size]
        batch = torch.from_numpy(batch).to(device)
        sorted_batch, indices = torch.sort(batch, dim=-1)
        sorted_batch = sorted_batch[:, :500].cpu()
        indices = indices[:, :500].cpu()
        sim_matrix.append(sorted_batch)
        indice_list.append(indices)
    sim_matrix = torch.cat(sim_matrix, dim=0).numpy()
    indice_array = torch.cat(indice_list, dim=0).numpy()
    sim = np.concatenate([np.expand_dims(sim_matrix, 0), np.expand_dims(indice_array, 0)], axis=0)
    return sim


class EmbedChannel(nn.Module):

    def __init__(self, ent_num_sr, ent_num_tg, dim, layer_num, drop_out, channels):
        super(EmbedChannel, self).__init__()
        assert len(channels) == 1
        if 'name' in channels:
            self.embedChannel = NameEmbed(dim, layer_num, drop_out, **channels['name'])
        if 'structure' in channels:
            self.embedChannel = StructureEmbed(ent_num_sr, ent_num_tg, dim, layer_num, drop_out, **channels['structure'])
        if 'attribute' in channels:
            self.embedChannel = AttributeEmbed(dim, **channels['attribute'])

    def forward(self, sr_ent_seeds, tg_ent_seeds):
        sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid = self.embedChannel.forward(sr_ent_seeds, tg_ent_seeds)
        return sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid

    def predict(self, sr_ent_seeds, tg_ent_seeds):
        with torch.no_grad():
            sr_seed_hid, tg_seed_hid, _, _ = self.forward(sr_ent_seeds, tg_ent_seeds)
        return sr_seed_hid, tg_seed_hid

    def negative_sample(self, sr_ent_seeds, tg_ent_seeds):
        with torch.no_grad():
            sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid = self.forward(sr_ent_seeds, tg_ent_seeds)
            if sr_seed_hid.ndimension() == 3:
                sr_seed_hid = sr_seed_hid.squeeze()
                tg_seed_hid = tg_seed_hid.squeeze()
            sim_sr = - cosine_similarity_nbyn(sr_seed_hid, sr_ent_hid)
            sim_tg = - cosine_similarity_nbyn(tg_seed_hid, tg_ent_hid)
        return sim_sr, sim_tg


class NameEmbed(nn.Module):
    def __init__(self, dim, layer_num, drop_out, edges_sr, edges_tg, entity_vec_sr, entity_vec_tg):
        super(NameEmbed, self).__init__()

        self.embedding_sr = nn.Parameter(entity_vec_sr, requires_grad=False)
        self.embedding_tg = nn.Parameter(entity_vec_tg, requires_grad=False)
        self.edges_sr = nn.Parameter(edges_sr, requires_grad=False)
        self.edges_tg = nn.Parameter(edges_tg, requires_grad=False)
        in_dim = entity_vec_sr.shape[1]
        self.gcn = MultiLayerGCN(in_dim, dim, layer_num, drop_out, featureless=False, residual=True)

    def forward(self, sr_ent_seeds, tg_ent_seeds):
        sr_ent_hid = self.gcn(self.edges_sr, self.embedding_sr)
        tg_ent_hid = self.gcn(self.edges_tg, self.embedding_tg)
        sr_seed_hid = sr_ent_hid[sr_ent_seeds.type(torch.long)]
        tg_seed_hid = tg_ent_hid[tg_ent_seeds.type(torch.long)]
        return sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid


class StructureEmbed(nn.Module):
    def __init__(self, ent_num_sr, ent_num_tg, dim, layer_num, drop_out, edges_sr,edges_tg):
        super(StructureEmbed, self).__init__()
        embedding_weight = torch.zeros((ent_num_sr + ent_num_tg, dim), dtype=torch.float)
        nn.init.xavier_uniform_(embedding_weight)
        self.feats_sr = nn.Parameter(embedding_weight[:ent_num_sr], requires_grad=True)
        self.feats_tg = nn.Parameter(embedding_weight[ent_num_sr:], requires_grad=True)
        self.edges_sr = nn.Parameter(edges_sr, requires_grad=False)
        self.edges_tg = nn.Parameter(edges_tg, requires_grad=False)
        assert len(self.feats_sr) == ent_num_sr
        assert len(self.feats_tg) == ent_num_tg
        self.gcn = MultiLayerGCN(self.feats_sr.shape[-1], dim, layer_num, drop_out, featureless=True, residual=False)


    def forward(self, sr_ent_seeds, tg_ent_seeds):
        sr_ent_hid = self.gcn(self.edges_sr, self.feats_sr)
        tg_ent_hid = self.gcn(self.edges_tg, self.feats_tg)
        sr_ent_hid = F.normalize(sr_ent_hid, p=2, dim=-1)
        tg_ent_hid = F.normalize(tg_ent_hid, p=2, dim=-1)

        sr_seed_hid = sr_ent_hid[sr_ent_seeds.type(torch.long)]
        tg_seed_hid = tg_ent_hid[tg_ent_seeds.type(torch.long)]
        return sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid


class AttributeEmbed(nn.Module):
    def __init__(self, dim,attention_property_vec_sr,attention_property_vec_tg,entity_vec_sr,entity_vec_tg,value_vec_sr,value_vec_tg,property_vec_sr,property_vec_tg):
        super(AttributeEmbed, self).__init__()
        self.attention_property_vec_sr = attention_property_vec_sr
        self.attention_property_vec_tg = attention_property_vec_tg
        self.entity_vec_sr = entity_vec_sr
        self.entity_vec_tg = entity_vec_tg
        self.value_vec_sr = value_vec_sr
        self.value_vec_tg = value_vec_tg
        self.property_vec_sr = property_vec_sr
        self.property_vec_tg = property_vec_tg
        W_sr = torch.ones((1, 768), dtype=torch.float)
        W_tg = torch.ones((1, 768), dtype=torch.float)
        self.Weight_Attribute_sr = nn.Parameter(W_sr, requires_grad=True)
        self.Weight_Attribute_tg = nn.Parameter(W_tg, requires_grad=True)
        indim = 768 + 2 * dim + dim
        self.linear = nn.Linear(indim, dim)
        self.proj_sr = nn.Linear(768, 2*dim)
        self.proj_tg = nn.Linear(768, 2*dim)
        self.dropout = nn.Dropout(0.7)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, sr_ent_seeds, tg_ent_seeds):
        # attention
        topic_vec_sr = torch.cat((entity_vec_sr, attention_property_vec_sr, exp_out_sr), dim=1)
        topic_vec_tg = torch.cat((entity_vec_tg, attention_property_vec_tg, exp_out_tg), dim=1)

        topic_vec_sr = self.linear(topic_vec_sr)
        topic_vec_tg = self.linear(topic_vec_tg)
        sr_seed_hid = topic_vec_sr[sr_ent_seeds.type(torch.long)]
        tg_seed_hid = topic_vec_tg[tg_ent_seeds.type(torch.long)]

        return sr_seed_hid, tg_seed_hid, topic_vec_sr, topic_vec_tg



class MCKE(object):

    def __init__(self):
        self.train_seeds_ratio = 0.3
        self.dim = 128
        self.drop_out = 0.0
        self.layer_num = 2
        self.epoch_num = 2000
        self.nega_sample_freq = 5
        self.nega_sample_num = 25

        self.learning_rate = 0.001
        self.l2_regularization = 0.0001
        self.margin_gamma = 1.0

        self.log_comment = "comment"

        self.coIter_att_num = 1

        self.name_channel = False
        self.structure_channel = False
        self.attribute_channel = False
        self.load_new_seed_split = False

    def set_load_new_seed_split(self, load_new_seed_split):
        self.load_new_seed_split = load_new_seed_split

    def set_channel(self, channel_name):
        if channel_name == 'Name':
            self.set_name_channel(True)
        elif channel_name == 'Structure':
            self.set_structure_channel(True)
        elif channel_name == 'Attribute':
            self.set_attribute_channel(True)
        else:
            raise Exception()

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def set_nega_sample_num(self, nega_sample_num):
        self.nega_sample_num = nega_sample_num

    def set_log_comment(self, log_comment):
        self.log_comment = log_comment

    def set_name_channel(self, use_name_channel):
        self.name_channel = use_name_channel

    def set_structure_channel(self, use_structure_channel):
        self.structure_channel = use_structure_channel

    def set_attribute_channel(self, use_attribute_channel):
        self.attribute_channel = use_attribute_channel

    def set_drop_out(self, drop_out):
        self.drop_out = drop_out

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_l2_regularization(self, l2_regularization):
        self.l2_regularization = l2_regularization

    def set_log_dir(self,log_dir):
        self.log_dir = log_dir

    def print_parameter(self, file=None):
        parameters = self.__dict__
        print_time_info('Parameter setttings:', dash_top=True, file=file)
        for key, value in parameters.items():
            if type(value) in {int, float, str, bool}:
                print('\t%s:' % key, value, file=file)
        print('---------------------------------------', file=file)


    def init_log(self, log_dir):
        comment = log_dir.name
        with open(log_dir / 'parameters.txt', 'w') as f:
            print_time_info(comment, file=f)
            self.print_parameter(f)

    def init(self, directory, device):
        set_random_seed()
        self.directory = Path(directory)
        self.loaded_data = LoadData(self.train_seeds_ratio, self.directory, self.nega_sample_num,
                                    name_channel=self.name_channel,
                                    structure_channel=self.structure_channel,
                                    attribute_channel=self.attribute_channel,
                                    load_new_seed_split=self.load_new_seed_split,device=device)
        self.sr_ent_num = self.loaded_data.sr_ent_num
        self.tg_ent_num = self.loaded_data.tg_ent_num

        # Init graph adjacent matrix
        print_time_info('Begin preprocessing data')
        self.channels = {}

        edges_sr = torch.tensor(self.loaded_data.triples_sr)[:, :2]
        edges_tg = torch.tensor(self.loaded_data.triples_tg)[:, :2]
        edges_sr = torch.unique(edges_sr, dim=0)
        edges_tg = torch.unique(edges_tg, dim=0)

        if self.name_channel:
            self.channels['name'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                     'entity_vec_sr': self.loaded_data.sr_entity_vec,
                                     'entity_vec_tg': self.loaded_data.tg_entity_vec }

        if self.structure_channel:
            self.channels['structure'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg}

        if self.attribute_channel:
            self.channels['attribute'] = {'attention_property_vec_sr': self.loaded_data.sr_attention_property_vec,
                                          'attention_property_vec_tg': self.loaded_data.tg_attention_property_vec,
                                          'entity_vec_sr': self.loaded_data.sr_entity_vec,
                                          'entity_vec_tg': self.loaded_data.tg_entity_vec,
                                          'value_vec_sr': self.loaded_data.sr_value_vec,
                                          'value_vec_tg': self.loaded_data.tg_value_vec,
                                          'property_vec_sr': self.loaded_data.sr_property_vec,
                                          'property_vec_tg': self.loaded_data.tg_property_vec
                                          }

        print_time_info('Finished preprocesssing data')


    def negative_sample(self, ):

        sim_sr, sim_tg = self.embedChannel.negative_sample(self.loaded_data.train_sr_ent_seeds_ori,
                                                           self.loaded_data.train_tg_ent_seeds_ori)
        sr_nns = get_nearest_neighbor(sim_sr, self.nega_sample_num)
        tg_nns = get_nearest_neighbor(sim_tg, self.nega_sample_num)
        self.loaded_data.update_negative_sample(sr_nns, tg_nns)

    def train(self, device):
        training_info = []

        set_random_seed()
        self.loaded_data.negative_sample()

        embed_channel = EmbedChannel(self.sr_ent_num, self.tg_ent_num, self.dim,
                                     self.layer_num, self.drop_out, self.channels)
        self.embedChannel = embed_channel
        embed_channel.to(device)
        embed_channel.train()

        # Prepare optimizer
        optimizer = Adagrad(filter(lambda p: p.requires_grad, embed_channel.parameters(), ),
                            lr=self.learning_rate, weight_decay=self.l2_regularization)
        criterion = AlignLoss(self.margin_gamma)

        best_hit_at_1 = 0
        best_epoch_num = 0

        for epoch_num in range(1, self.epoch_num + 1):
            print('current epoch', epoch_num)
            embed_channel.train()
            optimizer.zero_grad()
            sr_seed_hid, tg_seed_hid, _, _ = embed_channel.forward(self.loaded_data.train_sr_ent_seeds,
                                                                   self.loaded_data.train_tg_ent_seeds)
            loss = criterion(sr_seed_hid, tg_seed_hid)
            loss.backward()
            optimizer.step()

            if epoch_num % self.nega_sample_freq == 0:
                if str(self.directory).find('DWY100k') >= 0:
                    self.loaded_data.negative_sample()
                else:
                    self.negative_sample()

                hit_at_1 = self.evaluate(embed_channel, print_info=False, device=device)
                print('current hit@1:', hit_at_1)
                if hit_at_1 > best_hit_at_1:
                    best_hit_at_1 = hit_at_1
                    best_epoch_num = epoch_num

                epoch_info = 'Epoch: [{}/{}], loss:{}, current best Hit@1:{:.2f}'.format(epoch_num, self.epoch_num, loss,best_hit_at_1)
                training_info.append(epoch_info)
        best_info='Model best Hit@1 on valid set is %.2f at %d epoch.' % (best_hit_at_1, best_epoch_num)
        print(best_info)
        training_info.append(best_info)
        return best_hit_at_1, best_epoch_num,training_info


    def train_StructureEmbed(self, device):
        training_info = []

        set_random_seed()
        self.loaded_data.negative_sample()

        embed_channel = EmbedChannel(self.sr_ent_num, self.tg_ent_num,self.dim,
                                     self.layer_num, self.drop_out, self.channels)
        self.embedChannel = embed_channel

        co_att = CoIter_Attention(self.dim, self.coIter_att_num)
        self.co_att = co_att

        embed_channel.to(device)
        co_att.to(device)


        # Prepare optimizer
        optimizer_gcn = Adagrad(filter(lambda p: p.requires_grad, embed_channel.parameters(), ),
                            lr=self.learning_rate, weight_decay=self.l2_regularization)
        optimizer_coatt = Adagrad(filter(lambda p: p.requires_grad, co_att.parameters(), ),
                            lr=0.001, weight_decay=0.002)  #lr=self.learning_rate, weight_decay=self.l2_regularization
        criterion = AlignLoss(self.margin_gamma)

        best_hit_at_1 = 0
        best_epoch_num = 0

        for epoch_num in range(1, self.epoch_num + 1):
            print('current epoch', epoch_num)
            embed_channel.train()
            co_att.train()
            optimizer_gcn.zero_grad()
            optimizer_coatt.zero_grad()

            train_data = TensorDataset(self.loaded_data.train_sr_ent_seeds,
                                       self.loaded_data.train_tg_ent_seeds)
            train_loader = DataLoader(train_data, batch_size=8192, shuffle=False)  #8192

            for batch_idx, sr_tg_seeds in enumerate(train_loader):
                sr_seed, tg_seed = sr_tg_seeds
                sr_seed_hid, tg_seed_hid, _, _ = embed_channel.forward(sr_seed, tg_seed)
                h1, h2 = co_att(sr_seed_hid, tg_seed_hid)
                loss = criterion(h1, h2)
                loss.backward(retain_graph=True)
                optimizer_gcn.step()
                optimizer_coatt.step()

            if epoch_num % self.nega_sample_freq == 0:
                if str(self.directory).find('DWY100k') >= 0:
                    self.loaded_data.negative_sample()
                else:
                    self.negative_sample()
                hit_at_1 = self.evaluate_StructureEmbed(embed_channel, co_att, print_info=False, device=device)
                print('current hit@1:', hit_at_1)
                if hit_at_1 > best_hit_at_1:
                    best_hit_at_1 = hit_at_1
                    best_epoch_num = epoch_num

                epoch_info = 'Epoch: [{}/{}], loss:{}, current best Hit@1:{:.2f}'.format(epoch_num, self.epoch_num,
                                                                                         loss, best_hit_at_1)
                training_info.append(epoch_info)
        best_info = 'Model best Hit@1 on valid set is %.2f at %d epoch.' % (best_hit_at_1, best_epoch_num)
        print(best_info)
        training_info.append(best_info)
        return best_hit_at_1, best_epoch_num, training_info


    def evaluate(self,current_embed_channel, print_info=True, device='cpu'):
        current_embed_channel.eval()
        sr_seed_hid, tg_seed_hid = current_embed_channel.predict(self.loaded_data.valid_sr_ent_seeds,self.loaded_data.valid_tg_ent_seeds)
        if sr_seed_hid.ndimension()==3:
            sr_seed_hid = sr_seed_hid.squeeze()
            tg_seed_hid = tg_seed_hid.squeeze()
        sim = - cosine_similarity_nbyn(sr_seed_hid, tg_seed_hid)
        top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl = get_hits(sim, print_info=print_info, device=device)
        hit_at_1 = (top_lr[0] + top_rl[0]) / 2
        return hit_at_1


    def get_structure_sim(self,current_co_att,sr_seed_hid, tg_seed_hid):
        current_co_att.eval()
        sr_seed_hid = sr_seed_hid.unsqueeze(1)
        tg_seed_hid = tg_seed_hid.unsqueeze(1)
        sr_structure_hid, tg_structure_hid = current_co_att(sr_seed_hid, tg_seed_hid)
        sr_structure_hid = sr_structure_hid.squeeze()
        tg_structure_hid = tg_structure_hid.squeeze()
        sim = - cosine_similarity_nbyn(sr_structure_hid, tg_structure_hid)
        return sim

    def evaluate_StructureEmbed(self,current_embed_channel,current_co_att, print_info=True, device='cpu'):
        current_embed_channel.eval()
        sr_seed_hid, tg_seed_hid = current_embed_channel.predict(self.loaded_data.valid_sr_ent_seeds,self.loaded_data.valid_tg_ent_seeds)
        sim=self.get_structure_sim(current_co_att,sr_seed_hid, tg_seed_hid)
        top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl = get_hits(sim, print_info=print_info, device=device)
        hit_at_1 = (top_lr[0] + top_rl[0]) / 2
        return hit_at_1


    def save_sim_matrix(self, device, log_comment, data_set):
        # Get the similarity matrix of the current model
        self.embedChannel.eval()

        sr_seed_hid_train, tg_seed_hid_train = self.embedChannel.predict(self.loaded_data.train_sr_ent_seeds_ori,
                                                                         self.loaded_data.train_tg_ent_seeds_ori)

        sr_seed_hid_valid, tg_seed_hid_valid = self.embedChannel.predict(self.loaded_data.valid_sr_ent_seeds,
                                                                         self.loaded_data.valid_tg_ent_seeds)

        sr_seed_hid_test, tg_seed_hid_test = self.embedChannel.predict(self.loaded_data.test_sr_ent_seeds,
                                                                       self.loaded_data.test_tg_ent_seeds)

        if sr_seed_hid_train.ndimension()==3:
            sr_seed_hid_train = sr_seed_hid_train.squeeze()
            tg_seed_hid_train = tg_seed_hid_train.squeeze()

        if sr_seed_hid_valid.ndimension()==3:
            sr_seed_hid_valid = sr_seed_hid_valid.squeeze()
            tg_seed_hid_valid = tg_seed_hid_valid.squeeze()

        if sr_seed_hid_test.ndimension()==3:
            sr_seed_hid_test = sr_seed_hid_test.squeeze()
            tg_seed_hid_test = tg_seed_hid_test.squeeze()

        sim_train = - cosine_similarity_nbyn(sr_seed_hid_train, tg_seed_hid_train)
        sim_train = sim_train.cpu().detach().numpy()

        sim_valid = - cosine_similarity_nbyn(sr_seed_hid_valid, tg_seed_hid_valid)
        sim_valid = sim_valid.cpu().detach().numpy()

        sim_test  = - cosine_similarity_nbyn(sr_seed_hid_test, tg_seed_hid_test)
        print_time_info('Best result on the test set:', dash_top=False)
        top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl = get_hits(sim_test, print_info=True, device=device)
        getResult(self.log_dir, top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl)
        sim_test = sim_test.cpu().detach().numpy()

        def save_sim(sim, comment):
            if sim.shape[0] > 20000:
                partial_sim = sort_and_keep_indices(sim, device)
                partial_sim_t = sort_and_keep_indices(sim.T, device)
                np.save(str(self.log_dir / ('%s_sim.npy' % comment)), partial_sim)
                np.save(str(self.log_dir / ('%s_sim_t.npy' % comment)), partial_sim_t)
            else:
                np.save(str(self.log_dir / ('%s_sim.npy' % comment)), sim)
        save_sim(sim_train, 'train')
        save_sim(sim_valid, 'valid')
        save_sim(sim_test, 'test')
        print_time_info("Model configs and predictions saved to directory: %s." % str(self.log_dir))

    def save_model(self):
        save_path = self.log_dir / 'model.pt'
        state_dict = self.embedChannel.state_dict()
        state_dict = OrderedDict(filter(lambda x: x[1].layout != torch.sparse_coo, state_dict.items()))
        torch.save(state_dict, str(save_path))
        print_time_info("Model is saved to directory: %s." % str(self.log_dir))

    def save_structure_sim_matrix(self, device, log_comment, data_set):
        # Get the similarity matrix of the current model
        self.embedChannel.eval()

        sr_seed_hid_train, tg_seed_hid_train = self.embedChannel.predict(self.loaded_data.train_sr_ent_seeds_ori,
                                                                         self.loaded_data.train_tg_ent_seeds_ori)

        sr_seed_hid_valid, tg_seed_hid_valid = self.embedChannel.predict(self.loaded_data.valid_sr_ent_seeds,
                                                                         self.loaded_data.valid_tg_ent_seeds)

        sr_seed_hid_test, tg_seed_hid_test = self.embedChannel.predict(self.loaded_data.test_sr_ent_seeds,
                                                                       self.loaded_data.test_tg_ent_seeds)

        sim_train = self.get_structure_sim(self.co_att, sr_seed_hid_train, tg_seed_hid_train)
        sim_valid = self.get_structure_sim(self.co_att, sr_seed_hid_valid, tg_seed_hid_valid)
        sim_test = self.get_structure_sim(self.co_att, sr_seed_hid_test, tg_seed_hid_test)

        print_time_info('Best result on the test set:', dash_top=False)
        top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl = get_hits(sim_test, print_info=True, device=device)
        getResult(self.log_dir, top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl)

        sim_train = sim_train.cpu().detach().numpy()
        sim_valid = sim_valid.cpu().detach().numpy()
        sim_test = sim_test.cpu().detach().numpy()

        def save_sim(sim, comment):
            if sim.shape[0] > 20000:
                partial_sim = sort_and_keep_indices(sim, device)
                partial_sim_t = sort_and_keep_indices(sim.T, device)
                np.save(str(self.log_dir / ('%s_sim.npy' % comment)), partial_sim)
                np.save(str(self.log_dir / ('%s_sim_t.npy' % comment)), partial_sim_t)
            else:
                np.save(str(self.log_dir / ('%s_sim.npy' % comment)), sim)

        save_sim(sim_train, 'train')
        save_sim(sim_valid, 'valid')
        save_sim(sim_test, 'test')
        print_time_info("Model configs and predictions saved to directory: %s." % str(self.log_dir))

    def save_structure_model(self):
        save_GCN_path = self.log_dir / 'GCN_model.pt'
        state_dict_gcn = self.embedChannel.state_dict()
        state_dict_gcn = OrderedDict(filter(lambda x: x[1].layout != torch.sparse_coo, state_dict_gcn.items()))
        torch.save(state_dict_gcn, str(save_GCN_path))

        save_CoAtt_path = self.log_dir / 'CoIter_Attention_model.pt'
        state_dict_coatt = self.co_att.state_dict()
        state_dict_coatt = OrderedDict(filter(lambda x: x[1].layout != torch.sparse_coo, state_dict_coatt.items()))
        torch.save(state_dict_coatt, str(save_CoAtt_path))

        print_time_info("Model is saved to directory: %s." % str(self.log_dir))

    def save_training_info(self,training_info):
        save_path = self.log_dir / 'training_info.txt'
        with open(save_path, 'w') as f:
            for i in training_info:
                f.write(str(i))
                f.write('\n')


def MCKE_EA(log_comment, data_set, layer_num, device, load_new_seed_split=False, save_model=False):
    nsa_embed=MCKE()

    nsa_embed.set_channel(log_comment)
    nsa_embed.layer_num = layer_num
    nsa_embed.set_log_comment(log_comment)
    nsa_embed.set_load_new_seed_split(load_new_seed_split)

    log_dir = './log/%s_%s' % (data_set.replace('/', '_'), nsa_embed.log_comment)
    nsa_embed.set_log_dir(Path(log_dir))
    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists(nsa_embed.log_dir):
        os.mkdir(nsa_embed.log_dir)

    nsa_embed.init('./bin/%s' % data_set, device)
    data_set = data_set.split('/')[-1]


    if log_comment == 'Name':
        nsa_embed.init_log(nsa_embed.log_dir)
        _, _, training_info = nsa_embed.train(device)
        nsa_embed.save_sim_matrix(device, nsa_embed.log_comment, data_set)
        nsa_embed.save_training_info(training_info)
        if save_model:
            nsa_embed.save_model()

    elif log_comment == 'Structure':
        nsa_embed.init_log(nsa_embed.log_dir)
        _, _, training_info =nsa_embed.train_StructureEmbed(device)
        nsa_embed.save_structure_sim_matrix(device, nsa_embed.log_comment, data_set)
        nsa_embed.save_training_info(training_info)
        if save_model:
            nsa_embed.save_structure_model()

    elif log_comment == 'Attribute':
        nsa_embed.init_log(nsa_embed.log_dir)
        _, _, training_info = nsa_embed.train(device)
        nsa_embed.save_sim_matrix(device, nsa_embed.log_comment, data_set)
        nsa_embed.save_training_info(training_info)
        if save_model:
            nsa_embed.save_model()


if __name__ == '__main__':
    '''
    python train_channel.py --dataset DBP15k/zh_en --channel Structure  --gpu_id 0
    '''

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--load_hard_split', action='store_true')
    parser.add_argument('--layer_num', type=int, default=2)
    args = parser.parse_args()
    import os


    device = 'cuda:%d' % args.gpu_id if args.gpu_id >= 0 else 'cpu'

    if args.channel == 'all':
        MCKE_EA('Name', args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                save_model=True)
        MCKE_EA('Structure', args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                save_model=True)
        MCKE_EA('Attribute', args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                save_model=True)

    else:

        MCKE_EA(args.channel, args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                    save_model=True)



