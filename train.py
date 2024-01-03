from torchtools import *
from data import TieredImagenetLoader
from model import EmbeddingImagenet, Unet, Unet3
import shutil
import os
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time


class SupConLoss_inter(nn.Module):
    
    def __init__(self, contrast_mode='all'):
        super(SupConLoss_inter, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        

    def forward(self, temperature, features, label1, label2):

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # print('feture size: ',features.size())     #32*2*128
        batch_size = features.shape[0]   #32
        label1 = label1.contiguous().view(-1, 1)
        label2 = label2.contiguous().view(-1, 1)
        label = torch.cat([label1,label2],0)
        mask = torch.eq(label, label.T).float().to(tt.arg.device)
        #torch.eq()：对两个张量进行逐元素的比较，若相同位置的两个元素相同，则为1；若不同，则为0
       
        contrast_count = features.shape[1] # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 64*128
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # 64*128
            anchor_count = contrast_count  # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(tt.arg.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        # print('loss: ',loss)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss_intra(nn.Module):
    
    def __init__(self, contrast_mode='all'):
        super(SupConLoss_intra, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        

    def forward(self, temperature, features, label1, label2,label_c1, label_c2):

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # print('feture size: ',features.size())     #32*2*128
        batch_size = features.shape[0]   #32
        label1 = label1.contiguous().view(-1, 1)
        label2 = label2.contiguous().view(-1, 1)
        label = torch.cat([label1,label2],0)
        mask = torch.eq(label, label.T).float().to(tt.arg.device)

        label_c1 = label_c1.contiguous().view(-1, 1)
        label_c2 = label_c2.contiguous().view(-1, 1)
        label = torch.cat([label_c1,label_c2],0)
        mask_c = torch.eq(label, label.T).float().to(tt.arg.device)
        #torch.eq()：对两个张量进行逐元素的比较，若相同位置的两个元素相同，则为1；若不同，则为0
       
        contrast_count = features.shape[1] # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 64*128
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # 64*128
            anchor_count = contrast_count  # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),temperature).to(tt.arg.device)
        # print(anchor_dot_contrast[0,:])
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print(logits[0,:])
    

        # tile mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(tt.arg.device),
            0
        )
        mask = mask * logits_mask

        logits_mask_c = torch.scatter(
            mask_c,
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(tt.arg.device),
            0
        )
        # print('logits_mask',logits_mask[1,:])
        logits_mask_c = logits_mask_c * logits_mask #select sbiling
        
        exp_logits = torch.exp(logits) * logits_mask_c    
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
       
        loss_sbling = loss.view(anchor_count, batch_size).mean()

        logits_mask = logits_mask - logits_mask_c #select not sbiling
        
        exp_logits = torch.exp(logits) * logits_mask   
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
       
        loss_ns = loss.view(anchor_count, batch_size).mean()

        loss = 0.6*loss_sbling + 0.4*loss_ns

        return loss

class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 unet_module,
                 data_loader):
        # set encoder and unet
        self.enc_module = enc_module.to(tt.arg.device)
        self.unet_module = unet_module.to(tt.arg.device)

        if tt.arg.num_gpus > 1:
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=[0, 1], dim=0)
            self.unet_module = nn.DataParallel(self.unet_module, device_ids=[0, 1], dim=0)

            print('done!\n')

        # get data loader
        self.data_loader = data_loader

        # set module parameters
        self.module_params = list(self.enc_module.parameters()) + list(self.unet_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(params=self.module_params,
                                    lr=tt.arg.lr,
                                    weight_decay=tt.arg.weight_decay)

        # set loss
        self.node_loss = nn.NLLLoss()
        self.SupConLoss_intra = SupConLoss_intra().to(tt.arg.device)
        self.SupConLoss_inter = SupConLoss_inter().to(tt.arg.device)

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0
        self.best_step = 0

    def train(self):
        val_acc = self.val_acc

        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways * tt.arg.num_shots
        num_queries = tt.arg.num_queries
        num_samples = num_supports + num_queries

        time_start=time.time()

        # for each iteration
        for iter in range(self.global_step + 1, tt.arg.train_iteration + 1):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step = iter

            # load task data list
            [support_data,
             support_coarse_label,
             support_label,
             query_data,
             query_coarse_label,
             query_label] = self.data_loader['train'].get_task_batch(num_tasks=tt.arg.meta_batch_size,
                                                                     num_ways=tt.arg.num_ways,
                                                                     num_shots=tt.arg.num_shots,
                                                                     num_queries=int(tt.arg.num_queries /tt.arg.num_ways),
                                                                     seed=iter + tt.arg.seed)
            
            full_data = torch.cat([support_data, query_data], 1)
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)  # batch_size x num_samples x featdim

            #compute SupConLoss
            supconLoss = 0
            temperature = tt.arg.temperature/(1+np.log(self.global_step))
            que_data = full_data[:,num_supports:,:]
            que_data = que_data.contiguous().view(-1,que_data.size(-1))
            sup = full_data[:,:num_supports,:].view(full_data.size(0),tt.arg.num_ways,tt.arg.num_shots,-1)
            support_label = support_label.view(support_label.size(0),tt.arg.num_ways,-1)[:,:,0]
            support_coarse_label = support_coarse_label.view(support_coarse_label.size(0),tt.arg.num_ways,-1)[:,:,0]
            for i in range(tt.arg.num_shots):    
                sup_data = sup[:,:,i,:]
                sup_data = sup_data.contiguous().view(-1,sup_data.size(-1))
                features = torch.cat([que_data.unsqueeze(1), sup_data.unsqueeze(1)], dim=1)  #180*2*128
                # print(features.size())
                supconLoss += self.SupConLoss_intra(temperature,features,query_label,support_label,query_coarse_label,support_coarse_label)
                # supconLoss += self.SupConLoss_inter(temperature,features,query_coarse_label,support_coarse_label)
            supconLoss = supconLoss / tt.arg.num_shots
            l = np.array([range(tt.arg.num_ways)])
            support_label = torch.from_numpy(l.repeat(tt.arg.num_shots,axis=1).repeat(tt.arg.meta_batch_size,axis=0)).to(tt.arg.device)
            query_label = torch.from_numpy(l.repeat(int(tt.arg.num_queries /tt.arg.num_ways),axis=1).repeat(tt.arg.meta_batch_size,axis=0)).to(tt.arg.device) 
    
            # set as single data
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            # set init edge
            init_edge = full_edge.clone()  # batch_size x 2 x num_samples x num_samples
            init_edge[:, num_supports:, :] = 0.5
            init_edge[:, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, num_supports + i, num_supports + i] = 1.0


            # set as train mode
            self.enc_module.train()
            self.unet_module.train()

            # (1) encode data
            one_hot_label = self.one_hot_encode(tt.arg.num_ways, support_label.long())
            query_padding = (1 / tt.arg.num_ways) * torch.ones([full_data.shape[0]] + [num_queries] + [tt.arg.num_ways],
                                                               device=one_hot_label.device)
            one_hot_label = torch.cat([one_hot_label, query_padding], dim=1)
            full_data = torch.cat([full_data, one_hot_label], dim=-1)

            if tt.arg.transductive == True:
                # transduction
                if tt.arg.num_shots == 1:
                    full_node_out1,  full_node_out2,  full_node_out3 = self.unet_module(init_edge, full_data)
                else:
                    full_node_out1,  full_node_out2,  full_node_out3, full_node_out4= self.unet_module(init_edge, full_data)
            else:
                # non-transduction
                support_data = full_data[:, :num_supports]  # batch_size x num_support x featdim
                query_data = full_data[:, num_supports:]  # batch_size x num_query x featdim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1,
                                                                      1)  # batch_size x num_queries x num_support x featdim
                support_data_tiled = support_data_tiled.view(tt.arg.meta_batch_size * num_queries, num_supports,
                                                             -1)  # (batch_size x num_queries) x num_support x featdim
                query_data_reshaped = query_data.contiguous().view(tt.arg.meta_batch_size * num_queries, -1).unsqueeze(
                    1)  # (batch_size x num_queries) x 1 x featdim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped],
                                            1)  # (batch_size x num_queries) x (num_support + 1) x featdim

                input_edge_feat = 0.5 * torch.ones(tt.arg.meta_batch_size, num_supports + 1, num_supports + 1).to(
                    tt.arg.device)  # batch_size x (num_support + 1) x (num_support + 1)

                input_edge_feat[:, :num_supports, :num_supports] = init_edge[:, :num_supports,
                                                                   :num_supports]  # batch_size x (num_support + 1) x (num_support + 1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1,
                                                         1)  # (batch_size x num_queries) x (num_support + 1) x (num_support + 1)

                # 2. unet
                node_out = self.unet_module(input_edge_feat,
                                            input_node_feat)  # (batch_size x num_queries) x (num_support + 1) x num_classes
                node_out = node_out.view(tt.arg.meta_batch_size, num_queries, num_supports + 1,
                                         tt.arg.num_ways)  # batch_size x  num_queries x (num_support + 1) x num_classes
                full_node_out = torch.zeros(tt.arg.meta_batch_size, num_samples, tt.arg.num_ways).to(tt.arg.device)
                full_node_out[:, :num_supports, :] = node_out[:, :, :num_supports, :].mean(1)
                full_node_out[:, num_supports:, :] = node_out[:, :, num_supports:, :].squeeze(2)

            # 3. compute loss
            if tt.arg.num_shots == 1:
                query_node_out1, query_node_out2, query_node_out3 = full_node_out1[:,-num_queries:],full_node_out2[:,-num_queries:],full_node_out3[:,-num_queries:]
                query_node_out = 0.1*query_node_out1 + 0.4*query_node_out2 + 0.5*query_node_out3

            else:
                query_node_out1, query_node_out2, query_node_out3, query_node_out4 = full_node_out1[:,-num_queries:],full_node_out2[:,-num_queries:],full_node_out3[:,-num_queries:],full_node_out4[:,-num_queries:]
                query_node_out = 0.1*query_node_out1 + 0.1*query_node_out2 + 0.2*query_node_out3 + 0.6*query_node_out4
            
            node_pred = torch.argmax(query_node_out, dim=-1)
            node_accr = torch.sum(torch.eq(node_pred, full_label[:, num_supports:].long())).float() \
                        / node_pred.size(0) / num_queries
            node_loss = [self.node_loss(data.squeeze(1), label.squeeze(1).long()) for (data, label) in
                         zip(query_node_out.chunk(query_node_out.size(1), dim=1), full_label[:, num_supports:].chunk(full_label[:, num_supports:].size(1), dim=1))]
            node_loss = torch.stack(node_loss, dim=0)
            node_loss = torch.mean(node_loss)
            node_loss = node_loss + 0.4*supconLoss

            node_loss.backward()

            self.optimizer.step()

            # adjust learning rate
            self.adjust_learning_rate(optimizers=[self.optimizer],
                                      lr=tt.arg.lr,
                                      iter=self.global_step)

            # logging
            tt.log_scalar('train/loss', node_loss, self.global_step)
            tt.log_scalar('train/node_accr', node_accr, self.global_step)
            tt.log_scalar('train/time', time.time()-time_start, self.global_step)

            # evaluation
            if self.global_step % tt.arg.test_interval == 0:
                val_acc = self.eval(partition='val')

                is_best = 0

                if val_acc >= self.val_acc:
                    self.val_acc = val_acc
                    self.best_step = self.global_step
                    is_best = 1


                tt.log_scalar('val/best_accr', self.val_acc, self.global_step)
                tt.log_scalar('val/best_step', self.best_step, self.global_step)

                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'unet_module_state_dict': self.unet_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)

            tt.log_step(global_step=self.global_step)


    def eval(self,partition='test', log_flag=True):
        best_acc = 0

        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways * tt.arg.num_shots
        num_queries = tt.arg.num_queries
        num_samples = num_supports + num_queries

        query_node_accrs = []

        time_start_eval=time.time()

        # for each iteration
        for iter in range(tt.arg.test_iteration // tt.arg.test_batch_size):
            # load task data list
            [support_data,
             _,
             _,
             query_data,
             _,
             _] = self.data_loader[partition].get_task_batch(num_tasks=tt.arg.test_batch_size,
                                                                     num_ways=tt.arg.num_ways,
                                                                     num_shots=tt.arg.num_shots,
                                                                     num_queries=int(tt.arg.num_queries /tt.arg.num_ways),
                                                                     seed=iter)
         
            '''
            q0 = query_data[:,0,:].clone()
            q1 = query_data[:,1,:].clone()
            query_data[:, 1, :] = q0
            query_data[:, 0, :] = q1
            ql0 = query_label[:,0].clone()
            ql1 = query_label[:,1].clone()
            query_label[:, 1] = ql0
            query_label[:, 0] = ql1
            '''
            l = np.array([range(tt.arg.num_ways)])
            support_label = torch.from_numpy(l.repeat(tt.arg.num_shots,axis=1).repeat(tt.arg.test_batch_size,axis=0)).to(tt.arg.device)
            query_label = torch.from_numpy(l.repeat(int(tt.arg.num_queries /tt.arg.num_ways),axis=1).repeat(tt.arg.test_batch_size,axis=0)).to(tt.arg.device) 
    

            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)



            # set init edge
            init_edge = full_edge.clone()
            init_edge[:, num_supports:, :] = 0.5
            init_edge[:, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, num_supports + i, num_supports + i] = 1.0

            # set as eval mode
            self.enc_module.eval()
            self.unet_module.eval()

            # (1) encode data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)  # batch_size x num_samples x featdim
            one_hot_label = self.one_hot_encode(tt.arg.num_ways, support_label.long())
            query_padding = (1 / tt.arg.num_ways) * torch.ones([full_data.shape[0]] + [num_queries] + [tt.arg.num_ways],
                                                               device=one_hot_label.device)
            one_hot_label = torch.cat([one_hot_label, query_padding], dim=1)
            full_data = torch.cat([full_data, one_hot_label], dim=-1)

            if tt.arg.transductive == True:
                # transduction
                if tt.arg.num_shots == 1:
                    full_node_out1,  full_node_out2,  full_node_out3 = self.unet_module(init_edge, full_data)
                else:
                    full_node_out1,  full_node_out2,  full_node_out3, full_node_out4= self.unet_module(init_edge, full_data)
            else:
                # non-transduction
                support_data = full_data[:, :num_supports]  # batch_size x num_support x featdim
                query_data = full_data[:, num_supports:]  # batch_size x num_query x featdim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1,
                                                                      1)  # batch_size x num_queries x num_support x featdim
                support_data_tiled = support_data_tiled.view(tt.arg.meta_batch_size * num_queries, num_supports,
                                                             -1)  # (batch_size x num_queries) x num_support x featdim
                query_data_reshaped = query_data.contiguous().view(tt.arg.meta_batch_size * num_queries, -1).unsqueeze(
                    1)  # (batch_size x num_queries) x 1 x featdim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped],
                                            1)  # (batch_size x num_queries) x (num_support + 1) x featdim

                input_edge_feat = 0.5 * torch.ones(tt.arg.meta_batch_size, num_supports + 1, num_supports + 1).to(
                    tt.arg.device)  # batch_size x (num_support + 1) x (num_support + 1)

                input_edge_feat[:, :num_supports, :num_supports] = init_edge[:, :num_supports,
                                                                   :num_supports]  # batch_size x (num_support + 1) x (num_support + 1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1,
                                                         1)  # (batch_size x num_queries) x (num_support + 1) x (num_support + 1)

                # 2. unet
                node_out = self.unet_module(input_edge_feat,
                                            input_node_feat)  # (batch_size x num_queries) x (num_support + 1) x num_classes
                node_out = node_out.view(tt.arg.meta_batch_size, num_queries, num_supports + 1,
                                         tt.arg.num_ways)  # batch_size x  num_queries x (num_support + 1) x num_classes
                full_node_out = torch.zeros(tt.arg.meta_batch_size, num_samples, tt.arg.num_ways).to(tt.arg.device)
                full_node_out[:, :num_supports, :] = node_out[:, :, :num_supports, :].mean(1)
                full_node_out[:, num_supports:, :] = node_out[:, :, num_supports:, :].squeeze(2)

            # 3. compute loss
            if tt.arg.num_shots == 1:
                query_node_out1, query_node_out2, query_node_out3 = full_node_out1[:,-num_queries:],full_node_out2[:,-num_queries:],full_node_out3[:,-num_queries:]
                query_node_out = 0.1*query_node_out1 + 0.4*query_node_out2 + 0.5*query_node_out3
            else:
                query_node_out1, query_node_out2, query_node_out3, query_node_out4 = full_node_out1[:,-num_queries:],full_node_out2[:,-num_queries:],full_node_out3[:,-num_queries:],full_node_out4[:,-num_queries:]
                query_node_out = 0.1*query_node_out1 + 0.1*query_node_out2 + 0.2*query_node_out3 + 0.6*query_node_out4
                # query_node_out = query_node_out4
            node_pred = torch.argmax(query_node_out, dim=-1)
            node_accr = torch.sum(torch.eq(node_pred, full_label[:, num_supports:].long())).float() \
                        / node_pred.size(0) / num_queries

            query_node_accrs += [node_accr.item()]
            # print('acc:',node_accr.item(),'  time cost',time.time()-time_start_eval,'s')

        # logging
        if log_flag:
            # tt.log('---------------------------')
            tt.log_scalar('{}/node_accr'.format(partition), np.array(query_node_accrs).mean(), self.global_step)

            tt.log('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_node_accrs).mean() * 100,
                    np.array(query_node_accrs).std() * 100,
                    1.96 * np.array(query_node_accrs).std() / np.sqrt(
                        float(len(np.array(query_node_accrs)))) * 100))
            tt.log('---------------------------')

        return np.array(query_node_accrs).mean()

    def adjust_learning_rate(self, optimizers, lr, iter):
        new_lr = lr * (0.5 ** (int(iter / tt.arg.dec_lr)))

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge
        edge = torch.eq(label_i, label_j).float().to(tt.arg.device)

        return edge

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].to(tt.arg.device)

    def save_checkpoint(self, state, is_best):
        torch.save(state, './checkpoints/CSTS/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('./checkpoints/CSTS/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar',
                            './checkpoints/CSTS/{}/'.format(tt.arg.experiment) + 'model_best.pth.tar')

def set_exp_name():
    exp_name = 'D-{}'.format(tt.arg.dataset)
    exp_name += '_N-{}_K-{}_Q-{}'.format(tt.arg.num_ways, tt.arg.num_shots,tt.arg.num_queries)
    exp_name += '_B-{}_T-{}'.format(tt.arg.meta_batch_size,tt.arg.transductive)
    exp_name += '_P-{}_Un-{}'.format(tt.arg.pool_mode,tt.arg.unet_mode)
    exp_name += '_SEED-{}_2'.format(tt.arg.seed)

    return exp_name

if __name__ == '__main__':

    tt.arg.device = 'cuda:0' #if tt.arg.device is None else tt.arg.device
    tt.arg.dataset_root = 'data root'
    tt.arg.dataset = 'tiered' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 5 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_queries = tt.arg.num_ways*1
    tt.arg.num_supports = tt.arg.num_ways*tt.arg.num_shots
    tt.arg.transductive = True if tt.arg.transductive is None else tt.arg.transductive
    if tt.arg.transductive == False:
        tt.arg.meta_batch_size = 16
    else:
        tt.arg.meta_batch_size = 36
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1
    tt.arg.temperature = 25.5

    # model parameter related
    tt.arg.emb_size = 128
    tt.arg.in_dim = tt.arg.emb_size + tt.arg.num_ways

    tt.arg.pool_mode = 'kn' if tt.arg.pool_mode is None else tt.arg.pool_mode # 'way'/'support'/'kn'
    tt.arg.unet_mode = 'addold' if tt.arg.unet_mode is None else tt.arg.unet_mode # 'addold'/'noold'
    unet2_flag = False # the label of using unet2

    # confirm ks
    if tt.arg.num_shots == 1 and tt.arg.transductive == False:
        if tt.arg.pool_mode == 'support':  # 'support': pooling on support
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        elif tt.arg.pool_mode == 'kn':  # left close support node
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        else:
            print('wrong mode setting!!!')
            raise NameError('wrong mode setting!!!')
    elif tt.arg.num_shots == 5 and tt.arg.transductive == False:
        if tt.arg.pool_mode == 'way':  # 'way' pooling on support by  way
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'support'
            unet2_flag = True
        elif tt.arg.pool_mode == 'kn':
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way&kn'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'kn'
            unet2_flag = True
        else:
            print('wrong mode setting!!!')
            raise NameError('wrong mode setting!!!')

    elif tt.arg.num_shots == 1 and tt.arg.transductive == True:
        if tt.arg.pool_mode == 'support':  # 'support': pooling on support
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        elif tt.arg.pool_mode == 'kn':  # left close support node
            tt.arg.ks = [0.5, 0.3]  # 5->3->1
            # tt.arg.ks = [0.2]  # 5->1
        else:
            print('wrong mode setting!!!')
            raise NameError('wrong mode setting!!!')

    elif tt.arg.num_shots == 5 and tt.arg.transductive == True:
        if tt.arg.pool_mode == 'way':  # 'way' pooling on support by  way
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'support'
            unet2_flag = True
        elif tt.arg.pool_mode == 'kn':
            tt.arg.ks_1 = [0.2]  # 5->1
            mode_1 = 'way&kn'
            tt.arg.ks_2 = [0.6,0.5]  # 5->1 # supplementary pooling for fair comparing
            mode_2 = 'kn'
            unet2_flag = True
        else:
            print('wrong mode setting!!!')
            raise NameError('wrong mode setting!!!')

    else:
        print('wrong shot and T settings!!!')
        raise NameError('wrong shot and T settings!!!')


    # train, test parameters
    tt.arg.train_iteration = 100000 if tt.arg.dataset == 'mini' else 200000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 1000 # 5000
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 100

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 10000 if tt.arg.dataset == 'mini' else 20000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    tt.arg.experiment = set_exp_name() if tt.arg.experiment is None else tt.arg.experiment

    print(set_exp_name())

    # set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tt.arg.log_dir_user = tt.arg.log_dir if tt.arg.log_dir_user is None else tt.arg.log_dir_user
    tt.arg.log_dir = tt.arg.log_dir_user

    if not os.path.exists('./checkpoints/CSTS'):
        os.makedirs('./checkpoints/CSTS')
    if not os.path.exists('./checkpoints/CSTS/' + tt.arg.experiment):
        os.makedirs('./checkpoints/CSTS/' + tt.arg.experiment)

    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)  # extract features
    # enc_module = ResNet(emb_size=tt.arg.emb_size)

    if tt.arg.transductive == False:
        if unet2_flag == False:
            unet_module = Unet(tt.arg.ks, tt.arg.in_dim, tt.arg.num_ways, 1)
    else:
        if unet2_flag == False:
            unet_module = Unet(tt.arg.ks, tt.arg.in_dim, tt.arg.num_ways, tt.arg.num_queries)
        else:
            unet_module = Unet3(tt.arg.ks_1, tt.arg.ks_2, mode_1, mode_2, tt.arg.in_dim, tt.arg.num_ways, tt.arg.num_queries)

    if tt.arg.dataset == 'mini':
        train_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='val')
    elif tt.arg.dataset == 'tiered':
        train_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='val')
    else:
        print('Unknown dataset!')
        raise NameError('Unknown dataset!!!')

    data_loader = {'train': train_loader,
                   'val': valid_loader
                   }
    
    trainer = ModelTrainer(enc_module=enc_module,
                           unet_module=unet_module,
                           data_loader=data_loader)

    trainer.train()



