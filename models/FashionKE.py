import sys
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import random
random.seed(0)
import numpy as np
np.random.seed(0)

import datetime
import json


class OccCatAttrClassifier(nn.Module):
    def __init__(self, conf, pretrained_word_embed_weight=None, meta_embed=None, cat_noise_estimate=None, attr_noise_estimate_list=[]):
        super(OccCatAttrClassifier, self).__init__()
        self.conf = conf

        self.cat_noise_estimate = torch.from_numpy(cat_noise_estimate).to(device=conf["device"])
        self.attr_noise_estimate_list = [torch.from_numpy(i).to(device=conf["device"]) for i in attr_noise_estimate_list]

        cat_noise_shape = self.cat_noise_estimate.shape
        self.cat_noise_transition = nn.Linear(cat_noise_shape[0], cat_noise_shape[1], bias=False)
        with torch.no_grad():
            self.cat_noise_transition.weight.copy_(self.cat_noise_estimate)

        self.attr_noise_transitions = []
        for each in self.attr_noise_estimate_list:
            attr_noise_shape = each.shape
            tmp_attr_noise_transition = nn.Linear(attr_noise_shape[0], attr_noise_shape[1], bias=False)
            with torch.no_grad():
                tmp_attr_noise_transition.weight.copy_(each)
            self.attr_noise_transitions.append(tmp_attr_noise_transition)
        self.attr_noise_transitions = nn.ModuleList(self.attr_noise_transitions)

        # the basic CNN model, shared across all visual inputs
        pretrained_model = models.resnet18(pretrained=True)
        self.imageCNN = nn.Sequential(*list(pretrained_model.children())[:-1]) 

        self.occ_ttl_fea_len = 512
        self.context_fea_len = 512
        self.ori_context_fea_len = 300
        self.final_fea_len = 512
        self.cloth_ttl_fea_len = 512
        if self.conf["context"] in ["all", "visual_cooc"]:
            self.cloth_ttl_fea_len = 512 * 2
            self.occ_ttl_fea_len = 512*3

        self.visual_context_rnn = nn.GRU(512, conf["mid_layer"], num_layers=1, bidirectional=True)
        self.attr_context_rnn = nn.GRU(self.final_fea_len, conf["mid_layer"], num_layers=1, bidirectional=True)
        self.att_W = nn.Linear(self.final_fea_len, self.context_fea_len)
        self.att_w = nn.Linear(self.context_fea_len, 1)

        if self.conf["text"] == 1:
            self.occW = nn.Linear(self.occ_ttl_fea_len+self.context_fea_len, conf["mid_layer"])
        else:
            self.occW = nn.Linear(self.occ_ttl_fea_len, conf["mid_layer"])
        self.occ_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, conf["mid_layer"]),
            nn.Linear(conf["mid_layer"], conf["num_occasion"])
        )

        self.catW = nn.Linear(self.cloth_ttl_fea_len, conf["mid_layer"])
        self.cat_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, conf["mid_layer"]),
            nn.Linear(conf["mid_layer"], conf["num_cat"])
        )

        self.textEmbedding = nn.Embedding(self.conf["token_num"], self.conf["word_embed_size"])
        if pretrained_word_embed_weight is not None:
            self.textEmbedding.weight.data.copy_(torch.from_numpy(pretrained_word_embed_weight))

        D = self.conf["word_embed_size"] 
        Ci = 1
        Co = 32
        Ks = [2, 3, 4, 5]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.4)
        self.textW = nn.Linear(len(Ks)*Co, self.context_fea_len)#self.conf["text_rep_size"]))

        tmp_attr_cls = []
        tmp_attr_Ws = []
        tmp_attr_W1s = []
        for class_num in conf["attr_class_num"]:
            tmp_attr_Ws.append(nn.Linear(self.cloth_ttl_fea_len, conf["mid_layer"]))
            tmp_attr_W1s.append(nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(2*conf["mid_layer"], conf["mid_layer"])
            ))
            tmp_attr_cls.append(nn.Linear(conf["mid_layer"], class_num))
        self.attrWs = nn.ModuleList(tmp_attr_Ws)
        self.attrW1s = nn.ModuleList(tmp_attr_W1s)
        self.attr_classifiers = nn.ModuleList(tmp_attr_cls)


    def text_process(self, text):
        text_feature = self.textEmbedding(text)
        text_feature = text_feature.unsqueeze(1)
        text_feature = [F.relu(conv(text_feature)).squeeze(3) for conv in self.convs1]
        text_feature = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text_feature]
        text_feature = torch.cat(text_feature, 1)
        text_feature = self.dropout(text_feature)
        text_feature = self.textW(text_feature)

        return text_feature 


    def image_process(self, img):
        img_shape = img.shape
        img_fea = self.imageCNN(img)
        img_fea = img_fea.view([img_shape[0], 512])#self.imgfc1(img_fea.view([img_shape[0], 512]))

        return img_fea


    def predict(self, whole_img, imgs, season, age, gender, country, text):
        ori_shape = imgs.shape
        imgs = imgs.view([ori_shape[0]*ori_shape[1]] + list(ori_shape[2:]))
        img_feas = self.image_process(imgs)
        cloth_feas_s = img_feas.view(list(ori_shape[:2])+[512]) # [batch_size, num_cloth, 512]
        cloth_feas = cloth_feas_s
        ori_whole_img_fea = self.image_process(whole_img) # [batch_size, 512]
        whole_img_fea = ori_whole_img_fea
        text_fea = self.text_process(text)
        if self.conf["context"] in ["all", "visual_cooc"]:
            h0 = whole_img_fea.unsqueeze(1).permute(1, 0, 2)
            h0 = torch.cat([h0, h0], 0) 
            cloth_feas_lstm = cloth_feas_s.permute(1, 0, 2) # [num_cloth+1, batch_size, hidden_size]
            cloth_feas_lstm, cloth_context = self.visual_context_rnn(cloth_feas_lstm, h0)
            cloth_feas_lstm = cloth_feas_lstm.permute(1, 0, 2) # [batch_size, num_cloth+1, hidden_size]
            cloth_context = cloth_context.permute(1, 0, 2)
            cloth_feas = cloth_feas_lstm
            whole_img_fea = torch.cat([cloth_context[:,0,:], cloth_context[:,1,:], whole_img_fea], -1)

        if self.conf["text"] == 1:
            occ_fea = self.occW(torch.cat([whole_img_fea, text_fea], dim=-1))
        else:
            occ_fea = self.occW(whole_img_fea)
        cat_fea = self.catW(cloth_feas)# [20, 5, 512]
        cat_fea_input = cat_fea.contiguous().view(cat_fea.shape[0]*cat_fea.shape[1], -1) # [100, 512]
        cat_fea_input = cat_fea_input.unsqueeze(0)
             
        occ_res = self.occ_classifier(occ_fea)
        cat_res = self.cat_classifier(cat_fea)
        if self.conf["context"] in ["all", "task_ct"]:
            attr_feas = []
            attr_reses = []
            ori_shape = None

            for attrW in self.attrWs:
                tmp_fea = attrW(cloth_feas) 
                ori_shape = tmp_fea.shape # [20, 5, 512]
                tmp_fea = tmp_fea.view(ori_shape[0]*ori_shape[1], -1) # [batch, 5, 512] -> [100, 512]
                attr_feas.append(tmp_fea)
            attr_feas = torch.stack(attr_feas, dim=0) # [8, 100, 512]
            attr_feas = torch.cat([attr_feas, cat_fea_input], 0)

            attr_feas, _ = self.attr_context_rnn(attr_feas)
            for i in range(attr_feas.shape[0]-1):
                attr_fea = attr_feas[i, :, :].contiguous().view(ori_shape[0], ori_shape[1], -1) # [batch, 5, 1024]
                attr_fea = self.attrW1s[i](attr_fea)
                attr_reses.append(self.attr_classifiers[i](attr_fea))
            
        else:        
            attr_reses = []
            for attrW,  attr_classifier in zip(self.attrWs, self.attr_classifiers):
                tmp_fea = attrW(cloth_feas)
                tmp_res = attr_classifier(tmp_fea)
                attr_reses.append(tmp_res)

        return occ_res, cat_res, attr_reses


    def forward(self, whole_img, imgs, occ, attr_val, cats, season, age, gender, country, text):
        #import ipdb
        #ipdb.set_trace()
        occ_res, cat_res, attr_reses = self.predict(whole_img, imgs, season, age, gender, country, text)

        occ_true = occ
        occ_pred = occ_res
        occ_loss = F.cross_entropy(occ_res, occ, reduction="sum")

        cat_res_shape = cat_res.shape
        cat_pred = cat_res.view(cat_res_shape[0]*cat_res_shape[1], cat_res_shape[2])
        cat_true = cats.view(-1)

        if self.conf["noise_cancel_method"] == "forward":
            ori_cat_losses = F.cross_entropy(cat_pred, cat_true, reduction="none")
            #modified_cat_losses = F.cross_entropy(torch.mm(cat_pred, self.cat_noise_estimate), cat_true, reduction="none")
            modified_cat_losses = F.cross_entropy(self.cat_noise_transition(cat_pred), cat_true, reduction="none")
            ori_cat_losses = ori_cat_losses.view(cat_res_shape[0], cat_res_shape[1])
            modified_cat_losses = modified_cat_losses.view(cat_res_shape[0], cat_res_shape[1])

            cat_losses = [ori_cat_losses, modified_cat_losses]
        else:
            cat_losses = F.cross_entropy(cat_pred, cat_true, reduction="none")
            cat_losses = cat_losses.view(cat_res_shape[0], cat_res_shape[1]) # [batch, num_cloth]


        attr_losses = []
        ori_attr_losses = []
        modified_attr_losses = []
        attr_res_shape = attr_reses[0].shape
        for i, (attr_res, each_attr_noise_transition) in enumerate(zip(attr_reses, self.attr_noise_transitions)):
            attr_true = attr_val[:, :, i].contiguous().view(-1)
            attr_pred = attr_res.contiguous().view(attr_res_shape[0]*attr_res_shape[1], -1)

            #import ipdb
            #ipdb.set_trace()
            if self.conf["noise_cancel_method"] == "forward":
                #each_attr_noise_estimate = self.attr_noise_estimate_list[i]
                ori_attr_loss = F.cross_entropy(attr_pred, attr_true, reduction="none")
                #modified_attr_loss = F.cross_entropy(torch.mm(attr_pred, each_attr_noise_estimate), attr_true, reduction="none")
                modified_attr_loss = F.cross_entropy(each_attr_noise_transition(attr_pred), attr_true, reduction="none")
                ori_attr_losses.append(ori_attr_loss.view(attr_res_shape[0], attr_res_shape[1]))
                modified_attr_losses.append(modified_attr_loss.view(attr_res_shape[0], attr_res_shape[1]))
            else:
                attr_loss = F.cross_entropy(attr_pred, attr_true, reduction="none")
                attr_losses.append(attr_loss.view(attr_res_shape[0], attr_res_shape[1]))
        if len(attr_losses) != 0:
            attr_losses = torch.stack(attr_losses, dim=0).permute(1, 2, 0) # [batch, num_cloth, num_attr]
        else:
            ori_attr_losses = torch.stack(ori_attr_losses, dim=0).permute(1, 2, 0)
            modified_attr_losses = torch.stack(modified_attr_losses, dim=0).permute(1, 2, 0)
            attr_losses = [ori_attr_losses, modified_attr_losses]

        return occ_loss, cat_losses, attr_losses
