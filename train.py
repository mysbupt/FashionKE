import argparse
import datetime
import torch
torch.manual_seed(0)
from torch.autograd import Variable
from torch.optim import lr_scheduler

import sys
sys.path.insert(0, "./models")
from FashionKE import *
from utility import *

import json
import yaml
import numpy as np
np.random.seed(0)
import random
random.seed(0)
torch.cuda.manual_seed_all(0)
import pickle as pkl
from copy import deepcopy

from tensorboard_logger import configure, log_value
from scipy.spatial.distance import pdist, cdist, squareform

torch.multiprocessing.set_sharing_strategy('file_system')

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate", default=0.01, help="learning rate")
    parser.add_argument("-ld", "--lr_decay_interval", default=4, help="learning rate decay interval")
    parser.add_argument("-nr", "--noise_ratio", default=0.7, help="the noise ratio in the training set: 0, 0.1, 0.3, 0.5, 0.7")
    parser.add_argument("-ncm", "--noise_cancel_method", default="forward", help="which noise cancelling method to use: forward or none")
    parser.add_argument("-nbeta", "--noise_loss_beta", default=0.5, help="weight hyperparameter for modified noise")
    parser.add_argument("-ct", "--context", default="all", help="which context to use: visual_cooc, task_ct, all, none")
    parser.add_argument("-i", "--info", default="", help="some comments")
    parser.add_argument("-l", "--loss", default="all", help="which loss to use: cat, attr, all")
    parser.add_argument("-t", "--text", default=0, help="whether to use text")
    args = parser.parse_args()
    return args


def train_fashion_recognition(conf):
    dataset = FashionData(conf)
    train_dataloader = dataset.train_dataloader
    if conf["noise_cancel_method"] == "forward":
        train_dataloader = dataset.train_dataloader_clean_noise

    conf["num_occasion"] = 10
    conf["num_cat"] = len(dataset.cat_code)
    conf["num_attr"] = len(dataset.attr_code)
    conf["num_country"] = len(dataset.country_code) + 1

    conf["attr_class_num"] = [0] * conf["num_attr"]
    conf["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for attr, code in dataset.attr_code.items():
        conf["attr_class_num"][code] = len(dataset.attr_val_code[attr])

    if not os.path.isdir(conf["checkpoint"]):
        os.mkdir(conf["checkpoint"])
    if not os.path.isdir(conf["model_save_path"]):
        os.mkdir(conf["model_save_path"])

    model = OccCatAttrClassifier(conf, dataset.word_embedding, dataset.meta_embed, dataset.cat_noise_estimate, dataset.attr_noise_estimate_list)
    model.to(device=conf["device"])

    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_file_name = "Loss_%s__NCM_%s__LR_%.2f__LDI_%d__NR_%.2f__Beta_%.2f__Ctx_%s__Text_%d__%s__%s" %(conf["loss"], conf["noise_cancel_method"], conf["lr"], conf["lr_decay_interval"], conf["noise_ratio"], conf["noise_loss_beta"], conf["context"], conf["text"], conf["info"], start_time)
    configure(os.path.join(conf["checkpoint"], log_file_name), flush_secs=5)

    # init optimizer
    lr = conf["lr"]
    weight_decay = conf["weight_decay"]
    params = [
        {'params': model.imageCNN.parameters(), 'lr': 0.5*lr},
        {'params': model.catW.parameters(), 'lr': lr},
        {'params': model.occW.parameters(), 'lr': lr},
        {'params': model.attrWs.parameters(), 'lr': lr},
        {'params': model.attrW1s.parameters(), 'lr': lr},
        {'params': model.occ_classifier.parameters(), 'lr': lr},
        {'params': model.cat_classifier.parameters(), 'lr': lr},
        {'params': model.attr_classifiers.parameters(), 'lr': lr},
        {'params': model.convs1.parameters(), 'lr': lr},
        {'params': model.textW.parameters(), 'lr':  lr},
        {'params': model.attr_context_rnn.parameters(), 'lr': lr},
        {'params': model.visual_context_rnn.parameters(), 'lr': lr},
        {'params': model.attr_noise_transitions.parameters(), 'lr': 0.001 * lr},
        {'params': model.cat_noise_transition.parameters(), 'lr': 0.001 * lr}
    ]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=conf["momentum"])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=int(conf["lr_decay_interval"]*len(train_dataloader)), gamma=conf["lr_decay_gamma"])

    best_occ_acc = 0.0
    best_cat_acc = 0.0
    best_attr_val_acc = 0.0

    loss_print, occ_loss_print, attr_ttl_loss_print, cat_loss_print = [[] for i in range(4)]
    attr_loss_print = [[] for i in range(len(dataset.attr_code))]

    for epoch in range(conf["num_epoches"]):
        for batch_cnt, batch in enumerate(train_dataloader): 
            step = int(batch_cnt + epoch*len(train_dataloader) + 1)

            model.to(device=conf["device"])
            model.train(True)
            exp_lr_scheduler.step() #adjust learning rate
            optimizer.zero_grad()

            if conf["noise_cancel_method"] == "forward":
                # [batch_cnt, 2, 3, 224, 224]
                whole_img = Variable(torch.cat([batch[0][:, 0, :, :, :], batch[0][:, 1, :, :, :]], dim=0)).to(device=conf["device"])
                # [batch_cnt, 2, max_num_cloth, 3, 224, 224]
                imgs = Variable(torch.cat([batch[1][:, 0, :, :, :, :], batch[1][:, 1, :, :, :, :]], dim=0)).to(device=conf["device"])
                # [batch_cnt, 2]
                occ, season, country = [Variable(torch.cat([each[:, 0], each[:, 1]], dim=0).squeeze(-1)).to(device=conf["device"]) for each in [batch[2], batch[8], batch[11]]]
                # [batch_cnt, 2, max_num_cloth, attr_num]
                attr_val, attr_val_masks = [Variable(torch.cat([each[:, 0, :, :], each[:, 1, :, :]], dim=0)).to(device=conf["device"]) for each in [batch[3], batch[5]]]
                # [batch_cnt, 2, max_num_cloth]
                cats, cat_masks, age, gender = [Variable(torch.cat([each[:, 0, :], each[:, 1, :]], dim=0)).to(device=conf["device"]) for each in [batch[4], batch[6], batch[9], batch[10]]]
                # [batch_cnt, 2, sent_len(16)]
                text = Variable(torch.cat([batch[12][:, 0, :], batch[12][:, 1, :]], dim=0)).to(device=conf["device"])
            else:
                whole_img = Variable(batch[0]).to(device=conf["device"])
                imgs = Variable(batch[1]).to(device=conf["device"])
                occ, season, country = [Variable(each.squeeze(-1)).to(device=conf["device"]) for each in [batch[2], batch[8], batch[11]]]
                attr_val, attr_val_masks = [Variable(each).to(device=conf["device"]) for each in [batch[3], batch[5]]]
                cats, cat_masks, age, gender = [Variable(each).to(device=conf["device"]) for each in [batch[4], batch[6], batch[9], batch[10]]]
                text = Variable(batch[12]).to(device=conf["device"])

            occ_loss, cat_losses, attr_losses = model(whole_img, imgs, occ, attr_val, cats, season, age, gender, country, text)

            occ_loss /= conf["batch_size"]
            if conf["noise_cancel_method"] == "forward":
                ori_cat_losses, modified_cat_losses = cat_losses
                ori_attr_losses, modified_attr_losses = attr_losses

                clean_noise_cat_loss = ori_cat_losses * cat_masks
                clean_cat_loss = torch.sum(clean_noise_cat_loss[:conf["batch_size_clean"], :]) / torch.sum(cat_masks[:conf["batch_size_clean"], :])

                modified_cat_losses = modified_cat_losses * cat_masks
                modified_cat_loss = torch.sum(modified_cat_losses[conf["batch_size_clean"]:, :]) / torch.sum(cat_masks[conf["batch_size_clean"]:, :])

                cat_loss = clean_cat_loss + conf["noise_loss_beta"] * modified_cat_loss

                # attr_losses, attr_val_masks: [batch, num_cloth, num_attrs] [20, 5, 10]
                per_attr_losses = []
                ori_attr_losses = ori_attr_losses * attr_val_masks
                modified_attr_losses = modified_attr_losses * attr_val_masks
                num_valid_attr = 0
                for attr, code in sorted(dataset.attr_code.items(), key=lambda i: i[1]):
                    denorm = torch.sum(attr_val_masks[:conf["batch_size_clean"], :, code])
                    if denorm == 0:
                        clean_per_attr_loss = torch.sum(ori_attr_losses[:conf["batch_size_clean"], :, code]) 
                    else:
                        clean_per_attr_loss = torch.sum(ori_attr_losses[:conf["batch_size_clean"], :, code]) / denorm

                    denorm = torch.sum(attr_val_masks[conf["batch_size_clean"]:, :, code])
                    if denorm == 0:
                        modified_attr_loss = torch.sum(modified_attr_losses[conf["batch_size_clean"]:, :, code]) 
                    else:
                        modified_attr_loss = torch.sum(modified_attr_losses[conf["batch_size_clean"]:, :, code]) / denorm 
                        num_valid_attr += 1

                    per_attr_loss = clean_per_attr_loss + conf["noise_loss_beta"] * modified_attr_loss

                    per_attr_losses.append(per_attr_loss)

                attr_ttl_loss = torch.sum(torch.stack(per_attr_losses, dim=0)) / num_valid_attr 
                if conf["loss"] == "cat":
                    loss = cat_loss
                if conf["loss"] == "attr":
                    loss = attr_ttl_loss
                if conf["loss"] == "all":
                    loss = torch.sum(torch.stack([occ_loss, cat_loss] + per_attr_losses, dim=0)) / (num_valid_attr + 2)
            else:
                cat_loss = torch.sum(cat_losses * cat_masks)
                cat_loss = cat_loss / torch.sum(cat_masks)

                per_attr_losses = []
                attr_losses = attr_losses * attr_val_masks
                num_valid_attr = 0
                for attr, code in sorted(dataset.attr_code.items(), key=lambda i: i[1]):
                    denorm = torch.sum(attr_val_masks[:, :, code])
                    if denorm == 0:
                        per_attr_losses.append(torch.sum(attr_losses[:, :, code]))
                    else:
                        num_valid_attr += 1
                        per_attr_losses.append(torch.sum(attr_losses[:, :, code]) / denorm)
                attr_ttl_loss = torch.sum(torch.stack(per_attr_losses, dim=0)) / num_valid_attr 

                if conf["loss"] == "cat":
                    loss = cat_loss
                if conf["loss"] == "attr":
                    loss = attr_ttl_loss
                if conf["loss"] == "all":
                    loss = torch.sum(torch.stack([occ_loss, cat_loss] + per_attr_losses, dim=0)) / (num_valid_attr + 2)

            log_value("occ_loss", occ_loss.item(), step)
            log_value("cat_loss", cat_loss.item(), step)
            log_value("loss", loss.item(), step)

            occ_loss_print.append(occ_loss.item())
            loss_print.append(loss.item())

            log_value("attr_ttl_loss", attr_ttl_loss.item(), step)
            for attr, code in sorted(dataset.attr_code.items(), key=lambda i: i[1]):
                log_value("%s_loss" %(attr), per_attr_losses[code], step)
            attr_ttl_loss_print.append(attr_ttl_loss.item())
            for i, each_attr_loss in enumerate(per_attr_losses):
                attr_loss_print[i].append(each_attr_loss)
            
            cat_loss_print.append(cat_loss.item())
            if (batch_cnt+1) % 10 == 0:
                each_attr_loss = []
                for attr, code in sorted(dataset.attr_code.items(), key=lambda i: i[1]):
                    each_attr_loss.append("%s:%f.4" %(attr, mean(attr_loss_print[code])))
                print("epoch/batch/total:%d/%d/%d,loss:%f.4,cat_loss:%f.4,occ_loss:%f.4,attr_loss:%f.4" %(epoch, batch_cnt, len(train_dataloader), mean(loss_print), mean(cat_loss_print), mean(occ_loss_print), mean(attr_ttl_loss_print)))
                loss_print, occ_loss_print, attr_ttl_loss_print, cat_loss_print = [[] for i in range(4)]
                attr_loss_print = [[] for i in range(len(dataset.attr_code))]

            loss.backward()
            optimizer.step()

            if (batch_cnt+1) % int(conf["test_interval"]*len(train_dataloader)) == 0:
                #import ipdb
                #ipdb.set_trace()
                print("\n\nstart to test, context: %s, loss: %s" %(conf["context"], conf["loss"]))
                model.eval()
                occ_acc, cat_acc, attr_val_acc = test_fashion_recognition(model, dataset, conf)

                attr_val_ttl_acc = sum(attr_val_acc)/len(attr_val_acc)

                log_value("occ_acc", occ_acc, step)
                log_value("cat_acc", cat_acc, step)
                log_value("attr_val_acc", attr_val_ttl_acc, step)

                each_attr_acc = []
                for attr, code in sorted(dataset.attr_code.items(), key=lambda i: i[1]):
                    log_value("%s_acc" %(attr), attr_val_acc[code], step)
                    each_attr_acc.append("%s:%f" %(attr, attr_val_acc[code]))

                print("occ_acc:%f,cat_acc:%f,attr_val_tll_acc:%f" %(occ_acc, cat_acc, attr_val_ttl_acc))

                if occ_acc > best_occ_acc and cat_acc > best_cat_acc and attr_val_ttl_acc > best_attr_val_acc:
                    best_occ_acc = occ_acc
                    best_cat_acc = cat_acc 
                    best_attr_val_acc = attr_val_ttl_acc
                    print("achieve best performance, save model.")
                    print("best_occ: %f, best_cat: %f, best_attr: %f" %(best_occ_acc, best_cat_acc, best_attr_val_acc))
                    model_save_path = os.path.join(conf["model_save_path"], log_file_name)
                    torch.save(model.state_dict(), model_save_path)


def mean(num_list):
    return sum(num_list) / float(len(num_list))


def test_fashion_recognition(model, dataset, conf):
    occ, attr_val, cat, attr_val_mask, cat_mask = [], [], [], [], []
    occ_res, attr_val_res, cat_res = [], [], []
    for batch_cnt, batch in enumerate(dataset.test_dataloader):
        #model.to(device=conf["device"])

        whole_img = batch[0].to(device=conf["device"])
        imgs = batch[1].to(device=conf["device"])
        occs, attr_vals, cats, attr_val_masks, cat_masks, _ = batch[2:8]
        season, age, gender, country, text = [each.to(device=conf["device"]) for each in batch[8:13]]

        occ.append(occs)
        attr_val.append(attr_vals)
        cat.append(cats)
        attr_val_mask.append(attr_val_masks)
        cat_mask.append(cat_masks)

        occ_reses, cat_reses, attr_val_reses = model.predict(whole_img, imgs, season, age, gender, country, text)
        occ_res.append(occ_reses.data.cpu())
        cat_res.append(cat_reses.data.cpu())

        # x: [batch, num_cloth, num_vals_for_each_attr]
        tmp_attr_val = torch.stack([torch.argmax(x, dim=-1) for x in attr_val_reses], dim=0) #[num_attr, batch, num_cloth]
        x = tmp_attr_val.shape
        tmp_attr_val = tmp_attr_val.permute(1, 2, 0) #[batch, num_cloth, num_attr]
        attr_val_res.append(tmp_attr_val.data.cpu())

    #import ipdb
    #ipdb.set_trace()
    occ_res = np.argmax(torch.cat(occ_res, dim=0).numpy(), axis=-1) # [test_size]
    cat_res = np.argmax(torch.cat(cat_res, dim=0).numpy(), axis=-1) # [test_size, num_cloth]

    occ = torch.cat(occ, dim=0).numpy() # [test_size]
    cat = torch.cat(cat, dim=0).numpy() # [test_size, num_cloth]

    cat_mask = torch.cat(cat_mask, dim=0).numpy() # [test_size, num_cloth]

    occ_acc = np.sum(np.equal(occ, occ_res)) / occ.shape[0]
    cat_acc = np.sum(np.equal(cat, cat_res) * cat_mask) / np.sum(cat_mask)

    attr_acc = []
    cat_res_index = cat_res.astype(np.int)
    tmp_attr_val_res = torch.cat(attr_val_res, dim=0).numpy() # [test_size, num_cloth, num_attr]
    tmp_attr_val = torch.cat(attr_val, dim=0).numpy() # [test_size, num_cloth, num_attr]
    tmp_attr_val_mask = torch.cat(attr_val_mask, dim=0).numpy() # [test_size, num_cloth, num_attr]

    np.save(conf["result_path"] + "/attr_val_res", tmp_attr_val_res)
    np.save(conf["result_path"] + "/occ_res", occ_res)
    np.save(conf["result_path"] + "/cat_res", cat_res)
    np.save(conf["result_path"] + "/cat_mask", cat_mask)

    for attr, code in sorted(dataset.attr_code.items(), key=lambda i: i[1]):
        if conf["loss"] in ["cat_attr", "all"]:
            #import ipdb
            #ipdb.set_trace()
            # cat_res shape: [test_size, num_cloth]
            # dataset.cat_attr_mask shape: [num_cat, num_attr]
            # tmp_cat_attr_mask shape: [test_size, num_cloth]
            tmp_cat_attr_mask = dataset.cat_attr_mask[cat_res_index, code * np.ones(cat_res_index.shape, dtype=np.int)]
            each_attr_acc = np.sum(np.equal(tmp_attr_val[:, :, code], tmp_attr_val_res[:, :, code]) * tmp_attr_val_mask[:, :, code] * tmp_cat_attr_mask) / np.sum(tmp_attr_val_mask[:, :, code])
        else:
            each_attr_acc = np.sum(np.equal(tmp_attr_val[:, :, code], tmp_attr_val_res[:, :, code]) * tmp_attr_val_mask[:, :, code]) / np.sum(tmp_attr_val_mask[:, :, code])
        attr_acc.append(each_attr_acc)
    
    return occ_acc, cat_acc, attr_acc


def main():
    conf = yaml.load(open("./config.yaml"))

    #paras = get_cmd()
    assert conf["noise_ratio"] in [0.0, 0.1, 0.3, 0.5, 0.7]

    train_fashion_recognition(conf)


if __name__ == "__main__":
    main() 
