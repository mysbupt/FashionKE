#coding: utf-8

import os
import math
import json
import random
random.seed(0)
import numpy as np
np.random.seed(0)


from PIL import Image

import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class InstagramData(Dataset):
    def __init__(self, conf, train_ids, data_path, img_meta_map, id_text_map, occ_code, attr_code, attr_val_code, cat_code, country_code, labelled_pool, max_cloth_num):
        self.conf = conf
        self.train_ids = train_ids
        self.img_meta_map = img_meta_map
        self.data_path = data_path
        self.max_cloth_num = max_cloth_num
        self.attr_code = attr_code
        self.attr_val_code = attr_val_code
        self.cat_code = cat_code
        self.occasion_code = occ_code
        self.country_code = country_code
        self.id_text_map = id_text_map
        self.labelled_pool = labelled_pool

        self.img_size = (224, 224)
        self.image_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def crop_imgs(self, ori_img, bbox):
        img = ori_img.crop(bbox)
        res = self.image_transform(img)

        return res


    def __len__(self):
        return len(self.train_ids)


    def get_one_img(self, img_id):
        img_path = self.data_path + "/images/" + img_id + ".jpg"
        ori_img = Image.open(img_path).convert("RGB")
        whole_img = self.image_transform(ori_img)

        occasion = self.occasion_code[self.img_meta_map[img_id]["occasion"]]
        #clothes cat/attr labels
        attr_val_masks = torch.zeros([self.max_cloth_num, len(self.attr_code)])
        attr_val = torch.zeros([self.max_cloth_num, len(self.attr_code)], dtype=torch.long)
        categories = torch.zeros(self.max_cloth_num, dtype=torch.long) 
        cat_masks = torch.zeros(self.max_cloth_num)

        season = int((self.img_meta_map[img_id]["month"] - 1) / 3)
        if season >= 4 or season < 0:
            print(self.img_meta_map[img_id]["month"], season, img_id)
            exit()
        ages = torch.zeros(self.max_cloth_num, dtype=torch.long)
        genders = torch.zeros(self.max_cloth_num, dtype=torch.long)
        country = self.img_meta_map[img_id]["country"]
        if country == "" or country is None or country not in self.country_code:
            country = 0
        else:
            country = self.country_code[country]

        label_mask = torch.FloatTensor([1.0]) if img_id in self.labelled_pool else torch.FloatTensor([0.0])

        imgs = torch.zeros([self.max_cloth_num, 3, self.img_size[0], self.img_size[1]])

        img_loc = []
        for i, each in enumerate(self.img_meta_map[img_id]["cloth_body_face"]):
            # body_center_x, cloth_center_y
            body_center_x = each["body"][0] + (each["body"][2] - each["body"][0]) / 2.0
            cloth_center_y = each["cloth"]["box"][1] + (each["cloth"]["box"][3] - each["cloth"]["box"][1]) / 2.0
            tmp_loc = [i, body_center_x, cloth_center_y]
            img_loc.append(tmp_loc)
        
        
        #for i, each in enumerate(self.img_meta_map[img_id]["cloth_body_face"]):
        for x in sorted(img_loc, key=lambda i: (i[1], i[2])):
            # sort the clothes in the order of location, left->right (by different bodies' center), top->down (by different clothes' center)
            i = x[0]
            each = self.img_meta_map[img_id]["cloth_body_face"][i]

            age = each["age"]
            if age < 30:
                ages[i] = 0 # young
            elif age < 50:
                ages[i] = 1 # middle-aged
            else:
                ages[i] = 2 # old

            if each["gender"] == "M":
                genders[i] = 0 # men
            else:
                genders[i] = 1 # women

            cloth_bbox = each["cloth"]["box"]
            cloth_img = self.crop_imgs(ori_img, cloth_bbox)
            imgs[i] = cloth_img

            for k, v in each["cloth"].items():
                if ":" not in k:
                    continue
                else:
                    attr, val = k.split(":") 
                    if attr in self.attr_code:
                        code = self.attr_code[attr]
                        attr_val_masks[i][code] = 1
                        attr_val[i][code] = self.attr_val_code[attr][val]
                        
            categories[i] = self.cat_code[each["cloth"]["category"]]
            cat_masks[i] = 1

        text = torch.LongTensor(self.id_text_map[img_id]["ids"])
        return [whole_img, imgs, occasion, attr_val, categories, attr_val_masks, cat_masks, label_mask, season, ages, genders, country, text]


    def __getitem__(self, idx):
        img_id = self.train_ids[idx]
        return self.get_one_img(img_id)


class InstagramDataNew(InstagramData):
    def __init__(self, conf, train_ids, data_path, img_meta_map, id_text_map, occ_code, attr_code, attr_val_code, cat_code, country_code, labelled_pool, max_cloth_num):
        super().__init__(conf, train_ids, data_path, img_meta_map, id_text_map, occ_code, attr_code, attr_val_code, cat_code, country_code, labelled_pool, max_cloth_num)
        self.train_ids_clean, self.train_ids_noise = self.get_clean_noise_ids(train_ids, labelled_pool)


    def get_clean_noise_ids(self, train_ids, all_clean_ids):
        train_ids_clean = set()
        train_ids_noise = set()
        for id_ in train_ids:
            if id_ in all_clean_ids:
                train_ids_clean.add(id_)
            else:
                train_ids_noise.add(id_)

        return sorted(list(train_ids_clean)), train_ids_noise


    def __len__(self):
        return len(self.train_ids_clean)


    def __getitem__(self, idx):
        clean_img_id = self.train_ids_clean[idx]
        noise_img_id = random.sample(self.train_ids_noise, 1)[0]

        clean_sample = self.get_one_img(clean_img_id)
        noise_sample = self.get_one_img(noise_img_id)

        res = []
        for c, n in zip(clean_sample, noise_sample):
            if isinstance(c, int):
                res.append(torch.LongTensor([[c], [n]]))
            else:
                res.append(torch.stack([c, n], dim=0))

        return res


class FashionData():
    def __init__(self, conf):
        self.conf = conf
        train_ids = json.load(open(conf["data_path"] + "/train_ids_%.1f.json" %(conf["noise_ratio"])))
        test_ids = json.load(open(conf["data_path"] + "/test_ids.json"))

        self.labelled_pool = set(json.load(open(conf["data_path"] + "/labelled_pool.json")))

        train_ids = train_ids[:len(train_ids)-len(train_ids)%conf["batch_size"]]
        test_ids = test_ids[:len(test_ids)-len(test_ids)%conf["batch_size"]]
        img_meta_map = json.load(open(conf["data_path"] + "/cloth_body_face_meta.json"))

        self.word_embedding = np.load(conf["data_path"] + "/word_embeds.npy")
        self.meta_embed = self.get_meta_embed()
        self.id_text_map = json.load(open(conf["data_path"] + "/id_text_preprocessed.json"))
       
        self.attr_code = json.load(open(conf["data_path"] + "/code_attr.json"))
        self.attr_val_code = json.load(open(conf["data_path"] + "/code_attr_val.json"))
        self.cat_code = json.load(open(conf["data_path"] + "/code_cat.json"))
        self.occ_code = json.load(open(conf["data_path"] + "/code_occasion.json"))
        self.country_code = json.load(open(conf["data_path"] + "/code_country.json"))
        self.cat_noise_estimate, self.attr_noise_estimate_list = self.get_noise_estimate()

        self.cat_attr_mask = self.get_cat_attr_mask()

        self.batch_size = conf["batch_size"]
        self.train_set = InstagramData(conf, train_ids, conf["data_path"], img_meta_map, self.id_text_map, self.occ_code, self.attr_code, self.attr_val_code, self.cat_code, self.country_code, self.labelled_pool, conf["max_cloth_num"])
        self.train_dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=10)

        self.train_set_clean_noise = InstagramDataNew(conf, train_ids, conf["data_path"], img_meta_map, self.id_text_map, self.occ_code, self.attr_code, self.attr_val_code, self.cat_code, self.country_code, self.labelled_pool, conf["max_cloth_num"])
        self.train_dataloader_clean_noise = DataLoader(self.train_set_clean_noise, batch_size=int(self.batch_size/2), shuffle=True, num_workers=10)

        self.test_set = InstagramData(conf, test_ids, conf["data_path"], img_meta_map, self.id_text_map, self.occ_code, self.attr_code, self.attr_val_code, self.cat_code, self.country_code, self.labelled_pool, conf["max_cloth_num"])
        self.test_dataloader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=10)


    def get_meta_embed(self):
        res = {}
        data = json.load(open(self.conf["data_path"] + "/meta_embed_list.json"))
        for meta, embed in data.items():
            res[meta] = np.array(embed)

        return res


    def get_cat_attr_mask(self):
        cat_attr_mask = np.zeros((len(self.cat_code), len(self.attr_code)), dtype=np.float)
        cat_attr_map = json.load(open(self.conf["data_path"] + "/clothes_category_attribute_value.json"))
        for cat, res in cat_attr_map.items():
            if cat not in self.cat_code:
                continue
            cat_code = self.cat_code[cat]
            for each in res:
                for attr, vals in each.items():
                    if attr not in self.attr_code:
                        continue
                    attr_code = self.attr_code[attr]
                    cat_attr_mask[cat_code][attr_code] = 1

        return cat_attr_mask
        

    def get_noise_estimate(self):
        cat_confuse = json.load(open(self.conf["data_path"] + "/category_confuse_mat.json"))
        attr_confuse = json.load(open(self.conf["data_path"] + "/attribute_confuse_mat.json"))

        def convert_dict2np(trans_type, cat_code, cat_confuse):
            cat_confuse_np = np.zeros((len(cat_code), len(cat_code)), dtype=np.float32)
            for grd, res in cat_confuse.items():
                grd = cat_code[grd]
                for pred, cnt in res.items():
                    pred = cat_code[pred]
                    if trans_type == "noise2clean":
                        cat_confuse_np[pred][grd] = cnt
                    if trans_type == "clean2noise":
                        cat_confuse_np[grd][pred] = cnt
            for i in range(cat_confuse_np.shape[0]):
                cat_confuse_np[i] /= np.sum(cat_confuse_np[i])
            return cat_confuse_np

        cat_confuse_np = convert_dict2np(self.conf["trans_type"], self.cat_code, cat_confuse)

        attr_confuse_np_list = []
        for attr, code in sorted(self.attr_code.items(), key=lambda i: i[1]):
            confuse_mat = attr_confuse[attr]
            each_attr_confuse_np = convert_dict2np(self.conf["trans_type"], self.attr_val_code[attr], confuse_mat)
            attr_confuse_np_list.append(each_attr_confuse_np)

        return cat_confuse_np, attr_confuse_np_list
