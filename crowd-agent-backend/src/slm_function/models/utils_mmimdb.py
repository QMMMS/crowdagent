# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from collections import Counter

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
import csv
from tqdm import tqdm

POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[args["num_image_embeds"]])

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, labels, max_seq_length):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.labels = labels
        self.n_classes = len(labels)
        self.max_seq_length = max_seq_length

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.tokenizer.encode(self.data[index]["text"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[: self.max_seq_length]

        label = torch.zeros(self.n_classes)
        label[[self.labels.index(tgt) for tgt in self.data[index]["label"] if tgt in self.labels]] = 1

        image = Image.open(os.path.join(self.data_dir, self.data[index]["image"])).convert("RGB")
        image = self.transforms(image)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        
        for row in self.data:
            flitered_labels = []
            for label in row["label"]:
                if label in self.labels:
                    flitered_labels.append(label)
            label_freqs.update(flitered_labels)
        return label_freqs


def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])


    raw_index_list = [row["raw_index"] for row in batch]

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor, raw_index_list


class CsvDataset(Dataset):
    def __init__(self, csv_data_path, tokenizer, transforms, image_base_dir, max_seq_length, num_labels=2, mode=""):
        with open(csv_data_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.data = [row for row in reader]

        self.image_base_dir = image_base_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transforms = transforms
        self.mode = mode
        self.num_labels = num_labels

        self.sentence_list = []
        self.image_list = []
        self.label_list = []
        self.image_start_token_list = []
        self.image_end_token_list = []
        self.raw_index_list = []
        self.raw_text_list = []
        self.raw_image_list = []

        for row in tqdm(self.data):
            raw_index = row[0]
            label = int(row[1])
            text = row[2].lower()
            image_path = os.path.join(self.image_base_dir, row[3])
            
            if self.mode == "vsnli":
                premise, hypothesis = text.split("#####")
                sentence = torch.LongTensor(self.tokenizer.encode(premise,hypothesis,add_special_tokens=True))
            else :
                sentence = torch.LongTensor(self.tokenizer.encode(text, add_special_tokens=True))

            start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
            sentence = sentence[: self.max_seq_length]
            tensor_label = torch.zeros(self.num_labels)
            tensor_label[label] = 1
            image = Image.open(image_path).convert("RGB")
            image = self.transforms(image)

            self.raw_index_list.append(raw_index)
            self.label_list.append(tensor_label)
            self.sentence_list.append(sentence)
            self.image_list.append(image)
            self.image_start_token_list.append(start_token)
            self.image_end_token_list.append(end_token)
            self.raw_text_list.append(text)
            self.raw_image_list.append(image_path)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        start_token = self.image_start_token_list[index]
        end_token = self.image_end_token_list[index]
        sentence = self.sentence_list[index]
        image = self.image_list[index]
        label = self.label_list[index]
        raw_index = self.raw_index_list[index]
        raw_text = self.raw_text_list[index]
        raw_image = self.raw_image_list[index]

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
            "raw_index": raw_index,
            "raw_text": raw_text,
            "raw_image": raw_image,
        }


    # def get_label_frequencies(self):
    #     label_freqs = Counter()
    #     for row in self.data:
    #         label_freqs.update([int(row[1])])
    #     return label_freqs


def get_mmimdb_labels():
    return [
        1,
        0,
    ]


def get_labels_from_label_num(label_num):
    labels = []
    for i in range(label_num):
        labels.append(i)
    return labels


def get_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )