#! /usr/bin/env python3
# coding=utf-8

# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import csv
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange

from pplm_classification_head import ClassificationHead
from transformers import GPT2LMHeadModel, GPT2Tokenizer


torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 100


class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""
    def __init__(self, class_size, pretrained_model="gpt2-medium", cached_mode=False, device="cpu"):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.embed_size = self.encoder.transformer.config.hidden_size
        self.classifier_head = ClassificationHead(class_size=class_size, embed_size=self.embed_size)
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(1, 1, self.embed_size).float().to(self.device).detach()
        hidden = self.encoder.transformer(x)["last_hidden_state"]
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + EPSILON)
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs


class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(len(sequences), max(lengths)).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            try:
                padded_sequences[i, :end] = seq[:end]
            except TypeError:
                padded_sequences[i, :end] = torch.LongTensor(seq[:end])

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def train_epoch(data_loader, discriminator, optimizer, epoch=0, log_interval=10, device="cpu"):
    samples_so_far = 0
    discriminator.train_custom()
    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        loss = F.nll_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    samples_so_far,
                    len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset),
                    loss.item(),
                )
            )


def evaluate_performance(data_loader, discriminator, device="cpu"):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input_t, target_t in data_loader:
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

    test_loss /= len(data_loader.dataset)

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(data_loader.dataset), 100.0 * correct / len(data_loader.dataset)
        )
    )


def predict(input_sentence, model, classes, cached=False, device="cpu"):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print(
        "Predictions:",
        ", ".join("{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in zip(classes, log_probs)),
    )


def get_cached_data_loader(dataset, batch_size, discriminator, shuffle=False, device="cpu"):
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()
    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys), batch_size=batch_size, shuffle=shuffle, collate_fn=cached_collate_fn
    )

    return data_loader


def train_discriminator(
    dataset,
    dataset_path=None,
    pretrained_model="gpt2-medium",
    epochs=10,
    batch_size=64,
    log_interval=10,
    save_model=False,
    cached=False,
    no_cuda=False,
):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    if dataset == "HateOffensive":
        idx2class = ["hate-speech", "offensive-language","neither"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class), pretrained_model=pretrained_model, cached_mode=cached, device=device
        ).to(device)

        full_dataset =[]
        for file in ['train.tsv','test.tsv']:
            x = []
            y = []
            with open(os.path.join(dataset_path,file)) as f:
                for i, line in enumerate(tqdm(f, ascii=True)):
                    try:
                        _, label, text = line.split("\t")
                        text = text.replace("\n","")
                        seq = discriminator.tokenizer.encode(text)

                        if len(seq) < max_length_seq:
                            seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
                        x.append(seq)
                        y.append(int(label))
                    except Exception:
                        print("Error evaluating / tokenizing" " line {}, skipping it".format(i))
                        pass

            tmp_dataset = Dataset(x,y)
            full_dataset.append(tmp_dataset)

        train_dataset, test_dataset = full_dataset
        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }

    elif dataset == "SentiTreeBank":
        idx2class = ["negative", "neutral", "positive"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class), pretrained_model=pretrained_model, cached_mode=cached, device=device
        ).to(device)

        full_dataset =[]
        for file in ['train.tsv','test.tsv']:
            x = []
            y = []
            with open(os.path.join(dataset_path,file)) as f:
                for i, line in enumerate(tqdm(f, ascii=True)):
                    if i==0:
                        continue
                    else:
                        try:
                            _, text,_,_, label = line.split("\t")
                            text = text.replace("\n","")
                            label = label.replace("\n","")
                            label = class2idx[label]
                            seq = discriminator.tokenizer.encode(text)
                            if len(seq) < max_length_seq:
                                seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
                            x.append(seq)
                            y.append(int(label))
                        except Exception:
                            print("Error evaluating / tokenizing" " line {}, skipping it".format(i))
                            pass

            tmp_dataset = Dataset(x,y)
            full_dataset.append(tmp_dataset)

        train_dataset, test_dataset = full_dataset
        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }

    else:
        raise ValueError("Dataset type and path shoud be selected.")

    end = time.time()
    print("Preprocessed {} data points".format(len(train_dataset) + len(test_dataset)))
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(train_dataset, batch_size, discriminator, shuffle=True, device=device)

        test_loader = get_cached_data_loader(test_dataset, batch_size, discriminator, device=device)

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    if save_model:
        with open("{}_classifier_head_meta.json".format(dataset), "w") as meta_file:
            json.dump(discriminator_meta, meta_file)

    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in tqdm(range(epochs)):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device,
        )
        evaluate_performance(data_loader=test_loader, discriminator=discriminator, device=device)

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))
        print("\nExample prediction")
        predict(example_sentence, discriminator, idx2class, cached=cached, device=device)

        if save_model:
            torch.save(
                discriminator.get_classifier().state_dict(),
                "{}_classifier_head_epoch_{}.pt".format(dataset, epoch + 1),
            )

    if save_model:
        torch.save(discriminator.state_dict(), "{}_classifier.pt".format(dataset))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HateOffensive",
        choices=("HateOffensive","SentiTreeBank"),
        help="dataset to train the discriminator on.")
    parser.add_argument("--dataset_path", type=str, help="Dataset path.")
    parser.add_argument(
        "--pretrained_model", type=str,
        default="gpt2-medium", help="Pretrained model to use as encoder")
    parser.add_argument("--epochs", type=int,
            default=10, metavar="N", help="Number of training epochs")
    parser.add_argument(
        "--batch_size", type=int, default=64, metavar="N",
        help="input batch size for training (default: 64)")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save_model", action="store_true", help="whether to save the model")
    parser.add_argument("--cached", action="store_true", help="whether to cache the input representations")
    parser.add_argument("--no_cuda", action="store_true", help="use to turn off cuda")
    args = parser.parse_args()

    train_discriminator(**(vars(args)))
