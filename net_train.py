from graphadapter import LinearHead

import os
import datasets
import argparse
import torch
import pickle

from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer, default_data_collator


def parse_args():
    parser = argparse.ArgumentParser("finetuning GraphAdapter")
    # training
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument("--max_epoch", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--learning_ratio", type=float, default=1e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    # model
    parser.add_argument("--hiddensize_gnn", type=int, default=64)
    parser.add_argument("--hiddensize_fusion", type=int, default=64)
    parser.add_argument("--GNN_type", type=str, default="SAGE", choices=["SAGE", "GAT", "MLP"])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--step", type=int, default=20, help="epoch of saved graphadapter")
    parser.add_argument("--graph_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    # data
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset to be used")
    parser.add_argument("--plm_path", type=str, required=True, help="path of llama 2")
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()
    return args


def load_dataset(dataset_path):
    dataset = datasets.load_from_disk(dataset_path)
    split = pickle.load(open(f"{dataset_path}/split.pkl", "rb"))
    return dataset, split


def print_results(tokenizer, batch, logits):
    net_ids = batch["net_ids"]
    input_strings = batch["query"]

    print("Net_ids---{}".format(net_ids))
    print("query---{}".format(input_strings))

    labels = batch["labels"].squeeze(0)

    mask = labels != -100
    filtered_labels = labels[mask]

    predicted_tokens = torch.argmax(logits, dim=1)
    predicted_tokens = predicted_tokens[mask]
    pre = tokenizer.batch_decode(predicted_tokens)
    x = tokenizer.batch_decode(filtered_labels)

    eval_pred = [item.split("</s>")[0] for item in pre]
    eval_pred = [item.split("\n\n###\n\n ")[-1] for item in eval_pred]

    eval_label = [item.split("</s>")[0] for item in x]
    eval_label = [item.split("\n\n###\n\n ")[-1] for item in eval_label]

    print("Pre---{}".format("".join(eval_pred)))
    print("label---{}".format("".join(eval_label)))


def to_tensor(example):
    example['config_descs'] = torch.tensor(example['config_descs'])
    N = example['config_descs'].shape[0]
    example['query_embedding'] = torch.tensor(example['query_embedding']).unsqueeze(0).expand(N, -1)
    example['edge_index'] = torch.tensor(example['edge_index'])
    example['labels'] = torch.tensor(example['labels'])
    return example


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    device = args.device

    logger.remove(0)
    logger.add(f"{os.path.basename(args.dataset_name)}.{args.learning_rate}.{args.max_epoch}.train.log", mode='w')

    dataset, split = load_dataset(args.dataset_name)

    dataset = dataset.map(to_tensor, num_proc=16).with_format('torch')

    embedding_dim = dataset["query_embedding"][0].shape[1]

    train_dataset = dataset.select(split["train"])
    test_dataset = dataset.select(split["test"])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(args.plm_path, use_fast=False)
    tokenizer.pad_token = "[PAD]"

    model = LinearHead(embedding_dim, args.seq_len, args.graph_size, tokenizer.vocab_size, args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)

    logger.info("Training")
    args.max_epoch *= args.graph_size
    pbar = tqdm(total=args.max_epoch * len(train_loader))
    for epoch in range(args.max_epoch):
        for batch in train_loader:
            model.train()
            model.ga.train()
            optimizer.zero_grad()
            loss, logits, labels = model(
                x=batch['config_descs'].squeeze(0).to(device),
                edge_index=batch['edge_index'].squeeze(0).to(device),
                prompt_embedding=batch['query_embedding'].squeeze(0).to(device),
                labels=batch['labels'].squeeze(0),
            )
            # print_results(tokenizer, batch, logits)
            logger.info(f"Accum Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            pbar.update()
    torch.save(model.state_dict(), f"{args.save_path}/checkpoint.pth")

    logger.info("Testing")
    pbar = tqdm(total=len(test_loader))
    with torch.no_grad():
        model.eval()
        model.ga.eval()
        for batch in test_loader:
            loss, logits, labels = model(
                x=batch['config_descs'].squeeze(0).to(device),
                edge_index=batch['edge_index'].squeeze(0).to(device),
                prompt_embedding=batch['query_embedding'].squeeze(0).to(device),
                labels=batch['labels'].squeeze(0),
            )
            print_results(tokenizer, batch, logits)
            pbar.update()
