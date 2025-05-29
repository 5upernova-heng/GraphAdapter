"""
将 Arrow 数据集转换为本模型使用的格式
Arrow 格式的数据集参见 jzt@3090-nlp:~/Config/dataset_temp
旧数据集：{ net_id, query, labels, config_desc, node_ids}

新数据集：{ net_id, query_embedding, labels, edge_index, config_descs }

- query 从自然语言需要使用 13B 的模型进行嵌入，变成 query_embedding，[seq_len, embed_size]
- edge_index 为当前网络 id 对应的所有边，从 0 开始编号 [2, edge_num]
- config_descs 是当前网络所有 config_desc 的嵌入，所以格式应该是 [graph_size(node_num), embed_size]

"""

import numpy as np
import random
import argparse
import datasets
import pickle
import os
import torch

from typing import List
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from torch_geometric.utils import index_to_mask
from loguru import logger


def convert(train_dataset: Dataset, config_dataset: Dataset, edge_index, graph_size: int):
    net_num = len(train_dataset) // graph_size

    def group_net(range_):
        start, end = range_
        net_id = train_dataset['net_ids'][start]
        query = train_dataset['query'][start]
        label = train_dataset['labels'][start]

        subset = torch.arange(start, end)
        node_mask = index_to_mask(subset, size=len(train_dataset))
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index_sub = edge_index[:, edge_mask]
        edge_index_sub %= graph_size
        config_descs = config_dataset['config_desc'][start:end]
        return net_id, query, label, edge_index_sub, config_descs

    results = map(group_net, [(i, i + graph_size) for i in range(net_num)])
    net_ids, query, labels, edge_index, config_descs = map(list, zip(*results))

    return {'net_ids': net_ids, 'query': query, 'labels': labels, 'edge_index': edge_index, 'config_descs': config_descs}


def embed(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizer, args, strings: List[str], pbar: tqdm = None):
    batch_size = args.batch_size
    device = args.device
    embedding_dim = args.embedding_dim

    iter_num = len(strings) // batch_size
    step = batch_size
    if pbar is None:
        step = 1
        pbar = tqdm(total=iter_num)
    embeddings_list = []
    for i in range(iter_num):
        batch = strings[i : i + batch_size]
        encoded = tokenizer(batch, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        text_id = encoded['input_ids'].long().to(device)
        attention_mask = encoded['attention_mask'].to(device)
        with torch.no_grad():
            output = model.model(input_ids=text_id, attention_mask=attention_mask)[0]
        prompt_last_position = attention_mask.sum(dim=1) - 1
        embedding = output.gather(1, prompt_last_position.view(-1, 1, 1).repeat(1, 1, embedding_dim)).view(-1, embedding_dim)
        embeddings_list.append(embedding.to("cpu"))
        pbar.update(step)

    embedding = torch.cat(embeddings_list, dim=0)
    return embedding


def dict2dataset(data_dict, dest_dir):
    dataset = Dataset.from_dict(data_dict)
    dataset.save_to_disk(dest_dir)


def gen_split(dest_dir, dataset_size, split):
    assert len(split) == 3
    train, valid, test = split
    numbers = list(range(dataset_size))

    valid_size = dataset_size // 10 * valid
    test_size = dataset_size // 10 * test

    train_list = []
    valid_list = []
    test_list = []

    # test
    for i in range(0, test_size):
        num = random.choice(numbers)
        test_list.append(num)
        numbers.remove(num)
    test_list.sort()

    # valid
    for i in range(0, valid_size):
        num = random.choice(numbers)
        valid_list.append(num)
        numbers.remove(num)
    valid_list.sort()

    train_list = numbers
    print("train/valid/test:", f"{len(train_list)}/{len(valid_list)}/{len(test_list)}")
    train_list = np.array(train_list)
    valid_list = np.array(valid_list)
    test_list = np.array(test_list)

    train_list = torch.from_numpy(train_list)
    valid_list = torch.from_numpy(valid_list)
    test_list = torch.from_numpy(test_list)

    data = {'train': train_list, 'valid': valid_list, 'test': test_list}

    with open(f'{dest_dir}/split.pkl', 'wb') as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--plm_path", type=str, default="/data/user/jzt/models/Llama-2-7b-hf", help="path of llama 2")
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--graph_size", type=int, required=True)

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    logger.info(f"Device: {args.device}")

    in_path = args.input_path
    out_path = args.output_path
    device = args.device

    logger.info(f"Loading dataset from {in_path}")
    train_dataset = datasets.load_from_disk(f"{in_path}/train")
    config_dataset = datasets.load_from_disk(f"{in_path}/config")
    edge_index = pickle.load(open(f'{in_path}/edge_index.pkl', 'rb'))

    logger.info("Converting dataset")
    new_dataset_dict = convert(train_dataset, config_dataset, edge_index, args.graph_size)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    logger.info("Loading models")
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.plm_path, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(device)
    tokenizer.pad_token = "[PAD]"  # for batch preprocess
    args.embedding_dim = model.model.embed_tokens.embedding_dim

    logger.info("Embedding queries")
    new_dataset_dict['query_embedding'] = embed(model, tokenizer, args, new_dataset_dict['query'])

    logger.info("Embedding config descs")
    config_embeds = []
    pbar = tqdm(total=len(new_dataset_dict['config_descs']) * args.graph_size)
    for config_descs in new_dataset_dict['config_descs']:
        config_embeds.append(embed(model, tokenizer, args, config_descs, pbar))
    new_dataset_dict['config_descs'] = config_embeds

    logger.info("Tokenizing label strings")
    encoded = tokenizer(new_dataset_dict['labels'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    text_id = encoded['input_ids'].long()
    new_dataset_dict['labels'] = text_id

    logger.info(f"Saving dataset to {out_path}")
    dict2dataset(new_dataset_dict, out_path)

    split = (7, 1, 2)
    logger.info(f"Generating split, {split}")
    gen_split(out_path, len(new_dataset_dict['net_ids']), split)


if __name__ == "__main__":
    main()
