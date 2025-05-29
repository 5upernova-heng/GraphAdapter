"""
将我们模型的数据集转换成本模型需要的数据集。
我们的数据集构成：
- train:
- config: 

本模型输入给模型的数据集处理手段参见 preprocess.py

数据集构成：
- data.x 节点文本的嵌入
- data.y 节点对应的答案
- data.edge_index 图的邻接信息
- prompt_x data.x 结合 prompt 信息，输入给 LLM 后得到的嵌入
"""

import os
import pickle
import json
import argparse
import datasets
import torch
import numpy as np
import copy
import random

from typing import List, Tuple
from tqdm import tqdm
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def mask_dict(data_dict: dict, key: str):
    modified_dict = copy.deepcopy(data_dict)
    config_list = modified_dict.get(key)
    print(data_dict, key, config_list)
    assert isinstance(config_list, list)
    chosen_index = random.choice(range(len(config_list)))
    value = config_list[chosen_index][2]
    modified_dict[key][chosen_index][2] = "-"

    return value, str(modified_dict)


def gen_template(label: str, args) -> Tuple[int, str, str]:
    label = label.replace("'", '"')
    update_records = json.loads(label)
    if args.ospf:
        value, modified_dict = mask_dict(update_records, 'ospf')
    elif args.bgp:
        value, modified_dict = mask_dict(update_records, 'bgp_route')
    else:
        value, modified_dict = mask_dict(update_records, random.choice(['ospf', 'bgp_route']))

    template_l = f"Here is an intent from network operator that asks you to update network configurations, the update abstract are as follows {modified_dict}"
    template_r = '.\n\nQuestion: Based on given intents and update abstract, the masked value is "___" (answer is an integer)'
    return value, template_l, template_r


def get_llm_embeddings_for_finetune(
    raw_texts: List[str],
    labels: List[str],
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为给定的原始文本列表生成两种类型的LLM嵌入，以匹配finetune代码的需求。
    返回:
    - tuple[np.ndarray, np.ndarray]:
        - sentence_embeddings_np: 对应 data.x 的嵌入 (基于 template_l)。
        - full_prompt_embeddings_np: 对应 prompt_x 的嵌入 (基于 template_l 和 template_r)。
        - y
    """
    plm_path = args.plm_path
    batch_size = args.batch_size
    max_length = args.max_length
    device = args.device

    logger.info(f"Loading LLM model from: {plm_path}")
    model = AutoModelForCausalLM.from_pretrained(plm_path, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(plm_path, use_fast=False)
    tokenizer.pad_token = "[PAD]"

    all_sentence_embeddings_list = []
    all_full_prompt_embeddings_list = []
    all_y_values = []

    logger.info(f"Generating embeddings in batches of {batch_size} on {device}...")
    for i in tqdm(range(0, len(raw_texts), batch_size), desc="Processing batches"):
        batch_raw_texts = raw_texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        texts_for_sentence_emb = []
        texts_for_full_prompt_emb = []

        for text_idx_in_batch, text_content in enumerate(batch_raw_texts):
            current_label = batch_labels[text_idx_in_batch]
            y_value, template_l, template_r = gen_template(current_label, args)

            all_y_values.append(y_value)  # 收集y值

            texts_for_sentence_emb.append(template_l + text_content)
            texts_for_full_prompt_emb.append(template_l + text_content + template_r)

        # 1. 生成 "sentence_embeddings" (对应 data.x)
        inputs_sentence = tokenizer(texts_for_sentence_emb, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_sentence_hidden_states = model.model(input_ids=inputs_sentence.input_ids, attention_mask=inputs_sentence.attention_mask)[0]

        last_token_indices_sentence = inputs_sentence.attention_mask.sum(dim=1) - 1
        batch_sentence_embeddings = outputs_sentence_hidden_states.gather(1, last_token_indices_sentence.view(-1, 1, 1).repeat(1, 1, outputs_sentence_hidden_states.shape[-1])).squeeze(1)
        all_sentence_embeddings_list.append(batch_sentence_embeddings.cpu())

        # 2. 生成 "full_prompt_embeddings" (对应 prompt_x)
        # 这是基于 template_l + text + template_r 的嵌入
        inputs_full_prompt = tokenizer(texts_for_full_prompt_emb, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_full_prompt_hidden_states = model.model(input_ids=inputs_full_prompt.input_ids, attention_mask=inputs_full_prompt.attention_mask)[0]

        last_token_indices_full_prompt = inputs_full_prompt.attention_mask.sum(dim=1) - 1
        batch_full_prompt_embeddings = outputs_full_prompt_hidden_states.gather(
            1, last_token_indices_full_prompt.view(-1, 1, 1).repeat(1, 1, outputs_full_prompt_hidden_states.shape[-1])
        ).squeeze(1)
        all_full_prompt_embeddings_list.append(batch_full_prompt_embeddings.cpu())

    # 合并所有批次的结果
    sentence_embeddings_tensor = torch.cat(all_sentence_embeddings_list, dim=0)
    full_prompt_embeddings_tensor = torch.cat(all_full_prompt_embeddings_list, dim=0)

    # 转换为NumPy数组
    sentence_embeddings_np = sentence_embeddings_tensor.numpy()
    full_prompt_embeddings_np = full_prompt_embeddings_tensor.numpy()
    y_np = np.array(all_y_values, dtype=np.int64)

    logger.info("Embeddings generated successfully.")
    return sentence_embeddings_np, full_prompt_embeddings_np, y_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arrow_path')
    # 嵌入
    parser.add_argument("--plm_path", type=str, default="/data/user/jzt/models/Llama-2-7b-hf", help="path of llama 2")
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    # label
    parser.add_argument("--ospf", action='store_true')
    parser.add_argument("--bgp", action='store_true')

    args = parser.parse_args()
    arrow_path = args.arrow_path
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    logger.info(f"Device: {args.device}")

    train_dataset = datasets.load_from_disk(f'{arrow_path}/train')
    config_dataset = datasets.load_from_disk(f'{arrow_path}/config')
    edge_index = pickle.load(open(f'{arrow_path}/edge_index.pkl', 'rb'))

    # 2. 调用嵌入生成函数
    sentence_embeddings, full_prompt_embeddings, y_target = get_llm_embeddings_for_finetune(
        raw_texts=config_dataset['config_desc'],
        labels=train_dataset['labels'],
        args=args,
    )

    # 3. 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # 4. 定义输出文件名 (与你之前finetune代码中加载的名称对应)
    dataname = os.path.basename(args.arrow_path)
    path_sentence_embeddings = f'./token_embedding/{dataname}/sentence_embeddings.npy'
    path_prompt_embeddings = f'./prompt_embedding/{dataname}/prompt_embedding.npy'
    path_edge_index = f'./datasets/{dataname}/edge_index.npy'
    path_y = f'./datasets/{dataname}/y.npy'

    # 5. 保存 NumPy 数组
    try:
        np.save(path_edge_index, edge_index)
        logger.info(f"Saved edge index to: {path_edge_index} (Shape: {path_edge_index})")

        np.save(path_sentence_embeddings, sentence_embeddings)
        logger.info(f"Saved sentence embeddings to: {path_sentence_embeddings} (Shape: {sentence_embeddings.shape})")

        np.save(path_prompt_embeddings, full_prompt_embeddings)
        logger.info(f"Saved full prompt embeddings to: {path_prompt_embeddings} (Shape: {full_prompt_embeddings.shape})")

        np.save(path_y, y_target)
        logger.info(f"Saved y targets to: {path_y} (Shape: {y_target.shape})")

        logger.info("All files saved successfully. Preprocessing complete.")
    except Exception as e:
        logger.error(f"Error saving .npy files: {e}")
        exit(1)
    ...


if __name__ == "__main__":
    main()
