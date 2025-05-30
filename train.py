from train_utils import finetune, load_data_with_prompt_embedding, set_random_seed
import numpy as np
import argparse
import torch


def run_exp(args):
    acc_ls = []
    for split in range(1, 2):
        data = load_data_with_prompt_embedding(args.dataset_name, 70, 10, split)
        print("class_num:", data.y.max() + 1)
        for i in range(1):
            acc = finetune(data, args)
            acc_ls.append(acc)
    print(np.mean(acc_ls), np.std(acc_ls))
    return acc_ls


if __name__ == "__main__":
    set_random_seed(0)
    parser = argparse.ArgumentParser("finetuning GraphAdapter")
    # data
    parser.add_argument("--dataset_name", type=str, help="dataset to be used", default="instagram")
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
    parser.add_argument("--seq_len", type=int, default=512)

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    acc_ls = run_exp(args)
