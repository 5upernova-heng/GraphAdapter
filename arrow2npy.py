"""
将 Arrow 数据集转换为本模型使用的格式
Arrow 格式的数据集参见 jzt@3090-nlp:~/Config/dataset_temp

本模型使用的数据集格式本质上是一个 edge_index 文件以及一个 x_text 文件
前者记录图的边信息（2D），后者记录图的节点文本信息（1D）
"""

import numpy as np
import argparse
import datasets
import pickle   
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')

    args = parser.parse_args()
    in_path = args.input_path
    out_path = args.output_path

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dataset = datasets.load_from_disk(in_path)
    x_text = np.array(dataset['config_desc'])
    np.save(out_path + '/x_text.npy', x_text)

    with open(in_path + '/edge_index.pkl', 'rb') as f:
        edge_index = pickle.load(f)
    np.save(out_path + '/edge_index.npy', edge_index)

