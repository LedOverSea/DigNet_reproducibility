from torch_geometric.data import DataLoader
from pathway.pathway import create_batch_dataset_from_cancer, create_batch_dataset_simu
from tqdm import tqdm
from datetime import datetime
import torch
from denoising_diffusion_pytorch import GaussianDiffusion1D
from discrete.models.transformer_model import DigNetGraphTransformer
from discrete import network_preprocess
from discrete.diffusion_utils import Evaluation
from torch.optim.lr_scheduler import StepLR
import argparse
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
from torch_geometric.data import Batch
import warnings
import pickle
from make_final_net import cal_final_net
from joblib import Parallel, delayed
import os
from DigNet import DigNet,Config
import json
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # exec(open('config.py').read())
    config_file = 'config/config.json'  # Or any path where your configuration file is stored
    args = Config(config_file)

    # For Case 1:
    # train_filename = 'pathway/simulation/SERGIO_data_node_2000.data'   # Your gene expression profile and network input data. Refer to Supplement S1 for details.
    # args.pca_file = 'result/simu_pca_model.pkl'                        # This is a PCA parameter file, which needs to be loaded during the test step.
    # trainer = DigNet(args)
    # trainer.train(train_filename, n_train=200, n_test=[1000, 1010])  # Training with simulated data

    
    # 数据可能的路径pathway\simulation\SERGIO_data_for_test_law_array.data
    # Cancer_datasets\S33_Cancer_BRCA_output.csv

    # For Case 2:
    """ train_filename = 'Cancer_datasets\S33_Cancer_BRCA_output.csv'  # Please input your gene expression profile (ending in csv or xlsx). If you want to process your sequencing data beforehand, such as matrix completion and quality control, please refer to Supplement S2.
    args.pca_file = 'result/simu_pca_model.pkl'  # This is a PCA parameter file, which needs to be loaded during the test step.
    args.test_pathway = "hsa05224"  # During training, if you want to exclude certain gene sets from the gene list, please provide the corresponding KEGG library ID number.
    trainer = DigNet(args)
    best_mean_AUC, train_model_file, printf = trainer.train(train_filename)  # Training with simulated data """

    
    # 训练oom, 加载一个预训练模型
    #train_model_file = 'pre_train\S33_Cancer_cell_checkpoint_pre_train_20240326.pth' # key error
    train_model_file = 'pre_train\S33_Cancer_cell_checkpoint_pre_train_20240326.pth'

    #test_filename = 'Cancer_datasets\S33_B_cell_BRCA_output.csv'
    test_filename = 'Cancer_datasets\S33_Cancer_BRCA_output.csv'
    args.test_pathway = "hsa05224"
    args.pca_file = 'result/simu_pca_model.pkl'

    trainer = DigNet(args)
    results = {'AUC': [], 'AUPR': [], 'AUPR_norm': [], 'F1': [], 'nodenum': []}
    test_num = 1
    result_filename = 'result/FILE1_dignet.data'
    for i in range(0, test_num):
        print(f'Generating a network for the {i + 1:3d}-th gene expression profile!')
        diffusion_pre = torch.load(train_model_file, map_location=trainer.device)
        testdata, truelabel = trainer.load_test_data(test_filename, num=i, diffusion_pre=diffusion_pre)

        adj_final = trainer.test(diffusion_pre,testdata,truelabel)  # Testing with simulated data
        performance = trainer.evaluation(adj_final, truelabel)
        for key in performance.keys():
            results[key].append(performance[key])
        results['nodenum'].append(testdata.x.shape[0])
        with open(result_filename, 'wb') as f:
            pickle.dump(results, f)

    print(f'**************   Task is finished! The results are saved in {result_filename}!   **************')

    '''
    Supplement S1: If your data includes raw data and networks, you can directly import them as '*.data' or any pickle-saved file.
                   Note that the file should be a dataset containing multiple list-type variables, each with the following structure:
                 1. 'net' variable:
                    Description: Contains the adjacency matrix of network data.
                    File Type: Numpy ndarray.
                    Content: 0-1 weight matrix. Non-0-1 values can also be loaded, sized cell * cell.
                 2. 'exp' variable:
                    Description: Contains experimental data in DataFrame format.
                    File Type: CSV format.
                    Content: Preprocessed scRNA-seq results, sized gene * cell.

    Supplement S2: We recommend completing matrix completion and quality control before inputting sequencing data.
                   'Create_BRCA_data.py' provides a simple example for reference.
                   Once you have obtained high-quality gene expression information (gene * cell), it should be stored in table format (CSV or XLSX), with the first row and column being cell numbers and gene symbol IDs, respectively.
    '''