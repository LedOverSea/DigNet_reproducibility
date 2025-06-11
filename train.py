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
    #train_filename = 'pathway/simulation/SERGIO_data_node_2000.data'   # Your gene expression profile and network input data. Refer to Supplement S1 for details.
    train_filename = 'Cancer_datasets\S33_Cancer_BRCA_processed_20250609.data'
    args.pca_file = 'result/20250609.pkl'                        # This is a PCA parameter file, which needs to be loaded during the test step.
    trainer = DigNet(args)
    trainer.train(train_filename, n_train=200, n_test=[200,210])  # Training with simulated data