#from tools.data_handler import Sig
from _KS_module import *
import numpy as np
import os
import multiprocessing as mp
import time
from tqdm import tqdm_notebook
from sklearn.metrics import roc_auc_score

output_path = os.path.join(os.getcwd(), 'Sig_KS_results')

if not os.path.exists(output_path):
    os.mkdir(output_path)

tr_dic = get_data_dic('tr')
te_dic = get_data_dic('te')
