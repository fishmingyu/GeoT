#!/usr/bin/env python3
import torch
import numpy as np
import time
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix

import FS_Block
from scipy.sparse import *

class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data):
        super(MGCN_dataset, self).__init__()

        # self.graph = np.load('dgl_dataset/mythroughput/' + data +'.npz')
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/mythroughput/' + data +'.npz')
        # self.num_features = dimN
        self.init_edges()
        # self.init_embedding()
        
    def init_edges(self):
        # loading from a .npz graph file
        src_li=self.graph['src_li']
        dst_li=self.graph['dst_li']
        
        self.num_nodes_ori = self.graph['num_nodes']
        self.num_nodes=self.graph['num_nodes']+16-(self.graph['num_nodes']%16)
        self.num_edges = len(src_li)
        self.edge_index = np.stack([src_li, dst_li])

        # self.edge_index_new, self.perm_new, self.m_edge_index_new, self.m_perm_new, self.l_comesNew = MagicsphereMRabbit_cmake.reorder(torch.IntTensor(self.edge_index),self.num_nodes_ori,6)

        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()
        
        self.column_index1 = torch.IntTensor(adj.indices)
        self.row_pointers1 = torch.IntTensor(adj.indptr)
        dd = (self.row_pointers1[1:] - self.row_pointers1[:-1]).tolist()
        dd=torch.tensor(dd, dtype=torch.float32) 
        # dd= torch.rsqrt(dd).to(torch.float16)  
        self.dd= torch.ones_like(dd)
        self.row_pointers, self.column_index, self.degrees, self.t_window_rowTensor, self.t_atomicTensor=FS_Block.blockProcess_fp16_balance(self.row_pointers1, self.column_index1, self.dd.to(torch.float16), 8,8 ,2)
        # print()
        # print(self.row_pointers[-1]/8)
        # print((self.row_pointers[-1]/8)*64)
        
    def init_embedding(self,dimN):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        # 打印归一化后的特征
        # self.x = torch.ones(self.num_nodes_ori, dimN).to(dtype=torch.float16)
        self.x = torch.randint(low=1, high=2, size=(self.num_nodes_ori, dimN)).half()


    def to(self, device):
        self.row_pointers =  self.row_pointers.to(device)
        self.column_index =  self.column_index.to(device)     
        self.degrees = self.degrees.to(device)  
        self.x =  self.x.to(device)
        return self
    
  
