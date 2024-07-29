import torch

from torch.library import opcheck
import importlib
import os.path as osp
import torch

library = '_C'
# in the parent directory of the current file

torch.ops.load_library("/home/zhongming/GeoT/geot/_C.so")
