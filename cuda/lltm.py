import math
from torch import nn
from torch.autograd import Function
import torch

import lltm_cuda

torch.manual_seed(42)
