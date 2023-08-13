import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

shape_log = False

class AutoEncoders(nn.Module):
    def __init__(self, INPUT_AXIS, BATCH_SIZE, LATENT_DIMENSION):
        super(AutoEncoders, self).__init__()
        self.IA = INPUT_AXIS
        self.BS = BATCH_SIZE
        self.LD = LATENT_DIMENSION
        self.unit = int((self.IA-self.LD)/4)
        self.fc1_1 = nn.Linear(self.IA, self.unit*2)
        self.fc2   = nn.Linear(self.unit*2, self.LD)
        self.fc3   = nn.Linear(self.LD, self.unit*2)
        self.fc4_3 = nn.Linear(self.unit*2, self.IA)

    def make_distance_vector(self, input_tensor):
        input_diff_sum = torch.stack([torch.linalg.norm(input_tensor[n[0]]-input_tensor[n[1]]) for n in itertools.combinations(range(self.BS), 2)], dim=0).cuda()
        return input_diff_sum

    def encoder(self, x):
        x = self.fc1_1(x)
        x = F.relu(x)
        return x

    def full_connection(self, z):
        z = self.fc2(z)
        return z

    def tr_full_connection(self, y):
        y = torch.relu(y)
        y = self.fc3(y)
        y = torch.relu(y)
        return y

    def decoder(self, y):
        y = self.fc4_3(y)
        return y
    
    def forward(self, x):
        if shape_log == True: print(f'x:{x.shape}')
        x = x.view(-1, self.IA)
        z = self.encoder(x).cuda()
        if shape_log == True: print(f'z = encorder(x):{z.shape}')
        z = z.view(self.BS, -1)
        z = self.full_connection(z).cuda()
        if shape_log == True: print(f'full_connection(z):{z.shape}')
        #----------------------------------------
        y = self.tr_full_connection(z).cuda()
        if shape_log == True: print(f'tr_full_connection(z):{y.shape}')
        y = self.decoder(y).view(self.BS, self.IA).cuda()
        if shape_log == True: print(f'y:{y.shape}')
        y = y.view(self.BS, self.IA)##
        #-----------------------------------------
        lat_repr = z.view(self.BS, self.LD).cuda()
        return x, lat_repr, y