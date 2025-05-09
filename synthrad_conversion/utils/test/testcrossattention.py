import torch
import torch.nn as nn
# test cross attention
import torch.nn as nn
from torch.nn import Conv3d

import torch
import torch.nn as nn
import torch.nn.functional as F

class adapt_condition_shape(nn.Module):
    def __init__(self):
        super(adapt_condition_shape, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)  # Output: [32, 64, 64, 64]
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)  # Output: [64, 32, 32, 64]
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)  # Output: [64, 16, 16, 64]
        #self.fc = nn.Linear(256 * 8 * 8, 2048 * 64)  # Flatten and map to the final size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        print('shape of x',x.shape)
        batch, channel, height, width, depth = x.shape
        inner_dim = x.shape[1]
        x = x.permute(0, 2, 3, 4, 1).reshape(batch, height * width * depth, inner_dim)
        print('shape of x',x.shape)
        #x = self.fc(x)
        #x = x.view(-1, 2048, 64)  # Reshape to the desired output shape
        return x

'''class adapt_condition_shape(nn.Module):
    def __init__(self):
        super(adapt_condition_shape, self).__init__()
        self.conv1=Conv3d(1, 64, kernel_size=3, stride=2, padding=1)  # output shape: [64, 64, 32]
        self.conv2=Conv3d(64, 128, kernel_size=3, stride=2, padding=1)  # output shape: [128, 32, 16]
        self.conv3=Conv3d(128, 256, kernel_size=3, stride=2, padding=1)  # output shape: [256, 16, 8]
        self.conv4=Conv3d(256, 512, kernel_size=3, stride=2, padding=1)  # output shape: [512, 8, 4]
        self.flatten=nn.Flatten()
        self.ln=nn.Linear(512 * 8* 8 * 4, 2048*64)
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        print('shape of x',x.shape)
        x=self.flatten(x)
        x=self.ln(x)
        return x'''
context = torch.randn(2, 1, 128, 128, 64) # [1, 1, 128, 128, 64] 
#context = context.permute(0, 1, 4, 2, 3) # [1, 1, 64, 128, 128] 
adapt=adapt_condition_shape()
adapt_context=adapt(context)
print(adapt_context.shape)
#cross_attention_dim = 64
#inner_dim = 2
#to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)(context)
#print(to_k)