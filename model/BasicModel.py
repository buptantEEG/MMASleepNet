import torch
from torch import nn
import time

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.model_name = str(type(self)).split('.')[1].split('\'')[0]
        self.init_time = time.strftime('%m%d_%H%M')
        # print(self.model_name)
        
    def load(self, path):
        print('load: ', path)
        self.load_state_dict(torch.load(path))
        print('model loaded!')
    
    def save(self, path=None):
        name = f'{self.model_name}_{self.init_time}.pth'
        if path is not None:
            name = path + name
        torch.save(self.state_dict(), name)
        print('model saved!')
        
