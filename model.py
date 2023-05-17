import torch.nn as nn

class DrivingModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.waypoint_embedding = nn.Embedding(256, 256)
        
        self.mlp = nn.Sequential(
            nn.Linear(2*8*256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1),
        )
    
    def forward(self, x, w):
        x = self.conv_stack(x)
        w = self.waypoint_embedding(w)
        b,c = w.shape
        
        x = (x + w.view(b,c,1,1)).reshape(b,-1)
        x = self.mlp(x)
        
        x = nn.functional.tanh(x).view(b)
        
        return x
