import torch
import torch.nn as nn
import torchvision.models as models
from config import Config
class ChemicalStructureEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Load pretrained ResNet
        resnet = models.resnet34(weights='DEFAULT')
        
        # Remove the final FC layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Add new layers
        self.fc = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        # ResNet feature extraction
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        
        # Project to hidden dimension
        output = self.fc(features)
        return output