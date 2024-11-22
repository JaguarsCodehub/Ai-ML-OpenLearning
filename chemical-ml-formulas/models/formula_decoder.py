import torch
import torch.nn as nn
from config import Config
import random

class FormulaDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(FormulaDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, encoder_output, target_tokens, teacher_forcing_ratio=0):
        batch_size = encoder_output.size(0)
        max_length = target_tokens.size(1)
        vocab_size = self.fc.out_features
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, max_length, vocab_size).to(encoder_output.device)
        
        # Initialize hidden state with encoder output
        hidden = encoder_output.unsqueeze(0)
        
        # First input is always START token
        current_token = target_tokens[:, 0:1]
        
        for t in range(max_length):
            # Embed current token
            embedded = self.embedding(current_token)
            
            # Get output and hidden state from GRU
            output, hidden = self.gru(embedded, hidden)
            
            # Get prediction
            prediction = self.fc(output)
            
            # Store prediction
            outputs[:, t:t+1] = prediction
            
            # Teacher forcing: use actual target tokens as next input
            if t < max_length - 1:
                if random.random() < teacher_forcing_ratio:
                    current_token = target_tokens[:, t+1:t+2]
                else:
                    current_token = prediction.argmax(dim=-1)
        
        return outputs
