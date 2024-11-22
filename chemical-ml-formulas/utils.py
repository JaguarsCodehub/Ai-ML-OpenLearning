import torch
from pathlib import Path
from config import Config
def save_checkpoint(model_dict, save_path, is_best=False):
    """Save model checkpoint"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model_dict, save_path)
    if is_best:
        best_path = save_path.parent / f"best_{save_path.name}"
        torch.save(model_dict, best_path)

def load_checkpoint(path, encoder, decoder, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=Config.DEVICE)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['val_loss']

def indices_to_formula(indices, idx_to_symbol, remove_special_tokens=True):
    """Convert a list of indices to a chemical formula string."""
    if not indices:
        return ""
    
    special_tokens = {'<PAD>', '<START>', '<END>', '<UNK>'}
    formula = []
    
    for idx in indices:
        if idx in idx_to_symbol:
            symbol = idx_to_symbol[idx]
            if not remove_special_tokens or symbol not in special_tokens:
                formula.append(symbol)
    
    return ''.join(formula)