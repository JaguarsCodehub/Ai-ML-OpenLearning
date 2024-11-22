import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class ChemicalStructureDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Create vocabulary
        self.symbol_to_idx = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3,
            'C': 4, 'H': 5, 'O': 6, 'N': 7, 'P': 8, 'S': 9,
            'F': 10, 'Cl': 11, 'Br': 12, 'I': 13,
            '0': 14, '1': 15, '2': 16, '3': 17, '4': 18,
            '5': 19, '6': 20, '7': 21, '8': 22, '9': 23,
            '+': 24, '-': 25
        }
        self.idx_to_symbol = {v: k for k, v in self.symbol_to_idx.items()}
        
        # Build vocabulary from formulas
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build vocabulary from chemical formulas."""
        for formula in self.metadata['formula']:
            symbols = self._tokenize_formula(formula)
            for symbol in symbols:
                if symbol not in self.symbol_to_idx:
                    idx = len(self.symbol_to_idx)
                    self.symbol_to_idx[symbol] = idx
                    self.idx_to_symbol[idx] = symbol

    def _tokenize_formula(self, formula):
        """Split formula into tokens (e.g., 'H2O' -> ['H', '2', 'O'])."""
        tokens = []
        current_token = ''
        
        for char in formula:
            if char.isupper():  # Start of new element
                if current_token:
                    tokens.append(current_token)
                current_token = char
            elif char.islower():  # Continuation of element
                current_token += char
            else:  # Numbers or other characters
                if current_token:
                    tokens.append(current_token)
                tokens.append(char)
                current_token = ''
        
        if current_token:
            tokens.append(current_token)
        
        return tokens

    def formula_to_indices(self, formula):
        """Convert formula string to sequence of indices."""
        tokens = self._tokenize_formula(formula)
        indices = [self.symbol_to_idx['<START>']]
        
        for token in tokens:
            if token in self.symbol_to_idx:
                indices.append(self.symbol_to_idx[token])
            else:
                indices.append(self.symbol_to_idx['<UNK>'])
        
        indices.append(self.symbol_to_idx['<END>'])
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path and load image
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx]['image_path'])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            return None

        # Apply transforms if any
        if self.transform is not None:
            image = self.transform(image)

        # Get formula and convert to indices
        formula = self.metadata.iloc[idx]['formula']
        formula_indices = self.formula_to_indices(formula)

        return {
            'image': image,
            'formula': formula_indices
        }