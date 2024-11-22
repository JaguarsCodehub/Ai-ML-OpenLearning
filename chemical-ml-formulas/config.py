import torch
from pathlib import Path

class Config:
    # Data settings
    DATA_DIR = Path("data")
    PUBCHEM_SUBSET_SIZE = 10000  # Number of structures to download
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Model settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HIDDEN_DIM = 256
    NUM_CHANNELS = 3
    MAX_FORMULA_LENGTH = 50
    EMBEDDING_DIM = 256  # Dimension for token embeddings
    DROPOUT = 0.1  # Dropout rate for regularization
    NUM_LAYERS = 1  # Changed to 1 to avoid dropout warning
    
    # Training settings
    LEARNING_RATE = 3e-4  # Better default for Adam optimizer
    NUM_EPOCHS = 50
    TEACHER_FORCING_RATIO = 0.5  # Ratio for teacher forcing
    CLIP_GRAD_NORM = 1.0  # For gradient clipping
    
    # Special tokens
    PAD_IDX = 0  # Index for padding token
    START_IDX = 1  # Index for start token
    END_IDX = 2  # Index for end token
    UNK_IDX = 3  # Index for unknown token
    
    # Validation settings
    VAL_CHECK_INTERVAL = 5  # Epochs between validation checks
    EARLY_STOPPING_PATIENCE = 10  # Epochs to wait before early stopping
    
    # Save settings
    SAVE_DIR = Path("models/saved_models")  # Directory for saved models
    CHECKPOINT_PREFIX = "chemical_formula_model"  # Prefix for saved models
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    