import torch
from torch.utils.data import DataLoader
from models.formula_decoder import FormulaDecoder
from models.cnn_encoder import ChemicalStructureEncoder
from data.dataset import ChemicalStructureDataset
from config import Config
from utils import indices_to_formula
from tqdm import tqdm
from data import dataset

def collate_fn(batch):
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return {
            'image': torch.empty(0, 3, 224, 224),
            'formula': torch.empty(0, 1),
            'formula_length': torch.empty(0, dtype=torch.long)
        }
    
    # Get images and formulas
    images = [item['image'] for item in batch]
    formulas = [item['formula'] for item in batch]
    
    # Stack images
    images = torch.stack(images)
    
    # Find max formula length in this batch
    max_length = max(formula.size(0) for formula in formulas)
    
    # Pad formulas to max length
    padded_formulas = []
    formula_lengths = []
    
    for formula in formulas:
        length = formula.size(0)
        formula_lengths.append(length)
        padding_size = max_length - length
        padded_formula = torch.nn.functional.pad(
            formula, 
            (0, padding_size), 
            value=Config.PAD_IDX
        )
        padded_formulas.append(padded_formula)
    
    # Stack padded formulas and convert lengths to tensor
    padded_formulas = torch.stack(padded_formulas)
    formula_lengths = torch.tensor(formula_lengths, dtype=torch.long)
    
    return {
        'image': images,
        'formula': padded_formulas,
        'formula_length': formula_lengths
    }


def evaluate_model(encoder, decoder, test_loader, config):
    encoder.eval()
    decoder.eval()
    all_predictions = []
    all_actuals = []
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(config.DEVICE)
            formulas = batch['formula']
            
            # Get predictions
            encoder_output = encoder(images)
            current_token = torch.full((images.size(0), 1), 
                                    dataset.symbol_to_idx['<START>'],
                                    device=config.DEVICE)
            
            predicted_formula = []
            
            # Generate formula token by token
            for _ in range(50):
                decoder_output = decoder(encoder_output, current_token)
                next_token = decoder_output.argmax(dim=-1)
                predicted_formula.append(next_token)
                
                if next_token.item() == dataset.symbol_to_idx['<END>']:
                    break
                    
                current_token = next_token
            
            # Convert predictions to formulas
            predicted_indices = torch.cat(predicted_formula, dim=1).cpu().numpy()
            for pred_idx, actual in zip(predicted_indices, formulas):
                pred_formula = indices_to_formula(pred_idx, dataset.idx_to_symbol)
                actual_formula = indices_to_formula(actual.cpu().numpy(), dataset.idx_to_symbol)
                all_predictions.append(pred_formula)
                all_actuals.append(actual_formula)
    
    # Calculate metrics
    correct = sum(1 for p, a in zip(all_predictions, all_actuals) if p == a)
    total = len(all_predictions)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nEvaluation Results:")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return all_predictions, all_actuals


# Add this to main.py after training
def test_model():
    # Load your saved model
    model_path = "models/saved_models/chemical_formula_model.pth"
    
    # Create test dataset
    test_dataset = ChemicalStructureDataset(
        csv_file="data/raw/metadata.csv",
        img_dir="data/raw"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    # Load model
    checkpoint = torch.load(model_path)
    encoder = ChemicalStructureEncoder(Config.HIDDEN_DIM)
    decoder = FormulaDecoder(len(test_dataset.symbol_to_idx), Config.HIDDEN_DIM)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Test model
    predictions, actuals = evaluate_model(encoder, decoder, test_loader, Config)
    
    # Print some examples
    print("\nExample Predictions:")
    for i in range(min(5, len(predictions))):
        print(f"\nImage {i+1}:")
        print(f"Predicted: {predictions[i]}")
        print(f"Actual: {actuals[i]}")

if __name__ == "__main__":
    test_model()
