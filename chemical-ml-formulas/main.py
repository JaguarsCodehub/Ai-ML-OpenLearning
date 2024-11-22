from config import Config
from data import dataset
from models.cnn_encoder import ChemicalStructureEncoder
from models.formula_decoder import FormulaDecoder
from data.dataset import ChemicalStructureDataset
from utils import save_checkpoint, load_checkpoint, indices_to_formula
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
import os
from multiprocessing import freeze_support
from data.download_pubchem import download_pubchem_data
from torchvision import transforms
import numpy as np



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

def calculate_accuracy(predictions, targets, idx_to_symbol):
    """Calculate accuracy metrics for predictions"""
    total = len(predictions)
    exact_matches = 0
    partial_matches = 0
    char_accuracy = 0
    
    for pred, target in zip(predictions, targets):
        # Convert indices to formulas if needed
        if isinstance(pred, (torch.Tensor, np.integer, int)):
            # Handle single integer predictions
            if isinstance(pred, (np.integer, int)):
                pred = [pred]
            pred = indices_to_formula(pred, idx_to_symbol)
            
        if isinstance(target, (torch.Tensor, np.integer, int)):
            # Handle single integer targets
            if isinstance(target, (np.integer, int)):
                target = [target]
            target = indices_to_formula(target, idx_to_symbol)
        
        # Convert to strings if they aren't already
        pred = str(pred)
        target = str(target)
        
        # Exact match
        if pred == target:
            exact_matches += 1
        
        # Calculate common characters
        pred_chars = list(pred)
        target_chars = list(target)
        common_chars = sum(1 for p, t in zip(pred_chars, target_chars) if p == t)
        max_len = max(len(pred), len(target))
        
        # Partial match (at least 50% of characters correct)
        if max_len > 0 and common_chars / max_len >= 0.5:
            partial_matches += 1
        
        # Character-level accuracy
        char_accuracy += common_chars / max_len if max_len > 0 else 0
    
    # Calculate final metrics
    metrics = {
        'exact_match': exact_matches / total if total > 0 else 0,
        'partial_match': partial_matches / total if total > 0 else 0,
        'char_accuracy': char_accuracy / total if total > 0 else 0
    }
    
    return metrics

def train_model(encoder, decoder, train_loader, val_loader, config, idx_to_symbol):
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    optimizer = Adam(list(encoder.parameters()) + 
                    list(decoder.parameters()), 
                    lr=config.LEARNING_RATE)
    
    encoder.to(config.DEVICE)
    decoder.to(config.DEVICE)
    
    best_val_loss = float('inf')
    best_val_accuracy = 0
    
    for epoch in range(config.NUM_EPOCHS):
        # Training
        encoder.train()
        decoder.train()
        total_loss = 0
        batch_count = 0
        epoch_predictions = []
        epoch_targets = []
        
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("Training:")
        
        for batch in tqdm(train_loader):
            if batch['image'].size(0) == 0:
                continue
                
            images = batch['image'].to(config.DEVICE)
            formulas = batch['formula'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                encoder_output = encoder(images)
                decoder_input = formulas[:, :-1]
                target = formulas[:, 1:]
                
                # Get decoder outputs
                outputs = decoder(encoder_output, decoder_input, Config.TEACHER_FORCING_RATIO)
                
                # Ensure outputs and targets have the same sequence length
                min_len = min(outputs.size(1), target.size(1))
                outputs = outputs[:, :min_len, :]
                target = target[:, :min_len]
                
                # Reshape for loss calculation
                batch_size = outputs.size(0)
                outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))
                target_flat = target.contiguous().view(-1)
                
                # Calculate loss
                loss = criterion(outputs_flat, target_flat)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Collect predictions and targets for accuracy calculation
                predictions = outputs.argmax(dim=-1)  # [batch_size, seq_len]
                for pred_seq, target_seq in zip(predictions.cpu().numpy(), target.cpu().numpy()):
                    # Remove padding and special tokens
                    pred_seq = pred_seq[pred_seq != Config.PAD_IDX]
                    target_seq = target_seq[target_seq != Config.PAD_IDX]
                    epoch_predictions.append(pred_seq)
                    epoch_targets.append(target_seq)
                    
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        print(f"Average Training Loss: {avg_loss:.4f}")
        
        # Calculate training metrics using passed idx_to_symbol
        train_metrics = calculate_accuracy(epoch_predictions, epoch_targets, idx_to_symbol)
        
        print(f"Training Metrics:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Exact Match Accuracy: {train_metrics['exact_match']:.2%}")
        print(f"Partial Match Accuracy: {train_metrics['partial_match']:.2%}")
        print(f"Character Accuracy: {train_metrics['char_accuracy']:.2%}")
        
        # Validation
        if epoch % config.VAL_CHECK_INTERVAL == 0:
            encoder.eval()
            decoder.eval()
            val_loss = 0
            val_batch_count = 0
            val_predictions = []
            val_targets = []
            
            print("\nValidation:")
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader)):
                    if batch['image'].size(0) == 0:
                        continue
                        
                    try:
                        images = batch['image'].to(config.DEVICE)
                        formulas = batch['formula'].to(config.DEVICE)
                        
                        # Get predictions
                        encoder_output = encoder(images)
                        outputs = decoder(encoder_output, formulas[:, :-1])  # No teacher forcing during validation
                        target = formulas[:, 1:]
                        
                        # Calculate validation loss
                        outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))
                        target_flat = target.contiguous().view(-1)
                        loss = criterion(outputs_flat, target_flat)
                        
                        val_loss += loss.item()
                        val_batch_count += 1
                        
                        # Collect predictions and targets
                        predictions = outputs.argmax(dim=-1)  # [batch_size, seq_len]
                        for pred_seq, target_seq in zip(predictions.cpu().numpy(), target.cpu().numpy()):
                            # Remove padding and special tokens
                            pred_seq = pred_seq[pred_seq != Config.PAD_IDX]
                            target_seq = target_seq[target_seq != Config.PAD_IDX]
                            val_predictions.append(pred_seq)
                            val_targets.append(target_seq)
                        
                        # Show some predictions (first batch only)
                        if batch_idx == 0:
                            for j in range(min(3, predictions.size(0))):
                                pred_formula = indices_to_formula(predictions[j], idx_to_symbol)
                                actual_formula = indices_to_formula(formulas[j], idx_to_symbol)
                                print(f"\nSample {j}:")
                                print(f"Predicted: {pred_formula}")
                                print(f"Actual: {actual_formula}")
                            
                    except Exception as e:
                        print(f"Error in validation batch: {str(e)}")
                        continue
            
            # Calculate average validation loss
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            
            # Calculate validation metrics using passed idx_to_symbol
            val_metrics = calculate_accuracy(val_predictions, val_targets, idx_to_symbol)
            
            print(f"\nValidation Metrics:")
            print(f"Loss: {avg_val_loss:.4f}")
            print(f"Exact Match Accuracy: {val_metrics['exact_match']:.2%}")
            print(f"Partial Match Accuracy: {val_metrics['partial_match']:.2%}")
            print(f"Character Accuracy: {val_metrics['char_accuracy']:.2%}")
            
            # Save best model based on validation accuracy
            if val_metrics['exact_match'] > best_val_accuracy:
                best_val_accuracy = val_metrics['exact_match']
                save_checkpoint({
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'val_accuracy': best_val_accuracy,
                    'config': Config,
                    'vocab': {
                        'symbol_to_idx': dataset.symbol_to_idx,
                        'idx_to_symbol': idx_to_symbol
                    }
                }, os.path.join(Config.SAVE_DIR, f"{Config.CHECKPOINT_PREFIX}_best.pth"))
                print("Saved new best model!")

def evaluate_model(encoder, decoder, test_loader, config, idx_to_symbol):
    encoder.eval()
    decoder.eval()
    all_predictions = []
    all_targets = []
    test_loss = 0
    
    print("\nEvaluating model...")
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
                all_targets.append(actual_formula)
    
    # Calculate final metrics
    metrics = calculate_accuracy(all_predictions, all_targets, idx_to_symbol)
    
    print("\nTest Results:")
    print(f"Exact Match Accuracy: {metrics['exact_match']:.2%}")
    print(f"Partial Match Accuracy: {metrics['partial_match']:.2%}")
    print(f"Character Accuracy: {metrics['char_accuracy']:.2%}")
    
    # Print some example predictions
    print("\nExample Predictions:")
    for i in range(min(5, len(all_predictions))):
        pred = indices_to_formula(all_predictions[i], dataset.idx_to_symbol)
        target = indices_to_formula(all_targets[i], dataset.idx_to_symbol)
        print(f"\nExample {i+1}:")
        print(f"Predicted: {pred}")
        print(f"Actual: {target}")
    
    return metrics

def beam_search_decode(encoder_output, decoder, beam_width=3, max_length=50):
    batch_size = encoder_output.size(0)
    vocab_size = decoder.fc.out_features
    
    # Initialize beams with START token
    beams = [(torch.full((1,), dataset.symbol_to_idx['<START>'], device=encoder_output.device), 0)]
    
    for _ in range(max_length):
        candidates = []
        for sequence, score in beams:
            if sequence[-1] == dataset.symbol_to_idx['<END>']:
                candidates.append((sequence, score))
                continue
                
            # Get predictions for next token
            current_token = sequence[-1].unsqueeze(0)
            with torch.no_grad():
                pred = decoder(encoder_output, current_token)
                probs = torch.log_softmax(pred[-1], dim=-1)
            
            # Get top k candidates
            values, indices = probs.topk(beam_width)
            for value, idx in zip(values, indices):
                new_sequence = torch.cat([sequence, idx.unsqueeze(0)])
                new_score = score + value.item()
                candidates.append((new_sequence, new_score))
        
        # Select top beam_width candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Stop if all beams end with END token
        if all(b[0][-1] == dataset.symbol_to_idx['<END>'] for b in beams):
            break
    
    # Return best sequence
    return beams[0][0]

def main():
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)

    # Download the data and get metadata
    successful_compounds, metadata_df = download_pubchem_data("data/raw", n_compounds=100)

    # Only proceed if we have successfully downloaded compounds
    if len(successful_compounds) > 0 and metadata_df is not None:
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        # Create dataset with transforms
        dataset = ChemicalStructureDataset(
            csv_file="data/raw/metadata.csv",
            img_dir="data/raw",
            transform=transform
        )
        
        print("\nDataset created successfully!")
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of successful downloads: {len(successful_compounds)}")
    else:
        print("No compounds were downloaded successfully. Please check your internet connection and try again.")
        return

    # Create data splits
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders with the collate_fn
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    # Initialize models
    encoder = ChemicalStructureEncoder(Config.HIDDEN_DIM)
    decoder = FormulaDecoder(len(dataset.symbol_to_idx), Config.HIDDEN_DIM)

    try:
        # Train the model with vocabulary
        train_model(
            encoder, 
            decoder, 
            train_loader, 
            val_loader, 
            Config,
            dataset.idx_to_symbol  # Pass the vocabulary
        )

        # Evaluate the model
        print("\nEvaluating final model...")
        test_metrics = evaluate_model(
            encoder, 
            decoder, 
            val_loader, 
            Config,
            dataset.idx_to_symbol  # Pass the vocabulary here too
        )

        # Save the final model with metrics
        save_path = os.path.join(os.path.dirname(__file__), "models", "saved_models")
        os.makedirs(save_path, exist_ok=True)

        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'config': Config,
            'vocab': {
                'symbol_to_idx': dataset.symbol_to_idx,
                'idx_to_symbol': dataset.idx_to_symbol
            },
            'metrics': test_metrics
        }, os.path.join(save_path, 'chemical_formula_model.pth'))

        print("\nFinal Model Performance:")
        print(f"Exact Match Accuracy: {test_metrics['exact_match']:.2%}")
        print(f"Partial Match Accuracy: {test_metrics['partial_match']:.2%}")
        print(f"Character Accuracy: {test_metrics['char_accuracy']:.2%}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == '__main__':
    freeze_support()  # Add this for Windows
    main()