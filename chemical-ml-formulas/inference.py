import os
import traceback
import torch
from PIL import Image
from torchvision import transforms
from models.cnn_encoder import ChemicalStructureEncoder
from models.formula_decoder import FormulaDecoder
from utils import indices_to_formula
import matplotlib.pyplot as plt


class ChemicalFormulaPredictor:
    def __init__(self, model_path):
        # Load model with proper path handling
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.config = checkpoint['config']
        
        # Get vocabulary from checkpoint
        self.symbol_to_idx = checkpoint['vocab']['symbol_to_idx']
        self.idx_to_symbol = checkpoint['vocab']['idx_to_symbol']
        
        # Initialize models
        self.encoder = ChemicalStructureEncoder(self.config.HIDDEN_DIM)
        self.decoder = FormulaDecoder(
            len(self.symbol_to_idx), 
            self.config.HIDDEN_DIM
        )
        
        # Load weights
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Set to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("Model loaded successfully!")

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            try:
                # Get encoder output
                encoder_output = self.encoder(image)
                print(f"Encoder output shape: {encoder_output.shape}")
                
                # Initialize first token as START
                current_token = torch.tensor([[self.symbol_to_idx['<START>']]])
                
                # Generate formula token by token
                max_length = 50
                pred_indices = []
                
                for step in range(max_length):
                    # Get decoder output
                    decoder_output = self.decoder(encoder_output, current_token)
                    print(f"\nStep {step}:")
                    print(f"Decoder output shape: {decoder_output.shape}")
                    
                    # Get probabilities
                    if len(decoder_output.shape) == 3:
                        logits = decoder_output[0, -1]
                    else:
                        logits = decoder_output[-1]
                    
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Print top 3 most likely tokens
                    top_probs, top_indices = probs.topk(3)
                    print("\nTop 3 predictions:")
                    for prob, idx in zip(top_probs, top_indices):
                        token = self.idx_to_symbol[idx.item()]
                        print(f"{token}: {prob.item():.4f}")
                    
                    # Get next token
                    next_token_idx = logits.argmax().item()
                    
                    # Break if END token
                    if next_token_idx == self.symbol_to_idx['<END>']:
                        print("Found END token")
                        break
                    
                    pred_indices.append(next_token_idx)
                    current_token = torch.cat([current_token, torch.tensor([[next_token_idx]])], dim=1)
                
                return indices_to_formula(pred_indices, self.idx_to_symbol)
                
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                import traceback
                traceback.print_exc()
                return ""

    def predict_and_display(self, image_path):
        """Predict formula and display the image with result"""
        # Load and show image
        img = Image.open(image_path).convert('RGB')
        formula = self.predict(image_path)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Input Chemical Structure')
        
        # Display prediction information
        ax2.axis('off')
        ax2.text(0.1, 0.7, f'Predicted Formula:', fontsize=12, fontweight='bold')
        ax2.text(0.1, 0.5, formula, fontsize=14, color='blue')
        ax2.text(0.1, 0.3, f'Confidence Score: {self.get_confidence_score():.2f}%', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return formula

    def get_confidence_score(self):
        """Calculate confidence score based on prediction probabilities"""
        if not hasattr(self, '_last_prediction_probs'):
            return 0.0
        
        # Average probability of selected tokens
        return float(torch.mean(self._last_prediction_probs).item() * 100)

def test_single_image(image_path):
    """Test a single image and display results"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'saved_models', 'chemical_formula_model.pth')
    
    try:
        predictor = ChemicalFormulaPredictor(model_path)
        formula = predictor.predict_and_display(image_path)
        return formula
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None

# Example usage:
if __name__ == "__main__":
    # Test a single image
    image_path = "data/raw/compound_1.png"
    formula = test_single_image(image_path)
    if formula:
        print(f"\nPredicted Formula: {formula}")