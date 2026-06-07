import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import config

class FeatureExtractor:
    def __init__(self):
        # Load pre-trained ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # Remove the final classification layer to get the raw features (embeddings)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval() # Set model to evaluation mode

        # Standard preprocessing for PyTorch vision models
        self.preprocess = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, img_path):
        """Extracts a feature vector from an image."""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0) # Add batch dimension
            
            with torch.no_grad():
                feature = self.model(img_tensor)
                
            return feature.squeeze().numpy() # Return as a 1D numpy array
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

# Quick test block (runs only if you execute this script directly)
if __name__ == "__main__":
    print("Loading model...")
    extractor = FeatureExtractor()
    print("Model loaded successfully! Ready to extract features.")
