"""
Test fine-tuned AQUAFace model on two images.

Usage:
    python test_two_images.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg --checkpoint checkpoints/resnet18_epoch_1.pth
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cosine
import argparse
import os
import onnx
from onnx2pytorch import ConvertModel


class FaceVerifier:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize the face verifier with a fine-tuned checkpoint.
        
        Args:
            checkpoint_path: Path to the fine-tuned .pth checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load base ONNX model
        print("Loading base ONNX model...")
        onnx_path = '/home/tawfik/git/AQUAFace/pretrained_models/R100_Glint360K.onnx'
        onnx_model = onnx.load(onnx_path)
        self.model = ConvertModel(onnx_model)
        
        # Load fine-tuned weights
        print(f"Loading fine-tuned checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DataParallel wrapper
        if 'module.' in list(checkpoint.keys())[0]:
            # Remove 'module.' prefix from state dict keys
            new_checkpoint = {}
            for key, value in checkpoint.items():
                new_key = key.replace('module.', '')
                new_checkpoint[new_key] = value
            checkpoint = new_checkpoint
        
        # Load weights (strict=False to ignore metric_fc layer if present)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_image(self, image_path):
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)
    
    def get_embedding(self, image_path):
        """
        Extract face embedding from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Face embedding as numpy array
        """
        image_tensor = self.load_image(image_path)
        
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        # Convert to numpy and normalize
        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
        
        return embedding
    
    def compare_faces(self, img1_path, img2_path):
        """
        Compare two face images and return similarity.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Dictionary with similarity metrics
        """
        print(f"\n{'='*60}")
        print("Face Verification")
        print(f"{'='*60}")
        print(f"Image 1: {img1_path}")
        print(f"Image 2: {img2_path}")
        
        # Extract embeddings
        print("\nExtracting embeddings...")
        emb1 = self.get_embedding(img1_path)
        emb2 = self.get_embedding(img2_path)
        
        # Compute similarity metrics
        cosine_distance = cosine(emb1, emb2)
        cosine_similarity = 1 - cosine_distance
        euclidean_distance = np.linalg.norm(emb1 - emb2)
        
        # Thresholds (you can adjust these based on your validation results)
        # Common thresholds for face verification:
        threshold_strict = 0.6    # High confidence
        threshold_normal = 0.5    # Normal confidence
        threshold_loose = 0.4     # Low confidence
        
        # Make decision
        if cosine_similarity >= threshold_strict:
            decision = "✅ SAME PERSON (High Confidence)"
        elif cosine_similarity >= threshold_normal:
            decision = "✅ SAME PERSON (Normal Confidence)"
        elif cosine_similarity >= threshold_loose:
            decision = "⚠️  SAME PERSON (Low Confidence)"
        else:
            decision = "❌ DIFFERENT PERSON"
        
        results = {
            'cosine_similarity': cosine_similarity,
            'cosine_distance': cosine_distance,
            'euclidean_distance': euclidean_distance,
            'decision': decision
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        print(f"Cosine Similarity:    {cosine_similarity:.4f}  (Range: -1 to 1, higher is more similar)")
        print(f"Cosine Distance:      {cosine_distance:.4f}  (Range: 0 to 2, lower is more similar)")
        print(f"Euclidean Distance:   {euclidean_distance:.4f}")
        print(f"\nDecision: {decision}")
        print(f"{'='*60}\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Test fine-tuned face recognition model on two images')
    parser.add_argument('--img1', type=str, required=True, help='Path to first image')
    parser.add_argument('--img2', type=str, required=True, help='Path to second image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fine-tuned checkpoint (.pth)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize verifier
    verifier = FaceVerifier(args.checkpoint, device=args.device)
    
    # Compare faces
    results = verifier.compare_faces(args.img1, args.img2)
    
    # Return status code (0 = same person, 1 = different person)
    if "SAME PERSON" in results['decision']:
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())