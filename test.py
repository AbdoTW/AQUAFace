"""
Updated test.py for AgeDB evaluation.

Usage:
    python test.py --pairs train_data/val_pairs.txt --checkpoint checkpoints/resnet18_epoch_1.pth
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from PIL import Image
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
from models import metrics
from utils.visualizer import *
import torchvision
import numpy as np
import random
random.seed(16)
import time
from config.config import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.nn import DataParallel
from collections import defaultdict
from scipy import spatial
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm
import argparse
import onnx
from onnx2pytorch import ConvertModel

torch.manual_seed(16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FDataset(Dataset):
    """Dataset for validation pairs"""
    def __init__(self, data, transforms=None):
        super(FDataset, self).__init__()
        self.data = data
        self.transforms = transforms
        self.subject_id_map = self.create_subject_id_map()
    
    def create_subject_id_map(self):
        """Map each unique person name to integer ID"""
        subject_ids = set()
        for sample in self.data:
            try:
                # Extract person name (index 1 from filename)
                filename1 = sample[0].split('/')[-1]
                person_name1 = filename1.split('_')[1]
                subject_ids.add(person_name1)
                
                filename2 = sample[1].split('/')[-1]
                person_name2 = filename2.split('_')[1]
                subject_ids.add(person_name2)
            except:
                print(f"Error processing sample: {sample}")
                raise 
        
        subject_id_map = {name: idx for idx, name in enumerate(sorted(subject_ids))}
        print(f"Total unique identities: {len(subject_id_map)}")
        return subject_id_map
    
    def __getitem__(self, idx):
        image_path1 = self.data[idx][0]
        image_path2 = self.data[idx][1]
        label = int(self.data[idx][2])
        
        # Extract person names for identity labels
        filename1 = image_path1.split('/')[-1]
        person_name1 = filename1.split('_')[1]
        label1 = self.subject_id_map[person_name1]
        
        filename2 = image_path2.split('/')[-1]
        person_name2 = filename2.split('_')[1]
        label2 = self.subject_id_map[person_name2]
        
        # Load images
        sample1 = Image.open(image_path1).convert('RGB')
        sample2 = Image.open(image_path2).convert('RGB')
        
        if self.transforms is not None:
            sample1 = self.transforms(sample1)
            sample2 = self.transforms(sample2)
        
        return sample1, sample2, label1, label2, label
    
    def __len__(self):
        return len(self.data)


def compute_cosine_similarity(features1, features2):
    """Compute cosine similarity between two feature vectors"""
    feature1 = features1.data.cpu().numpy()
    feature2 = features2.data.cpu().numpy()
    similarities = []
    
    for i in range(feature1.shape[0]):
        sim = 1 - spatial.distance.cosine(feature1[i].flatten(), feature2[i].flatten())
        if sim < 0:
            sim = 0.0
        similarities.append(sim)
    
    return np.array(similarities)


def evaluate_on_pairs(model, pairs_file, batch_size=16, num_workers=4):
    """
    Evaluate model on pairs file and generate ROC curve.
    
    Args:
        model: Trained model
        pairs_file: Path to pairs text file
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
    """
    print(f"\n{'='*60}")
    print("Evaluation on Pairs File")
    print(f"{'='*60}")
    print(f"Pairs file: {pairs_file}")
    
    # Load pairs
    data = np.loadtxt(pairs_file, dtype=object)
    print(f"Total pairs: {len(data)}")
    
    # ===== UPDATED: RGB transforms, 112x112, 3-channel normalization =====
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB
    ])
    
    # Create dataset and dataloader
    val_dataset = FDataset(data=data, transforms=transform)
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Number of batches: {len(valloader)}")
    
    # Evaluate
    model.eval()
    y_pred = []
    y_true = []
    similarities = []
    
    with torch.no_grad():
        for ii, data in enumerate(tqdm(valloader, desc="Evaluating")):
            image1, image2, label1, label2, label = data
            
            # Extract features
            feature1 = model(image1)
            feature2 = model(image2)
            
            # Compute similarities
            sims = compute_cosine_similarity(feature1, feature2)
            
            y_true.extend(label.numpy().tolist())
            similarities.extend(sims.tolist())
    
    y_true = np.array(y_true)
    similarities = np.array(similarities)
    
    # Compute ROC curve
    print(f"\n{'='*60}")
    print("Computing ROC Curve")
    print(f"{'='*60}")
    
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Print TPR at specific FPR values
    print(f"\nTPR at Fixed FPR:")
    fpr_values = [0.0001, 0.001, 0.01, 0.1]
    
    for fpr_value in fpr_values:
        idx = np.where(fpr >= fpr_value)[0]
        if len(idx) > 0:
            idx = idx[0]
            print(f"  TPR at FPR={fpr_value*100:.2f}%: {tpr[idx]:.4f}")
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal Threshold (Youden's J):")
    print(f"  Threshold: {optimal_threshold:.4f}")
    print(f"  TPR: {tpr[optimal_idx]:.4f}")
    print(f"  FPR: {fpr[optimal_idx]:.4f}")
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xscale('log')
    plt.xlim([1e-5, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Face Verification')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    output_path = 'roc_curve_evaluation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ ROC curve saved to: {output_path}")
    
    return {
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'fpr': fpr,
        'tpr': tpr
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate face recognition model on pairs file')
    parser.add_argument('--pairs', type=str, required=True, help='Path to pairs file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    
    # Load base ONNX model
    onnx_path = '/home/tawfik/git/AQUAFace/pretrained_models/R100_Glint360K.onnx'
    print(f"Loading base model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    model = ConvertModel(onnx_model)
    
    # Load fine-tuned checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle DataParallel wrapper
    if 'module.' in list(checkpoint.keys())[0]:
        new_checkpoint = {}
        for key, value in checkpoint.items():
            new_key = key.replace('module.', '')
            new_checkpoint[new_key] = value
        checkpoint = new_checkpoint
    
    model.load_state_dict(checkpoint, strict=False)
    model = DataParallel(model).to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Evaluate on pairs
    results = evaluate_on_pairs(
        model,
        args.pairs,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"\n{'='*60}")
    print("✅ Evaluation Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()