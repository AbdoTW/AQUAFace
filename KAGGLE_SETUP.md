# Running AQUAFace on Kaggle

## Setup Instructions

### 1. Create a New Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com)
2. Click "Code" → "New Notebook"
3. Enable GPU: Settings → Accelerator → GPU T4 x2

---

### 2. Clone the Repository

```bash
!git clone https://github.com/YOUR_USERNAME/AQUAFace.git
%cd AQUAFace
```

---

### 3. Install Dependencies

```bash
!pip install -q onnx onnx2pytorch accelerate scikit-learn scipy matplotlib tqdm
```

---

### 4. Download Pretrained Models

```python
# Option 1: Upload from your local machine
# Go to Kaggle → Add Data → Upload

# Option 2: Download from Google Drive or OneDrive
!gdown --id YOUR_GOOGLE_DRIVE_FILE_ID -O pretrained_models/R100_Glint360K.onnx

# Option 3: Use Kaggle datasets
# Add as Input Dataset in Kaggle notebook settings
```

---

### 5. Download AgeDB Dataset

```python
# Option 1: From Kaggle Datasets
# Add "AgeDB" dataset as input in notebook settings

# Option 2: From Google Drive
!gdown --folder https://drive.google.com/drive/folders/YOUR_FOLDER_ID

# Option 3: Upload as Kaggle Dataset
# 1. Create AgeDB dataset on Kaggle
# 2. Add as input to notebook
# 3. Symlink to expected location
!ln -s /kaggle/input/agedb/AgeDB dataset/AgeDB
```

---

### 6. Generate Training Pairs

```python
!python generate_agedb_pairs.py
```

This creates:
- `train_data/train_pairs.txt`
- `train_data/val_pairs.txt`
- `train_data/all_pairs.txt`

---

### 7. Update Config Paths

Edit `config/config.py` for Kaggle paths:

```python
self.train_root = '/kaggle/working/AQUAFace/dataset/AgeDB'
self.train_list = '/kaggle/working/AQUAFace/train_data/train_pairs.txt'
self.checkpoints_path = '/kaggle/working/checkpoints'
self.load_model_path = '/kaggle/input/pretrained-models/R100_Glint360K.onnx'
```

---

### 8. Start Training

```python
!python train.py
```

---

### 9. Save Checkpoints

Kaggle notebooks have limited session time. Save checkpoints regularly:

```python
# Checkpoints are automatically saved to /kaggle/working/checkpoints/
# Download them before session ends or commit notebook output
```

---

### 10. Download Results

```python
from IPython.display import FileLink

# Download checkpoint
FileLink('/kaggle/working/checkpoints/resnet18_epoch_10.pth')

# Or zip and download
!zip -r checkpoints.zip checkpoints/
FileLink('checkpoints.zip')
```

---

## Kaggle-Specific Tips

### Memory Management
- Reduce batch size if OOM: `self.train_batch_size = 8`
- Use gradient accumulation for larger effective batch size

### Session Time
- Kaggle GPU sessions: 30 hours/week limit
- Save checkpoints frequently
- Use Kaggle Datasets to persist pretrained models

### Data Loading
```python
# Fast data loading on Kaggle
self.num_workers = 2  # Kaggle optimal
```

### Multi-GPU (if available)
```python
# Kaggle GPU T4 x2
self.gpu_id = '0,1'
```

---

## Example Kaggle Notebook Structure

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_USERNAME/AQUAFace.git
%cd AQUAFace
!pip install -q onnx onnx2pytorch accelerate

# Cell 2: Link datasets
!ln -s /kaggle/input/agedb dataset/AgeDB
!ln -s /kaggle/input/pretrained-r100 pretrained_models/

# Cell 3: Generate pairs
!python generate_agedb_pairs.py

# Cell 4: Train
!python train.py

# Cell 5: Evaluate
!python evaluate.py --checkpoint checkpoints/resnet18_epoch_10.pth

# Cell 6: Download
from IPython.display import FileLink
!zip -r results.zip checkpoints/ *.png
FileLink('results.zip')
```

---

## Troubleshooting

### Issue: Out of Memory
```python
# Reduce batch size in config.py
self.train_batch_size = 8
```

### Issue: Session Timeout
- Enable "Save version" to persist checkpoints
- Use smaller dataset subset for testing

### Issue: Slow Data Loading
```python
# Reduce workers
self.num_workers = 2
```

---

## Useful Kaggle Commands

```bash
# Check GPU
!nvidia-smi

# Check disk space
!df -h

# Monitor training
!tail -f train.log

# Kill process if stuck
!pkill -9 python
```