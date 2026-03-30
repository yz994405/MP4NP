import torch
import pandas as pd
from pathlib import Path
import yaml
import sys
from torch.utils.data import DataLoader
import pickle

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.models.main import DTAnet
from src.data.dataset import DTAPairDictDataset
from src.utils.scaler import fit_and_save_scalers, load_scalers, multimodal_collate_fn_factory
from src.utils.trainer import MultiModalTrainer 
from src.utils.data_splitter import DTADataSplitter


base_dir = Path(__file__).resolve().parent.parent
config_path = str(base_dir / "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
input_data_path = config.get('data', {}).get('input_data_path', 'data/raw/davis_data.csv')
chembert_pkl = config.get('chembert_drug_feature_pkl', 'data/processed/chembert/extracted_features/drug_features.pkl')
grover_pkl = config.get('grover_drug_feature_pkl', 'data/processed/grover/extracted_features/drug_features.pkl')
graphmvp_pkl = config.get('graphmvp_drug_feature_pkl', 'data/processed/graphmvp/extracted_features/drug_features.pkl')
protein_pkl = config.get('chembert_protein_feature_pkl', 'data/processed/chembert/extracted_features/protein_features.pkl')

# Get feature dimensions
with open(chembert_pkl, 'rb') as f:
    chembert_dim = next(iter(pickle.load(f).values())).shape[0]
with open(grover_pkl, 'rb') as f:
    grover_dim = next(iter(pickle.load(f).values())).shape[0]
with open(graphmvp_pkl, 'rb') as f:
    graphmvp_dim = next(iter(pickle.load(f).values())).shape[0]
with open(protein_pkl, 'rb') as f:
    protein_dim = next(iter(pickle.load(f).values())).shape[0]

# Initialize dataset
drug_feature_pkl_dict = {'chembert': chembert_pkl, 'grover': grover_pkl, 'graphmvp': graphmvp_pkl}
protein_feature_pkl_dict = {'esm2': protein_pkl}
dataset = DTAPairDictDataset(input_data_path, drug_feature_pkl_dict, protein_feature_pkl_dict)

# Data split configuration
data_split_config = config.get('data_split', {})
split_strategy = data_split_config.get('strategy', 'random')
val_ratio = data_split_config.get('val_ratio', 0.2)
train_ratio = data_split_config.get('train_ratio', 0.8)
use_cross_validation = data_split_config.get('use_cross_validation', True)
rand_seed = data_split_config.get('rand_seed', 42)
n_folds = data_split_config.get('n_folds', 5)
print(f"Seed: {rand_seed}")

splitter = DTADataSplitter(random_seed=rand_seed)
if split_strategy == 'random':
    train_indices, val_indices = splitter.split_data(dataset=dataset,split_strategy=split_strategy,use_cross_validation=False,train_ratio=train_ratio)
else:
    train_indices, val_indices = splitter.split_data(dataset=dataset,split_strategy=split_strategy,use_cross_validation=False,val_ratio=val_ratio)

# Simple split format
folds = [(train_indices, val_indices)]
n_folds = 1

train_config = config.get('train', {})
lr_rate = float(train_config.get('lr_rate', 0.001))
weight_decay = float(train_config.get('weight_decay', 1e-5))
use_early_stopping = train_config.get('use_early_stopping', False)
patience = train_config.get('patience', 20)
num_epochs = train_config.get('num_epochs', 200)
batch_size = train_config.get('batch_size', 64)
map_dim = config.get('feature_mapping_dim', 1024)
hidden_dims = config.get('mlp', {}).get('hidden_dims', [1024, 512, 256, 128, 64])
dropout_rate = config.get('mlp', {}).get('dropout_rate', 0.2)

print(f"\nTraining hyperparameters:")
print(f"  - Learning rate: {lr_rate}")
print(f"  - Weight decay: {weight_decay}")
print(f"  - Early stopping: {use_early_stopping}")
print(f"  - Patience: {patience}")
print(f"  - Total epochs: {num_epochs}")
print(f"  - Batch size: {batch_size}")
print(f"  - Feature mapping dimension: {map_dim}")
print(f"  - MLP hidden layers: {hidden_dims}")
print(f"  - Dropout rate: {dropout_rate}")

# Model save root directory
model_save_root = config.get('model_save_root', 'models/model/')
model_save_root = base_dir / model_save_root
model_save_root.mkdir(parents=True, exist_ok=True)
print(f"\nModel save root directory: {model_save_root}")

# Multi-fold training
for fold_idx, (train_indices, val_indices) in enumerate(folds, 1):
    print(f"\n{'='*60}")
    print(f"Starting fold {fold_idx}/{n_folds} training")
    print(f"{'='*60}")

    # Current fold save directory
    fold_save_dir = model_save_root / f"fold_{fold_idx}"
    fold_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Current fold save directory: {fold_save_dir}")

    # Scaler directory
    fold_scaler_dir = fold_save_dir / "scalers"
    fold_scaler_dir.mkdir(parents=True, exist_ok=True)

    # Fit/Load Scaler
    if (fold_scaler_dir / 'scaler_chembert.pkl').exists() and \
       (fold_scaler_dir / 'scaler_grover.pkl').exists() and \
       (fold_scaler_dir / 'scaler_graphmvp.pkl').exists() and \
       (fold_scaler_dir / 'scaler_protein.pkl').exists():
        chembert_scaler, grover_scaler, graphmvp_scaler, protein_scaler = load_scalers(fold_scaler_dir)
    else:
        chembert_scaler, grover_scaler, graphmvp_scaler, protein_scaler = fit_and_save_scalers(dataset=dataset,train_indices=train_indices,save_dir=fold_scaler_dir)

    # Create Collate function
    collate_fn = multimodal_collate_fn_factory(chembert_scaler=chembert_scaler,grover_scaler=grover_scaler,graphmvp_scaler=graphmvp_scaler,protein_scaler=protein_scaler)

    # Generate train/val dataset subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create train/val DataLoader
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn,pin_memory=True)

    # Initialize model
    model = DTAnet(
        ch_dim=chembert_dim,
        gr_dim=grover_dim,
        gm_dim=graphmvp_dim,
        prot_dim=protein_dim,
        map_dim=map_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    ).to(device)
    
    # MultiModalTrainer trainer
    trainer = MultiModalTrainer(model=model,device=device,lr_rate=lr_rate,weight_decay=weight_decay)
    result_file = fold_save_dir / "results.csv"
    history = trainer.train(train_loader=train_loader,val_loader=val_loader,num_epochs=num_epochs,patience=patience,use_early_stopping=use_early_stopping,save_dir=str(fold_save_dir),result_file=str(result_file))
   

    