import torch
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def fit_and_save_scalers(dataset, train_indices, save_dir):
    chembert_arr = np.stack([dataset[i][0]['chembert'] for i in train_indices])
    grover_arr = np.stack([dataset[i][0]['grover'] for i in train_indices])
    graphmvp_arr = np.stack([dataset[i][0]['graphmvp'] for i in train_indices])
    protein_arr = np.stack([dataset[i][1]['esm2'] for i in train_indices])
    chembert_scaler = StandardScaler().fit(chembert_arr)
    grover_scaler = StandardScaler().fit(grover_arr)
    graphmvp_scaler = StandardScaler().fit(graphmvp_arr)
    protein_scaler = StandardScaler().fit(protein_arr)
    with open(str(save_dir / 'scaler_chembert.pkl'), 'wb') as f:
        pickle.dump(chembert_scaler, f)
    with open(str(save_dir / 'scaler_grover.pkl'), 'wb') as f:
        pickle.dump(grover_scaler, f)
    with open(str(save_dir / 'scaler_graphmvp.pkl'), 'wb') as f:
        pickle.dump(graphmvp_scaler, f)
    with open(str(save_dir / 'scaler_protein.pkl'), 'wb') as f:
        pickle.dump(protein_scaler, f)
    return chembert_scaler, grover_scaler, graphmvp_scaler, protein_scaler

def load_scalers(save_dir):
    with open(str(save_dir / 'scaler_chembert.pkl'), 'rb') as f:
        chembert_scaler = pickle.load(f)
    with open(str(save_dir / 'scaler_grover.pkl'), 'rb') as f:
        grover_scaler = pickle.load(f)
    with open(str(save_dir / 'scaler_graphmvp.pkl'), 'rb') as f:
        graphmvp_scaler = pickle.load(f)
    with open(str(save_dir / 'scaler_protein.pkl'), 'rb') as f:
        protein_scaler = pickle.load(f)
    
    return chembert_scaler, grover_scaler, graphmvp_scaler, protein_scaler

def multimodal_collate_fn_factory(chembert_scaler, grover_scaler, graphmvp_scaler, protein_scaler):
    def collate_fn(batch):
        chembert_x = torch.tensor(chembert_scaler.transform(np.array([item[0]['chembert'] for item in batch])), dtype=torch.float32)
        grover_x = torch.tensor(grover_scaler.transform(np.array([item[0]['grover'] for item in batch])), dtype=torch.float32)
        graphmvp_x = torch.tensor(graphmvp_scaler.transform(np.array([item[0]['graphmvp'] for item in batch])), dtype=torch.float32)
        protein_x = torch.tensor(protein_scaler.transform(np.array([item[1]['esm2'] for item in batch])), dtype=torch.float32)
        affinities = torch.tensor(np.array([item[2] for item in batch]), dtype=torch.float32)
        return chembert_x, grover_x, graphmvp_x, protein_x, affinities
    return collate_fn

