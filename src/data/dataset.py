from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
   
class DTAPairDictDataset(Dataset):
  
    def __init__(self, pair_csv, drug_feature_pkl_dict, protein_feature_pkl_dict):
    
        self.pair_csv = pair_csv
        self.df = pd.read_csv(pair_csv)
        self.drug_features = {}
        for name, pkl_path in drug_feature_pkl_dict.items():
            with open(pkl_path, 'rb') as f:
                self.drug_features[name] = pickle.load(f)
        self.protein_features = {}
        for name, pkl_path in protein_feature_pkl_dict.items():
            with open(pkl_path, 'rb') as f:
                self.protein_features[name] = pickle.load(f)
        mask = np.ones(len(self.df), dtype=bool)
        for name, d_feat in self.drug_features.items():
            mask &= self.df['smiles'].isin(d_feat)
        for name, p_feat in self.protein_features.items():
            mask &= self.df['sequence'].isin(p_feat)
        self.df = self.df[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug_vecs = {name: self.drug_features[name][row['smiles']] for name in self.drug_features}
        protein_vecs = {name: self.protein_features[name][row['sequence']] for name in self.protein_features}
        label = row['value']
        return drug_vecs, protein_vecs, label



