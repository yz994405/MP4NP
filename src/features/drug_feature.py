import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Union, Dict, List
import yaml
import pickle

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class ChemBERTExtractor:
    def __init__(self, config_path: str):
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self._init_model()
        
    def _init_model(self):
        try:
            model_path = self.config.get('drug_feature', {}).get('model_path')
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
        
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            raise
    
    def get_feat(self, smiles: str) -> Union[torch.Tensor, np.ndarray]:
    
        max_length = self.config.get('drug_feature', {}).get('max_length', 128)
        tokens = self.tokenizer(smiles, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            last_layer_repr = outputs.last_hidden_state.squeeze(0)  
            attention_mask = tokens['attention_mask'].squeeze(0) 
            masked_repr = last_layer_repr * attention_mask.unsqueeze(-1) 
            sum_repr = masked_repr.sum(dim=0) 
            seq_len = attention_mask.sum().float() + 1e-9  
            global_repr = sum_repr / seq_len 
            
        return global_repr
    
    def batch_extract_feat(self, smiles_list: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
       
        print(f"Starting batch extraction for {len(smiles_list)} SMILES...")
        feat_dict = {}
        
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(smiles_list)-1)//batch_size + 1}")
            
            for smiles in batch_smiles:
                try:
                    rep = self.get_feat(smiles)
                    feat_dict[smiles] = rep.cpu().numpy() if isinstance(rep, torch.Tensor) else rep
                    
                except Exception as e:
                    print(f"Failed to process SMILES: {smiles}, error: {str(e)}")
                    continue
        
        print(f"ChemBERT feature extraction completed, {len(feat_dict)}/{len(smiles_list)} SMILES processed")
        return feat_dict
    

class GraphMVPExtractor:

    def __init__(self, config):
        
        self.config = config
        self.graphmvp_feature_path = self.config.get('graphmvp_drug_feature_pkl')
        if not self.graphmvp_feature_path:
            raise ValueError("Please configure graphmvp_feature_path in config.yaml")
            
        self._load_graphmvp_features()
    
    def _load_graphmvp_features(self):
        with open(self.graphmvp_feature_path, 'rb') as f:
            self.graphmvp_features = pickle.load(f)
        print(f"Successfully loaded GraphMVP features, {len(self.graphmvp_features)} SMILES in total")

    
    def extract_feat(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
    
        print(f"Starting batch extraction for {len(smiles_list)} SMILES...")
        feat_dict = {smiles: self.features[smiles] for smiles in smiles_list if smiles in self.features}

        missing = len(smiles_list) - len(feat_dict)
        if missing:
            print(f"{missing} SMILES not found in the feature file")
        
        print(f"GraphMVP feature extraction completed, {len(feat_dict)}/{len(smiles_list)} SMILES extracted")
        
        return feat_dict


class GroverExtractor:
  
    def __init__(self, config):
       
        self.config = config
        self.grover_feature_path = self.config.get('grover_drug_feature_pkl')
        if not self.grover_feature_path:
            raise ValueError("Please configure grover_feature_path in config.yaml")
        
        self._load_grover_features()
    
    def _load_grover_features(self):
        with open(self.grover_feature_path, 'rb') as f:
            self.grover_features = pickle.load(f)
        print(f"Successfully loaded Grover features, {len(self.grover_features)} SMILES in total")
        
    
    def extract_feat(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
    
        print(f"Starting batch extraction for {len(smiles_list)} SMILES...")
        feat_dict = {smiles: self.grover_features[smiles] for smiles in smiles_list if smiles in self.grover_features}
 
        missing = len(smiles_list) - len(feat_dict)
        if missing:
            print(f"{missing} SMILES not found in the feature file")
        
        print(f"Grover feature extraction completed, {len(feat_dict)}/{len(smiles_list)} SMILES extracted")
        
        return feat_dict
