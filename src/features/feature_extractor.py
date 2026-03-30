import os
import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from .drug_feature import GraphMVPExtractor, GroverExtractor, ChemBERTExtractor
from .protein_feature import ESM2Extractor

class FeatureExtractor:
    def __init__(self, config_path: str):
        
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        drug_feature_type = self.config.get('drug_feature_type', 'chembert')
        print(f"Drug Feature Extract Model: {drug_feature_type}")
        
        if drug_feature_type == 'chembert':
            self.drug_extractor = ChemBERTExtractor(config_path)
        elif drug_feature_type == 'grover':
            self.drug_extractor = GroverExtractor(self.config)
        elif drug_feature_type == 'graphmvp':
            self.drug_extractor = GraphMVPExtractor(self.config)

        self.protein_extractor = ESM2Extractor(config_path)    
    
    def extract_drug_features(self, smiles_list: List[str]) -> Tuple[Dict[str, np.ndarray], List[str]]:
        print("Starting drug feature extraction...")
        if isinstance(self.drug_extractor, ChemBERTExtractor):
            features_dict = self.drug_extractor.batch_extract_feat(smiles_list)
            failed_smiles = [smiles for smiles in smiles_list if smiles not in features_dict]
        elif isinstance(self.drug_extractor, GroverExtractor):
            features_dict = self.drug_extractor.extract_feat(smiles_list)
            failed_smiles = [smiles for smiles in smiles_list if smiles not in features_dict]
        elif isinstance(self.drug_extractor, GraphMVPExtractor):
            features_dict = self.drug_extractor.extract_feat(smiles_list)
            failed_smiles = [smiles for smiles in smiles_list if smiles not in features_dict]
                
        print(f"Successfully extracted {len(features_dict)} drug features")
        if failed_smiles:
            print(f"{len(failed_smiles)} SMILES could not be processed.")
        return features_dict, failed_smiles
    
    def extract_protein_features(self, sequence_list: List[str]) -> Tuple[Dict[str, np.ndarray], List[str]]:
      
        print("Starting protein feature extraction...")
        features, failed_seq = {}, []
        
        for seq in tqdm(sequence_list, desc="Extracting protein features"):
            try:
                rep = self.protein_extractor.get_sequence_representation(seq)
                features[seq] = rep.cpu().numpy() if not isinstance(rep, np.ndarray) else rep
            except Exception as e:
                print(f"Sequence processing failed {seq[:20]}...: {str(e)}")
                failed_seq.append(seq)
        
        print(f"Successfully extracted {len(features)} protein features")
        if failed_seq:
            print(f"{len(failed_seq)} sequences could not be processed")
        return features, failed_seq
    
    def extract_and_save_features(self, data_path: str, output_dir: str, force_recompute: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        deduplicated_dir = self.config.get('data', {}).get('deduplicated_data_dir', 'data/processed/deduplicated')
        print(f"Using deduplicated data from directory: {deduplicated_dir}")
        smiles_df = pd.read_csv(os.path.join(deduplicated_dir, 'unique_smiles.csv'))
        sequences_df = pd.read_csv(os.path.join(deduplicated_dir, 'unique_sequences.csv'))
        unique_smiles = smiles_df['smiles'].unique()
        unique_sequences = sequences_df['sequence'].unique()
        print(f"Number of unique SMILES after deduplication: {len(unique_smiles)}")
        print(f"Number of unique protein sequences after deduplication: {len(unique_sequences)}")

        drug_cache_path = output_dir / 'drug_features.pkl'
        protein_cache_path = output_dir / 'protein_features.pkl'
        
        drug_features = {}
        protein_features = {}
        
        # Load or compute drug features
        if not force_recompute and drug_cache_path.exists():
            print("Loading drug features from cache...")
            with open(drug_cache_path, 'rb') as f:
                drug_features = pickle.load(f)
            failed_smiles = []
        else:
            drug_features, failed_smiles = self.extract_drug_features(unique_smiles)
            
            with open(drug_cache_path, 'wb') as f:
                pickle.dump(drug_features, f)
            
            if failed_smiles:
                pd.DataFrame({'failed_smiles': failed_smiles}).to_csv(Path(output_dir) / 'failed_smiles.csv', index=False)
                print(f"Failed SMILES saved to: {Path(output_dir) / 'failed_smiles.csv'}")
                
        
        # Load or compute protein features
        if not force_recompute and protein_cache_path.exists():
            print("Loading protein features from cache...")
            with open(protein_cache_path, 'rb') as f:
                protein_features = pickle.load(f)
            failed_sequences = []
        else:
            protein_features, failed_sequences = self.extract_protein_features(unique_sequences)
            
            with open(protein_cache_path, 'wb') as f:
                pickle.dump(protein_features, f)
            
            if failed_sequences:
                pd.DataFrame({'failed_sequences': failed_sequences}).to_csv(Path(output_dir) / 'failed_proteins.csv', index=False)
                print(f"Failed protein sequences saved to: {Path(output_dir) / 'failed_proteins.csv'}")

        return drug_features, protein_features