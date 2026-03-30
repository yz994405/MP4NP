import numpy as np
import pandas as pd
from typing import List, Tuple, Union


class DTADataSplitter:
    
    def __init__(self, random_seed: int = 42):
        
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    # ==================== random ====================  
    def split_random(self, dataset, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.random_seed)
        total_size = len(dataset)
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        train_size = int(total_size * train_ratio)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        print(f"random split - total dataset: {total_size}, train set: {len(train_indices)}, val set: {len(val_indices)}")

        return train_indices, val_indices
    
    # ==================== drug cold start ====================
    def split_drug_cold(self, dataset, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(dataset.pair_csv) if hasattr(dataset, 'pair_csv') else self._extract_pairs_from_dataset(dataset)
        unique_drugs = df['smiles'].unique()
        np.random.seed(self.random_seed)
        np.random.shuffle(unique_drugs)
        n_val_drugs = int(len(unique_drugs) * val_ratio)
        val_drugs = unique_drugs[:n_val_drugs]
        train_drugs = unique_drugs[n_val_drugs:]
        train_mask = df['smiles'].isin(train_drugs)
        val_mask = df['smiles'].isin(val_drugs)
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        print(f"drug cold start split - " f"train drugs: {len(train_drugs)}, val drugs: {len(val_drugs)}, " f"sample of train: {len(train_indices)}, sample of val: {len(val_indices)}")
        
        return train_indices, val_indices
    
    # ==================== protein cold start ====================
    def split_protein_cold(self, dataset, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(dataset.pair_csv) if hasattr(dataset, 'pair_csv') else self._extract_pairs_from_dataset(dataset)
        unique_proteins = df['sequence'].unique()
        np.random.seed(self.random_seed)
        np.random.shuffle(unique_proteins)
        n_val_proteins = int(len(unique_proteins) * val_ratio)
        val_proteins = unique_proteins[:n_val_proteins]
        train_proteins = unique_proteins[n_val_proteins:]
        train_mask = df['sequence'].isin(train_proteins)
        val_mask = df['sequence'].isin(val_proteins)
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        print(f"protein cold start split - " f"train proteins: {len(train_proteins)}, val proteins: {len(val_proteins)}, " f"sample of train: {len(train_indices)}, sample of val: {len(val_indices)}")
        
        return train_indices, val_indices
    
    # ==================== full cold start ====================
    def split_full_cold(self, dataset, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(dataset.pair_csv) if hasattr(dataset, 'pair_csv') else self._extract_pairs_from_dataset(dataset)
        unique_drugs = df['smiles'].unique()
        unique_proteins = df['sequence'].unique()
        np.random.seed(self.random_seed)
        n_val_drugs = max(1, int(len(unique_drugs) * val_ratio * 0.5))
        n_val_proteins = max(1, int(len(unique_proteins) * val_ratio * 0.5))
        val_drugs = np.random.choice(unique_drugs, n_val_drugs, replace=False)
        val_proteins = np.random.choice(unique_proteins, n_val_proteins, replace=False)
        
        # Strict cold start: validation set drugs and proteins do not appear in the training set
        val_mask = df['smiles'].isin(val_drugs) & df['sequence'].isin(val_proteins)
        train_mask = ~df['smiles'].isin(val_drugs) & ~df['sequence'].isin(val_proteins)
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        if len(train_indices) < 100 or len(val_indices) < 10:
            self.logger.warning(f"Full cold start: insufficient number of samples, adjust parameters")
        print(f"full cold start split - " f"val drugs: {len(val_drugs)}, val proteins: {len(val_proteins)}, " f"sample of train: {len(train_indices)}, sample of val: {len(val_indices)}")
        
        return train_indices, val_indices
    
    # ==================== uniform interface ====================
    
    def split_data(self, dataset, split_strategy: str = 'random', use_cross_validation: bool = True, **kwargs) -> Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
        
        print(f"Starting data split, strategy: {split_strategy}, cross validation: {use_cross_validation}")
        
        if split_strategy == 'random':
            return self.split_random(dataset, **kwargs)
        elif split_strategy == 'drug_cold':
            return self.split_drug_cold(dataset, **kwargs)
        elif split_strategy == 'protein_cold':
            return self.split_protein_cold(dataset, **kwargs)
        elif split_strategy == 'full_cold':
            return self.split_full_cold(dataset, **kwargs)
        else:
            raise ValueError(f"Unsupported split strategy: {split_strategy}")
    
    def _extract_pairs_from_dataset(self, dataset) -> pd.DataFrame:
        
        pairs = []
        for i in range(len(dataset)):
            if hasattr(dataset, 'df'):
                row = dataset.df.iloc[i]
                pairs.append({
                    'smiles': row['smiles'],
                    'sequence': row['sequence'], 
                    'value': row['value']
                })
            else:
                try:
                    _, _, _ = dataset[i]  
                    print("Can")
                    break
                except:
                    break
        if not pairs:
            raise ValueError("Cannot extract drug-protein pairs from dataset")
        
        return pd.DataFrame(pairs)
    