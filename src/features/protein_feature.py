import torch
import esm
import numpy as np
from typing import Union, Dict, List
import yaml
from torch.serialization import add_safe_globals

add_safe_globals([esm.data.Alphabet])

class ESM2Extractor:
    def __init__(self, config_path: str):
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self._init_model()
        
    def _init_model(self):
        
        try:
            model_name = self.config.get('protein_feature', {}).get('model_name', 'esm2_t30_150M_UR50D')
            max_seq_len = self.config.get('protein_feature', {}).get('max_seq_len', None)
            
            print("Starting loading ESM2...")
            
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.batch_converter = self.alphabet.get_batch_converter(truncation_seq_length=max_seq_len)

            self.model.eval()  
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            raise
    
    def get_sequence_representation(self, sequence: str) -> Union[torch.Tensor, np.ndarray]:
        try:
            sequences = [("protein", sequence)]
            _, _, batch_tokens = self.batch_converter(sequences)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(self.device)
            with torch.no_grad():
                last_layer = self.model.num_layers
                results = self.model(batch_tokens,repr_layers=[last_layer],return_contacts=False)
            
            token_representations = results["representations"][last_layer]
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1:tokens_len-1].mean(0))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return sequence_representations[0]
            
        except Exception as e:
            print(f"Feature extraction failed, sequence: {sequence[:30]}..., error  : {str(e)}")
            raise
    
    def batch_extract_features(self, sequence_list: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        print(f"Starting batch extraction for {len(sequence_list)} sequences...")
        features_dict = {}
        
        for i in range(0, len(sequence_list), batch_size):
            batch_sequences = sequence_list[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(sequence_list)-1)//batch_size + 1}")
            
            for seq in batch_sequences:
                try:
                    rep = self.get_sequence_representation(seq)
                    if isinstance(rep, torch.Tensor):
                        rep = rep.cpu().numpy()
                    features_dict[seq] = rep
                    
                except Exception as e:
                    print(f"Failed to Process sequence:  {seq[:30]}...: error: {str(e)}")
                    continue
        
        print(f"Feature extraction completed, {len(features_dict)}/{len(sequence_list)} sequences extracted")
        return features_dict