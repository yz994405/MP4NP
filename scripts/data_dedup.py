import sys
from pathlib import Path
import pandas as pd
import yaml

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


config_path = project_root / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}

input_path = config.get('data', {}).get('input_data_path')
output_dir = config.get('data', {}).get('deduplicated_data_dir', 'data/processed/deduplicated')

df = pd.read_csv(input_path)
    
print("Begin deduplication...")
unique_smiles = df['smiles'].unique().tolist()
unique_sequences = df['sequence'].unique().tolist()
    
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
    
smiles_df = pd.DataFrame({'smiles': unique_smiles})
smiles_file = output_path / 'unique_smiles.csv'
smiles_df.to_csv(smiles_file, index=False)
print(f"SMILES file saved: {smiles_file}")
    
sequences_df = pd.DataFrame({'sequence': unique_sequences})
sequences_file = output_path / 'unique_sequences.csv'
sequences_df.to_csv(sequences_file, index=False)
print(f"Protein sequence file saved: {sequences_file}")

print("Completed!")   
