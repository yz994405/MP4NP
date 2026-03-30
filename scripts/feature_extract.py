import numpy as np
from pathlib import Path
import yaml
import sys
import os

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.features.feature_extractor import FeatureExtractor


def extract_and_save_features(data_path: str, config_path: str, output_dir: str, force_recompute: bool = False):
    
    print("=" * 50)
    print("Starting feature extraction and saving...")
    print("=" * 50)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor(config_path)
    drug_feat, protein_feat = extractor.extract_and_save_features(data_path=data_path,output_dir=output_dir,force_recompute=force_recompute)
    
    print("Feature extraction and saving completed!")
    print("=" * 50)
    return drug_feat, protein_feat

def main():

    base_dir = Path(__file__).resolve().parent.parent
    config_path = str(base_dir / "config.yaml")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    drug_feature_type = config.get('drug_feature_type', 'chembert')
    if drug_feature_type == 'chembert':
        feature_dir = config.get('chembert_feature_dir', 'data/processed/chembert/')
    elif drug_feature_type == 'grover':
        feature_dir = config.get('grover_feature_dir', 'data/processed/grover/extracted_features/')
    elif drug_feature_type == 'graphmvp':
        feature_dir = config.get('graphmvp_feature_dir', 'data/processed/graphmvp/extracted_features/')

    feature_dir = str(base_dir / feature_dir)
    os.makedirs(feature_dir, exist_ok=True)

    data_path = str(base_dir / config['data']['input_data_path'])
    drug_feat, protein_feat = extract_and_save_features(
        data_path=data_path,
        config_path=config_path,
        output_dir=feature_dir,
        force_recompute=False
    )

    print("\nFeature extraction statistics:")
    print(f"Number of drugs: {len(drug_feat)}")
    print(f"Number of proteins: {len(protein_feat)}")
    if drug_feat:
        drug_dim = np.array(next(iter(drug_feat.values()))).shape[0]
        print(f"Drug feature dimension: {drug_dim}")
    if protein_feat:
        protein_dim = next(iter(protein_feat.values())).shape[0]
        print(f"Protein feature dimension: {protein_dim}")
    print(f"\nFeature files saved to: {feature_dir}")

if __name__ == "__main__":
    main()
