import yaml
import os
import sys
import tensorflow as tf

# [DEBUG] confirm the script is actually running
print("\n>>> Initializing DCESR Pipeline...")

# Add root directory to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scr.data_processing import run_preprocessing
from scr.trainer import run_training_pipeline
from model.proposed import ProposedModel

def load_config(config_path="scr/config.yaml"):
    """Loads system configuration from YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("="*60)
    print("DCESR: Dual-Channel Emotion- & Semantic-aware Recommender")
    print("="*60)

    # 1. Load Configuration
    config = load_config("scr/config.yaml")
    
    # 2. Hardware Check (TensorFlow Logic)
    gpus = tf.config.list_physical_devices('GPU')
    device = "GPU" if gpus else "CPU"
    config['device'] = "cuda" if gpus else "cpu" # For BART/DistilRoBERTa torch parts
    print(f"[SYSTEM] Hardware Check: Running on {device}")

    # 3. Phase 1: Preprocessing
    output_data_path = config['data']['output_path']
    if not os.path.exists(output_data_path):
        print("\n[PHASE 1] Starting Feature Extraction (BART & Emotion)...")
        run_preprocessing(config)
    else:
        print("\n[PHASE 1] Preprocessing")
        print(f"[INFO] Processed data found at: {output_data_path}. Skipping.")

    # 4. Phase 2: Model Architecture
    print("\n[PHASE 2] Building Proposed Model Architecture")
    model = ProposedModel(config)
    model.summary()

    # 5. Phase 3: Training
    print("\n[PHASE 3] Starting Training Engine")
    try:
        run_training_pipeline(model, config)
    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline failed: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("DCESR Pipeline Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()