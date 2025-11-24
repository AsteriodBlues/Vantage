"""
Model serialization and deployment utilities.

Handles saving and loading trained models with metadata.
"""

import joblib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def save_model_pipeline(
    model: Any,
    model_name: str,
    metadata: Dict,
    save_dir: str = 'models/production/'
) -> str:
    """
    Save model with metadata for deployment.

    Args:
        model: Trained model object
        model_name: Name identifier for the model
        metadata: Dictionary with model information
        save_dir: Base directory for saving

    Returns:
        Path to saved model directory
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create timestamped folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(save_dir, f'{model_name}_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, model_path, compress=3)
    print(f"Model saved to: {model_path}")

    # Add runtime info to metadata
    metadata['save_timestamp'] = timestamp
    metadata['model_name'] = model_name
    metadata['model_path'] = model_path
    metadata['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)

    # Save metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)

    print(f"Metadata saved to: {metadata_path}")
    print(f"Model size: {metadata['model_size_mb']:.2f} MB")

    return model_dir


def load_model_pipeline(model_dir: str) -> tuple:
    """
    Load model and metadata from directory.

    Args:
        model_dir: Path to model directory

    Returns:
        Tuple of (model, metadata)
    """
    model_path = os.path.join(model_dir, 'model.pkl')
    metadata_path = os.path.join(model_dir, 'metadata.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load model
    model = joblib.load(model_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded model: {metadata['model_name']}")
    print(f"Trained: {metadata.get('training_date', 'Unknown')}")

    return model, metadata


def create_symlink_latest(model_dir: str, save_dir: str = 'models/production/'):
    """
    Create 'latest' symlink pointing to most recent model.

    Args:
        model_dir: Path to model directory
        save_dir: Base directory containing models
    """
    model_name = Path(model_dir).name.rsplit('_', 2)[0]
    latest_link = os.path.join(save_dir, f'{model_name}_latest')

    # Remove existing symlink
    if os.path.islink(latest_link):
        os.unlink(latest_link)

    # Create new symlink
    os.symlink(os.path.basename(model_dir), latest_link)
    print(f"Created symlink: {latest_link} -> {model_dir}")


def save_preprocessing_artifacts(
    artifacts: Dict[str, Any],
    save_dir: str = 'models/preprocessing/'
):
    """
    Save all preprocessing objects.

    Args:
        artifacts: Dictionary mapping names to objects
        save_dir: Directory to save artifacts
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, obj in artifacts.items():
        output_path = os.path.join(save_dir, f'{name}.pkl')
        joblib.dump(obj, output_path, compress=3)
        print(f"Saved {name} to {output_path}")


def test_model_loading(model_dir: str, sample_data: Any = None) -> bool:
    """
    Test that model can be loaded and used.

    Args:
        model_dir: Path to model directory
        sample_data: Optional sample data for prediction test

    Returns:
        True if test successful
    """
    try:
        model, metadata = load_model_pipeline(model_dir)

        if sample_data is not None:
            predictions = model.predict(sample_data)
            print(f"Sample predictions: {predictions[:3]}")

        print(f"✓ Model test passed")
        return True

    except Exception as e:
        print(f"✗ Model test failed: {str(e)}")
        return False
