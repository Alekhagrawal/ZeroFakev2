# File: datasets/__init__.py
# What this file does: Marks datasets/ as a regular Python package so `from datasets.deepfake_dataset import ...`
#   resolves to ZeroFake's local dataset module instead of colliding with Google Colab's pre-installed
#   Hugging Face `datasets` library when the project root is at sys.path[0].
# Inputs: Imported by training/train.py and evaluation/*.py; Outputs: Exposes DeepfakeDataset and GeneratorBalancedBatchSampler.

from datasets.deepfake_dataset import DeepfakeDataset, GeneratorBalancedBatchSampler

__all__ = ["DeepfakeDataset", "GeneratorBalancedBatchSampler"]
