"""
Diplomacy GRPO Pipeline

A self-play pipeline for training LLMs on Diplomacy using Group Relative Policy 
Optimization (GRPO) with country-based grouping.
"""

__version__ = "0.1.0"

# Import reward system (no torch dependency)
from .rewards.multi_level import MultiLevelReward

# Defer torch import to avoid conflicts
def _get_country_sampler():
    try:
        from .core.country_sampler import CountryBalancedSampler
        return CountryBalancedSampler
    except ImportError:
        return None

CountryBalancedSampler = None
_TORCH_AVAILABLE = False

# Lazy loading function for torch components
def get_country_sampler():
    global CountryBalancedSampler, _TORCH_AVAILABLE
    if CountryBalancedSampler is None and not _TORCH_AVAILABLE:
        CountryBalancedSampler = _get_country_sampler()
        _TORCH_AVAILABLE = CountryBalancedSampler is not None
    return CountryBalancedSampler

# Defer verifiers import to avoid transformers/datasets conflicts
def _get_country_trainer():
    try:
        from .core.country_trainer import CountryGroupedGRPOTrainer
        return CountryGroupedGRPOTrainer
    except ImportError:
        return None

CountryGroupedGRPOTrainer = None
_VERIFIERS_AVAILABLE = False

# Lazy loading function
def get_country_trainer():
    global CountryGroupedGRPOTrainer, _VERIFIERS_AVAILABLE
    if CountryGroupedGRPOTrainer is None and not _VERIFIERS_AVAILABLE:
        CountryGroupedGRPOTrainer = _get_country_trainer()
        _VERIFIERS_AVAILABLE = CountryGroupedGRPOTrainer is not None
    return CountryGroupedGRPOTrainer

__all__ = [
    "MultiLevelReward",
    "get_country_trainer",
    "get_country_sampler",
]