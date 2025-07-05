"""
Country-grouped GRPO trainer.

Extends the verifiers GRPO trainer to implement country-based advantage computation
as outlined in the GRPO Multi-Agent Analysis document.
"""

from typing import List, Optional, Dict, Any
import torch
import logging
from collections import defaultdict

# Import base GRPO trainer - will need to adjust import path when integrating
try:
    from verifiers.trainers.grpo_trainer import GRPOTrainer
    from verifiers.trainers.grpo_config import GRPOConfig
    from verifiers import Environment
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.modeling_utils import PreTrainedModel
    from transformers.trainer_callback import TrainerCallback
    from peft import PeftConfig
    _VERIFIERS_AVAILABLE = True
except ImportError:
    # For development/testing without verifiers installed
    class GRPOTrainer:
        def __init__(self, *args, **kwargs):
            pass
    
    class GRPOConfig:
        pass
    
    class Environment:
        pass
        
    class PreTrainedModel:
        pass
        
    class PreTrainedTokenizerBase:
        pass
        
    class TrainerCallback:
        pass
        
    class PeftConfig:
        pass
        
    _VERIFIERS_AVAILABLE = False


class CountryGroupedGRPOTrainer(GRPOTrainer):
    """
    GRPO trainer with country-based advantage computation.
    
    Instead of grouping episodes by prompt (standard GRPO), this trainer groups
    episodes by the country assignment. This enables meaningful comparisons
    within the same strategic context while handling multi-agent non-stationarity.
    
    Key features:
    - Country-specific advantage computation
    - Balanced country representation in batches  
    - Country-specific normalization to handle difficulty differences
    - Maintains single model instance for general Diplomacy skills
    
    Args:
        countries: List of Diplomacy countries. Defaults to standard 7 countries.
        country_specific_normalization: Whether to normalize advantages per country.
        **kwargs: Arguments passed to base GRPOTrainer.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        env: Environment,
        args: GRPOConfig,
        processing_class: PreTrainedTokenizerBase,
        countries: Optional[List[str]] = None,
        country_specific_normalization: bool = True,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        self.countries = countries or [
            'Austria', 'England', 'France', 'Germany',
            'Italy', 'Russia', 'Turkey'
        ]
        self.country_specific_normalization = country_specific_normalization
        self.logger = logging.getLogger(__name__)
        
        # Track country-specific metrics
        self.country_metrics = defaultdict(list)
        
        super().__init__(
            model=model,
            env=env, 
            args=args,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        
    def _compute_advantages(
        self, 
        rewards: torch.Tensor,
        countries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute advantages grouped by country assignment.
        
        Each country's episodes are compared against other episodes of the same 
        country, preserving strategic context and enabling meaningful comparisons.
        
        Args:
            rewards: Tensor of shape (batch_size,) containing episode rewards
            countries: Tensor of shape (batch_size,) containing country assignments
            
        Returns:
            advantages: Tensor of shape (batch_size,) containing computed advantages
        """
        if countries is None:
            # Fall back to standard GRPO if no country info available
            self.logger.warning("No country information provided, falling back to standard GRPO")
            return super()._compute_advantages(rewards, **kwargs)
            
        advantages = torch.zeros_like(rewards)
        
        for i, country in enumerate(self.countries):
            # Find episodes for this country
            # Handle both string names and integer indices
            if isinstance(countries[0].item() if torch.is_tensor(countries) else countries[0], str):
                country_mask = (countries == country)
            else:
                country_mask = (countries == i)
            
            if country_mask.sum() == 0:
                continue  # No episodes for this country in this batch
                
            country_rewards = rewards[country_mask]
            
            if country_rewards.numel() == 1:
                # Only one episode for this country - advantage is 0
                advantages[country_mask] = 0.0
                continue
                
            # Compute country-specific advantages
            country_mean = country_rewards.mean()
            country_advantages = country_rewards - country_mean
            
            # Optional: normalize by country-specific standard deviation
            if self.country_specific_normalization:
                country_std = country_rewards.std()
                if country_std > 1e-8:  # Avoid division by zero
                    country_advantages = country_advantages / country_std
                    
            advantages[country_mask] = country_advantages
            
            # Log country-specific metrics
            if country not in self.country_metrics:
                self.country_metrics[country] = []
            self.country_metrics[country].extend([
                country_mean.item(),
                country_std.item() if self.country_specific_normalization else 0.0,
                country_rewards.max().item(),
                country_rewards.min().item(),
            ])
            
        return advantages
        
    def _extract_countries_from_batch(self, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Extract country assignments from batch inputs.
        
        This method should be overridden based on how country information
        is stored in your specific dataset format.
        """
        # Default implementation - assumes 'countries' key in inputs
        if 'countries' in inputs:
            return inputs['countries']
            
        # Alternative: extract from prompts or other metadata
        if 'country_info' in inputs:
            return inputs['country_info'] 
            
        self.logger.warning("Could not extract country information from batch")
        return None
        
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute loss using country-grouped advantages.
        
        Extracts country information from inputs and passes it to advantage computation.
        """
        # Extract country assignments from inputs
        countries = self._extract_countries_from_batch(inputs)
        
        # Store countries for use in _compute_advantages
        if countries is not None:
            inputs['countries'] = countries
            
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
    def get_country_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for each country's performance.
        
        Returns:
            Dictionary mapping country names to their performance metrics.
        """
        summary = {}
        
        for country, metrics in self.country_metrics.items():
            if not metrics:
                continue
                
            # Metrics are stored as [mean, std, max, min] for each batch
            means = metrics[0::4]  # Every 4th element starting from 0
            stds = metrics[1::4]   # Every 4th element starting from 1  
            maxes = metrics[2::4]  # Every 4th element starting from 2
            mins = metrics[3::4]   # Every 4th element starting from 3
            
            summary[country] = {
                'avg_reward': sum(means) / len(means) if means else 0.0,
                'avg_std': sum(stds) / len(stds) if stds else 0.0,
                'max_reward': max(maxes) if maxes else 0.0,
                'min_reward': min(mins) if mins else 0.0,
                'num_batches': len(means),
            }
            
        return summary
        
    def log_country_metrics(self) -> None:
        """Log country-specific performance metrics."""
        summary = self.get_country_metrics_summary()
        
        for country, metrics in summary.items():
            self.logger.info(
                f"Country {country}: avg_reward={metrics['avg_reward']:.3f}, "
                f"std={metrics['avg_std']:.3f}, "
                f"range=[{metrics['min_reward']:.3f}, {metrics['max_reward']:.3f}], "
                f"batches={metrics['num_batches']}"
            )
            
        # Clear metrics for next logging period
        self.country_metrics.clear()