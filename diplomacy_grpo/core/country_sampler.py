"""
Country-balanced sampler for GRPO training.

This sampler ensures each country gets equal representation across batches,
enabling country-based grouping for advantage computation.
"""

import random
from typing import List, Optional, Sized, Iterator
import torch
from torch.utils.data import Sampler


class CountryBalancedSampler(Sampler[int]):
    """
    Sampler that ensures balanced representation across Diplomacy countries.
    
    For GRPO training, we need each batch to contain episodes for each country
    so we can compute country-specific advantages. This sampler guarantees
    that each country appears exactly `num_generations_per_country` times
    in each batch.
    
    Args:
        data_source: Dataset to sample from. Each item should have a 'country' field.
        num_generations_per_country: Number of episodes per country per batch.
        countries: List of country names. Defaults to standard Diplomacy countries.
        shuffle: Whether to shuffle the order within each country group.
        seed: Random seed for reproducibility.
        
    Example:
        >>> dataset = [
        ...     {'country': 'Austria', 'prompt': '...'},
        ...     {'country': 'England', 'prompt': '...'},
        ...     # ... more examples
        ... ]
        >>> sampler = CountryBalancedSampler(
        ...     dataset, 
        ...     num_generations_per_country=5,
        ...     countries=['Austria', 'England', 'France']
        ... )
        >>> # Each batch will have 15 items: 5 Austria + 5 England + 5 France
    """
    
    def __init__(
        self,
        data_source: Sized,
        num_generations_per_country: int = 5,
        countries: Optional[List[str]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_generations_per_country = num_generations_per_country
        self.countries = countries or [
            'Austria', 'England', 'France', 'Germany', 
            'Italy', 'Russia', 'Turkey'
        ]
        self.shuffle = shuffle
        self.seed = seed
        
        if shuffle and seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None
            
        # Pre-compute country indices for efficiency
        self._build_country_indices()
        
    def _build_country_indices(self) -> None:
        """Build mapping from countries to dataset indices."""
        self.country_indices = {country: [] for country in self.countries}
        
        for idx, item in enumerate(self.data_source):
            country = item.get('country')
            if country in self.country_indices:
                self.country_indices[country].append(idx)
                
        # Verify we have examples for all countries
        missing_countries = [
            country for country, indices in self.country_indices.items()
            if not indices
        ]
        if missing_countries:
            raise ValueError(
                f"No examples found for countries: {missing_countries}. "
                f"Available countries: {[c for c, idx in self.country_indices.items() if idx]}"
            )
            
    def __iter__(self) -> Iterator[int]:
        """Generate indices ensuring balanced country representation."""
        indices = []
        
        for country in self.countries:
            country_pool = self.country_indices[country].copy()
            
            if self.shuffle:
                if self.generator is not None:
                    # Use torch.randperm for reproducibility
                    perm = torch.randperm(len(country_pool), generator=self.generator)
                    country_pool = [country_pool[i] for i in perm]
                else:
                    random.shuffle(country_pool)
            
            # Sample with replacement if needed
            for _ in range(self.num_generations_per_country):
                if country_pool:
                    if self.shuffle:
                        if self.generator is not None:
                            # Use generator for reproducible random choice
                            rand_idx = torch.randint(0, len(country_pool), (1,), generator=self.generator).item()
                            idx = country_pool.pop(rand_idx)
                        else:
                            idx = country_pool.pop(random.randrange(len(country_pool)))
                    else:
                        idx = country_pool.pop()
                    indices.append(idx)
                else:
                    # If we run out, sample with replacement
                    if self.generator is not None:
                        rand_idx = torch.randint(0, len(self.country_indices[country]), (1,), generator=self.generator).item()
                        idx = self.country_indices[country][rand_idx]
                    else:
                        idx = random.choice(self.country_indices[country])
                    indices.append(idx)
        
        # Final shuffle of the complete batch
        if self.shuffle:
            if self.generator is not None:
                perm = torch.randperm(len(indices), generator=self.generator)
                indices = [indices[i] for i in perm]
            else:
                random.shuffle(indices)
                
        return iter(indices)
        
    def __len__(self) -> int:
        """Return total number of samples per epoch."""
        return len(self.countries) * self.num_generations_per_country
        
    def get_country_distribution(self) -> dict[str, int]:
        """Get the number of available examples per country."""
        return {
            country: len(indices) 
            for country, indices in self.country_indices.items()
        }