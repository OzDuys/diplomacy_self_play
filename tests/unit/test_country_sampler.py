"""
Unit tests for CountryBalancedSampler.
"""

import pytest
from collections import Counter
from diplomacy_grpo.core.country_sampler import CountryBalancedSampler


@pytest.fixture
def sample_dataset():
    """Create a sample dataset with country information."""
    return [
        {'country': 'Austria', 'prompt': 'Austria prompt 1'},
        {'country': 'Austria', 'prompt': 'Austria prompt 2'},
        {'country': 'England', 'prompt': 'England prompt 1'},
        {'country': 'England', 'prompt': 'England prompt 2'},
        {'country': 'England', 'prompt': 'England prompt 3'},
        {'country': 'France', 'prompt': 'France prompt 1'},
        {'country': 'Germany', 'prompt': 'Germany prompt 1'},
        {'country': 'Germany', 'prompt': 'Germany prompt 2'},
        {'country': 'Italy', 'prompt': 'Italy prompt 1'},
        {'country': 'Russia', 'prompt': 'Russia prompt 1'},
        {'country': 'Russia', 'prompt': 'Russia prompt 2'},
        {'country': 'Turkey', 'prompt': 'Turkey prompt 1'},
    ]


@pytest.fixture
def three_country_dataset():
    """Create a smaller dataset with 3 countries for simpler testing."""
    return [
        {'country': 'Austria', 'prompt': 'Austria 1'},
        {'country': 'Austria', 'prompt': 'Austria 2'},
        {'country': 'England', 'prompt': 'England 1'},
        {'country': 'England', 'prompt': 'England 2'},
        {'country': 'France', 'prompt': 'France 1'},
        {'country': 'France', 'prompt': 'France 2'},
    ]


class TestCountryBalancedSampler:
    """Test the CountryBalancedSampler class."""
    
    def test_initialization_with_defaults(self, sample_dataset):
        """Test sampler initialization with default parameters."""
        sampler = CountryBalancedSampler(sample_dataset)
        
        assert sampler.num_generations_per_country == 5
        assert len(sampler.countries) == 7
        assert 'Austria' in sampler.countries
        assert 'Turkey' in sampler.countries
        assert sampler.shuffle is True
        
    def test_initialization_with_custom_countries(self, three_country_dataset):
        """Test sampler with custom country list."""
        countries = ['Austria', 'England', 'France']
        sampler = CountryBalancedSampler(
            three_country_dataset,
            countries=countries,
            num_generations_per_country=2
        )
        
        assert sampler.countries == countries
        assert len(sampler) == 6  # 3 countries × 2 generations
        
    def test_country_indices_building(self, three_country_dataset):
        """Test that country indices are built correctly."""
        countries = ['Austria', 'England', 'France']
        sampler = CountryBalancedSampler(
            three_country_dataset,
            countries=countries
        )
        
        assert len(sampler.country_indices['Austria']) == 2
        assert len(sampler.country_indices['England']) == 2  
        assert len(sampler.country_indices['France']) == 2
        
    def test_missing_country_raises_error(self, three_country_dataset):
        """Test that missing countries raise an error."""
        countries = ['Austria', 'England', 'France', 'Germany']  # Germany not in dataset
        
        with pytest.raises(ValueError, match="No examples found for countries"):
            CountryBalancedSampler(three_country_dataset, countries=countries)
            
    def test_iteration_returns_correct_length(self, three_country_dataset):
        """Test that iteration returns the expected number of indices."""
        countries = ['Austria', 'England', 'France']
        sampler = CountryBalancedSampler(
            three_country_dataset,
            countries=countries,
            num_generations_per_country=2
        )
        
        indices = list(sampler)
        assert len(indices) == 6  # 3 countries × 2 generations
        
    def test_country_distribution_in_batch(self, three_country_dataset):
        """Test that each country appears the correct number of times."""
        countries = ['Austria', 'England', 'France']
        sampler = CountryBalancedSampler(
            three_country_dataset,
            countries=countries,
            num_generations_per_country=2,
            shuffle=False,  # Disable shuffle for deterministic testing
            seed=42
        )
        
        indices = list(sampler)
        
        # Count how many times each country appears
        country_counts = Counter()
        for idx in indices:
            country = three_country_dataset[idx]['country']
            country_counts[country] += 1
            
        # Each country should appear exactly num_generations_per_country times
        for country in countries:
            assert country_counts[country] == 2
            
    def test_reproducibility_with_seed(self, three_country_dataset):
        """Test that results are reproducible with the same seed."""
        countries = ['Austria', 'England', 'France']
        
        sampler1 = CountryBalancedSampler(
            three_country_dataset,
            countries=countries,
            num_generations_per_country=2,
            seed=42
        )
        
        sampler2 = CountryBalancedSampler(
            three_country_dataset,
            countries=countries, 
            num_generations_per_country=2,
            seed=42
        )
        
        indices1 = list(sampler1)
        indices2 = list(sampler2)
        
        assert indices1 == indices2
        
    def test_get_country_distribution(self, sample_dataset):
        """Test the get_country_distribution method."""
        sampler = CountryBalancedSampler(sample_dataset)
        distribution = sampler.get_country_distribution()
        
        assert distribution['Austria'] == 2
        assert distribution['England'] == 3
        assert distribution['France'] == 1
        assert distribution['Germany'] == 2
        assert distribution['Russia'] == 2
        
    def test_sampling_with_replacement(self):
        """Test that sampler works when requesting more samples than available."""
        # Dataset with only 1 example per country
        small_dataset = [
            {'country': 'Austria', 'prompt': 'Austria 1'},
            {'country': 'England', 'prompt': 'England 1'},
        ]
        
        countries = ['Austria', 'England']
        sampler = CountryBalancedSampler(
            small_dataset,
            countries=countries,
            num_generations_per_country=3,  # More than available
            shuffle=False
        )
        
        indices = list(sampler)
        assert len(indices) == 6  # 2 countries × 3 generations
        
        # Should sample with replacement
        country_counts = Counter()
        for idx in indices:
            country = small_dataset[idx]['country']
            country_counts[country] += 1
            
        assert country_counts['Austria'] == 3
        assert country_counts['England'] == 3
        
    def test_len_method(self, three_country_dataset):
        """Test the __len__ method."""
        countries = ['Austria', 'England', 'France']
        sampler = CountryBalancedSampler(
            three_country_dataset,
            countries=countries,
            num_generations_per_country=4
        )
        
        assert len(sampler) == 12  # 3 countries × 4 generations