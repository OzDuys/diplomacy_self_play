"""
Basic unit tests for AI_Diplomacy environment without external dependencies.

Tests core functionality using mocks to avoid dependency issues.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Test without external dependencies by mocking
class MockDataset:
    def __init__(self, data):
        self.data = data
        self.column_names = list(data.keys()) if data else []
    
    def __len__(self):
        return len(self.data.get('prompt', []))
    
    def __getitem__(self, key):
        return self.data[key]
    
    @classmethod
    def from_list(cls, items):
        if not items:
            return cls({})
        # Convert list of dicts to dict of lists
        result = {}
        for key in items[0].keys():
            result[key] = [item[key] for item in items]
        return cls(result)


# Mock the dependencies
with patch.dict('sys.modules', {
    'datasets': Mock(Dataset=MockDataset),
    'openai': Mock(),
    'verifiers.envs.environment': Mock(),
    'verifiers.parsers': Mock(),
    'verifiers.rubrics': Mock()
}):
    from diplomacy_grpo.integration.ai_diplomacy_env import DiplomacyGRPOEnvironment


class TestBasicEnvironmentFunctionality:
    """Test basic environment functionality without external dependencies."""
    
    def test_initialization_minimal(self):
        """Test basic initialization."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        assert env.countries == ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
        assert env.max_year == 1902
        assert env.working_dir.exists()
        assert env.reward_system is not None
    
    def test_default_dataset_creation(self):
        """Test default dataset creation."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        # Should create dataset with scenarios for each country
        assert env.dataset is not None
        assert len(env.dataset) == 21  # 7 countries × 3 scenarios
        
        # Check dataset structure
        assert 'prompt' in env.dataset.column_names
        assert 'country' in env.dataset.column_names
        assert 'answer' in env.dataset.column_names
        
        # Each country should be represented
        countries_in_dataset = set(env.dataset['country'])
        assert countries_in_dataset == set(env.countries)
    
    def test_custom_countries(self):
        """Test with custom country list."""
        custom_countries = ['Austria', 'England', 'France']
        env = DiplomacyGRPOEnvironment(countries=custom_countries, max_year=1902)
        
        assert env.countries == custom_countries
        
        # Dataset should reflect custom countries
        countries_in_dataset = set(env.dataset['country'])
        assert countries_in_dataset == set(custom_countries)
        assert len(env.dataset) == 9  # 3 countries × 3 scenarios
    
    def test_model_assignment_creation(self):
        """Test model assignment logic."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        # Test with default reference models
        assignments = env._create_model_assignment('Austria', 'test-model')
        
        assert assignments['Austria'] == 'test-model'
        assert all(model == 'gpt-4o-mini' for country, model in assignments.items() if country != 'Austria')
        assert len(assignments) == 7
    
    def test_model_assignment_with_references(self):
        """Test model assignment with custom reference models."""
        reference_models = {
            'England': 'claude-3-5-sonnet',
            'France': 'gpt-4o',
            'Russia': 'custom-model'
        }
        
        env = DiplomacyGRPOEnvironment(reference_models=reference_models, max_year=1902)
        assignments = env._create_model_assignment('Austria', 'training-model')
        
        assert assignments['Austria'] == 'training-model'
        assert assignments['England'] == 'claude-3-5-sonnet'
        assert assignments['France'] == 'gpt-4o'
        assert assignments['Russia'] == 'custom-model'
        assert assignments['Germany'] == 'gpt-4o-mini'  # Default
    
    def test_outcome_determination(self):
        """Test game outcome determination."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        # Test victory
        game_data = {
            'phases': [{
                'state': {
                    'centers': {
                        'Austria': ['A'] * 19  # 19 centers = victory
                    }
                }
            }]
        }
        outcome = env._determine_outcome(game_data, 'Austria')
        assert outcome == 'victory'
        
        # Test elimination
        game_data = {
            'phases': [{
                'state': {
                    'centers': {
                        'England': ['LON', 'EDI', 'LVP']  # Austria not present = eliminated
                    }
                }
            }]
        }
        outcome = env._determine_outcome(game_data, 'Austria')
        assert outcome == 'eliminated'
        
        # Test survival
        game_data = {
            'phases': [{
                'state': {
                    'centers': {
                        'Austria': ['VIE', 'BUD', 'TRI', 'SER']  # 4 centers = surviving
                    }
                }
            }]
        }
        outcome = env._determine_outcome(game_data, 'Austria')
        assert outcome == 'surviving'
    
    def test_episode_data_extraction(self):
        """Test episode data extraction."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        # Mock game result
        game_result = {
            'game_data': {
                'phases': [
                    {'state': {'centers': {'Austria': ['A', 'B', 'C']}}},  # 3 centers
                    {'state': {'centers': {'Austria': ['A', 'B', 'C', 'D']}}},  # 4 centers
                    {'state': {'centers': {'Austria': ['A', 'B']}}},  # 2 centers
                ]
            },
            'focus_country': 'Austria'
        }
        
        # Mock CSV data
        csv_data = pd.DataFrame({
            'power': ['Austria', 'Austria', 'England'],
            'raw_response': ['Move to Budapest', 'Hold position', 'Attack Austria']
        })
        game_result['csv_data'] = csv_data
        
        episode_data = env._extract_episode_data(game_result, 'Austria')
        
        assert episode_data['supply_centers'] == [3, 4, 2]
        assert episode_data['initial_centers'] == 3
        assert episode_data['final_centers'] == 2
        assert episode_data['phases_completed'] == 3
        assert episode_data['game_outcome'] == 'surviving'
        assert 'Move to Budapest' in episode_data['completion_sequence']
        assert 'Hold position' in episode_data['completion_sequence']
        assert 'Attack Austria' not in episode_data['completion_sequence']
    
    def test_episode_data_extraction_no_csv(self):
        """Test episode data extraction without CSV data."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        game_result = {
            'game_data': {
                'phases': [
                    {'state': {'centers': {'Austria': ['A', 'B', 'C']}}},
                ]
            },
            'focus_country': 'Austria',
            'csv_data': None
        }
        
        episode_data = env._extract_episode_data(game_result, 'Austria')
        
        assert episode_data['completion_sequence'] == ''
        assert episode_data['supply_centers'] == [3]
        assert episode_data['initial_centers'] == 3
        assert episode_data['final_centers'] == 3
    
    def test_reward_computation(self):
        """Test reward computation."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        episode_data = {
            'initial_centers': 3,
            'final_centers': 5,  # Gained 2 centers
            'game_outcome': 'surviving'
        }
        
        game_result = {
            'focus_country': 'Austria'
        }
        
        rewards = env._compute_rewards(episode_data, game_result)
        
        # Check reward structure
        assert 'total' in rewards
        assert 'year_level' in rewards
        assert 'game_level' in rewards
        assert 'order_validity' in rewards
        assert 'diplomatic' in rewards
        
        # Should have positive year-level reward for gaining centers
        assert rewards['year_level'] > 0
        assert rewards['total'] > 0
    
    def test_working_directory_creation(self):
        """Test working directory management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env = DiplomacyGRPOEnvironment(working_dir=temp_dir, max_year=1902)
            
            assert str(env.working_dir) == temp_dir
            assert env.working_dir.exists()
    
    def test_default_working_directory(self):
        """Test default working directory creation."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        assert env.working_dir.exists()
        assert 'diplomacy_grpo_' in str(env.working_dir)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_empty_game_data(self):
        """Test handling of empty game data."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        game_result = {
            'game_data': {'phases': []},
            'focus_country': 'Austria'
        }
        
        episode_data = env._extract_episode_data(game_result, 'Austria')
        
        assert episode_data['completion_sequence'] == ''
        assert episode_data['supply_centers'] == []
        assert episode_data['game_outcome'] == 'unknown'
    
    def test_missing_country_in_game_data(self):
        """Test handling when country is missing from game state."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        game_data = {
            'phases': [{
                'state': {
                    'centers': {
                        'England': ['LON', 'EDI', 'LVP']  # Austria missing
                    }
                }
            }]
        }
        
        outcome = env._determine_outcome(game_data, 'Austria')
        assert outcome == 'eliminated'


@pytest.mark.parametrize("country", ["Austria", "England", "France", "Germany", "Italy", "Russia", "Turkey"])
class TestCountrySpecificBehavior:
    """Test country-specific behavior."""
    
    def test_model_assignment_for_each_country(self, country):
        """Test model assignment for each country."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        assignments = env._create_model_assignment(country, 'training-model')
        
        assert assignments[country] == 'training-model'
        assert len(assignments) == 7
        
        # Other countries should have default
        for other_country in env.countries:
            if other_country != country:
                assert assignments[other_country] == 'gpt-4o-mini'
    
    def test_dataset_contains_country(self, country):
        """Test that dataset contains scenarios for each country."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        country_prompts = [prompt for prompt, c in zip(env.dataset['prompt'], env.dataset['country']) if c == country]
        
        assert len(country_prompts) == 3  # 3 scenarios per country
        
        # Each prompt should mention the country
        for prompt in country_prompts:
            assert country in prompt


class TestIntegrationPoints:
    """Test integration with our existing components."""
    
    def test_reward_system_integration(self):
        """Test integration with MultiLevelReward system."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        # Test different outcomes
        test_cases = [
            ({'initial_centers': 3, 'final_centers': 5, 'game_outcome': 'surviving'}, 'gained_centers'),
            ({'initial_centers': 3, 'final_centers': 1, 'game_outcome': 'eliminated'}, 'lost_centers'),
            ({'initial_centers': 3, 'final_centers': 18, 'game_outcome': 'victory'}, 'victory')
        ]
        
        for episode_data, description in test_cases:
            game_result = {'focus_country': 'Austria'}
            rewards = env._compute_rewards(episode_data, game_result)
            
            assert isinstance(rewards, dict), f"Rewards should be dict for {description}"
            assert 'total' in rewards, f"Total reward missing for {description}"
            assert isinstance(rewards['total'], (int, float)), f"Total should be numeric for {description}"
    
    def test_dataset_format_for_sampler(self):
        """Test that dataset format works with our country sampler."""
        env = DiplomacyGRPOEnvironment(max_year=1902)
        
        # Should have structure expected by CountryBalancedSampler
        assert 'country' in env.dataset.column_names
        assert 'prompt' in env.dataset.column_names
        
        # Each entry should have required fields
        for i in range(len(env.dataset)):
            assert env.dataset['country'][i] in env.countries
            assert isinstance(env.dataset['prompt'][i], str)
            assert len(env.dataset['prompt'][i]) > 0