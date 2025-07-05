"""
Integration tests for AI_Diplomacy environment wrapper.

Tests the complete integration between our GRPO components and
the AI_Diplomacy framework using real (but minimal) games.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import pandas as pd

from diplomacy_grpo.integration.ai_diplomacy_env import (
    DiplomacyGRPOEnvironment,
    create_test_environment
)


class TestDiplomacyGRPOEnvironment:
    """Test AI_Diplomacy integration environment."""
    
    def test_initialization(self):
        """Test environment initialization with default parameters."""
        env = create_test_environment()
        
        assert env.countries == ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
        assert env.max_year == 1902
        assert env.num_negotiation_rounds == 1
        assert env.working_dir.exists()
        assert env.reward_system is not None
        
        # Should have created a default dataset
        assert env.dataset is not None
        assert len(env.dataset) == 21  # 7 countries × 3 scenarios
    
    def test_custom_initialization(self):
        """Test environment initialization with custom parameters."""
        custom_countries = ['Austria', 'England', 'France']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            env = DiplomacyGRPOEnvironment(
                countries=custom_countries,
                max_year=1903,
                num_negotiation_rounds=2,
                working_dir=temp_dir
            )
            
            assert env.countries == custom_countries
            assert env.max_year == 1903
            assert env.num_negotiation_rounds == 2
            assert str(env.working_dir) == temp_dir
            
            # Dataset should reflect custom countries
            countries_in_dataset = set(env.dataset['country'])
            assert countries_in_dataset == set(custom_countries)
    
    def test_model_assignment_creation(self):
        """Test model assignment logic."""
        env = create_test_environment()
        
        # Test with default reference models
        assignments = env._create_model_assignment('Austria', 'test-model')
        
        assert assignments['Austria'] == 'test-model'
        assert all(model == 'gpt-4o-mini' for country, model in assignments.items() if country != 'Austria')
        assert len(assignments) == 7
    
    def test_model_assignment_with_custom_references(self):
        """Test model assignment with custom reference models."""
        reference_models = {
            'England': 'claude-3-5-sonnet',
            'France': 'gpt-4o',
            'Russia': 'custom-model'
        }
        
        env = DiplomacyGRPOEnvironment(reference_models=reference_models)
        assignments = env._create_model_assignment('Austria', 'training-model')
        
        assert assignments['Austria'] == 'training-model'
        assert assignments['England'] == 'claude-3-5-sonnet'
        assert assignments['France'] == 'gpt-4o'
        assert assignments['Russia'] == 'custom-model'
        # Countries not in reference_models should use default
        assert assignments['Germany'] == 'gpt-4o-mini'
    
    def test_outcome_determination(self):
        """Test game outcome determination logic."""
        env = create_test_environment()
        
        # Test victory (18+ centers)
        game_data = {
            'phases': [{
                'state': {
                    'centers': {
                        'Austria': ['VIE', 'BUD', 'TRI'] + ['WAR', 'MOS'] * 8  # 19 centers
                    }
                }
            }]
        }
        outcome = env._determine_outcome(game_data, 'Austria')
        assert outcome == 'victory'
        
        # Test elimination (0 centers)
        game_data = {
            'phases': [{
                'state': {
                    'centers': {
                        'England': ['LON', 'EDI', 'LVP']
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
                        'Austria': ['VIE', 'BUD', 'TRI', 'SER']  # 4 centers
                    }
                }
            }]
        }
        outcome = env._determine_outcome(game_data, 'Austria')
        assert outcome == 'surviving'
    
    def test_episode_data_extraction(self):
        """Test episode data extraction from game results."""
        env = create_test_environment()
        
        # Mock game result with realistic structure
        game_result = {
            'game_data': {
                'phases': [
                    {'state': {'centers': {'Austria': ['VIE', 'BUD', 'TRI']}}},  # 3 centers
                    {'state': {'centers': {'Austria': ['VIE', 'BUD', 'TRI', 'SER']}}},  # 4 centers  
                    {'state': {'centers': {'Austria': ['VIE', 'BUD']}}},  # 2 centers
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
        assert 'Attack Austria' not in episode_data['completion_sequence']  # Different country
    
    def test_reward_computation(self):
        """Test reward computation integration."""
        env = create_test_environment()
        
        episode_data = {
            'initial_centers': 3,
            'final_centers': 5,  # Gained 2 centers
            'game_outcome': 'surviving'
        }
        
        game_result = {
            'focus_country': 'Austria'
        }
        
        rewards = env._compute_rewards(episode_data, game_result)
        
        assert 'total' in rewards
        assert 'year_level' in rewards
        assert 'game_level' in rewards
        assert 'order_validity' in rewards
        assert 'diplomatic' in rewards
        
        # Should have positive year-level reward for gaining centers
        assert rewards['year_level'] > 0
        assert rewards['total'] > 0
    
    @patch('subprocess.run')
    def test_rollout_success(self, mock_subprocess):
        """Test successful rollout execution."""
        env = create_test_environment()
        
        # Mock successful subprocess execution
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Mock game output files
        with patch('builtins.open') as mock_open, \
             patch('pathlib.Path.exists') as mock_exists:
            
            mock_exists.return_value = True
            
            # Mock lmvsgame.json
            mock_game_data = {
                'phases': [
                    {'state': {'centers': {'Austria': ['VIE', 'BUD', 'TRI']}}}
                ]
            }
            
            # Mock CSV data  
            mock_csv_data = pd.DataFrame({
                'power': ['Austria'],
                'raw_response': ['Strategic opening move']
            })
            
            # Setup file mocks
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_game_data)
            
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.return_value = mock_csv_data
                
                # Create mock client
                mock_client = Mock()
                
                # Run rollout
                completion, state = env.rollout(
                    client=mock_client,
                    model="test-model",
                    prompt="You are Austria in Spring 1901. Plan your opening strategy.",
                    answer="Strategic play",
                    info={'country': 'Austria'}
                )
                
                # Verify results
                assert isinstance(completion, str)
                assert 'Strategic opening move' in completion
                
                assert 'country' in state
                assert state['country'] == 'Austria'
                assert 'game_result' in state
                assert 'rewards' in state
                assert 'total' in state['rewards']
    
    @patch('subprocess.run')
    def test_rollout_game_failure(self, mock_subprocess):
        """Test rollout handling of game execution failure."""
        env = create_test_environment()
        
        # Mock failed subprocess execution
        mock_subprocess.return_value = Mock(
            returncode=1, 
            stdout="", 
            stderr="Game failed to start"
        )
        
        mock_client = Mock()
        
        completion, state = env.rollout(
            client=mock_client,
            model="test-model",
            prompt="You are Austria in Spring 1901.",
            answer="",
            info={'country': 'Austria'}
        )
        
        # Should return error completion
        assert completion.startswith("[ERROR]")
        assert 'error' in state
        assert state['country'] == 'Austria'
        assert state['rewards']['total'] == 0.0
    
    def test_country_extraction_from_prompt(self):
        """Test country extraction from prompt when not in info."""
        env = create_test_environment()
        
        # Mock all the game execution parts
        with patch.object(env, '_run_diplomacy_game') as mock_game, \
             patch.object(env, '_extract_episode_data') as mock_extract, \
             patch.object(env, '_compute_rewards') as mock_rewards:
            
            mock_game.return_value = {'focus_country': 'England'}
            mock_extract.return_value = {'completion_sequence': 'test'}
            mock_rewards.return_value = {'total': 1.0}
            
            mock_client = Mock()
            
            # Test string prompt
            completion, state = env.rollout(
                client=mock_client,
                model="test-model",
                prompt="You are England in Spring 1901. Plan your naval strategy.",
                answer="",
                info={}  # No country specified
            )
            
            assert state['country'] == 'England'
            mock_game.assert_called_once()
            
            # Test chat format prompt
            mock_game.reset_mock()
            chat_prompt = [
                {'role': 'system', 'content': 'You are a Diplomacy player'},
                {'role': 'user', 'content': 'You are Russia in Fall 1901. What is your strategy?'}
            ]
            
            completion, state = env.rollout(
                client=mock_client,
                model="test-model", 
                prompt=chat_prompt,
                answer="",
                info={}
            )
            
            assert state['country'] == 'Russia'
    
    def test_country_extraction_failure(self):
        """Test error handling when country cannot be extracted."""
        env = create_test_environment()
        
        mock_client = Mock()
        
        # Prompt without country information
        with pytest.raises(ValueError, match="Could not determine country"):
            env.rollout(
                client=mock_client,
                model="test-model",
                prompt="Generic diplomacy strategy question",
                answer="",
                info={}
            )


class TestEnvironmentIntegration:
    """Test integration with verifiers framework."""
    
    def test_environment_interface_compliance(self):
        """Test that our environment properly implements verifiers.Environment interface."""
        env = create_test_environment()
        
        # Should have required methods
        assert hasattr(env, 'rollout')
        assert hasattr(env, 'generate')
        assert hasattr(env, 'get_dataset')
        
        # Should have dataset
        assert env.dataset is not None
        assert len(env.dataset) > 0
        
        # Dataset should have required columns
        assert 'prompt' in env.dataset.column_names
        assert 'answer' in env.dataset.column_names
    
    def test_dataset_structure(self):
        """Test that dataset has proper structure for country-based training."""
        env = create_test_environment()
        dataset = env.get_dataset()
        
        assert dataset is not None
        
        # Should have all countries represented
        countries_in_dataset = set(dataset['country'])
        assert countries_in_dataset == set(env.countries)
        
        # Each country should have multiple scenarios
        for country in env.countries:
            country_scenarios = [item for item in dataset if item['country'] == country]
            assert len(country_scenarios) >= 3  # At least 3 scenarios per country
            
            # Each scenario should have proper structure
            for scenario in country_scenarios:
                assert 'prompt' in scenario
                assert 'country' in scenario
                assert 'answer' in scenario
                assert country in scenario['prompt']  # Country should be mentioned in prompt


@pytest.mark.parametrize("country", ["Austria", "England", "France", "Germany", "Italy", "Russia", "Turkey"])
class TestCountrySpecificBehavior:
    """Test country-specific behavior across all countries."""
    
    def test_model_assignment_for_country(self, country):
        """Test model assignment works for each country."""
        env = create_test_environment()
        
        assignments = env._create_model_assignment(country, 'training-model')
        
        assert assignments[country] == 'training-model'
        assert len(assignments) == 7
        
        # Other countries should have default model
        for other_country in env.countries:
            if other_country != country:
                assert assignments[other_country] == 'gpt-4o-mini'
    
    def test_outcome_determination_for_country(self, country):
        """Test outcome determination for each country."""
        env = create_test_environment()
        
        # Test survival scenario
        game_data = {
            'phases': [{
                'state': {
                    'centers': {
                        country: ['A', 'B', 'C', 'D']  # 4 centers
                    }
                }
            }]
        }
        
        outcome = env._determine_outcome(game_data, country)
        assert outcome == 'surviving'


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_game_files(self):
        """Test handling when game output files are missing."""
        env = create_test_environment()
        
        with patch('subprocess.run') as mock_subprocess, \
             patch('pathlib.Path.exists') as mock_exists:
            
            mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
            mock_exists.return_value = False  # No game files created
            
            mock_client = Mock()
            
            completion, state = env.rollout(
                client=mock_client,
                model="test-model",
                prompt="You are Austria in Spring 1901.",
                answer="",
                info={'country': 'Austria'}
            )
            
            assert completion.startswith("[ERROR]")
            assert 'error' in state
    
    def test_subprocess_timeout(self):
        """Test handling of game execution timeout."""
        env = create_test_environment()
        
        with patch('subprocess.run') as mock_subprocess:
            import subprocess
            mock_subprocess.side_effect = subprocess.TimeoutExpired('cmd', 1800)
            
            mock_client = Mock()
            
            completion, state = env.rollout(
                client=mock_client,
                model="test-model",
                prompt="You are Austria in Spring 1901.",
                answer="",
                info={'country': 'Austria'}
            )
            
            assert completion.startswith("[ERROR]")
            assert 'timed out' in completion.lower()
            assert 'error' in state
    
    def test_empty_game_data(self):
        """Test handling of empty or malformed game data."""
        env = create_test_environment()
        
        # Test with empty game data
        game_result = {
            'game_data': {'phases': []},
            'focus_country': 'Austria'
        }
        
        episode_data = env._extract_episode_data(game_result, 'Austria')
        
        assert episode_data['completion_sequence'] == ''
        assert episode_data['supply_centers'] == []
        assert episode_data['game_outcome'] == 'unknown'


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_test_environment(self):
        """Test test environment creation utility."""
        env = create_test_environment()
        
        assert isinstance(env, DiplomacyGRPOEnvironment)
        assert env.max_year == 1902  # Short for testing
        assert env.num_negotiation_rounds == 1
        
        # Should accept additional kwargs
        env_custom = create_test_environment(
            countries=['Austria', 'England'],
            max_year=1901
        )
        
        assert env_custom.countries == ['Austria', 'England']
        assert env_custom.max_year == 1901