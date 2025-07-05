"""
AI_Diplomacy integration environment for GRPO training.

Wraps AI_Diplomacy framework to provide verifiers.Environment interface
for country-based GRPO training with real Diplomacy games.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from datasets import Dataset
from openai import OpenAI

try:
    from verifiers.envs.environment import Environment
    from verifiers.parsers import Parser
    from verifiers.rubrics import Rubric
except ImportError:
    # Mock for testing without verifiers
    class Environment:
        def __init__(self, *args, **kwargs):
            # Store common attributes that might be expected
            for key, value in kwargs.items():
                setattr(self, key, value)
    class Parser:
        pass
    class Rubric:
        pass

from diplomacy_grpo.rewards.multi_level import MultiLevelReward


class DiplomacyGRPOEnvironment(Environment):
    """
    GRPO environment that wraps AI_Diplomacy for real game generation.
    
    This environment integrates our country-based GRPO approach with 
    AI_Diplomacy's full game system, enabling training on realistic
    multi-agent strategic interactions.
    
    Key Features:
    - Uses AI_Diplomacy's game engine and agent system
    - Leverages existing vLLM integration from verifiers
    - Extracts training episodes with multi-level rewards
    - Maintains country-balanced sampling for GRPO
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        reference_models: Optional[Dict[str, str]] = None,
        max_year: int = 1905,
        num_negotiation_rounds: int = 1,  # Keep simple for initial testing
        countries: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Diplomacy GRPO environment.
        
        Args:
            client: OpenAI client (uses verifiers default if None)
            model: Model name for training country
            dataset: Dataset with country-specific scenarios (optional)
            reference_models: Models for non-training countries
            max_year: Maximum game year (shorter for faster iteration)
            num_negotiation_rounds: Negotiation rounds per phase
            countries: List of countries (default: all 7)
            working_dir: Directory for game outputs
        """
        self.countries = countries or [
            'Austria', 'England', 'France', 'Germany', 
            'Italy', 'Russia', 'Turkey'
        ]
        
        # Game configuration
        self.max_year = max_year
        self.num_negotiation_rounds = num_negotiation_rounds
        self.reference_models = reference_models or {}
        
        # Reward system
        self.reward_system = MultiLevelReward()
        
        # Working directory for game files
        if working_dir:
            self.working_dir = Path(working_dir)
        else:
            self.working_dir = Path(tempfile.mkdtemp(prefix="diplomacy_grpo_"))
        self.working_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Create simple dataset if none provided
        if dataset is None:
            dataset = self._create_default_dataset()
        
        # Store dataset explicitly 
        self.dataset = dataset
        
        # Initialize parent with simple parser/rubric
        super().__init__(
            client=client,
            model=model,
            dataset=dataset,
            parser=Parser(),  # Simple parser - game handles structure
            rubric=Rubric(),  # Simple rubric - we handle rewards
            **kwargs
        )
    
    def _create_default_dataset(self) -> Dataset:
        """Create a simple dataset for country-based scenarios."""
        scenarios = []
        
        scenario_templates = [
            "You are {country} in Spring 1901. Plan your opening strategy.",
            "You are {country} in Fall 1902. Russia is expanding. Your response?",
            "You are {country} in Spring 1903. Form an alliance to contain a threat.",
        ]
        
        for country in self.countries:
            for i, template in enumerate(scenario_templates):
                scenarios.append({
                    'prompt': template.format(country=country),
                    'country': country,
                    'scenario_id': i,
                    'answer': f"Strategic play as {country}"
                })
        
        return Dataset.from_list(scenarios)
    
    def rollout(
        self,
        client: OpenAI,
        model: str,
        prompt: Union[str, List[Dict[str, Any]]],
        answer: str,
        task: str = "default",
        info: Dict[str, Any] = {},
        sampling_args: Dict[str, Any] = {},
        **kwargs
    ) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Run a single rollout by playing a Diplomacy game.
        
        This is the main entry point called by verifiers framework.
        """
        # Extract country from info or prompt
        country = info.get('country')
        if not country:
            # Try to extract from prompt text
            for c in self.countries:
                if isinstance(prompt, str) and c in prompt:
                    country = c
                    break
                elif isinstance(prompt, list) and any(c in msg.get('content', '') for msg in prompt):
                    country = c
                    break
        
        if not country:
            raise ValueError("Could not determine country from prompt or info")
        
        # Run the game synchronously (verifiers handles async coordination)
        try:
            game_result = self._run_diplomacy_game(country, model, client, sampling_args)
            
            # Extract episode data
            episode_data = self._extract_episode_data(game_result, country)
            
            # Compute rewards
            rewards = self._compute_rewards(episode_data, game_result)
            
            # Format completion (keep it simple for now)
            completion = episode_data.get('completion_sequence', '')
            
            # Return state with all relevant data
            state = {
                'country': country,
                'game_result': game_result,
                'episode_data': episode_data,
                'rewards': rewards,
                'game_file': game_result.get('game_file')
            }
            
            return completion, state
            
        except Exception as e:
            self.logger.error(f"Game execution failed for {country}: {e}")
            # Return error state
            return f"[ERROR] Game failed: {str(e)}", {
                'country': country,
                'error': str(e),
                'rewards': {'total': 0.0}
            }
    
    def _run_diplomacy_game(
        self, 
        focus_country: str, 
        focus_model: str,
        client: OpenAI,
        sampling_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a complete Diplomacy game using AI_Diplomacy framework.
        
        Uses the proper AI_Diplomacy infrastructure instead of subprocess.
        """
        import subprocess
        import sys
        
        # Create model assignment (AI_Diplomacy expects comma-separated models in country order)
        model_assignments = self._create_model_assignment(focus_country, focus_model)
        
        # Create temporary config for this game
        game_id = f"{focus_country}_{hash(focus_model) % 10000}"
        game_dir = self.working_dir / game_id
        game_dir.mkdir(exist_ok=True)
        
        # Build AI_Diplomacy command using proper format
        ai_diplomacy_path = Path(__file__).parent.parent.parent / "AI_Diplomacy"
        lm_game_path = ai_diplomacy_path / "lm_game.py"
        
        # Model order for AI_Diplomacy: AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY
        country_order = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        model_list = [model_assignments[country.title()] for country in country_order]
        
        cmd = [
            sys.executable, 
            str(lm_game_path),
            f"--run_dir={game_dir}",
            f"--max_year={self.max_year}", 
            f"--num_negotiation_rounds={self.num_negotiation_rounds}",
            f"--models={','.join(model_list)}",
            "--generate_phase_summaries=false",  # Keep output minimal for RL training
        ]
        
        try:
            # Run the game
            self.logger.info(f"Running Diplomacy game for {focus_country}: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode != 0:
                self.logger.error(f"Game stderr: {result.stderr}")
                self.logger.error(f"Game stdout: {result.stdout}")
                raise RuntimeError(f"Game failed with return code {result.returncode}")
            
            # Load game results
            game_file = game_dir / "lmvsgame.json"
            if not game_file.exists():
                self.logger.error(f"Game output not found. Game directory contents: {list(game_dir.iterdir())}")
                raise FileNotFoundError(f"Game output file not found at {game_file}")
                
            with open(game_file, 'r') as f:
                game_data = json.load(f)
            
            # Load CSV data for episode extraction
            csv_file = game_dir / "llm_responses.csv"
            csv_data = None
            if csv_file.exists():
                import pandas as pd
                csv_data = pd.read_csv(csv_file)
                
                # Convert to RL format using existing csv_to_rl_json functionality
                rl_data = self._convert_csv_to_rl_format(csv_data, game_id)
            else:
                rl_data = None
                self.logger.warning(f"No CSV data found at {csv_file}")
            
            return {
                'game_data': game_data,
                'csv_data': csv_data,
                'rl_data': rl_data,
                'game_file': str(game_file),
                'game_dir': str(game_dir),
                'focus_country': focus_country,
                'model_assignments': model_assignments
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Game timed out after 30 minutes")
        except Exception as e:
            self.logger.error(f"Game execution error: {e}")
            raise
    
    def _create_model_assignment(self, focus_country: str, focus_model: str) -> Dict[str, str]:
        """Create model assignment for all countries."""
        assignments = {}
        
        for country in self.countries:
            if country == focus_country:
                assignments[country] = focus_model
            else:
                # Use reference model or default
                assignments[country] = self.reference_models.get(
                    country, 
                    "gpt-4o-mini"  # Use cheaper model for non-focus countries
                )
        
        return assignments
        
    def _convert_csv_to_rl_format(self, csv_data, game_id: str) -> List[Dict[str, Any]]:
        """Convert CSV data to RL training format using existing csv_to_rl_json logic."""
        rl_data = []
        
        for index, row in csv_data.iterrows():
            raw_response_data = row.get('raw_response')
            try:
                if isinstance(raw_response_data, str) and \
                   raw_response_data.strip().startswith(('{', '[')) and \
                   raw_response_data.strip().endswith(('}', ']')):
                    llm_response_parsed = json.loads(raw_response_data)
                else:
                    llm_response_parsed = raw_response_data
            except json.JSONDecodeError:
                llm_response_parsed = raw_response_data

            # Parse success value
            success_val = row.get('success')
            if isinstance(success_val, str):
                if success_val.lower() == 'true':
                    success_parsed = True
                elif success_val.lower() == 'false':
                    success_parsed = False
                else:
                    success_parsed = success_val
            else:
                success_parsed = success_val

            entry = {
                "game_id": game_id,
                "model": row.get('model'),
                "power": row.get('power'),
                "phase": row.get('phase'),
                "response_type": row.get('response_type'),
                "prompt": row.get('raw_input'),
                "llm_response": llm_response_parsed,
                "success": success_parsed
            }
            rl_data.append(entry)
        
        return rl_data
    
    def _extract_episode_data(self, game_result: Dict, country: str) -> Dict[str, Any]:
        """Extract training-relevant episode data for specific country."""
        game_data = game_result['game_data']
        csv_data = game_result.get('csv_data')
        rl_data = game_result.get('rl_data', [])
        
        # Extract basic game progression
        phases = game_data.get('phases', [])
        if not phases:
            return {
                'completion_sequence': '', 
                'supply_centers': [],
                'game_outcome': 'unknown',
                'rl_episodes': []
            }
        
        # Get supply center progression
        supply_centers = []
        for phase in phases:
            centers = phase.get('state', {}).get('centers', {}).get(country, [])
            supply_centers.append(len(centers))
        
        # Extract RL training episodes for this country
        country_rl_episodes = []
        if rl_data:
            country_episodes = [ep for ep in rl_data if ep.get('power') == country.upper()]
            country_rl_episodes = country_episodes
        
        # Extract completion sequence from CSV (for backward compatibility)
        completion_sequence = ""
        if csv_data is not None:
            country_responses = csv_data[csv_data['power'] == country.upper()]
            if not country_responses.empty:
                # Combine all responses for this country
                responses = country_responses['raw_response'].tolist()
                import pandas as pd
                completion_sequence = " | ".join(str(r) for r in responses if pd.notna(r))
        
        return {
            'completion_sequence': completion_sequence,
            'supply_centers': supply_centers,
            'initial_centers': supply_centers[0] if supply_centers else 3,
            'final_centers': supply_centers[-1] if supply_centers else 3,
            'phases_completed': len(phases),
            'game_outcome': self._determine_outcome(game_data, country),
            'rl_episodes': country_rl_episodes,
            'total_rl_episodes': len(country_rl_episodes)
        }
    
    def _determine_outcome(self, game_data: Dict, country: str) -> str:
        """Determine game outcome for specific country."""
        # Simple outcome determination
        final_phase = game_data.get('phases', [])
        if not final_phase:
            return 'unknown'
        
        final_state = final_phase[-1].get('state', {})
        centers = final_state.get('centers', {})
        
        if country not in centers:
            return 'eliminated'
        
        country_centers = len(centers[country])
        
        # Check for victory (18+ centers)
        if country_centers >= 18:
            return 'victory'
        
        # Check if eliminated (0 centers)  
        if country_centers == 0:
            return 'eliminated'
        
        # Otherwise surviving
        return 'surviving'
    
    def _compute_rewards(self, episode_data: Dict, game_result: Dict) -> Dict[str, float]:
        """Compute multi-level rewards using our reward system."""
        country = game_result['focus_country']
        rl_episodes = episode_data.get('rl_episodes', [])
        
        # Extract orders and success rates from RL episodes
        valid_orders = []
        invalid_orders = []
        
        for ep in rl_episodes:
            if ep.get('response_type') == 'orders' and ep.get('success') is not None:
                if ep['success']:
                    valid_orders.append(ep.get('llm_response', {}))
                else:
                    invalid_orders.append(ep.get('llm_response', {}))
        
        # Build episode data for reward system
        reward_episode_data = {
            'initial_supply_centers': episode_data.get('initial_centers', 3),
            'final_supply_centers': episode_data.get('final_centers', 3),
            'orders': valid_orders,
            'invalid_orders': invalid_orders,
            'total_interactions': len(rl_episodes),
            'order_success_rate': len(valid_orders) / max(len(valid_orders) + len(invalid_orders), 1)
        }
        
        # Build game context
        game_context = {
            'game_result': episode_data.get('game_outcome', 'unknown'),
            'winner': country if episode_data.get('game_outcome') == 'victory' else None,
            'final_year': self.max_year,
            'phases_completed': episode_data.get('phases_completed', 0)
        }
        
        rewards = self.reward_system.compute_reward(
            country=country,
            episode_data=reward_episode_data,
            game_context=game_context
        )
        
        # Add RL-specific rewards
        rewards['rl_specific'] = {
            'order_success_bonus': reward_episode_data['order_success_rate'] * 10.0,
            'interaction_volume_bonus': min(len(rl_episodes) / 50.0, 1.0) * 5.0,  # Normalize to 50 interactions
            'episode_count': len(rl_episodes)
        }
        
        return rewards


# Simple utility functions for testing
def create_test_environment(**kwargs) -> DiplomacyGRPOEnvironment:
    """Create test environment with minimal configuration."""
    return DiplomacyGRPOEnvironment(
        max_year=1902,  # Very short for testing
        num_negotiation_rounds=1,
        **kwargs
    )