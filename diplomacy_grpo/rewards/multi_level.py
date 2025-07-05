"""
Multi-level reward system for Diplomacy.

Implements year-level and game-level rewards to address reward sparsity
while providing meaningful intermediate feedback.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class RewardWeights:
    """Configuration for reward component weights."""
    year_level: float = 1.0
    game_level: float = 10.0
    order_validity: float = 0.1
    diplomatic_success: float = 0.5


class MultiLevelReward:
    """
    Multi-level reward function for Diplomacy episodes.
    
    Combines multiple reward signals to provide rich feedback:
    1. Year-level rewards: Supply center gains/losses during the year
    2. Game-level rewards: Final game outcome (win/draw/loss)
    3. Order validity rewards: Penalty for invalid orders
    4. Diplomatic rewards: Success in negotiations and alliance formation
    
    This addresses reward sparsity by providing intermediate feedback
    while maintaining focus on the ultimate game objective.
    """
    
    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()
        
        # Country-specific starting supply center counts for normalization
        self.country_starting_centers = {
            'Austria': 3,
            'England': 3, 
            'France': 3,
            'Germany': 3,
            'Italy': 3,
            'Russia': 4,  # Russia starts with 4 centers
            'Turkey': 3,
        }
        
    def compute_reward(
        self,
        country: str,
        episode_data: Dict[str, Any],
        game_context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute total reward for an episode.
        
        Args:
            country: The country this episode is for
            episode_data: Data for this specific episode/phase
            game_context: Broader game context and final outcomes
            
        Returns:
            Dictionary with individual reward components and total
        """
        rewards = {}
        
        # Year-level reward: supply center progression
        rewards['year_level'] = self._compute_year_reward(
            country, episode_data, game_context
        )
        
        # Game-level reward: final outcome
        rewards['game_level'] = self._compute_game_reward(
            country, episode_data, game_context  
        )
        
        # Order validity reward: penalty for invalid orders
        rewards['order_validity'] = self._compute_order_validity_reward(
            country, episode_data, game_context
        )
        
        # Diplomatic success reward: negotiation outcomes
        rewards['diplomatic_success'] = self._compute_diplomatic_reward(
            country, episode_data, game_context
        )
        
        # Compute weighted total
        rewards['total'] = (
            self.weights.year_level * rewards['year_level'] +
            self.weights.game_level * rewards['game_level'] +
            self.weights.order_validity * rewards['order_validity'] +
            self.weights.diplomatic_success * rewards['diplomatic_success']
        )
        
        return rewards
        
    def _compute_year_reward(
        self,
        country: str,
        episode_data: Dict[str, Any],
        game_context: Dict[str, Any],
    ) -> float:
        """
        Compute reward based on supply center changes during the year.
        
        This provides intermediate feedback to address reward sparsity.
        Normalized by country's starting position to ensure fairness.
        """
        initial_centers = episode_data.get('initial_supply_centers', 0)
        final_centers = episode_data.get('final_supply_centers', 0)
        
        if initial_centers == 0:
            return 0.0
            
        # Calculate net change
        center_change = final_centers - initial_centers
        
        # Normalize by starting position  
        starting_centers = self.country_starting_centers.get(country, 3)
        normalized_change = center_change / starting_centers
        
        # Apply sigmoid scaling to prevent extreme values
        return math.tanh(normalized_change)
        
    def _compute_game_reward(
        self,
        country: str, 
        episode_data: Dict[str, Any],
        game_context: Dict[str, Any],
    ) -> float:
        """
        Compute reward based on final game outcome.
        
        Large reward for winning, moderate for draws, penalty for elimination.
        """
        game_result = game_context.get('game_result', 'ongoing')
        winner = game_context.get('winner')
        is_eliminated = game_context.get('eliminated_countries', [])
        
        if game_result == 'victory' and winner == country:
            return 1.0
        elif game_result == 'draw':
            # Draw reward based on final supply center count
            final_centers = episode_data.get('final_supply_centers', 0)
            # Scale between 0.1 and 0.5 based on final position
            return 0.1 + 0.4 * min(final_centers / 18.0, 1.0)
        elif country in is_eliminated:
            return -0.3
        elif game_result == 'defeat':
            return -0.1
            
        return 0.0  # Ongoing game
        
    def _compute_order_validity_reward(
        self,
        country: str,
        episode_data: Dict[str, Any], 
        game_context: Dict[str, Any],
    ) -> float:
        """
        Compute reward/penalty based on order validity.
        
        Encourages the model to generate valid orders.
        """
        orders = episode_data.get('orders', [])
        invalid_orders = episode_data.get('invalid_orders', [])
        
        if not orders:
            return 0.0
            
        validity_ratio = 1.0 - (len(invalid_orders) / len(orders))
        
        # Penalty for invalid orders, bonus for all valid
        if validity_ratio == 1.0:
            return 0.1  # Small bonus for perfect validity
        else:
            return -0.2 * (1.0 - validity_ratio)  # Penalty proportional to invalidity
            
    def _compute_diplomatic_reward(
        self,
        country: str,
        episode_data: Dict[str, Any],
        game_context: Dict[str, Any], 
    ) -> float:
        """
        Compute reward based on diplomatic success.
        
        Rewards successful alliance formation, coordination, and negotiation.
        """
        # Alliance formation bonus
        alliances_formed = episode_data.get('alliances_formed', [])
        alliance_bonus = 0.05 * len(alliances_formed)
        
        # Coordination success bonus
        successful_coordinations = episode_data.get('successful_coordinations', 0)
        coordination_bonus = 0.02 * successful_coordinations
        
        # Betrayal penalty (if detected)
        betrayals_committed = episode_data.get('betrayals_committed', 0)
        betrayal_penalty = -0.1 * betrayals_committed
        
        # Message response rate bonus
        messages_sent = episode_data.get('messages_sent', 0)
        messages_responded = episode_data.get('messages_responded', 0)
        
        response_bonus = 0.0
        if messages_sent > 0:
            response_rate = messages_responded / messages_sent
            response_bonus = 0.03 * response_rate
            
        total_diplomatic = (
            alliance_bonus + 
            coordination_bonus + 
            betrayal_penalty + 
            response_bonus
        )
        
        # Clamp to reasonable range
        return max(-0.2, min(0.2, total_diplomatic))
        
    def get_weights(self) -> RewardWeights:
        """Get current reward weights."""
        return self.weights
        
    def set_weights(self, weights: RewardWeights) -> None:
        """Update reward weights."""
        self.weights = weights