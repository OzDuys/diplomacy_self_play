"""
Unit tests for MultiLevelReward system.
"""

import pytest
from diplomacy_grpo.rewards.multi_level import MultiLevelReward, RewardWeights


@pytest.fixture
def reward_system():
    """Create a MultiLevelReward instance with default weights."""
    return MultiLevelReward()


@pytest.fixture
def custom_reward_system():
    """Create a MultiLevelReward instance with custom weights."""
    weights = RewardWeights(
        year_level=2.0,
        game_level=5.0,
        order_validity=0.2,
        diplomatic_success=1.0
    )
    return MultiLevelReward(weights)


@pytest.fixture
def sample_episode_data():
    """Create sample episode data for testing."""
    return {
        'initial_supply_centers': 3,
        'final_supply_centers': 5,
        'orders': ['A VIE-BUD', 'A TRI-VEN', 'F TRI-ALB'],
        'invalid_orders': [],
        'alliances_formed': ['Germany'],
        'successful_coordinations': 2,
        'betrayals_committed': 0,
        'messages_sent': 5,
        'messages_responded': 4,
    }


@pytest.fixture 
def sample_game_context():
    """Create sample game context for testing."""
    return {
        'game_result': 'victory',
        'winner': 'Austria',
        'eliminated_countries': [],
    }


class TestRewardWeights:
    """Test the RewardWeights dataclass."""
    
    def test_default_weights(self):
        """Test default weight values."""
        weights = RewardWeights()
        assert weights.year_level == 1.0
        assert weights.game_level == 10.0
        assert weights.order_validity == 0.1
        assert weights.diplomatic_success == 0.5
        
    def test_custom_weights(self):
        """Test custom weight values."""
        weights = RewardWeights(
            year_level=2.0,
            game_level=15.0,
            order_validity=0.2,
            diplomatic_success=1.0
        )
        assert weights.year_level == 2.0
        assert weights.game_level == 15.0
        assert weights.order_validity == 0.2
        assert weights.diplomatic_success == 1.0


class TestMultiLevelReward:
    """Test the MultiLevelReward class."""
    
    def test_initialization(self, reward_system):
        """Test reward system initialization."""
        assert reward_system.weights.year_level == 1.0
        assert reward_system.weights.game_level == 10.0
        assert 'Austria' in reward_system.country_starting_centers
        assert reward_system.country_starting_centers['Russia'] == 4
        
    def test_compute_year_reward_positive_gain(self, reward_system):
        """Test year reward computation for positive supply center gain."""
        episode_data = {
            'initial_supply_centers': 3,
            'final_supply_centers': 5,  # +2 centers
        }
        game_context = {}
        
        reward = reward_system._compute_year_reward('Austria', episode_data, game_context)
        
        # Should be positive for gain
        assert reward > 0
        # Should be normalized by starting centers (3 for Austria)
        # 2/3 ≈ 0.67, tanh(0.67) ≈ 0.58
        assert 0.5 < reward < 0.7
        
    def test_compute_year_reward_negative_loss(self, reward_system):
        """Test year reward computation for supply center loss."""
        episode_data = {
            'initial_supply_centers': 3,
            'final_supply_centers': 1,  # -2 centers
        }
        game_context = {}
        
        reward = reward_system._compute_year_reward('Austria', episode_data, game_context)
        
        # Should be negative for loss
        assert reward < 0
        
    def test_compute_year_reward_no_change(self, reward_system):
        """Test year reward when no supply centers change."""
        episode_data = {
            'initial_supply_centers': 3,
            'final_supply_centers': 3,  # No change
        }
        game_context = {}
        
        reward = reward_system._compute_year_reward('Austria', episode_data, game_context)
        
        # Should be zero for no change
        assert reward == 0.0
        
    def test_compute_game_reward_victory(self, reward_system):
        """Test game reward for victory."""
        episode_data = {'final_supply_centers': 18}
        game_context = {
            'game_result': 'victory',
            'winner': 'Austria',
        }
        
        reward = reward_system._compute_game_reward('Austria', episode_data, game_context)
        
        assert reward == 1.0
        
    def test_compute_game_reward_draw(self, reward_system):
        """Test game reward for draw."""
        episode_data = {'final_supply_centers': 10}
        game_context = {
            'game_result': 'draw',
        }
        
        reward = reward_system._compute_game_reward('Austria', episode_data, game_context)
        
        # Should be positive but less than victory
        assert 0.1 < reward < 1.0
        
    def test_compute_game_reward_elimination(self, reward_system):
        """Test game reward for elimination."""
        episode_data = {'final_supply_centers': 0}
        game_context = {
            'game_result': 'defeat',
            'eliminated_countries': ['Austria'],
        }
        
        reward = reward_system._compute_game_reward('Austria', episode_data, game_context)
        
        assert reward == -0.3  # Elimination penalty
        
    def test_compute_order_validity_reward_all_valid(self, reward_system):
        """Test order validity reward when all orders are valid."""
        episode_data = {
            'orders': ['A VIE-BUD', 'A TRI-VEN'],
            'invalid_orders': [],
        }
        game_context = {}
        
        reward = reward_system._compute_order_validity_reward('Austria', episode_data, game_context)
        
        assert reward == 0.1  # Bonus for all valid orders
        
    def test_compute_order_validity_reward_some_invalid(self, reward_system):
        """Test order validity reward when some orders are invalid."""
        episode_data = {
            'orders': ['A VIE-BUD', 'A TRI-VEN', 'INVALID'],
            'invalid_orders': ['INVALID'],
        }
        game_context = {}
        
        reward = reward_system._compute_order_validity_reward('Austria', episode_data, game_context)
        
        # Should be negative penalty
        assert reward < 0
        # 1/3 invalid, so penalty should be -0.2 * (1/3) ≈ -0.067
        assert -0.1 < reward < 0
        
    def test_compute_diplomatic_reward_positive(self, reward_system):
        """Test diplomatic reward with positive interactions."""
        episode_data = {
            'alliances_formed': ['Germany', 'Italy'],  # 2 alliances
            'successful_coordinations': 3,
            'betrayals_committed': 0,
            'messages_sent': 10,
            'messages_responded': 8,  # 80% response rate
        }
        game_context = {}
        
        reward = reward_system._compute_diplomatic_reward('Austria', episode_data, game_context)
        
        # Should be positive
        assert reward > 0
        # 2*0.05 + 3*0.02 + 0 + 0.8*0.03 = 0.1 + 0.06 + 0.024 = 0.184
        assert 0.15 < reward < 0.2
        
    def test_compute_diplomatic_reward_with_betrayals(self, reward_system):
        """Test diplomatic reward with betrayals."""
        episode_data = {
            'alliances_formed': ['Germany'],
            'successful_coordinations': 1, 
            'betrayals_committed': 2,  # Heavy betrayal penalty
            'messages_sent': 5,
            'messages_responded': 3,
        }
        game_context = {}
        
        reward = reward_system._compute_diplomatic_reward('Austria', episode_data, game_context)
        
        # Should likely be negative due to betrayals
        # 0.05 + 0.02 - 0.2 + 0.018 = -0.112
        assert reward < 0
        
    def test_compute_reward_integration(self, reward_system, sample_episode_data, sample_game_context):
        """Test complete reward computation integration."""
        rewards = reward_system.compute_reward('Austria', sample_episode_data, sample_game_context)
        
        # Check all components are present
        assert 'year_level' in rewards
        assert 'game_level' in rewards
        assert 'order_validity' in rewards
        assert 'diplomatic_success' in rewards
        assert 'total' in rewards
        
        # Check total is weighted sum
        expected_total = (
            reward_system.weights.year_level * rewards['year_level'] +
            reward_system.weights.game_level * rewards['game_level'] +
            reward_system.weights.order_validity * rewards['order_validity'] +
            reward_system.weights.diplomatic_success * rewards['diplomatic_success']
        )
        
        assert abs(rewards['total'] - expected_total) < 1e-6
        
    def test_compute_reward_with_custom_weights(self, custom_reward_system, sample_episode_data, sample_game_context):
        """Test reward computation with custom weights."""
        rewards = custom_reward_system.compute_reward('Austria', sample_episode_data, sample_game_context)
        
        # With higher weights, total should be different
        assert 'total' in rewards
        assert isinstance(rewards['total'], float)
        
    def test_weights_getter_setter(self, reward_system):
        """Test getting and setting weights."""
        original_weights = reward_system.get_weights()
        assert original_weights.year_level == 1.0
        
        new_weights = RewardWeights(year_level=3.0, game_level=15.0)
        reward_system.set_weights(new_weights)
        
        updated_weights = reward_system.get_weights()
        assert updated_weights.year_level == 3.0
        assert updated_weights.game_level == 15.0
        
    def test_russia_starting_centers(self, reward_system):
        """Test that Russia has correct starting centers (4 vs 3 for others)."""
        episode_data = {
            'initial_supply_centers': 4,
            'final_supply_centers': 6,  # +2 centers
        }
        game_context = {}
        
        # Russia should have different normalization due to starting with 4 centers
        russia_reward = reward_system._compute_year_reward('Russia', episode_data, game_context)
        
        # For other countries with same gain but 3 starting centers
        episode_data_other = {
            'initial_supply_centers': 3,
            'final_supply_centers': 5,  # +2 centers
        }
        austria_reward = reward_system._compute_year_reward('Austria', episode_data_other, game_context)
        
        # Russia's reward should be smaller due to larger starting base
        assert russia_reward < austria_reward