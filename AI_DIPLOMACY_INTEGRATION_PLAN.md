# AI_Diplomacy Integration Plan

## Overview

This document outlines the integration between our Phase 1 `diplomacy_grpo` foundation and the AI_Diplomacy framework to create a complete GRPO self-play pipeline. The integration leverages our country-based advantage computation with AI_Diplomacy's sophisticated game engine and agent system.

## Current Architecture Analysis

### Phase 1 Foundation (diplomacy_grpo) ✅
- **CountryGroupedGRPOTrainer**: GRPO with country-specific advantage computation
- **CountryBalancedSampler**: Equal country representation in batches
- **MultiLevelReward**: Multi-level reward system (year/game/diplomatic/order validity)
- **VLLMBatchClient**: Async batch inference for efficient generation
- **MockDiplomacyEnvironment**: Complete testing infrastructure (121 tests)

### AI_Diplomacy Framework 
- **Core Game Engine**: `diplomacy.Game` class for real Diplomacy gameplay
- **Agent System**: `DiplomacyAgent` with memory, relationships, goal tracking
- **LLM Clients**: `BaseModelClient` supporting multiple providers (OpenAI, Anthropic, vLLM)
- **Game Orchestration**: `lm_game.py` coordinates complete game lifecycle
- **Analysis Tools**: Strategic moment detection, lie analysis, visualizations

### Verifiers Framework
- **Base GRPO Trainer**: `verifiers.trainers.grpo_trainer.GRPOTrainer` 
- **Environment Interface**: `verifiers.envs.environment.Environment`
- **Training Infrastructure**: Async batch generation, configuration management

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO Self-Play Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│  Training Loop (diplomacy_grpo/core/country_trainer.py)         │
│  - CountryGroupedGRPOTrainer extends verifiers.GRPOTrainer      │
│  - Country-specific advantage computation                        │
│  - Multi-level rewards from AI_Diplomacy games                  │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│             Environment Layer (NEW)                             │
│  - DiplomacyGRPOEnvironment implements verifiers.Environment    │
│  - Wraps AI_Diplomacy game engine                              │  
│  - Manages batch game generation                                │
│  - Extracts training episodes and rewards                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│           AI_Diplomacy Integration                              │
│  - Real Diplomacy games via diplomacy.Game                     │
│  - DiplomacyAgent with memory/relationships                     │
│  - Multiple LLM providers (OpenAI, Anthropic, vLLM)           │
│  - Game analysis and strategic moment detection                │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 2A: Core Environment Integration (Week 1)

#### 1. Create DiplomacyGRPOEnvironment

**File**: `diplomacy_grpo/integration/ai_diplomacy_env.py`

```python
class DiplomacyGRPOEnvironment(Environment):
    """
    GRPO environment that wraps AI_Diplomacy for real game generation.
    
    Integrates our country-based GRPO approach with full Diplomacy games,
    enabling training on realistic multi-agent interactions.
    """
    
    def __init__(
        self,
        model_configs: Dict[str, str],  # country -> model mapping
        max_year: int = 1905,
        num_negotiation_rounds: int = 3,
        countries: Optional[List[str]] = None,
        **kwargs
    ):
        self.countries = countries or ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
        self.model_configs = model_configs
        self.max_year = max_year
        self.num_negotiation_rounds = num_negotiation_rounds
        
    async def generate_episode(
        self, 
        country: str, 
        model: PreTrainedModel,
        prompt_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate training episode by playing a full Diplomacy game."""
        
        # 1. Create AI_Diplomacy game
        game_config = self._create_game_config(country, model)
        
        # 2. Run complete game using lm_game.py orchestration
        game_result = await self._run_diplomacy_game(game_config)
        
        # 3. Extract country-specific episode data
        episode_data = self._extract_episode_data(game_result, country)
        
        # 4. Compute multi-level rewards
        rewards = self._compute_rewards(episode_data, game_result)
        
        return {
            'country': country,
            'prompt': episode_data['initial_prompt'],
            'completion': episode_data['completion_sequence'], 
            'reward': rewards['total'],
            'detailed_rewards': rewards,
            'game_metadata': game_result['metadata']
        }
```

#### 2. Game Configuration and Model Assignment

```python
def _create_game_config(self, focus_country: str, focus_model: PreTrainedModel) -> Dict:
    """Create AI_Diplomacy game configuration with model assignments."""
    
    # Assign our training model to focus country
    model_assignments = {focus_country: focus_model}
    
    # Assign reference models to other countries
    for country in self.countries:
        if country != focus_country:
            model_assignments[country] = self._get_reference_model(country)
    
    return {
        'model_assignments': model_assignments,
        'max_year': self.max_year,
        'num_negotiation_rounds': self.num_negotiation_rounds,
        'enable_analysis': True,  # For strategic moment detection
        'save_game_data': True
    }

def _get_reference_model(self, country: str) -> str:
    """Get reference model for non-focus countries."""
    # Use strong reference models to create realistic opponents
    return self.model_configs.get(country, "gpt-4o")
```

#### 3. Episode Data Extraction

```python
def _extract_episode_data(self, game_result: Dict, country: str) -> Dict:
    """Extract training-relevant data for specific country."""
    
    agent_data = game_result['agents'][country]
    
    return {
        'initial_prompt': self._construct_initial_prompt(country, game_result),
        'completion_sequence': self._extract_completion_sequence(agent_data),
        'orders_history': agent_data['orders_history'],
        'negotiation_history': agent_data['negotiations'],
        'strategic_decisions': agent_data['strategic_decisions'],
        'supply_center_progression': agent_data['supply_centers_by_phase'],
        'final_outcome': game_result['game_outcome'][country]
    }

def _construct_initial_prompt(self, country: str, game_result: Dict) -> str:
    """Construct prompt that captures the strategic scenario."""
    
    # Use AI_Diplomacy's prompt system
    initial_state = game_result['initial_state']
    
    # Get country-specific system prompt
    system_prompt = self._load_country_system_prompt(country)
    
    # Create scenario prompt
    scenario_prompt = f"""
    You are {country} in a Diplomacy game starting in Spring 1901.
    
    Initial Board State:
    {initial_state['board_description']}
    
    Your starting position:
    - Supply Centers: {initial_state['supply_centers'][country]}
    - Units: {initial_state['units'][country]}
    
    Objective: Achieve victory through strategic play, diplomacy, and tactical excellence.
    """
    
    return f"{system_prompt}\n\n{scenario_prompt}"
```

### Phase 2B: Reward Integration (Week 1-2)

#### 4. AI_Diplomacy Reward Extraction

```python
def _compute_rewards(self, episode_data: Dict, game_result: Dict) -> Dict:
    """Compute multi-level rewards using our MultiLevelReward system."""
    
    from diplomacy_grpo.rewards.multi_level import MultiLevelReward
    
    reward_system = MultiLevelReward()
    
    # Extract data for reward computation
    country = episode_data['country']
    initial_centers = len(episode_data['supply_center_progression'][0])
    final_centers = len(episode_data['supply_center_progression'][-1])
    
    # Build episode data dict
    episode_reward_data = {
        'initial_supply_centers': initial_centers,
        'final_supply_centers': final_centers,
        'orders': episode_data['orders_history'],
        'invalid_orders': self._extract_invalid_orders(episode_data),
        'negotiation_success': self._compute_negotiation_metrics(episode_data),
        'alliance_formations': self._extract_alliance_data(episode_data)
    }
    
    # Build game context
    game_context = {
        'game_result': episode_data['final_outcome'],
        'winner': game_result['winner'],
        'final_year': game_result['final_year'],
        'elimination_year': game_result.get('elimination_years', {}).get(country)
    }
    
    return reward_system.compute_reward(
        country=country,
        episode_data=episode_reward_data,
        game_context=game_context
    )
```

#### 5. Strategic Metrics Extraction

```python
def _compute_negotiation_metrics(self, episode_data: Dict) -> float:
    """Compute negotiation success metrics from AI_Diplomacy data."""
    
    negotiations = episode_data['negotiation_history']
    
    # Metrics from AI_Diplomacy's analysis tools
    successful_alliances = 0
    betrayal_incidents = 0
    coordination_successes = 0
    
    for phase_negotiations in negotiations:
        # Use AI_Diplomacy's strategic moment analysis
        moments = self._analyze_strategic_moments(phase_negotiations)
        
        successful_alliances += moments.get('alliance_formations', 0)
        betrayal_incidents += moments.get('betrayals', 0) 
        coordination_successes += moments.get('coordinated_attacks', 0)
    
    # Compute success ratio
    total_diplomatic_actions = successful_alliances + betrayal_incidents + coordination_successes
    if total_diplomatic_actions == 0:
        return 0.0
    
    success_score = (successful_alliances + coordination_successes) / total_diplomatic_actions
    return success_score

def _analyze_strategic_moments(self, negotiations: List) -> Dict:
    """Use AI_Diplomacy's strategic moment analysis."""
    
    # Import AI_Diplomacy analysis tools
    from analyze_game_moments_llm import analyze_strategic_moments
    
    return analyze_strategic_moments(negotiations)
```

### Phase 2C: Batch Processing Integration (Week 2)

#### 6. Multi-Game Batch Coordinator

**File**: `diplomacy_grpo/integration/batch_coordinator.py`

```python
class DiplomacyBatchCoordinator:
    """
    Coordinates multiple simultaneous Diplomacy games for GRPO training.
    
    Manages the target scale: 35 concurrent instances (5 games × 7 countries)
    """
    
    def __init__(
        self,
        target_batch_size: int = 35,  # 5 games × 7 countries
        games_per_batch: int = 5,
        environment: DiplomacyGRPOEnvironment,
        vllm_client: VLLMBatchClient
    ):
        self.target_batch_size = target_batch_size
        self.games_per_batch = games_per_batch
        self.environment = environment
        self.vllm_client = vllm_client
        
    async def generate_training_batch(
        self, 
        model: PreTrainedModel
    ) -> Dict[str, Any]:
        """Generate complete training batch from multiple games."""
        
        # 1. Create country-balanced game assignments
        game_assignments = self._create_game_assignments()
        
        # 2. Run games in parallel
        game_tasks = []
        for game_id, country_assignments in game_assignments.items():
            task = self._run_game_with_focus_rotation(
                game_id, country_assignments, model
            )
            game_tasks.append(task)
        
        # Execute all games concurrently
        game_results = await asyncio.gather(*game_tasks)
        
        # 3. Extract and combine episodes
        training_batch = self._combine_game_episodes(game_results)
        
        return training_batch

    def _create_game_assignments(self) -> Dict[int, List[str]]:
        """Create balanced country assignments across games."""
        
        assignments = {}
        countries = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
        
        # Each game focuses on all countries sequentially
        for game_id in range(self.games_per_batch):
            # Rotate country focus to ensure balance
            focus_countries = countries[game_id:] + countries[:game_id]
            assignments[game_id] = focus_countries
            
        return assignments

    async def _run_game_with_focus_rotation(
        self, 
        game_id: int, 
        focus_countries: List[str], 
        model: PreTrainedModel
    ) -> List[Dict]:
        """Run single game collecting episodes for all countries."""
        
        episodes = []
        
        for country in focus_countries:
            episode = await self.environment.generate_episode(
                country=country,
                model=model,
                game_instance_id=f"{game_id}_{country}"
            )
            episodes.append(episode)
            
        return episodes
```

### Phase 2D: Configuration Management (Week 2)

#### 7. Model Configuration System

**File**: `diplomacy_grpo/config/training_config.py`

```python
@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training with AI_Diplomacy."""
    
    # Model configuration
    training_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reference_models: Dict[str, str] = field(default_factory=lambda: {
        'Austria': 'gpt-4o',
        'England': 'claude-3-5-sonnet-20241022', 
        'France': 'gpt-4o',
        'Germany': 'claude-3-5-sonnet-20241022',
        'Italy': 'gpt-4o', 
        'Russia': 'claude-3-5-sonnet-20241022',
        'Turkey': 'gpt-4o'
    })
    
    # Game configuration  
    max_year: int = 1905
    num_negotiation_rounds: int = 3
    games_per_batch: int = 5
    target_batch_size: int = 35
    
    # GRPO configuration
    country_specific_normalization: bool = True
    learning_rate: float = 1e-5
    num_training_epochs: int = 3
    
    # vLLM configuration
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    max_concurrent_requests: int = 35
    
    # Reward configuration
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'year_level': 0.3,
        'game_level': 0.4, 
        'order_validity': 0.1,
        'diplomatic': 0.2
    })

def load_training_config(config_path: str) -> GRPOTrainingConfig:
    """Load training configuration from file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return GRPOTrainingConfig(**config_dict)
```

### Phase 2E: Integration Testing (Week 3)

#### 8. End-to-End Integration Tests

**File**: `tests/integration/test_ai_diplomacy_integration.py`

```python
class TestAIDiplomacyIntegration:
    """Test complete integration with AI_Diplomacy framework."""
    
    @pytest.mark.asyncio
    async def test_full_game_episode_generation(self):
        """Test generating training episode from real Diplomacy game."""
        
        # Create environment with test configuration
        env = DiplomacyGRPOEnvironment(
            model_configs={'Austria': 'gpt-4o-mini'},  # Use cheaper model for tests
            max_year=1902,  # Short game for testing
            num_negotiation_rounds=1
        )
        
        # Mock model for testing
        mock_model = create_mock_model()
        
        # Generate episode
        episode = await env.generate_episode(
            country='Austria',
            model=mock_model
        )
        
        # Validate episode structure
        assert 'country' in episode
        assert 'prompt' in episode  
        assert 'completion' in episode
        assert 'reward' in episode
        assert 'detailed_rewards' in episode
        
        # Validate rewards structure
        rewards = episode['detailed_rewards']
        assert 'year_level' in rewards
        assert 'game_level' in rewards
        assert 'order_validity' in rewards
        assert 'diplomatic' in rewards
        
    @pytest.mark.asyncio 
    async def test_batch_coordination(self):
        """Test coordinated batch generation across multiple games."""
        
        coordinator = DiplomacyBatchCoordinator(
            target_batch_size=14,  # 2 games × 7 countries for testing
            games_per_batch=2,
            environment=create_test_environment(),
            vllm_client=create_mock_vllm_client()
        )
        
        # Generate training batch
        batch = await coordinator.generate_training_batch(
            model=create_mock_model()
        )
        
        # Validate batch structure
        assert len(batch['countries']) == 14
        assert len(batch['rewards']) == 14
        assert len(batch['prompts']) == 14
        assert len(batch['completions']) == 14
        
        # Verify country balance
        country_counts = Counter(batch['countries'])
        assert all(count == 2 for count in country_counts.values())  # 2 per country

    def test_reward_system_integration(self):
        """Test multi-level rewards with AI_Diplomacy game data."""
        
        # Load real game data for testing
        with open('tests/fixtures/sample_game_result.json', 'r') as f:
            game_result = json.load(f)
        
        env = DiplomacyGRPOEnvironment({})
        
        # Extract episode data
        episode_data = env._extract_episode_data(game_result, 'Austria')
        
        # Compute rewards
        rewards = env._compute_rewards(episode_data, game_result)
        
        # Validate reward computation
        assert 0.0 <= rewards['total'] <= 20.0  # Within expected range
        assert rewards['year_level'] >= 0.0
        assert rewards['game_level'] >= 0.0
```

## Implementation Timeline

### Week 1: Core Environment Integration
- [ ] Implement `DiplomacyGRPOEnvironment` class
- [ ] Create game configuration and model assignment system
- [ ] Build episode data extraction pipeline
- [ ] Integrate multi-level reward computation

### Week 2: Batch Processing and Configuration
- [ ] Implement `DiplomacyBatchCoordinator` for multi-game management
- [ ] Create configuration management system
- [ ] Add strategic metrics extraction using AI_Diplomacy analysis
- [ ] Implement data persistence and logging

### Week 3: Integration Testing and Validation
- [ ] Create comprehensive integration test suite
- [ ] Test with real AI_Diplomacy games (short games for rapid iteration)
- [ ] Validate country-balanced batch generation
- [ ] Performance testing and optimization

### Week 4: Production Readiness
- [ ] Scale testing to full batch size (35 concurrent instances)
- [ ] Performance optimization and memory management
- [ ] Error handling and recovery systems
- [ ] Documentation and deployment guides

## Key Integration Points

### 1. Environment Interface Compliance
```python
# Our environment must implement verifiers.Environment
class DiplomacyGRPOEnvironment(Environment):
    def sample(self, model: PreTrainedModel) -> List[Dict[str, Any]]:
        """Required by verifiers framework"""
        
    async def evaluate(self, samples: List[Dict[str, Any]]) -> List[float]:
        """Required by verifiers framework"""
```

### 2. Model Integration 
```python
# Use AI_Diplomacy's BaseModelClient interface
from ai_diplomacy.clients import BaseModelClient

class VLLMDiplomacyClient(BaseModelClient):
    """Bridge our VLLMBatchClient with AI_Diplomacy interface"""
```

### 3. Game State Integration
```python
# Use AI_Diplomacy's game state management
from diplomacy.engine.game import Game
from ai_diplomacy.agent import DiplomacyAgent

# Integration in our environment
game = Game()
agents = {country: DiplomacyAgent(country, client, goals, relationships) 
          for country in self.countries}
```

## Expected Benefits

### 1. Realistic Training Data
- Real multi-agent Diplomacy games with sophisticated AI opponents
- Strategic complexity from negotiation, alliance formation, betrayal
- Authentic game dynamics from AI_Diplomacy's proven framework

### 2. Country-Specific Learning
- Our country-grouped GRPO enables specialized learning per country
- Addresses non-stationarity in multi-agent environments
- Balanced representation ensures equal learning across all countries

### 3. Multi-Level Reward Signals
- Rich feedback from year-level, game-level, diplomatic, and order validity rewards
- Addresses reward sparsity common in strategic games
- Enables learning of both tactical and strategic behaviors

### 4. Scalable Architecture
- Designed for target scale: 35 concurrent instances
- Async batch processing for efficient resource utilization
- Integration with vLLM for fast inference

## Risk Mitigation

### 1. AI_Diplomacy Dependency
- **Risk**: AI_Diplomacy updates affecting compatibility
- **Mitigation**: Maintain our mock environment for independent development
- **Fallback**: Can continue development with mocks until integration possible

### 2. Performance at Scale
- **Risk**: 35 concurrent games may strain resources
- **Mitigation**: Implement gradual scaling with performance monitoring
- **Optimization**: Use game length limits and efficient batching

### 3. Model Assignment Complexity
- **Risk**: Complex model configurations for different countries
- **Mitigation**: Start with simple uniform assignments, add complexity gradually
- **Validation**: Extensive testing with different model combinations

## Success Metrics

### Technical Metrics
- ✅ **Integration Tests Passing**: All integration tests with real AI_Diplomacy games
- ✅ **Batch Generation**: Successful 35-instance concurrent batch generation  
- ✅ **Reward Computation**: Multi-level rewards computed from real game data
- ✅ **Country Balance**: Equal representation across all 7 countries

### Research Metrics  
- ✅ **Training Convergence**: GRPO training converges with country-grouped advantages
- ✅ **Strategic Learning**: Model demonstrates improved strategic play over training
- ✅ **Country Specialization**: Evidence of country-specific learning patterns

### Performance Metrics
- ✅ **Throughput**: Target batch generation time < 2 hours per batch
- ✅ **Resource Usage**: Memory and compute usage within acceptable limits
- ✅ **Reliability**: <5% failure rate in game generation and completion

This integration plan provides a complete roadmap for bridging our Phase 1 foundation with the AI_Diplomacy framework, enabling the full GRPO self-play pipeline with country-based advantage computation at scale.