# GRPO in Multi-Agent Environments: Analysis and Country-Based Grouping Solution

## Executive Summary

This document analyzes how Group Relative Policy Optimization (GRPO) can be adapted for multi-agent, non-deterministic environments like Diplomacy. We identify fundamental challenges with standard GRPO in multi-agent settings and propose an elegant solution: **country-based grouping** that preserves GRPO's theoretical foundations while handling multi-agent complexity.

## Table of Contents

1. [GRPO Algorithm Overview](#grpo-algorithm-overview)
2. [Multi-Turn GRPO Implementation in Verifiers](#multi-turn-grpo-implementation-in-verifiers)
3. [Challenges in Multi-Agent Environments](#challenges-in-multi-agent-environments)
4. [Country-Based Grouping Solution](#country-based-grouping-solution)
5. [Implementation Design](#implementation-design)
6. [Extensions and Variations](#extensions-and-variations)
7. [Conclusion](#conclusion)

## GRPO Algorithm Overview

**Group Relative Policy Optimization (GRPO)** is a memory-efficient alternative to PPO that:

- **Eliminates value function**: No separate critic network needed, reducing memory by ~50%
- **Group-based advantage calculation**: Generates multiple responses per prompt to form "groups"
- **Relative comparison**: Uses mean reward within each group as baseline for advantage calculation
- **Memory efficient**: Particularly beneficial for large language models

### Key Innovation

Instead of relying on a separate value function network, GRPO generates multiple responses for each prompt and uses the mean reward of these responses as the baseline:

```python
# For each prompt, generate multiple completions
group_rewards = [reward_1, reward_2, reward_3, reward_4]  # 4 generations per prompt
group_mean = mean(group_rewards)
advantages = [r - group_mean for r in group_rewards]  # Relative advantages
```

This approach:
- Compares multiple solutions to the same problem
- Learns to prefer better solutions within each group
- Reduces memory and computational costs compared to PPO

## Multi-Turn GRPO Implementation in Verifiers

### Key Finding: Groups are Per-Episode, Not Per-Turn

The verifiers framework implements multi-turn GRPO where **groups operate at the episode level** - each group contains multiple complete multi-turn conversations for the same prompt.

### Architecture Components

1. **RepeatSampler**: Ensures each prompt gets `num_generations` complete episodes
2. **MultiTurnEnv.rollout()**: Generates complete conversations until task completion or max_turns
3. **AsyncBatchGenerator**: Manages asynchronous generation of full episodes
4. **Advantage Computation**: Groups rewards by complete episodes for comparison

### Group Formation Process

For a multi-turn environment with `num_generations=4`:

```
Single Prompt → 4 Complete Episodes
├── Episode 1: Prompt → Assistant → Env → Assistant → ... → Final Answer
├── Episode 2: Prompt → Assistant → Env → Assistant → ... → Final Answer  
├── Episode 3: Prompt → Assistant → Env → Assistant → ... → Final Answer
└── Episode 4: Prompt → Assistant → Env → Assistant → ... → Final Answer

Reward Assignment: Each complete episode gets one reward
Group Comparison: GRPO compares these 4 episode rewards within the group
Advantage Calculation: advantages = episode_rewards - mean(group_rewards)
```

### Benefits for Multi-Turn RL

- **Holistic Optimization**: Learns complete conversational policies rather than isolated turn decisions
- **Natural Grouping**: Compares different conversation strategies for the same problem
- **Memory Efficiency**: No separate value function needed for multi-turn environments
- **Scalable**: Async generation enables efficient training with long episodes

## Challenges in Multi-Agent Environments

### Current GRPO Limitations for Multi-Agent Settings

#### 1. Stationarity Assumption
Standard GRPO assumes:
- **Deterministic environment responses**: Environment state transitions are deterministic given agent actions
- **Consistent reward functions**: Same action sequence should yield similar rewards across group members
- **Static opponents**: No consideration of other learning agents

#### 2. Group Comparison Validity
GRPO's core mechanism relies on comparing episodes within a group for the same prompt. In Diplomacy:

```python
# Current GRPO assumption:
prompt = "Negotiate with Austria about the Balkans"
episode_1_reward = 0.8  # Success against static opponents  
episode_2_reward = 0.3  # Different outcome, same actions?
```

This breaks down when:
- **Different opponents**: Each episode faces different opponent strategies
- **Non-stationary environment**: Opponent policies change during training
- **Action interdependence**: Reward depends on all players' actions, not just the learning agent

### Specific Challenges for Diplomacy-like Environments

#### 1. Non-Stationary Opponents
```python
# Diplomacy scenario:
turn_1: Agent vs [Austria_v1, Russia_v1, Turkey_v1] → reward = 0.7
turn_2: Agent vs [Austria_v2, Russia_v2, Turkey_v2] → reward = 0.2
# Same strategy, different opponents → Different rewards
```

**Problem**: GRPO advantage calculation assumes reward differences reflect policy quality, but they might reflect opponent strength changes.

#### 2. Environment State Coupling
```python
# Current GRPO: Independent episodes
episode_1: Spring_1901 → Agent_action_A → reward_1
episode_2: Spring_1901 → Agent_action_B → reward_2

# Diplomacy reality: Coupled episodes  
episode_1: Spring_1901 → All_players_act → Shared_game_state → Agent_reward_1
episode_2: Same state impossible - other players learned from episode_1
```

#### 3. Reward Attribution Problem
In Diplomacy:
- **Joint rewards**: Success depends on alliances and opponent mistakes
- **Delayed consequences**: Actions have long-term strategic implications
- **Credit assignment**: Hard to attribute game outcome to specific conversational moves

## Country-Based Grouping Solution

### The Core Insight

Instead of grouping by prompt (current GRPO), group by **country assignment** in Diplomacy:

```python
# Current GRPO grouping:
group = [
    (prompt_A, country_Austria), 
    (prompt_A, country_France),
    (prompt_A, country_Germany), 
    (prompt_A, country_Russia)
]  # Different countries, same negotiation scenario

# Country-based grouping:
austria_group = [
    (prompt_A, country_Austria),
    (prompt_B, country_Austria), 
    (prompt_C, country_Austria),
    (prompt_D, country_Austria)
]  # Same country, different scenarios
```

### Why This Works So Well

#### 1. Natural Strategic Context
Each country has distinct:
- **Starting positions**: Austria vs Russia have fundamentally different strategic situations
- **Neighbor relationships**: France borders different countries than Turkey
- **Win conditions**: Each country's path to victory is unique

**Result**: Comparing Austria episodes against each other is strategically meaningful!

#### 2. Balanced Opponent Exposure
```python
# In a single batch, Austria faces:
austria_episode_1: vs [France_1, Germany_1, Russia_1, ...]
austria_episode_2: vs [France_2, Germany_2, Russia_2, ...] 
austria_episode_3: vs [France_3, Germany_3, Russia_3, ...]
austria_episode_4: vs [France_4, Germany_4, Russia_4, ...]

# Austria episodes face different opponent combinations
# But all are solving the "Austria strategic problem"
```

#### 3. Controlled Comparison Context
- **Same strategic constraints**: All Austria episodes deal with Austria's geographic/diplomatic position
- **Varied execution**: Different approaches to handling Austria's challenges
- **Meaningful ranking**: "Which Austria strategy worked better?" is a valid question

#### 4. Naturally Handles Non-Stationarity
```python
# GRPO advantage calculation becomes:
austria_rewards = [0.7, 0.4, 0.8, 0.3]  # Austria performance across scenarios
austria_mean = 0.55
advantages = [0.15, -0.15, 0.25, -0.25]  # Which Austria strategies were above/below average?
```

### Advantages and Challenges

#### Advantages
- ✅ **Strategic Coherence**: Each country has unique strategic context making within-country comparisons meaningful
- ✅ **Natural Baseline**: "Average Austria performance" is a valid reference point
- ✅ **Balanced Exposure**: Each country faces diverse opponents but maintains consistent strategic constraints
- ✅ **Preserves GRPO Theory**: Group comparisons remain valid within strategic contexts
- ✅ **Handles Non-Stationarity**: Different opponents across episodes, but consistent country role

#### Challenges and Solutions

##### 1. Unequal Country Difficulty
**Problem**: Some countries (Austria) might be inherently harder than others (Russia)

**Solution**: Country-specific normalization
```python
def _compute_advantages_by_country(self, rewards, countries):
    advantages = []
    for country in ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']:
        country_mask = (countries == country)
        country_rewards = rewards[country_mask]
        
        # Country-specific normalization
        country_mean = country_rewards.mean()
        country_std = country_rewards.std()
        country_advantages = (country_rewards - country_mean) / (country_std + 1e-8)
        advantages.append(country_advantages)
    
    return torch.cat(advantages)
```

##### 2. Batch Size Requirements
**Challenge**: Need enough examples per country per batch

**Solution**: Modified sampling strategy
```python
class CountryBalancedSampler(Sampler):
    def __init__(self, dataset, num_generations_per_country=4):
        self.countries = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
        self.num_generations_per_country = num_generations_per_country
        
    def __iter__(self):
        for country in self.countries:
            country_prompts = self.get_prompts_for_country(country)
            # Sample prompts and repeat for multiple generations
            for prompt in random.sample(country_prompts, self.num_generations_per_country):
                yield prompt
```

## Implementation Design

### 1. Modified Environment

```python
class DiplomacyMultiAgentEnv(MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.countries = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
    
    def generate(self, inputs, **kwargs):
        # Ensure inputs contain country assignments
        if 'country' not in inputs:
            raise ValueError("Country assignment required for each prompt")
            
        results = super().generate(inputs, **kwargs)
        
        # Add country info to results for grouping
        results['country'] = inputs['country']
        return results
```

### 2. Modified GRPO Trainer

```python
class CountryGroupedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.countries = ['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
        self.num_generations_per_country = kwargs.get('num_generations_per_country', 4)
        
    def _get_train_sampler(self, train_dataset=None):
        return CountryBalancedSampler(
            self.train_dataset,
            num_generations_per_country=self.num_generations_per_country,
            num_countries=len(self.countries)
        )
    
    def _compute_advantages(self, rewards, countries):
        """Compute advantages grouped by country"""
        advantages = torch.zeros_like(rewards)
        
        for country in self.countries:
            country_mask = (countries == country)
            if country_mask.sum() > 0:
                country_rewards = rewards[country_mask]
                country_mean = country_rewards.mean()
                country_advantages = country_rewards - country_mean
                
                if self.scale_rewards:
                    country_std = country_rewards.std()
                    country_advantages = country_advantages / (country_std + 1e-4)
                
                advantages[country_mask] = country_advantages
        
        return advantages
```

### 3. Dataset Structure

```python
# Training data format
diplomacy_dataset = [
    {
        'prompt': "You are Austria in Spring 1901. The game has just begun...",
        'answer': "Win condition: Control 18 supply centers",
        'country': 'Austria',
        'game_state': {...},
        'task': 'diplomacy'
    },
    {
        'prompt': "You are Austria in Fall 1902. Russia is threatening Galicia...", 
        'answer': "Win condition: Control 18 supply centers",
        'country': 'Austria',
        'game_state': {...},
        'task': 'diplomacy'
    },
    # ... more Austria scenarios
    {
        'prompt': "You are England in Spring 1901. You control London, Edinburgh, Liverpool...",
        'answer': "Win condition: Control 18 supply centers", 
        'country': 'England',
        'game_state': {...},
        'task': 'diplomacy'
    }
    # ... scenarios for all 7 countries
]
```

### 4. Training Configuration

```python
# Example training setup
model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Diplomacy environment with country-based grouping
diplomacy_env = DiplomacyMultiAgentEnv(
    dataset=diplomacy_dataset,
    system_prompt=DIPLOMACY_PROMPT_TEMPLATE,
    max_turns=20,
    countries=['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
)

# Training arguments optimized for country grouping
training_args = vf.grpo_defaults(run_name="diplomacy-country-grpo")
training_args.num_generations_per_country = 4  # 4 episodes per country per batch
training_args.per_device_train_batch_size = 7   # One batch covers all 7 countries
training_args.gradient_accumulation_steps = 4

# Country-grouped GRPO trainer
trainer = CountryGroupedGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=diplomacy_env,
    args=training_args,
)
trainer.train()
```

## Extensions and Variations

### 1. Hierarchical Grouping

```python
# Group by country + game phase
groups = {
    ('Austria', 'opening'): [...],   # Spring 1901-1903
    ('Austria', 'midgame'): [...],   # 1904-1908 
    ('Austria', 'endgame'): [...],   # 1909+
    ('England', 'opening'): [...],
    # ...
}

class HierarchicalGRPOTrainer(CountryGroupedGRPOTrainer):
    def _compute_advantages(self, rewards, countries, game_phases):
        advantages = torch.zeros_like(rewards)
        
        for country in self.countries:
            for phase in ['opening', 'midgame', 'endgame']:
                mask = (countries == country) & (game_phases == phase)
                if mask.sum() > 0:
                    phase_rewards = rewards[mask]
                    phase_mean = phase_rewards.mean()
                    advantages[mask] = phase_rewards - phase_mean
        
        return advantages
```

### 2. Alliance-Aware Grouping

```python
# Group by country + strategic situation
groups = {
    ('Austria', 'western_triple'): [...],  # Austria in Western Triple alliance
    ('Austria', 'eastern_alliance'): [...], # Austria allied with Russia/Turkey
    ('Austria', 'isolated'): [...],        # Austria with no allies
}

def extract_alliance_context(game_state, country):
    """Extract alliance information from game state"""
    alliances = game_state.get('alliances', {})
    if country in alliances.get('western_triple', []):
        return 'western_triple'
    elif country in alliances.get('eastern_alliance', []):
        return 'eastern_alliance'
    else:
        return 'isolated'
```

### 3. Dynamic Difficulty Adjustment

```python
class AdaptiveDifficultyGRPO(CountryGroupedGRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.country_performance_history = defaultdict(list)
        self.opponent_strength = {country: 0.5 for country in self.countries}
    
    def adjust_country_difficulty(self, country_performance):
        """Adjust opponent strength based on country performance"""
        for country in self.countries:
            recent_performance = country_performance[country][-10:]  # Last 10 games
            avg_performance = sum(recent_performance) / len(recent_performance)
            
            if avg_performance > 0.8:  # Too easy
                self.opponent_strength[country] = min(1.0, self.opponent_strength[country] + 0.1)
            elif avg_performance < 0.3:  # Too hard
                self.opponent_strength[country] = max(0.1, self.opponent_strength[country] - 0.1)
    
    def sample_opponents_for_country(self, country):
        """Sample opponents with appropriate difficulty for country"""
        strength = self.opponent_strength[country]
        return self.opponent_pool.sample_by_strength(strength)
```

### 4. Cross-Country Transfer Learning

```python
class TransferGRPO(CountryGroupedGRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_weight = kwargs.get('transfer_weight', 0.1)
        
    def compute_loss(self, model, inputs, **kwargs):
        # Standard country-grouped loss
        country_loss = super().compute_loss(model, inputs, **kwargs)
        
        # Add cross-country similarity loss
        transfer_loss = self.compute_transfer_loss(model, inputs)
        
        return country_loss + self.transfer_weight * transfer_loss
    
    def compute_transfer_loss(self, model, inputs):
        """Encourage similar diplomatic strategies across countries"""
        # Extract diplomatic embeddings for each country
        country_embeddings = {}
        for country in self.countries:
            country_mask = (inputs['country'] == country)
            if country_mask.sum() > 0:
                country_hidden = self.get_diplomatic_embeddings(
                    model, inputs, country_mask
                )
                country_embeddings[country] = country_hidden
        
        # Compute similarity loss between diplomatically similar countries
        similar_pairs = [
            ('Austria', 'Italy'),    # Both vulnerable central powers
            ('England', 'Russia'),   # Both corner powers
            ('France', 'Germany'),   # Both strong western powers
        ]
        
        transfer_loss = 0
        for country_1, country_2 in similar_pairs:
            if country_1 in country_embeddings and country_2 in country_embeddings:
                emb_1 = country_embeddings[country_1].mean(dim=0)
                emb_2 = country_embeddings[country_2].mean(dim=0)
                transfer_loss += F.mse_loss(emb_1, emb_2)
        
        return transfer_loss
```

### 5. Multi-Agent Population Training

```python
class PopulationCountryGRPO(CountryGroupedGRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_size = kwargs.get('population_size', 8)
        self.population = self.initialize_population()
        
    def initialize_population(self):
        """Initialize population of diverse country specialists"""
        population = {}
        for country in self.countries:
            # Create multiple specialists per country
            population[country] = [
                self.create_country_specialist(country) 
                for _ in range(self.population_size // len(self.countries))
            ]
        return population
    
    def train_step(self):
        """Train each agent against population opponents"""
        for country in self.countries:
            for agent in self.population[country]:
                # Sample opponents from other countries
                opponents = self.sample_population_opponents(exclude_country=country)
                
                # Run GRPO training step
                agent.train_against_opponents(opponents)
                
                # Periodically update population with best performers
                if self.should_update_population():
                    self.update_population_member(country, agent)
```

## Performance Monitoring and Evaluation

### 1. Country-Specific Metrics

```python
class CountryMetricsLogger:
    def __init__(self, countries):
        self.countries = countries
        self.country_metrics = {country: defaultdict(list) for country in countries}
    
    def log_episode_results(self, results):
        """Log results grouped by country"""
        for country in self.countries:
            country_episodes = [r for r in results if r['country'] == country]
            if country_episodes:
                avg_reward = np.mean([ep['reward'] for ep in country_episodes])
                avg_length = np.mean([ep['episode_length'] for ep in country_episodes])
                win_rate = np.mean([ep['won_game'] for ep in country_episodes])
                
                self.country_metrics[country]['reward'].append(avg_reward)
                self.country_metrics[country]['length'].append(avg_length)
                self.country_metrics[country]['win_rate'].append(win_rate)
    
    def get_country_summary(self):
        """Get performance summary by country"""
        summary = {}
        for country in self.countries:
            metrics = self.country_metrics[country]
            summary[country] = {
                'avg_reward': np.mean(metrics['reward'][-100:]),  # Last 100 episodes
                'reward_std': np.std(metrics['reward'][-100:]),
                'win_rate': np.mean(metrics['win_rate'][-100:]),
                'avg_episode_length': np.mean(metrics['length'][-100:])
            }
        return summary
```

### 2. Strategic Evaluation Framework

```python
class DiplomacyEvaluator:
    def __init__(self, test_scenarios):
        self.test_scenarios = test_scenarios
        
    def evaluate_country_performance(self, model, country):
        """Evaluate model performance for specific country"""
        country_scenarios = [s for s in self.test_scenarios if s['country'] == country]
        
        results = {
            'opening_performance': self.evaluate_opening_play(model, country_scenarios),
            'alliance_formation': self.evaluate_alliance_skills(model, country_scenarios),
            'crisis_management': self.evaluate_crisis_response(model, country_scenarios),
            'endgame_execution': self.evaluate_endgame_play(model, country_scenarios)
        }
        
        return results
    
    def comparative_evaluation(self, models):
        """Compare multiple models across all countries"""
        comparison = {}
        for country in self.countries:
            comparison[country] = {}
            for model_name, model in models.items():
                comparison[country][model_name] = self.evaluate_country_performance(model, country)
        
        return comparison
```

## Conclusion

Country-based grouping provides an elegant solution to the fundamental challenges of applying GRPO in multi-agent environments. By leveraging the natural strategic structure of Diplomacy, this approach:

### Key Benefits

1. **Preserves GRPO Theory**: Group comparisons remain valid within strategic contexts
2. **Handles Non-Stationarity**: Different opponents across episodes, but consistent country role  
3. **Natural Curriculum**: Can progressively increase opponent difficulty per country
4. **Interpretable**: "Which Austria strategy worked better?" is meaningful
5. **Extensible**: Supports hierarchical grouping, transfer learning, and population training

### Implementation Feasibility

- ✅ **Theoretically Sound**: Maintains GRPO's mathematical foundations
- ✅ **Practically Implementable**: Requires minimal changes to existing verifiers framework
- ✅ **Computationally Efficient**: Preserves GRPO's memory advantages over PPO
- ✅ **Scalable**: Can handle full 7-player Diplomacy games

### Future Research Directions

1. **Empirical Validation**: Test country-grouped GRPO against baseline approaches
2. **Cross-Game Generalization**: Apply to other multi-agent strategy games
3. **Dynamic Grouping**: Develop adaptive grouping strategies based on game state
4. **Meta-Learning**: Learn optimal grouping strategies across different environments

This approach demonstrates how domain knowledge can elegantly solve complex RL challenges that would be difficult to address with purely algorithmic modifications. The country grouping insight transforms a fundamental obstacle (multi-agent non-stationarity) into a natural organizational principle (country-specific strategies), making GRPO viable for sophisticated multi-agent environments like Diplomacy.

### Impact

Country-based GRPO grouping opens new possibilities for training sophisticated diplomatic AI agents that can:
- Develop country-specific strategic expertise
- Handle complex multi-agent negotiations
- Learn from diverse opponent encounters while maintaining strategic coherence
- Scale to full-complexity strategic environments

This represents a significant advancement in applying modern RL techniques to classical strategy games and multi-agent coordination problems.