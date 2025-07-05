# Diplomacy GRPO Pipeline

A self-play pipeline for training Large Language Models on the game of Diplomacy using Group Relative Policy Optimization (GRPO) with country-based grouping.

## Overview

This project implements a novel approach to multi-agent reinforcement learning for strategic games, specifically addressing the challenges of training LLMs in non-stationary environments like Diplomacy.

### Key Features

- **Country-Based Grouping**: Groups episodes by country assignment rather than prompt for meaningful strategic comparisons
- **Multi-Level Rewards**: Combines year-level and game-level rewards to address reward sparsity
- **Modular Design**: Clean separation between core RL components and game-specific logic
- **Comprehensive Testing**: Full unit test coverage for critical components

## Quick Start

### Development Environment Setup

1. **Clone and navigate to the project:**
   ```bash
   cd diplomacy_self_play
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Run the setup script:**
   ```bash
   python setup_dev.py
   ```

This will install the package in development mode, install dependencies, and run tests to verify everything works.

### Manual Setup

If the setup script fails, install manually:

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]

# Install PyTorch (adjust URL for your system)
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For GPU
# OR
pip install torch --index-url https://download.pytorch.org/whl/cpu   # For CPU

# Run tests
pytest tests/unit/ -v
```

## Project Structure

```
diplomacy_grpo/
├── core/
│   ├── country_sampler.py      # Country-balanced sampling for GRPO
│   └── country_trainer.py      # GRPO trainer with country grouping
├── rewards/
│   └── multi_level.py          # Multi-level reward system
└── __init__.py

tests/
└── unit/
    ├── test_country_sampler.py
    └── test_multi_level_reward.py
```

## Core Components

### CountryBalancedSampler

Ensures each batch contains equal representation from all Diplomacy countries, enabling country-based advantage computation in GRPO.

```python
from diplomacy_grpo import CountryBalancedSampler

# Create sampler for balanced country representation
sampler = CountryBalancedSampler(
    dataset,
    num_generations_per_country=5,
    countries=['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey']
)
```

### CountryGroupedGRPOTrainer

GRPO trainer that computes advantages by grouping episodes by country assignment rather than prompt, addressing multi-agent non-stationarity.

```python
from diplomacy_grpo import CountryGroupedGRPOTrainer

trainer = CountryGroupedGRPOTrainer(
    model=model,
    env=env,
    args=training_args,
    processing_class=tokenizer,
    countries=['Austria', 'England', 'France', 'Germany', 'Italy', 'Russia', 'Turkey'],
    country_specific_normalization=True
)
```

### MultiLevelReward

Combines multiple reward signals to provide rich feedback and address reward sparsity:

```python
from diplomacy_grpo import MultiLevelReward

reward_system = MultiLevelReward()
rewards = reward_system.compute_reward(
    country='Austria',
    episode_data={
        'initial_supply_centers': 3,
        'final_supply_centers': 5,
        'orders': ['A VIE-BUD', 'A TRI-VEN'],
        'invalid_orders': [],
    },
    game_context={
        'game_result': 'victory',
        'winner': 'Austria'
    }
)
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=diplomacy_grpo

# Run specific test file
pytest tests/unit/test_country_sampler.py -v
```

### Code Quality

```bash
# Format code
black diplomacy_grpo/

# Lint code
ruff check diplomacy_grpo/

# Type check
mypy diplomacy_grpo/
```

## Current Status: Phase 1

This is Phase 1 of the implementation, focusing on core RL components that are independent of AI_Diplomacy integration:

✅ **Completed:**
- Country-balanced sampling strategy
- Country-grouped GRPO trainer foundation  
- Multi-level reward system
- Comprehensive unit tests
- Development environment setup

🚧 **Next Steps:**
- Integration with verifiers framework
- AI_Diplomacy environment wrapper
- Batch game generation system
- End-to-end training pipeline

## Dependencies

### Core Dependencies
- `torch>=2.6.0` - Neural network framework
- `transformers>=4.36.0` - Hugging Face transformers
- `datasets>=3.6.0` - Dataset handling
- `trl>=0.17.0` - Reinforcement learning for language models

### Development Dependencies
- `pytest>=7.4.0` - Testing framework
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Fast Python linter
- `mypy>=1.5.0` - Static type checking

### Optional Dependencies
- `vllm>=0.8.5` - Fast LLM inference (for production)
- `deepspeed>=0.12.0` - Distributed training (for large models)

## Architecture

The pipeline implements country-based grouping for GRPO as outlined in `GRPO_Multi_Agent_Analysis.md`:

1. **Country-Based Episodes**: Each episode is associated with a specific Diplomacy country
2. **Strategic Context Preservation**: Advantages computed within country groups maintain strategic meaning
3. **Non-Stationarity Handling**: Different opponents across episodes but consistent country role
4. **Single Model Training**: All updates applied to one model for general Diplomacy skills

## Contributing

1. Install development environment: `python setup_dev.py`
2. Make changes with tests: Add tests for new functionality
3. Run quality checks: `black`, `ruff`, `mypy`, `pytest`
4. Follow the existing patterns and keep it simple

## License

MIT License - see LICENSE file for details.