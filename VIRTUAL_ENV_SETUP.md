# Virtual Environment Setup

## Current Configuration

We've set up a **project-level virtual environment** located at:
```
diplomacy_self_play/.venv/
```

This single environment serves the entire project, including:
- Our custom `diplomacy_grpo` package
- AI_Diplomacy integration 
- Verifiers framework integration
- All dependencies and development tools

## ✅ Setup Complete

The environment is already configured and ready to use with:

### Core Dependencies
- **PyTorch 2.7.1** - Neural network framework
- **Transformers 4.53.1** - Hugging Face transformers
- **TRL 0.19.0** - Reinforcement learning for language models
- **Datasets 3.6.0** - Dataset handling
- **Accelerate 1.8.1** - Multi-GPU training
- **PEFT 0.16.0** - Parameter-efficient fine-tuning

### Development Tools
- **pytest 8.4.1** - Testing framework
- **black 25.1.0** - Code formatting
- **ruff 0.12.2** - Fast Python linter
- **mypy 1.16.1** - Static type checking
- **pytest-cov 6.2.1** - Test coverage

### Package Status
- **diplomacy-grpo-pipeline 0.1.0** - Installed in development mode ✅
- **All tests passing** - 27/27 unit tests ✅

## Usage

### Activate Environment
```bash
source .venv/bin/activate
```

### Verify Setup
```bash
# Check package installation
python -c "import diplomacy_grpo; print(f'Package version: {diplomacy_grpo.__version__}')"

# Run tests
pytest tests/unit/ -v

# Check available components
python -c "
import diplomacy_grpo
print('Available components:', diplomacy_grpo.__all__)
print('PyTorch available:', diplomacy_grpo._TORCH_AVAILABLE)
print('Verifiers available:', diplomacy_grpo._VERIFIERS_AVAILABLE)
"
```

### Development Workflow
```bash
# Format code
black diplomacy_grpo/

# Lint code  
ruff check diplomacy_grpo/

# Type check
mypy diplomacy_grpo/

# Run tests with coverage
pytest tests/ --cov=diplomacy_grpo
```

## Integration with Other Repositories

### AI_Diplomacy
The AI_Diplomacy repository has its own `.venv` directory that is **no longer needed**. To use our shared environment:

```bash
cd AI_Diplomacy
# Remove old environment (optional)
rm -rf .venv

# Use parent project environment
source ../.venv/bin/activate

# Install AI_Diplomacy in development mode
pip install -e .
```

### Verifiers
When we integrate with verifiers:

```bash
cd verifiers
source ../.venv/bin/activate

# Install verifiers in development mode  
pip install -e .
```

## Environment Management

### Adding Dependencies
Edit `pyproject.toml` and reinstall:
```bash
# Add dependency to pyproject.toml
pip install -e .
```

### Updating Dependencies
```bash
pip install --upgrade torch transformers trl
```

### Fresh Install
If you need to recreate the environment:
```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Benefits of This Setup

1. **Single Source of Truth**: One environment for all repositories
2. **Consistent Dependencies**: No version conflicts between repositories
3. **Easy Development**: Switch between repositories seamlessly
4. **Shared Tools**: Common linting, testing, and formatting tools
5. **Memory Efficient**: One PyTorch installation instead of multiple

## Files Structure
```
diplomacy_self_play/
├── .venv/                          # Shared virtual environment
├── diplomacy_grpo/                 # Our main package
├── AI_Diplomacy/                   # AI Diplomacy repository
├── verifiers/                      # Verifiers repository  
├── tests/                          # Test suite
├── pyproject.toml                  # Package configuration
├── .gitignore                      # Git ignore rules
└── README.md                       # Main documentation
```

The shared virtual environment approach provides a clean, efficient development setup for the entire project.