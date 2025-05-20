# BTC-AI Version 5

A Bitcoin trading AI system using reinforcement learning.

## Directory Structure

```
BTC-AI/
├── src/                    # Source code
│   ├── agent/              # Agent implementation
│   ├── environment/        # Trading environment components
│   ├── models/             # Neural network models
│   ├── training/           # Training logic and backtesting
│   ├── ui/                 # User interface
│   └── utils/              # Utility functions
│
├── tests/                  # Tests organized by scope
│   ├── unit/               # Unit tests for individual components
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
│
├── tools/                  # Utility scripts
│   ├── scripts/            # General scripts
│   └── path_management/    # Path and import management utilities
│
├── configs/                # Configuration files
│
├── logs/                   # Log files
│
└── data/                   # Data directory for historical price data
    └── 2020-2024_BTCUSDT_DATA/ # Historical price data
```

## Setup and Installation

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the setup wizard:
   ```
   python src/ui/setup_wizard.py
   ```

## Running Tests

```
pytest tests/
```

## Training

```
python src/training/progressive_training.py
```

## License

Proprietary 