# Natural Learning Features

This document describes the natural learning features implemented in the BTC-AI trading system to make the AI's learning process more human-like and adaptive.

## Overview

The goal of these features is to allow the trading AI to learn more naturally, without explicit hardcoded goals. Instead of directly programming specific trading strategies, we've implemented features that allow the AI to discover optimal strategies through exploration, experience, and adaptation.

## Key Features

### 1. Adaptive Exploration Rates

The AI dynamically adjusts its exploration rate based on:
- Current market regime (volatile, trending, ranging)
- Recent performance
- Confidence in predictions

This allows for context-sensitive exploration that resembles human learning patterns:
- High exploration in unfamiliar situations
- Lower exploration in well-understood scenarios
- Periodic exploration to discover new strategies

Implementation location: `PrioritizedReplayBuffer` class in `Agent V2.py`

### 2. Experience Prioritization

Not all trading experiences are equally valuable. The AI now prioritizes:
- Surprising outcomes (trades that performed differently than expected)
- Important transactions (high-value or high-risk trades)
- Novel market conditions

This mimics how humans focus more on memorable or unusual events when learning.

Implementation location: `PrioritizedReplayBuffer` class in `Agent V2.py`

### 3. Post-Trade Analysis

After completing trades, the AI performs analysis to extract lessons:
- Compares actual outcomes to predicted outcomes
- Identifies factors that contributed to success or failure
- Stores these lessons for future reference

This creates a "hindsight" mechanism similar to how humans reflect on past actions.

Implementation location: `LessonMemory` class in `Agent V2.py`

### 4. Adaptive Time Horizons

Rather than using fixed prediction horizons, the AI now:
- Dynamically generates prediction horizons based on market conditions
- Emphasizes shorter horizons in volatile markets
- Emphasizes longer horizons in trending markets
- Adjusts based on which horizons have performed well recently

This allows each bucket to naturally adapt its time preferences based on what works.

Implementation locations:
- `generate_dynamic_horizons` in `utils v2.py`
- `adapt_prediction_horizons` in `utils v2.py`
- `update_horizons` method in `models v2.py`

### 5. Simple Meta-Learning

The AI can now adjust its own hyperparameters based on performance:
- Learning rates
- Risk tolerance
- Memory utilization
- Discount factors

This self-optimization resembles how humans adjust their own learning strategies.

Implementation location: `HyperparamOptimizer` class in `Agent V2.py`

### 6. Contextual Memory

The AI now maintains a memory of important market situations and their outcomes:
- Stores embeddings of key market states with their outcomes
- Recalls similar past scenarios when making decisions
- Prioritizes memorable and successful scenarios
- Adjusts confidence based on similar past experiences

This creates an episodic memory system similar to human recall of market patterns.

Implementation location: `ContextualMemory` class in `Agent V2.py`

### 7. Cross-Bucket Knowledge Transfer

Different bucket types (Scalping, Short, Medium, Long) can now share knowledge:
- Shorter-term buckets can inform longer-term strategies
- Longer-term buckets can provide context to shorter-term decisions
- Shares high-performing prediction horizons between buckets
- Transfers feature importance across time scales
- Selectively shares model weights for common features

This creates a more holistic learning system where insights at one time scale can benefit others.

Implementation location: `CrossBucketKnowledgeTransfer` class in `Agent V2.py`

### 8. Bucket Goal Abstraction

Each bucket type (Scalping, Short, Medium, Long) has specific performance goals, but these are now abstracted:
- Goals are managed by the `BucketGoalProvider` class
- The AI isn't explicitly told these goals
- Instead, it discovers optimal strategies through reward signals
- Rewards are calculated based on matching the bucket's inherent goals

This allows each bucket to maintain its distinct purpose while learning naturally.

Implementation location: `BucketGoalProvider` class in `bucket_goals.py`

## Configuration

These features can be configured through the "Learning" tab in the menu interface, with options to:
- Enable/disable individual features
- Adjust parameters for each feature
- View performance metrics related to learning efficiency

## Testing

Test scripts are provided to verify the correct implementation of these features:
- `Scripts/newest stuff/testing/test_natural_learning.py`: Tests dynamic horizons and bucket goals
- `Scripts/newest stuff/testing/test_cross_bucket_transfer.py`: Tests knowledge transfer between buckets

## Integration in Training Loop

The features are integrated into the training loop:
- Contextual memory is used during action selection and experience processing
- Cross-bucket knowledge transfer happens periodically during training
- Dynamic horizons are updated based on market conditions and performance
- Meta-learning adjusts hyperparameters based on recent rewards

## Future Enhancements

Potential future enhancements to the natural learning system:
- More sophisticated meta-learning with Bayesian optimization
- Integration with external market regime detection
- Automated feature engineering
- Reinforcement learning from human feedback
- Memory consolidation during off-training periods
- Concept drift detection and adaption 