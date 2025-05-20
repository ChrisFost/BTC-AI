# Model Orchestration Layer Design

## Overview
This document outlines the design for a Model Orchestration Layer (MOL) that will enable different trading models ("buckets") to cooperate rather than compete for capital. The system allows models trained independently on different timeframes (Scalping, Short, Medium, Long) to work together as a unified trading system.

## Problem Statement
Currently, each trading model (Scalping, Short, Medium, Long) is trained independently to optimize its own performance without awareness of other models. When deployed together, they would compete for the same capital, potentially:

1. Creating conflicting positions
2. Over-allocating capital during overlapping signals
3. Missing opportunities for synergistic strategies
4. Failing to properly diversify exposure across timeframes

## Orchestration Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Capital Allocation Manager                  │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐    │
│  │ Performance │   │Market Regime│   │Risk Allocation  │    │
│  │  Tracker    │   │  Detector   │   │   Optimizer     │    │
│  └─────────────┘   └─────────────┘   └─────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Position Coordination Layer                 │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐    │
│  │   Signal    │   │   Trade     │   │    Conflict     │    │
│  │  Alignment  │   │  Scheduler  │   │   Resolution    │    │
│  └─────────────┘   └─────────────┘   └─────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Unified Risk Management                     │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐    │
│  │   Exposure  │   │  Drawdown   │   │   Correlation   │    │
│  │   Monitor   │   │  Defender   │   │    Analyzer     │    │
│  └─────────────┘   └─────────────┘   └─────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Strategy Models                           │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────┐  ┌──────┐ │
│  │  Scalping   │   │    Short    │   │  Medium  │  │ Long │ │
│  │   Model     │   │    Model    │   │   Model  │  │ Model│ │
│  └─────────────┘   └─────────────┘   └──────────┘  └──────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 1. Capital Allocation Manager

### Purpose
Dynamically allocate the total capital across different timeframe models based on market conditions, model performance, and risk parameters.

### Components

#### Performance Tracker
- Maintains a rolling history of each model's performance metrics:
  - Sharpe ratio, win rate, profit factor, max drawdown
  - Recent P&L trends and volatility
  - Comparison between predicted and actual returns
- Applies exponential weighting to favor recent performance
- Computes relative strength of each model's current performance

#### Market Regime Detector
- Identifies current market environment:
  - Trending vs. ranging markets
  - Volatility regime (high/medium/low)
  - Liquidity conditions
- Maps market conditions to historical model performance
- Determines which models are likely to outperform in current conditions

#### Risk Allocation Optimizer
- Calculates optimal capital allocation percentages
- Ensures diversification benefits are maximized
- Applies constraints:
  - Minimum allocation per model (e.g., 10%)
  - Maximum allocation per model (e.g., 50%)
  - Total allocation = 100% of available capital

### Implementation Approach
```python
class CapitalAllocationManager:
    def __init__(self, models, initial_allocation=None, min_allocation=0.1, max_allocation=0.5):
        self.models = models  # Dictionary of {name: model} pairs
        self.performance_tracker = PerformanceTracker(models)
        self.regime_detector = MarketRegimeDetector()
        self.risk_optimizer = RiskAllocationOptimizer(min_allocation, max_allocation)
        
        # Initialize with equal allocation if not specified
        if initial_allocation is None:
            model_count = len(models)
            self.allocations = {name: 1.0/model_count for name in models}
        else:
            self.allocations = initial_allocation
            
    def update_allocations(self, market_data, performance_metrics):
        # Update performance history
        self.performance_tracker.update(performance_metrics)
        
        # Detect current market regime
        current_regime = self.regime_detector.analyze(market_data)
        
        # Calculate performance scores for each model in current regime
        performance_scores = self.performance_tracker.get_regime_scores(current_regime)
        
        # Optimize allocation based on performance and risk
        new_allocations = self.risk_optimizer.optimize(
            performance_scores, 
            current_regime
        )
        
        # Apply smoothing to prevent drastic changes
        self.allocations = self._smooth_allocation_change(
            self.allocations, 
            new_allocations
        )
        
        return self.allocations
        
    def _smooth_allocation_change(self, old_allocations, new_allocations, max_change=0.1):
        """Limit the rate of change in allocations to prevent instability"""
        smoothed = {}
        for model_name in old_allocations:
            old_value = old_allocations[model_name]
            new_value = new_allocations[model_name]
            change = max(min(new_value - old_value, max_change), -max_change)
            smoothed[model_name] = old_value + change
            
        # Renormalize to ensure we still sum to 1.0
        total = sum(smoothed.values())
        return {k: v/total for k, v in smoothed.items()}
```

## 2. Position Coordination Layer

### Purpose
Coordinate position sizing, entry/exit timing, and trade direction across models to maximize synergy and prevent conflicts.

### Components

#### Signal Alignment
- Detects when multiple models generate similar signals
- Amplifies position sizes when signals align across timeframes
- Provides feedback to models about alignment with other timeframes
- Calculates a "harmony score" for each potential trade

#### Trade Scheduler
- Manages execution timing to prevent simultaneous large trades
- Prioritizes trades based on:
  - Timeframe (shorter timeframes get execution priority)
  - Signal strength
  - Expected holding period
- Staggers entry/exit to minimize market impact

#### Conflict Resolution
- Identifies contradictory positions between models
- Implements resolution strategies:
  - Partial allocation to both positions
  - Prioritization based on model confidence
  - Dynamic hedging between opposing positions
- Manages net exposure limits when models disagree

### Implementation Approach
```python
class PositionCoordinationLayer:
    def __init__(self, models, capital_allocator):
        self.models = models
        self.capital_allocator = capital_allocator
        self.signal_aligner = SignalAligner()
        self.trade_scheduler = TradeScheduler()
        self.conflict_resolver = ConflictResolver()
        
    def process_signals(self, raw_signals, market_data):
        """
        Take raw signals from models and coordinate them into final positions
        
        Args:
            raw_signals: Dict of {model_name: signal_details} where signal_details
                         contains action, size, confidence, and timeframe
            market_data: Current market state and conditions
            
        Returns:
            Dict of coordinated trade actions for execution
        """
        # Get current capital allocations
        allocations = self.capital_allocator.allocations
        
        # Find alignment between signals
        alignment_scores = self.signal_aligner.calculate_alignment(raw_signals)
        
        # Adjust position sizes based on alignment
        aligned_signals = self.signal_aligner.adjust_positions(
            raw_signals, 
            alignment_scores, 
            allocations
        )
        
        # Resolve conflicts between signals
        resolved_signals = self.conflict_resolver.resolve(aligned_signals)
        
        # Schedule trade execution
        final_trade_plan = self.trade_scheduler.schedule(
            resolved_signals,
            market_data
        )
        
        return final_trade_plan
```

## 3. Unified Risk Management

### Purpose
Provide central oversight of risk exposure across all models and enforce global risk constraints.

### Components

#### Exposure Monitor
- Tracks total exposure across all models
- Maintains exposure limits by:
  - Asset
  - Direction (long/short)
  - Timeframe
- Enforces position size scaling during high exposure periods

#### Drawdown Defender
- Implements global drawdown protection
- Reduces allowable position sizes as drawdown increases
- Provides early warning system for correlated losses
- Implements emergency deleveraging during extreme market events

#### Correlation Analyzer
- Measures correlation between model returns
- Identifies periods of increased correlation risk
- Adjusts capital allocation to minimize correlation
- Provides correlation-based feedback to position sizing

### Implementation Approach
```python
class UnifiedRiskManager:
    def __init__(self, max_exposure=1.0, max_drawdown_threshold=0.1):
        self.max_exposure = max_exposure  # 1.0 = 100% of capital
        self.max_drawdown_threshold = max_drawdown_threshold
        self.exposure_monitor = ExposureMonitor(max_exposure)
        self.drawdown_defender = DrawdownDefender(max_drawdown_threshold)
        self.correlation_analyzer = CorrelationAnalyzer()
        
        self.current_drawdown = 0.0
        self.peak_capital = 0.0
        self.current_capital = 0.0
        
    def update_capital_values(self, current_capital):
        """Update peak capital and drawdown metrics"""
        self.current_capital = current_capital
        self.peak_capital = max(self.peak_capital, current_capital)
        self.current_drawdown = 1.0 - (current_capital / self.peak_capital)
        
    def check_trade_risk(self, trade_plan, current_positions, market_data):
        """
        Validate and adjust trade plan based on risk constraints
        
        Returns:
            Adjusted trade plan and a boolean indicating if any
            risk limits were hit
        """
        # Check total exposure after proposed trades
        exposure_status = self.exposure_monitor.check_exposure(
            trade_plan, 
            current_positions
        )
        
        # Check drawdown limits
        drawdown_status = self.drawdown_defender.check_drawdown(
            self.current_drawdown,
            trade_plan,
            current_positions
        )
        
        # Analyze correlation risk
        correlation_status = self.correlation_analyzer.analyze_correlation(
            trade_plan,
            current_positions,
            market_data
        )
        
        # Adjust trade plan based on risk checks
        adjusted_plan = self._adjust_for_risk(
            trade_plan,
            exposure_status,
            drawdown_status,
            correlation_status
        )
        
        # Determine if any risk limits were breached
        risk_limits_hit = exposure_status['limits_hit'] or \
                          drawdown_status['limits_hit'] or \
                          correlation_status['limits_hit']
                          
        return adjusted_plan, risk_limits_hit
        
    def _adjust_for_risk(self, trade_plan, exposure, drawdown, correlation):
        """Apply risk adjustments to trade plan"""
        adjusted_plan = copy.deepcopy(trade_plan)
        
        # Apply scaling factors from each risk component
        for model_name, trades in adjusted_plan.items():
            for trade_id, trade in trades.items():
                # Get the minimum scaling factor from all risk components
                scale_factor = min(
                    exposure['scaling_factors'].get(model_name, 1.0),
                    drawdown['scaling_factor'],
                    correlation['scaling_factors'].get(model_name, 1.0)
                )
                
                # Apply scaling to position size
                trade['size'] *= scale_factor
                
        return adjusted_plan
```

## 4. Integration with Existing Models

### Non-Invasive Approach
The orchestration layer wraps around existing models without modifying their internal logic. This allows models to continue functioning as they were trained while the orchestration layer mediates their interactions.

### Interface Requirements
Each model must expose:
1. Action generation function that outputs:
   - Direction (long/short/neutral)
   - Position size recommendation
   - Confidence score (0-1)
   - Expected holding period
2. State update function to receive:
   - Actual position taken (after orchestration)
   - Market feedback
   - Performance metrics

### Adaptation Layer for Existing Models
For existing models that don't natively expose the required interface, we'll create thin adapter classes:

```python
class ModelAdapter:
    def __init__(self, original_model, timeframe):
        self.model = original_model
        self.timeframe = timeframe  # 'scalping', 'short', 'medium', 'long'
        
    def generate_action(self, state):
        """Adapt original model's action to the required format"""
        raw_action = self.model.select_action(state)
        
        # Extract or infer confidence from model output
        if hasattr(raw_action, 'confidence'):
            confidence = raw_action.confidence
        else:
            # Infer confidence from model's internals if available
            confidence = self._infer_confidence(raw_action, state)
            
        # Map timeframe to expected holding period
        holding_periods = {
            'scalping': {'min': 1, 'max': 60, 'typical': 12},    # In 5-min bars
            'short': {'min': 12, 'max': 288, 'typical': 48},     # 1 hour to 1 day
            'medium': {'min': 288, 'max': 1440, 'typical': 720}, # 1 day to 5 days
            'long': {'min': 1440, 'max': 8640, 'typical': 4320}  # 5 days to 30 days
        }
        
        return {
            'action': raw_action.action_type,  # buy, sell, hold
            'size': raw_action.size,
            'confidence': confidence,
            'expected_holding': holding_periods[self.timeframe]['typical'],
            'min_holding': holding_periods[self.timeframe]['min'],
            'max_holding': holding_periods[self.timeframe]['max']
        }
        
    def update_state(self, state, action_taken, feedback):
        """Update model with the action that was actually taken"""
        if hasattr(self.model, 'update_with_external_action'):
            self.model.update_with_external_action(state, action_taken, feedback)
        else:
            # For models that don't support external updates,
            # we can optionally implement a state tracking mechanism
            pass
            
    def _infer_confidence(self, action, state):
        """Attempt to infer confidence from model output"""
        # This is model-specific and would need custom logic
        # For PPO models, could use the ratio between chosen action prob and mean
        # For value-based models, could use normalized Q-value differences
        return 0.7  # Default confidence if can't be determined
```

## 5. Reward Function Extensions

### Hybrid Reward System
While we won't retrain models initially, we can extend the reward system to encourage cooperation:

1. **Base Reward**: Original model's reward function (profit/loss)
2. **Alignment Bonus**: Small additional reward when action aligns with other timeframes
3. **Diversity Bonus**: Reward for providing uncorrelated returns
4. **Stability Reward**: Reward for consistent performance across market regimes

### Phased Implementation
1. **Phase 1**: Pure orchestration without reward modification
2. **Phase 2**: Add alignment/diversity metrics to state space
3. **Phase 3**: Introduce small cooperative reward components
4. **Phase 4**: Optional fine-tuning with cooperative rewards

## 6. Monitoring and Analytics

### Performance Dashboard
- Overall system performance
- Individual model contribution
- Allocation history
- Synergy metrics: how much better the combined system performs vs. sum of parts

### Diagnostic Visualizations
- Correlation heatmap between models
- Capital allocation over time
- Conflict resolution events
- Risk limit activations

### Signal Alignment Analysis
- Alignment frequency by timeframe
- Impact of alignment on performance
- Trade clustering visualization

## 7. Implementation Roadmap

### Phase 1: Basic Integration (1-2 weeks)
- Create adapter layer for existing models
- Implement capital allocation manager with basic rules
- Set up unified risk management with simple limits
- Develop monitoring dashboard

### Phase 2: Coordination Enhancement (2-4 weeks)
- Add signal alignment mechanisms
- Implement conflict resolution strategies
- Enhance capital allocation with performance tracking
- Deploy trade scheduling system

### Phase 3: Advanced Features (4-8 weeks)
- Integrate market regime detection
- Implement correlation-based position sizing
- Add unified drawdown protection
- Develop synergy metrics and analytics

### Phase 4: Production Optimization (2-4 weeks)
- Performance optimization
- Stress testing
- Robustness improvements
- Documentation and operating procedures

## 8. Considerations and Challenges

### Technical Challenges
- Ensuring low latency for real-time decision making
- Managing state complexity across multiple models
- Preventing feedback loops between models and orchestration layer
- Balancing cooperation vs. model specialization

### Risk Management Challenges
- Avoiding over-fitting in the orchestration layer
- Managing transition periods between different allocations
- Ensuring system stability during extreme market events
- Preventing "crowding out" of longer-term strategies

### Opportunities
- Exploiting cross-timeframe signals for better entries/exits
- Capturing alpha across multiple time horizons
- Adapting to changing market regimes automatically
- Creating a more robust combined system with lower drawdowns

## Conclusion

The Model Orchestration Layer provides a framework for transforming independently trained models into a cohesive trading system without requiring retraining. By implementing allocation, coordination, and risk management layers, the system enables models to cooperate rather than compete, potentially extracting greater value from the existing models while maintaining their specialization advantages.

This approach allows for incremental improvement and testing while preserving the value of the original trained models. As the system matures, optional fine-tuning with cooperative rewards could further enhance performance. 