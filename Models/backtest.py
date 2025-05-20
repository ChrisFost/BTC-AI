"""
Backtesting framework for probabilistic trading models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import os
import json
from collections import defaultdict
import time

from trading_model import TradingModel

class Backtester:
    """Backtesting framework for evaluating model performance on historical data"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_sizing: str = 'confidence',  # Options: 'fixed', 'confidence', 'kelly'
        max_position_size: float = 0.2,       # Maximum position size as fraction of capital
        stop_loss_pct: Optional[float] = 0.05, # Stop loss percentage (None to disable)
        take_profit_pct: Optional[float] = 0.1, # Take profit percentage (None to disable)
        trading_fee_pct: float = 0.001,       # Trading fee percentage
        slippage_pct: float = 0.001,          # Slippage percentage
        log_dir: str = 'backtest_logs',
        save_results: bool = True,
        save_trades: bool = True,
        save_equity_curve: bool = True,
        risk_free_rate: float = 0.01          # Annual risk-free rate for Sharpe ratio
    ):
        self.initial_capital = initial_capital
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trading_fee_pct = trading_fee_pct
        self.slippage_pct = slippage_pct
        self.log_dir = log_dir
        self.save_results = save_results
        self.save_trades = save_trades
        self.save_equity_curve = save_equity_curve
        self.risk_free_rate = risk_free_rate
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_file = os.path.join(
            self.log_dir,
            f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        model: TradingModel,
        feature_columns: List[str] = None,
        price_column: str = 'close',
        timestamp_column: str = 'timestamp',
        save_file_prefix: str = None
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame containing historical data
            model: Trading model to use for predictions
            feature_columns: List of column names to use as features
            price_column: Name of column containing price data
            timestamp_column: Name of column containing timestamp data
            save_file_prefix: Prefix for saved files
            
        Returns:
            Dictionary containing backtest results
        """
        logging.info(f"Starting backtest with initial capital: ${self.initial_capital:.2f}")
        
        # Set default save file prefix if not provided
        if save_file_prefix is None:
            save_file_prefix = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize backtest state
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = None
        stop_loss = None
        take_profit = None
        
        # Initialize results containers
        equity_curve = []
        trades = []
        daily_returns = []
        
        # Feature columns
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col not in [price_column, timestamp_column]]
        
        # Sort data by timestamp
        data = data.sort_values(timestamp_column).reset_index(drop=True)
        
        # Get sequence length from model
        if hasattr(model, 'buffer_size'):
            seq_len = model.buffer_size
        else:
            seq_len = 100  # Default
        
        # Start backtesting
        logging.info(f"Backtesting on {len(data)} data points with {seq_len} sequence length")
        start_time = time.time()
        
        # Minimum index to have enough history
        start_idx = seq_len
        
        for i in range(start_idx, len(data)):
            # Current data point
            current_data = data.iloc[i]
            current_price = current_data[price_column]
            current_time = current_data[timestamp_column]
            
            # Check stop loss / take profit
            if position != 0:
                # Calculate current P&L
                unrealized_pnl = position * (current_price - entry_price)
                pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)
                
                # Check stop loss
                if self.stop_loss_pct is not None and pnl_pct <= -self.stop_loss_pct:
                    # Stop loss triggered
                    exit_price = current_price * (1 - self.slippage_pct * np.sign(position))
                    pnl = position * (exit_price - entry_price)
                    fee = abs(position * exit_price) * self.trading_fee_pct
                    capital += pnl - fee
                    
                    # Record trade
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'fee': fee,
                        'duration': (current_time - entry_time).total_seconds() / 3600,  # Hours
                        'exit_reason': 'stop_loss'
                    }
                    trades.append(trade)
                    
                    logging.info(f"Stop loss: Closing position of {position:.4f} at ${exit_price:.2f}, " +
                                 f"PnL: ${pnl:.2f}, Capital: ${capital:.2f}")
                    
                    position = 0.0
                    entry_price = 0.0
                    entry_time = None
                    stop_loss = None
                    take_profit = None
                
                # Check take profit
                elif self.take_profit_pct is not None and pnl_pct >= self.take_profit_pct:
                    # Take profit triggered
                    exit_price = current_price * (1 - self.slippage_pct * np.sign(position))
                    pnl = position * (exit_price - entry_price)
                    fee = abs(position * exit_price) * self.trading_fee_pct
                    capital += pnl - fee
                    
                    # Record trade
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'fee': fee,
                        'duration': (current_time - entry_time).total_seconds() / 3600,  # Hours
                        'exit_reason': 'take_profit'
                    }
                    trades.append(trade)
                    
                    logging.info(f"Take profit: Closing position of {position:.4f} at ${exit_price:.2f}, " +
                                 f"PnL: ${pnl:.2f}, Capital: ${capital:.2f}")
                    
                    position = 0.0
                    entry_price = 0.0
                    entry_time = None
                    stop_loss = None
                    take_profit = None
            
            # Prepare input features
            features_data = data.iloc[i-seq_len:i][feature_columns].values
            features = np.array(features_data, dtype=np.float32)
            
            # Convert to model input format
            observation = {
                'features': features,
                'current_price': current_price,
                'timestamp': current_time
            }
            
            # Get model prediction
            prediction, orders = model.predict(observation, capital, position)
            
            # Execute orders
            for order in orders:
                order_type = order.get('type')
                order_size = order.get('size', 0.0)
                
                if order_type == 'buy' and order_size > 0:
                    # Execute buy order
                    if position != 0:
                        # Close existing position first
                        exit_price = current_price * (1 + self.slippage_pct)
                        pnl = position * (exit_price - entry_price)
                        fee = abs(position * exit_price) * self.trading_fee_pct
                        capital += pnl - fee
                        
                        # Record trade
                        trade = {
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'pnl': pnl,
                            'fee': fee,
                            'duration': (current_time - entry_time).total_seconds() / 3600,  # Hours
                            'exit_reason': 'signal_change'
                        }
                        trades.append(trade)
                        
                        logging.info(f"Signal change: Closing position of {position:.4f} at ${exit_price:.2f}, " +
                                     f"PnL: ${pnl:.2f}, Capital: ${capital:.2f}")
                    
                    # Open new position
                    entry_price = current_price * (1 + self.slippage_pct)
                    max_position_value = capital * self.max_position_size
                    position_value = min(max_position_value, order_size * capital)
                    position = position_value / entry_price
                    fee = position_value * self.trading_fee_pct
                    capital -= fee
                    
                    entry_time = current_time
                    
                    # Set stop loss and take profit levels
                    if self.stop_loss_pct is not None:
                        stop_loss = entry_price * (1 - self.stop_loss_pct)
                    if self.take_profit_pct is not None:
                        take_profit = entry_price * (1 + self.take_profit_pct)
                    
                    logging.info(f"Buy: Opening position of {position:.4f} at ${entry_price:.2f}, " +
                                 f"Fee: ${fee:.2f}, Capital: ${capital:.2f}")
                
                elif order_type == 'sell' and order_size > 0:
                    # Execute sell order
                    if position != 0:
                        # Close existing position first
                        exit_price = current_price * (1 - self.slippage_pct)
                        pnl = position * (exit_price - entry_price)
                        fee = abs(position * exit_price) * self.trading_fee_pct
                        capital += pnl - fee
                        
                        # Record trade
                        trade = {
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'pnl': pnl,
                            'fee': fee,
                            'duration': (current_time - entry_time).total_seconds() / 3600,  # Hours
                            'exit_reason': 'signal_change'
                        }
                        trades.append(trade)
                        
                        logging.info(f"Signal change: Closing position of {position:.4f} at ${exit_price:.2f}, " +
                                     f"PnL: ${pnl:.2f}, Capital: ${capital:.2f}")
                    
                    # Open new short position
                    entry_price = current_price * (1 - self.slippage_pct)
                    max_position_value = capital * self.max_position_size
                    position_value = min(max_position_value, order_size * capital)
                    position = -position_value / entry_price  # Negative for short
                    fee = position_value * self.trading_fee_pct
                    capital -= fee
                    
                    entry_time = current_time
                    
                    # Set stop loss and take profit levels
                    if self.stop_loss_pct is not None:
                        stop_loss = entry_price * (1 + self.stop_loss_pct)
                    if self.take_profit_pct is not None:
                        take_profit = entry_price * (1 - self.take_profit_pct)
                    
                    logging.info(f"Sell: Opening short position of {position:.4f} at ${entry_price:.2f}, " +
                                 f"Fee: ${fee:.2f}, Capital: ${capital:.2f}")
            
            # Record equity
            equity = capital
            if position != 0:
                equity += position * current_price
            
            equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'capital': capital,
                'position': position,
                'price': current_price
            })
            
            # Calculate daily return if we have at least two equity points
            if len(equity_curve) >= 2:
                prev_equity = equity_curve[-2]['equity']
                daily_return = (equity - prev_equity) / prev_equity
                daily_returns.append(daily_return)
        
        # Close final position if still open
        if position != 0:
            final_price = data.iloc[-1][price_column]
            exit_price = final_price * (1 - self.slippage_pct * np.sign(position))
            pnl = position * (exit_price - entry_price)
            fee = abs(position * exit_price) * self.trading_fee_pct
            capital += pnl - fee
            
            # Record trade
            trade = {
                'entry_time': entry_time,
                'exit_time': data.iloc[-1][timestamp_column],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'pnl': pnl,
                'fee': fee,
                'duration': (data.iloc[-1][timestamp_column] - entry_time).total_seconds() / 3600,  # Hours
                'exit_reason': 'end_of_backtest'
            }
            trades.append(trade)
            
            logging.info(f"End of backtest: Closing position of {position:.4f} at ${exit_price:.2f}, " +
                         f"PnL: ${pnl:.2f}, Capital: ${capital:.2f}")
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(
            equity_curve, trades, daily_returns
        )
        
        # Save results
        if self.save_results:
            self.save_backtest_results(
                metrics, trades, equity_curve, save_file_prefix
            )
        
        # Log summary
        self.log_backtest_summary(metrics)
        
        logging.info(f"Backtest completed in {time.time() - start_time:.2f} seconds")
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def calculate_performance_metrics(
        self,
        equity_curve: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        daily_returns: List[float]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results
        
        Args:
            equity_curve: List of equity curve points
            trades: List of trades
            daily_returns: List of daily returns
            
        Returns:
            Dictionary of performance metrics
        """
        if not equity_curve:
            return {}
        
        # Convert lists to numpy arrays and pandas DataFrames
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        daily_returns_arr = np.array(daily_returns)
        
        # Initial and final equity
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        
        # Total return
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Annualized return
        # Calculate days in backtest
        if len(equity_df) > 1:
            start_date = pd.to_datetime(equity_df['timestamp'].iloc[0])
            end_date = pd.to_datetime(equity_df['timestamp'].iloc[-1])
            days = (end_date - start_date).total_seconds() / (60 * 60 * 24)
            years = days / 365.25
            
            if years > 0:
                annualized_return = (1 + total_return) ** (1 / years) - 1
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0
        
        # Maximum drawdown
        if len(equity_df) > 0:
            equity_arr = equity_df['equity'].values
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / peak
            max_drawdown = abs(min(drawdown))
        else:
            max_drawdown = 0.0
        
        # Sharpe ratio
        if len(daily_returns_arr) > 0:
            daily_risk_free = (1 + self.risk_free_rate) ** (1/252) - 1  # Daily risk-free rate
            excess_returns = daily_returns_arr - daily_risk_free
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio
        if len(daily_returns_arr) > 0:
            downside_returns = daily_returns_arr[daily_returns_arr < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = 0.0
        
        # Win rate
        if len(trades_df) > 0:
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        else:
            win_rate = 0.0
        
        # Average win/loss ratio
        if len(trades_df) > 0:
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 1
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        else:
            win_loss_ratio = 0.0
        
        # Profit factor
        if len(trades_df) > 0:
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            profit_factor = 0.0
        
        # Recovery factor
        if max_drawdown > 0:
            recovery_factor = total_return / max_drawdown
        else:
            recovery_factor = 0.0
        
        # Compile metrics
        metrics = {
            'initial_capital': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'recovery_factor': recovery_factor,
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]) if len(trades_df) > 0 else 0,
            'losing_trades': len(trades_df[trades_df['pnl'] < 0]) if len(trades_df) > 0 else 0,
            'avg_trade_pnl': trades_df['pnl'].mean() if len(trades_df) > 0 else 0.0,
            'avg_trade_duration': trades_df['duration'].mean() if len(trades_df) > 0 else 0.0,
            'total_fees': trades_df['fee'].sum() if len(trades_df) > 0 else 0.0
        }
        
        return metrics
    
    def save_backtest_results(
        self,
        metrics: Dict[str, float],
        trades: List[Dict[str, Any]],
        equity_curve: List[Dict[str, Any]],
        file_prefix: str
    ):
        """
        Save backtest results to files
        
        Args:
            metrics: Dictionary of performance metrics
            trades: List of trades
            equity_curve: List of equity curve points
            file_prefix: Prefix for saved files
        """
        # Save metrics
        metrics_file = os.path.join(
            self.log_dir,
            f'{file_prefix}_metrics.json'
        )
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save trades
        if self.save_trades and trades:
            trades_file = os.path.join(
                self.log_dir,
                f'{file_prefix}_trades.csv'
            )
            
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(trades_file, index=False)
        
        # Save equity curve
        if self.save_equity_curve and equity_curve:
            equity_file = os.path.join(
                self.log_dir,
                f'{file_prefix}_equity.csv'
            )
            
            equity_df = pd.DataFrame(equity_curve)
            equity_df.to_csv(equity_file, index=False)
        
        logging.info(f"Backtest results saved to {self.log_dir}/{file_prefix}_*")
    
    def log_backtest_summary(self, metrics: Dict[str, float]):
        """
        Log summary of backtest results
        
        Args:
            metrics: Dictionary of performance metrics
        """
        logging.info("=== Backtest Summary ===")
        logging.info(f"Initial Capital: ${metrics.get('initial_capital', 0):.2f}")
        logging.info(f"Final Equity: ${metrics.get('final_equity', 0):.2f}")
        logging.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
        logging.info(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        logging.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logging.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logging.info(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        logging.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        logging.info(f"Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}")
        logging.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        logging.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        logging.info(f"Winning Trades: {metrics.get('winning_trades', 0)}")
        logging.info(f"Losing Trades: {metrics.get('losing_trades', 0)}")
        logging.info(f"Avg Trade PnL: ${metrics.get('avg_trade_pnl', 0):.2f}")
        logging.info(f"Avg Trade Duration: {metrics.get('avg_trade_duration', 0):.2f} hours")
        logging.info(f"Total Fees: ${metrics.get('total_fees', 0):.2f}")
    
    def plot_equity_curve(
        self,
        equity_curve: List[Dict[str, Any]],
        show_drawdowns: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve
        
        Args:
            equity_curve: List of equity curve points
            show_drawdowns: Whether to highlight drawdowns
            save_path: Path to save the plot (optional)
        """
        if not equity_curve:
            logging.warning("Cannot plot equity curve: no data")
            return
        
        equity_df = pd.DataFrame(equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_df.index, equity_df['equity'], label='Equity')
        
        if show_drawdowns:
            # Calculate drawdowns
            equity_arr = equity_df['equity'].values
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / peak
            
            # Find drawdown periods
            is_drawdown = drawdown < -0.05  # 5% threshold
            
            # Highlight drawdown periods
            for i in range(len(is_drawdown)):
                if is_drawdown[i] and (i == 0 or not is_drawdown[i-1]):
                    start_idx = i
                elif not is_drawdown[i] and (i > 0 and is_drawdown[i-1]):
                    end_idx = i
                    plt.axvspan(
                        equity_df.index[start_idx],
                        equity_df.index[end_idx],
                        alpha=0.2,
                        color='red'
                    )
        
        plt.title('Equity Curve')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        
        # Calculate drawdown
        equity_arr = equity_df['equity'].values
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak
        
        plt.plot(equity_df.index, drawdown, 'r-', label='Drawdown')
        plt.fill_between(equity_df.index, drawdown, 0, color='red', alpha=0.3)
        
        plt.title('Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.ylim(min(drawdown) * 1.1, 0.01)
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_trade_analysis(
        self,
        trades: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ):
        """
        Plot trade analysis
        
        Args:
            trades: List of trades
            save_path: Path to save the plot (optional)
        """
        if not trades:
            logging.warning("Cannot plot trade analysis: no trades")
            return
        
        trades_df = pd.DataFrame(trades)
        
        plt.figure(figsize=(15, 10))
        
        # Plot PnL distribution
        plt.subplot(2, 2, 1)
        sns.histplot(trades_df['pnl'], kde=True)
        plt.title('PnL Distribution')
        plt.xlabel('PnL ($)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Plot PnL by exit reason
        plt.subplot(2, 2, 2)
        sns.boxplot(x='exit_reason', y='pnl', data=trades_df)
        plt.title('PnL by Exit Reason')
        plt.xlabel('Exit Reason')
        plt.ylabel('PnL ($)')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Plot trade duration vs PnL
        plt.subplot(2, 2, 3)
        plt.scatter(trades_df['duration'], trades_df['pnl'], alpha=0.5)
        plt.title('Trade Duration vs PnL')
        plt.xlabel('Duration (hours)')
        plt.ylabel('PnL ($)')
        plt.grid(True)
        
        # Plot cumulative PnL
        plt.subplot(2, 2, 4)
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        plt.plot(trades_df['cumulative_pnl'])
        plt.title('Cumulative PnL')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative PnL ($)')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 