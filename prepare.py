import pandas as pd
import numpy as np
from datetime import datetime


class XAUUSDDataPreparation:
    def __init__(self):
        self.df = None
        
    def load_and_prepare(self, filepath):
        """
        Complete pipeline: Load data, calculate indicators, create labels
        
        Returns:
        - DataFrame ready for ML training
        """
        print("XAUUSD DATA PREPARATION PIPELINE")
        
        # Load data
        print("[1/5] Loading data...")
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Loaded {len(df)} rows")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Calculate technical indicators
        print("[2/5] Calculating technical indicators...")
        df = self.calculate_all_indicators(df)
        print(f"Added {len([c for c in df.columns if c not in ['timestamp','open','high','low','close','volume']])} indicators")
        
        # Generate synthetic trade signals
        print("[3/5] Generating trade signals and outcomes...")
        df = self.generate_trade_signals(df)
        
        # Create training samples
        print("[4/5] Creating training samples...")
        df_training = self.create_training_samples(df)
        print(f"Created {len(df_training)} training samples")
        
        # Clean and validate
        print("[5/5] Cleaning data...")
        df_training = self.clean_data(df_training)
        print(f"Final dataset: {len(df_training)} samples")
        
        # Statistics
        self.print_statistics(df_training)
        
        self.df = df_training
        return df_training
    
    def calculate_all_indicators(self, df):
        """Calculate all technical indicators matching your EA"""
        
        # EMAs (matching your EA parameters)
        df['ema_fast'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=200, adjust=False).mean()
        df['ema_distance'] = ((df['ema_fast'] - df['ema_slow']) / df['ema_slow'] * 100)
        
        # RSI (14 period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR (14 period)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        # df['atr'] = true_range.rolling(14).mean()
        df['atr'] = true_range.ewm(alpha=1/14, adjust=False).mean()

        df['atr_pct'] = (df['atr'] / df['close'] * 100)
        
        # Stochastic (14, 3, 3)
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Bollinger Bands (20, 2)
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_middle + (bb_std * 2)
        df['bb_lower'] = bb_middle - (bb_std * 2)
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / bb_middle * 100)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
        
        # Price momentum
        df['momentum_5'] = df['close'].pct_change(5) * 100
        df['momentum_10'] = df['close'].pct_change(10) * 100
        df['price_change_1'] = df['close'].pct_change(1) * 100
        
        # Time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_london_session'] = ((df['hour_of_day'] >= 8) & (df['hour_of_day'] < 17)).astype(int)
        
        # Additional features
        df['high_low_range'] = df['high'] - df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        
        return df
    
    def generate_trade_signals(self, df):
        """
        Generate trade signals based on your EA's strategies
        Then determine if each signal would have been profitable
        """
        
        # Initialize signal columns
        df['signal_trend'] = 0
        df['signal_mean_rev'] = 0
        df['signal_breakout'] = 0
        
        # TREND FOLLOWING SIGNAL (EMA crossover + RSI)
        ema_cross_up = (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) & (df['ema_fast'] > df['ema_slow'])
        ema_cross_down = (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) & (df['ema_fast'] < df['ema_slow'])
        
        df.loc[ema_cross_up & (df['rsi'] > 50), 'signal_trend'] = 1  # Buy
        df.loc[ema_cross_down & (df['rsi'] < 50), 'signal_trend'] = -1  # Sell
        
        # MEAN REVERSION SIGNAL (RSI extremes + Stochastic)
        rsi_oversold = df['rsi'] < 30
        rsi_overbought = df['rsi'] > 70
        stoch_cross_up = (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) & (df['stoch_k'] > df['stoch_d'])
        stoch_cross_down = (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)) & (df['stoch_k'] < df['stoch_d'])
        
        df.loc[rsi_oversold & stoch_cross_up & (df['stoch_k'] < 30), 'signal_mean_rev'] = 1
        df.loc[rsi_overbought & stoch_cross_down & (df['stoch_k'] > 70), 'signal_mean_rev'] = -1
        
        # BREAKOUT SIGNAL (Bollinger Bands breakout)
        bb_break_up = df['close'] > df['bb_upper']
        bb_break_down = df['close'] < df['bb_lower']
        high_volume = df['volume_ratio'] > 1.5
        
        df.loc[bb_break_up & high_volume, 'signal_breakout'] = 1
        df.loc[bb_break_down & high_volume, 'signal_breakout'] = -1
        
        return df
    
    def create_training_samples(self, df):
        """
        For each signal, create a training sample with features and outcome
        """
        training_samples = []
        
        # Look forward window to determine trade outcome (e.g., 10 periods)
        lookforward = 10
        
        for idx in range(len(df) - lookforward):
            row = df.iloc[idx]
            
            # Check if any signal is present
            signals = [
                ('trend_following', row['signal_trend']),
                ('mean_reversion', row['signal_mean_rev']),
                ('breakout', row['signal_breakout'])
            ]
            
            for strategy_name, signal in signals:
                if signal == 0:
                    continue  # No signal
                
                # Calculate trade outcome
                entry_price = row['close']
                atr_value = row['atr']
                
                if pd.isna(atr_value) or atr_value <= 0:
                    continue
                
                # SL/TP based on EA (ATR * 2 for SL, ATR * 6 for TP)
                if signal == 1:  # Buy
                    sl_price = entry_price - (atr_value * 2)
                    tp_price = entry_price + (atr_value * 6)  # 3:1 RR
                else:  # Sell
                    sl_price = entry_price + (atr_value * 2)
                    tp_price = entry_price - (atr_value * 6)
                
                # Check next N candles for SL or TP hit
                trade_success = 0
                future_data = df.iloc[idx+1:idx+1+lookforward]
                
                for _, future_row in future_data.iterrows():
                    if signal == 1:  # Buy
                        if future_row['low'] <= sl_price:
                            trade_success = 0  # SL hit
                            break
                        elif future_row['high'] >= tp_price:
                            trade_success = 1  # TP hit
                            break
                    else:  # Sell
                        if future_row['high'] >= sl_price:
                            trade_success = 0  # SL hit
                            break
                        elif future_row['low'] <= tp_price:
                            trade_success = 1  # TP hit
                            break
                
                # If neither hit, use final P&L
                if signal == 1:
                    final_price = future_data.iloc[-1]['close'] if len(future_data) > 0 else entry_price
                    profit = final_price - entry_price
                else:
                    final_price = future_data.iloc[-1]['close'] if len(future_data) > 0 else entry_price
                    profit = entry_price - final_price
                
                if trade_success == 0 and profit > atr_value:  # Profitable even without TP
                    trade_success = 1
                
                # Create training sample
                sample = {
                    'timestamp': row['timestamp'],
                    'ema_fast': row['ema_fast'],
                    'ema_slow': row['ema_slow'],
                    'ema_distance': row['ema_distance'],
                    'rsi': row['rsi'],
                    'atr': row['atr'],
                    'atr_pct': row['atr_pct'],
                    'stoch_k': row['stoch_k'],
                    'stoch_d': row['stoch_d'],
                    'bb_upper': row['bb_upper'],
                    'bb_lower': row['bb_lower'],
                    'bb_width': row['bb_width'],
                    'volume_ratio': row['volume_ratio'],
                    'momentum_5': row['momentum_5'],
                    'momentum_10': row['momentum_10'],
                    'hour_of_day': row['hour_of_day'],
                    'day_of_week': row['day_of_week'],
                    'is_london_session': row['is_london_session'],
                    'high_low_range': row['high_low_range'],
                    'close_position': row['close_position'],
                    'spread': 2.5,  # Typical XAUUSD spread
                    'strategy_type': strategy_name,
                    'signal': signal,
                    'trade_success': trade_success
                }
                
                training_samples.append(sample)
        print(f"  Generated {len(training_samples)} samples from signals")
        return pd.DataFrame(training_samples)
    
    def clean_data(self, df):
        """Clean and validate data"""
        initial_rows = len(df)
        
        # Remove NaN values
        df = df.dropna()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Remove unrealistic values
        df = df[df['rsi'] >= 0]
        df = df[df['rsi'] <= 100]
        df = df[df['atr'] > 0]
        df = df[df['volume_ratio'] >= 0]
        
        removed = initial_rows - len(df)
        print(f"  Removed {removed} invalid rows ({removed/initial_rows*100:.1f}%)")
        
        return df
    
    def print_statistics(self, df):
        """Print dataset statistics"""
        print("DATASET STATISTICS")
        
        total = len(df)
        wins = df['trade_success'].sum()
        losses = total - wins
        win_rate = wins / total * 100
        
        print(f" Total Samples: {total:,}")
        print(f"Winning trades: {wins:,} ({win_rate:.1f}%)")
        print(f"Losing trades: {losses:,} ({100-win_rate:.1f}%)")
        
        print("Strategy Distribution:")
        for strategy in df['strategy_type'].unique():
            count = len(df[df['strategy_type'] == strategy])
            strat_wins = df[df['strategy_type'] == strategy]['trade_success'].sum()
            strat_wr = strat_wins / count * 100 if count > 0 else 0
            print(f"  {strategy:20s}: {count:6,} samples ({strat_wr:.1f}% win rate)")
        
        print("Signal Distribution:")
        buys = len(df[df['signal'] == 1])
        sells = len(df[df['signal'] == -1])
        print(f"  Buy signals:  {buys:,} ({buys/total*100:.1f}%)")
        print(f"  Sell signals: {sells:,} ({sells/total*100:.1f}%)")
        
        print("Feature Statistics:")
        print(f"  RSI range: {df['rsi'].min():.1f} - {df['rsi'].max():.1f}")
        print(f"  ATR range: {df['atr'].min():.2f} - {df['atr'].max():.2f}")
        print(f"  Volume ratio range: {df['volume_ratio'].min():.2f} - {df['volume_ratio'].max():.2f}")
        
    
    def save_training_data(self, output_path='xauusd_training_data.csv'):
        """Save prepared data"""
        if self.df is None:
            print("ERROR: No data to save. Run load_and_prepare() first.")
            return
        
        self.df.to_csv(output_path, index=False)
        print(f"Training data saved to: {output_path}")
        print(f"  File size: {len(self.df):,} rows × {len(self.df.columns)} columns")
        
        return output_path




if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  XAUUSD AI Training Data Preparation                             ║
    ║  Converts raw OHLCV data → ML-ready training dataset             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize
    prep = XAUUSDDataPreparation()
    
    # Path to your CSV file
    input_file = 'XAUUSD_H4.csv' 
    
    try:
        # Run complete pipeline
        df_training = prep.load_and_prepare(input_file)
        
        # Save training data
        output_file = prep.save_training_data('xauusd_training_data.csv')
        
        print("SUCCESS! Ready for model training.")
        print(f"Next step:")
        print(f"  python train_model.py")
        
        # Show sample data
        print("SAMPLE DATA (first 3 rows):")
        print("="*70)
        print(df_training.head(3).to_string())
        
    except FileNotFoundError:
        print(f"ERROR: File '{input_file}' not found.")
        print("Please ensure your CSV file has these columns:")
        print("  - timestamp")
        print("  - open")
        print("  - high") 
        print("  - low")
        print("  - close")
        print("  - volume")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()