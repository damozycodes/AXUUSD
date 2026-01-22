import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def train_model(data_path='xauusd_training_data.csv'):
    print("TRAINING AI CONFIDENCE MODEL")

    
    # Load data
    print("[1/6] Loading training data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} training samples")
    
    # Select features
    print(" [2/6] Selecting features...")
    feature_columns = [
        'ema_fast', 'ema_slow', 'ema_distance',
        'rsi', 'atr', 'atr_pct',
        'stoch_k', 'stoch_d',
        'bb_upper', 'bb_lower', 'bb_width',
        'volume_ratio', 'momentum_5', 'momentum_10',
        'hour_of_day', 'day_of_week', 'is_london_session',
        'high_low_range', 'close_position', 'spread'
    ]
    
    # Filter to existing columns
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(feature_columns)} features")
    
    X = df[feature_columns]
    y = df['trade_success']
    
    print(f" Target distribution:")
    print(f"  Wins:   {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  Losses: {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")
    
    # Split data
    print(" [3/6] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ Train: {len(X_train):,} samples")
    print(f"✓ Test:  {len(X_test):,} samples")
    
    # Scale features
    print("[4/6] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(" Features scaled")
    
    # Train model
    print("[5/6] Training Gradient Boosting model...")
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    print("Model trained")
    
    # Evaluate
    print("[6/6] Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("MODEL PERFORMANCE")
    
    # Accuracy
    accuracy = (y_pred == y_test).mean()
    print(f" Accuracy: {accuracy*100:.2f}%")
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    print(" Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              Loss    Win")
    print(f"Actual Loss   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Win    {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Feature importance
    print(" Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    # Test different confidence thresholds
    print("CONFIDENCE THRESHOLD ANALYSIS")

    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    print(f"{'Threshold':<12} {'Trades':<10} {'Win Rate':<12} {'Precision':<12}")

    
    for threshold in thresholds:
        high_conf_mask = y_pred_proba >= threshold
        if high_conf_mask.sum() == 0:
            continue
        
        trades_taken = high_conf_mask.sum()
        wins = (y_test[high_conf_mask] == 1).sum()
        win_rate = wins / trades_taken if trades_taken > 0 else 0
        
        # Precision = TP / (TP + FP)
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        tp = ((y_pred_thresh == 1) & (y_test == 1)).sum()
        fp = ((y_pred_thresh == 1) & (y_test == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"{threshold:<12.2f} {trades_taken:<10,} {win_rate*100:<11.1f}% {precision*100:<11.1f}%")
    
    # Save model
    print("SAVING MODEL")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_columns,
        'threshold': 0.75,  # Default threshold
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    model_path = 'trade_confidence_model.pkl'
    joblib.dump(model_data, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(f"Feature importance saved to: feature_importance.csv")
    

    print("✓ TRAINING COMPLETE!")

    print("\nNext steps:")
    print("  1. Review model performance above")
    print("  2. Choose optimal confidence threshold")
    print("  3. Start API server: python ai_confidence_api.py")
    print("  4. Test with MT4 EA")
    
    return model_data


if __name__ == "__main__":
    import sys
    
    # Check if data file exists
    data_file = 'xauusd_training_data.csv'
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    
    try:
        model_data = train_model(data_file)
        
        # Quick test prediction
        print("TESTING MODEL - Sample Prediction")

        
        test_features = {
            'ema_fast': 2650.5,
            'ema_slow': 2648.2,
            'ema_distance': 0.1,
            'rsi': 65.3,
            'atr': 12.5,
            'atr_pct': 0.47,
            'stoch_k': 72.1,
            'stoch_d': 68.9,
            'bb_upper': 2660.0,
            'bb_lower': 2640.0,
            'bb_width': 0.75,
            'volume_ratio': 1.2,
            'momentum_5': 0.5,
            'momentum_10': 1.2,
            'hour_of_day': 14,
            'day_of_week': 2,
            'is_london_session': 1,
            'high_low_range': 15.0,
            'close_position': 0.6,
            'spread': 2.5
        }
        
        # Prepare features
        features_array = np.array([test_features[f] for f in model_data['feature_names']]).reshape(1, -1)
        features_scaled = model_data['scaler'].transform(features_array)
        
        # Predict
        confidence = model_data['model'].predict_proba(features_scaled)[0, 1]
        should_trade = confidence >= model_data['threshold']
        
        print(f" Test Input: Bullish trend following signal")
        print(f"  EMA Fast: {test_features['ema_fast']}")
        print(f"  EMA Slow: {test_features['ema_slow']}")
        print(f"  RSI: {test_features['rsi']}")
        
        print(f"Prediction:")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  Decision: {'✓ TAKE TRADE' if should_trade else '✗ SKIP TRADE'}")
        
    except FileNotFoundError:
        print(f"ERROR: File '{data_file}' not found")
        print("Please run data preparation first:")
        print("  python prepare_data.py")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()