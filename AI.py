from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
MODEL_PATH = 'trade_confidence_model.pkl'
model_data = None

try:
    model_data = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")


class ParameterOptimizer:
    """
    Intelligent parameter optimizer that suggests best trade parameters
    based on current market conditions
    """
    
    def __init__(self, model_data):
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data.get('threshold', 0.75)
    
    def optimize_for_confidence(self, current_features, strategy_type, target_confidence=0.80):
        """
        Find optimal parameter adjustments to reach target confidence
        
        Returns:
        - optimized_params: Dictionary of suggested parameter changes
        - expected_confidence: Confidence with optimized parameters
        - rationale: Explanation of changes
        """
        
        logger.info(f"Optimizing parameters for {strategy_type}")
        
        # Current confidence
        current_confidence = self._calculate_confidence(current_features)
        
        if current_confidence >= target_confidence:
            return {
                'optimization_needed': False,
                'current_confidence': float(current_confidence),
                'message': 'Current setup already meets confidence threshold'
            }
        
        # Strategy-specific optimization
        if strategy_type == 'trend_following':
            optimized = self._optimize_trend_following(current_features, target_confidence)
        elif strategy_type == 'mean_reversion':
            optimized = self._optimize_mean_reversion(current_features, target_confidence)
        elif strategy_type == 'breakout':
            optimized = self._optimize_breakout(current_features, target_confidence)
        else:
            optimized = self._generic_optimization(current_features, target_confidence)
        
        return optimized
    
    def _calculate_confidence(self, features):
        """Calculate confidence score for given features"""
        features_array = np.array([features.get(f, 0) for f in self.feature_names]).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        confidence = self.model.predict_proba(features_scaled)[0, 1]
        return confidence
    
    def _optimize_trend_following(self, features, target_confidence):
        """
        Optimize trend following strategy parameters
        Focus: EMA settings, ATR multiplier, entry timing
        """
        
        suggestions = {
            'optimization_needed': True,
            'strategy': 'trend_following',
            'current_confidence': float(self._calculate_confidence(features)),
            'target_confidence': target_confidence,
            'optimized_parameters': {},
            'wait_conditions': [],
            'rationale': []
        }
        
        rsi = features.get('rsi', 50)
        ema_distance = features.get('ema_distance', 0)
        atr = features.get('atr', 10)
        stoch_k = features.get('stoch_k', 50)
        hour = features.get('hour_of_day', 12)
        volume_ratio = features.get('volume_ratio', 1.0)
        
        # RSI Optimization
        if rsi > 70:
            suggestions['wait_conditions'].append('RSI_OVERBOUGHT')
            suggestions['optimized_parameters']['wait_for_rsi_below'] = 65
            suggestions['rationale'].append(f'Current RSI {rsi:.1f} is overbought. Wait for pullback to RSI < 65 for better entry.')
            
        elif rsi < 50:
            suggestions['wait_conditions'].append('RSI_TOO_LOW_FOR_TREND')
            suggestions['optimized_parameters']['wait_for_rsi_above'] = 55
            suggestions['rationale'].append(f'Current RSI {rsi:.1f} lacks bullish momentum. Wait for RSI > 55.')
        
        # EMA Distance Optimization
        if abs(ema_distance) < 0.1:
            suggestions['wait_conditions'].append('WEAK_TREND')
            suggestions['optimized_parameters']['wait_for_ema_distance'] = 0.3
            suggestions['rationale'].append(f'EMAs too close (distance: {ema_distance:.2f}%). Wait for clearer trend (distance > 0.3%).')
        
        # ATR-based Stop Loss Optimization
        if atr > 15:
            suggestions['optimized_parameters']['atr_multiplier'] = 2.5
            suggestions['optimized_parameters']['reduce_lot_size_by'] = 0.40
            suggestions['rationale'].append(f'High volatility (ATR: {atr:.1f}). Use wider stop (2.5x ATR) and reduce lot size by 40%.')
        else:
            suggestions['optimized_parameters']['atr_multiplier'] = 2.0
        
        # Session Timing
        if hour < 8 or hour > 17:
            suggestions['wait_conditions'].append('OUTSIDE_OPTIMAL_HOURS')
            suggestions['optimized_parameters']['wait_for_london_session'] = True
            suggestions['rationale'].append(f'Current hour {hour}:00 GMT is outside London session. Wait for 8:00-17:00 GMT.')
        
        # Volume Confirmation
        if volume_ratio < 1.0:
            suggestions['wait_conditions'].append('LOW_VOLUME')
            suggestions['optimized_parameters']['wait_for_volume_ratio'] = 1.2
            suggestions['rationale'].append(f'Volume too low (ratio: {volume_ratio:.2f}). Wait for volume spike > 1.2x average.')
        
        # Stochastic Confirmation
        if stoch_k > 80:
            suggestions['optimized_parameters']['wait_for_stoch_pullback'] = 70
            suggestions['rationale'].append(f'Stochastic overbought ({stoch_k:.1f}). Wait for pullback to < 70.')
        
        # Calculate expected confidence after optimization
        optimized_features = features.copy()
        if len(suggestions['wait_conditions']) == 0:
            # Apply numeric optimizations
            if 'reduce_lot_size_by' in suggestions['optimized_parameters']:
                # This improves risk management, estimate +10% confidence
                suggestions['expected_confidence_after_wait'] = min(0.95, suggestions['current_confidence'] + 0.10)
        else:
            # If waiting for conditions, estimate confidence improvement
            improvement_per_condition = 0.08
            expected_improvement = len(suggestions['wait_conditions']) * improvement_per_condition
            suggestions['expected_confidence_after_wait'] = min(0.95, suggestions['current_confidence'] + expected_improvement)
        
        # Action recommendation
        if len(suggestions['wait_conditions']) > 0:
            suggestions['action'] = 'WAIT_FOR_CONDITIONS'
            suggestions['summary'] = f"Wait for {len(suggestions['wait_conditions'])} conditions to improve confidence from {suggestions['current_confidence']:.1%} to ~{suggestions['expected_confidence_after_wait']:.1%}"
        else:
            suggestions['action'] = 'ADJUST_PARAMETERS_AND_TRADE'
            suggestions['summary'] = f"Trade with adjusted parameters (wider stop, reduced size)"
        
        return suggestions
    
    def _optimize_mean_reversion(self, features, target_confidence):
        """
        Optimize mean reversion strategy parameters
        Focus: RSI extremes, Stochastic levels, entry/exit timing
        """
        
        suggestions = {
            'optimization_needed': True,
            'strategy': 'mean_reversion',
            'current_confidence': float(self._calculate_confidence(features)),
            'target_confidence': target_confidence,
            'optimized_parameters': {},
            'wait_conditions': [],
            'rationale': []
        }
        
        rsi = features.get('rsi', 50)
        stoch_k = features.get('stoch_k', 50)
        bb_width = features.get('bb_width', 1.0)
        atr = features.get('atr', 10)
        volume_ratio = features.get('volume_ratio', 1.0)
        
        # RSI for mean reversion (looking for extremes)
        if 40 < rsi < 60:
            suggestions['wait_conditions'].append('RSI_NOT_EXTREME')
            suggestions['optimized_parameters']['wait_for_rsi_below'] = 35
            suggestions['optimized_parameters']['or_wait_for_rsi_above'] = 70
            suggestions['rationale'].append(f'RSI {rsi:.1f} is neutral. For mean reversion, wait for RSI < 35 (oversold) or > 70 (overbought).')
        
        # Oversold entry optimization
        if rsi < 30:
            if stoch_k < 20:
                suggestions['optimized_parameters']['excellent_entry'] = True
                suggestions['rationale'].append(f'Excellent oversold setup: RSI {rsi:.1f} + Stoch {stoch_k:.1f}. Consider entering.')
            else:
                suggestions['wait_conditions'].append('STOCH_NOT_CONFIRMING')
                suggestions['optimized_parameters']['wait_for_stoch_below'] = 25
                suggestions['rationale'].append(f'RSI oversold but Stochastic {stoch_k:.1f} not confirming. Wait for Stoch < 25.')
        
        # Overbought entry optimization
        if rsi > 70:
            if stoch_k > 80:
                suggestions['optimized_parameters']['excellent_entry'] = True
                suggestions['rationale'].append(f'Excellent overbought setup: RSI {rsi:.1f} + Stoch {stoch_k:.1f}. Consider shorting.')
            else:
                suggestions['wait_conditions'].append('STOCH_NOT_CONFIRMING')
                suggestions['optimized_parameters']['wait_for_stoch_above'] = 75
                suggestions['rationale'].append(f'RSI overbought but Stochastic {stoch_k:.1f} not confirming. Wait for Stoch > 75.')
        
        # Bollinger Band width (volatility context)
        if bb_width > 2.0:
            suggestions['optimized_parameters']['reduce_lot_size_by'] = 0.50
            suggestions['optimized_parameters']['atr_multiplier'] = 3.0
            suggestions['rationale'].append(f'BB width {bb_width:.2f} indicates high volatility. Reduce lot 50%, use 3x ATR stops.')
        
        # Volume confirmation for reversal
        if volume_ratio < 0.8:
            suggestions['wait_conditions'].append('INSUFFICIENT_VOLUME')
            suggestions['optimized_parameters']['wait_for_volume_spike'] = 1.5
            suggestions['rationale'].append(f'Volume ratio {volume_ratio:.2f} too low. Wait for volume spike > 1.5x for reversal confirmation.')
        
        # Calculate expected improvement
        if len(suggestions['wait_conditions']) > 0:
            improvement = len(suggestions['wait_conditions']) * 0.09
            suggestions['expected_confidence_after_wait'] = min(0.95, suggestions['current_confidence'] + improvement)
            suggestions['action'] = 'WAIT_FOR_CONDITIONS'
        else:
            suggestions['expected_confidence_after_wait'] = suggestions['current_confidence']
            suggestions['action'] = 'TRADE_WITH_CAUTION'
        
        suggestions['summary'] = f"Mean reversion setup: {len(suggestions['wait_conditions'])} conditions to improve. Expected confidence: {suggestions['expected_confidence_after_wait']:.1%}"
        
        return suggestions
    
    def _optimize_breakout(self, features, target_confidence):
        """
        Optimize breakout strategy parameters
        Focus: Volume confirmation, volatility expansion, false breakout filtering
        """
        
        suggestions = {
            'optimization_needed': True,
            'strategy': 'breakout',
            'current_confidence': float(self._calculate_confidence(features)),
            'target_confidence': target_confidence,
            'optimized_parameters': {},
            'wait_conditions': [],
            'rationale': []
        }
        
        volume_ratio = features.get('volume_ratio', 1.0)
        bb_width = features.get('bb_width', 1.0)
        atr = features.get('atr', 10)
        momentum_5 = features.get('momentum_5', 0)
        close_position = features.get('close_position', 0.5)
        
        # Volume is CRITICAL for breakouts
        if volume_ratio < 1.5:
            suggestions['wait_conditions'].append('INSUFFICIENT_BREAKOUT_VOLUME')
            suggestions['optimized_parameters']['wait_for_volume_ratio'] = 2.0
            suggestions['rationale'].append(f'Volume ratio {volume_ratio:.2f} too low for valid breakout. Need > 2.0x average volume.')
        else:
            suggestions['optimized_parameters']['strong_volume'] = True
            suggestions['rationale'].append(f'Strong volume confirmation: {volume_ratio:.2f}x average.')
        
        # BB Width (volatility expansion)
        if bb_width < 0.5:
            suggestions['wait_conditions'].append('BOLLINGER_SQUEEZE')
            suggestions['optimized_parameters']['wait_for_bb_expansion'] = 0.8
            suggestions['rationale'].append(f'BB width {bb_width:.2f} indicates squeeze. Wait for expansion > 0.8 for true breakout.')
        
        # Momentum confirmation
        if abs(momentum_5) < 0.5:
            suggestions['wait_conditions'].append('WEAK_MOMENTUM')
            suggestions['optimized_parameters']['wait_for_momentum_above'] = 1.0
            suggestions['rationale'].append(f'Momentum {momentum_5:.2f}% too weak. Wait for momentum > 1.0% for conviction.')
        
        # Candle close position (avoid wicks)
        if close_position < 0.7:
            suggestions['wait_conditions'].append('CLOSE_NOT_NEAR_HIGH')
            suggestions['optimized_parameters']['wait_for_close_position'] = 0.75
            suggestions['rationale'].append(f'Close at {close_position:.0%} of range suggests weakness. Wait for strong close (> 75%).')
        
        # ATR-based position sizing
        if atr > 18:
            suggestions['optimized_parameters']['reduce_lot_size_by'] = 0.35
            suggestions['optimized_parameters']['atr_multiplier'] = 2.5
            suggestions['rationale'].append(f'High volatility (ATR {atr:.1f}). Reduce lot 35%, use 2.5x ATR stops for breakout volatility.')
        else:
            suggestions['optimized_parameters']['atr_multiplier'] = 2.0
        
        # False breakout filter (check for retest)
        suggestions['optimized_parameters']['wait_for_retest'] = True
        suggestions['rationale'].append('Consider waiting for retest of breakout level to confirm validity.')
        
        # Calculate expected confidence
        if len(suggestions['wait_conditions']) > 0:
            improvement = len(suggestions['wait_conditions']) * 0.12  # Breakouts need stronger conditions
            suggestions['expected_confidence_after_wait'] = min(0.95, suggestions['current_confidence'] + improvement)
            suggestions['action'] = 'WAIT_FOR_CONDITIONS'
        else:
            suggestions['action'] = 'TRADE_BREAKOUT'
            suggestions['expected_confidence_after_wait'] = suggestions['current_confidence']
        
        suggestions['summary'] = f"Breakout setup needs {len(suggestions['wait_conditions'])} conditions. Expected: {suggestions['expected_confidence_after_wait']:.1%}"
        
        return suggestions
    
    def _generic_optimization(self, features, target_confidence):
        """Generic optimization when strategy is unknown"""
        return {
            'optimization_needed': True,
            'current_confidence': float(self._calculate_confidence(features)),
            'action': 'INSUFFICIENT_DATA',
            'message': 'Strategy type not recognized. Manual review recommended.'
        }


# Initialize optimizer
optimizer = None
if model_data:
    optimizer = ParameterOptimizer(model_data)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'optimizer_ready': optimizer is not None
    })


@app.route('/predict', methods=['POST'])
def predict_confidence():
    """Enhanced prediction with parameter optimization"""
    
    if model_data is None or optimizer is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        logger.info(f"Received prediction request")
        
        # Extract features
        feature_names = model_data['feature_names']
        features = {}
        for feature in feature_names:
            features[feature] = data.get(feature, 0)
        
        # Calculate confidence
        features_array = np.array([features[f] for f in feature_names]).reshape(1, -1)
        features_scaled = model_data['scaler'].transform(features_array)
        
        confidence_score = float(model_data['model'].predict_proba(features_scaled)[0, 1])
        threshold = model_data.get('threshold', 0.75)
        should_trade = confidence_score >= threshold
        
        # Confidence level
        if confidence_score >= 0.8:
            confidence_level = "VERY_HIGH"
        elif confidence_score >= 0.7:
            confidence_level = "HIGH"
        elif confidence_score >= 0.6:
            confidence_level = "MEDIUM"
        elif confidence_score >= 0.5:
            confidence_level = "LOW"
        else:
            confidence_level = "VERY_LOW"
        
        # Get strategy type
        strategy_type = data.get('strategy_type', 'unknown')
        
        # PARAMETER OPTIMIZATION - Core feature
        optimization_result = None
        if not should_trade:
            optimization_result = optimizer.optimize_for_confidence(
                current_features=data,
                strategy_type=strategy_type,
                target_confidence=threshold
            )
            logger.info(f"Optimization suggested: {optimization_result.get('action')}")
        
        result = {
            'confidence': confidence_score,
            'confidence_level': confidence_level,
            'should_trade': should_trade,
            'threshold': threshold,
            'strategy_type': strategy_type,
            'optimization': optimization_result,  # NEW: Parameter suggestions
            'timestamp': data.get('timestamp', 'N/A')
        }
        
        logger.info(f"Prediction: {confidence_score:.2f}, Trade: {should_trade}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'should_trade': False
        }), 500


@app.route('/optimize_only', methods=['POST'])
def optimize_only():
    """
    Dedicated endpoint for parameter optimization
    Use this to get suggestions without making a trade decision
    """
    
    if optimizer is None:
        return jsonify({'error': 'Optimizer not initialized'}), 500
    
    try:
        data = request.get_json()
        strategy_type = data.get('strategy_type', 'trend_following')
        target_confidence = data.get('target_confidence', 0.80)
        
        optimization = optimizer.optimize_for_confidence(
            current_features=data,
            strategy_type=strategy_type,
            target_confidence=target_confidence
        )
        
        return jsonify(optimization)
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)