from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import joblib  # for saving/loading scaler

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None

def create_model():
    """Create the LSTM model with the exact architecture from training"""
    print("Creating new model with original architecture...")
    new_model = Sequential()
    new_model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(100, 1)))
    new_model.add(Dropout(0.2))
    new_model.add(LSTM(units=60, activation='relu', return_sequences=True))
    new_model.add(Dropout(0.3))
    new_model.add(LSTM(units=80, activation='relu', return_sequences=True))
    new_model.add(Dropout(0.4))
    new_model.add(LSTM(units=120, activation='relu'))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(units=1))
    new_model.compile(optimizer='adam', loss='mean_squared_error')
    return new_model

def train_model_on_data(df):
    """Train the model on available data"""
    global model, scaler
    
    print("Training model on available data...")
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.8)])
    
    # Fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Save scaler
    scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at {scaler_path}")
    
    # Create training sets
    x_train, y_train = [], []
    for i in range(100, len(data_training_array)):
        x_train.append(data_training_array[i-100:i, 0])
        y_train.append(data_training_array[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1)
    print("Model training completed!")

def load_model_with_fallback():
    """Try to load the model and scaler"""
    global model, scaler
    
    # Load scaler
    scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print("Loaded fitted scaler")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    
    # Load model (.keras or .h5)
    model_path = os.path.join(os.path.dirname(__file__), "model", "my_model.keras")
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print("Successfully loaded .keras model")
            return True
        except Exception as e:
            print(f"Error loading .keras model: {e}")
    
    model_path = os.path.join(os.path.dirname(__file__), "model", "my_model.h5")
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print("Successfully loaded .h5 model")
            return True
        except Exception as e:
            print(f"Error loading .h5 model: {e}")
    
    print("No model file found or loading failed. Creating new model...")
    model = create_model()
    return False

# Load at startup
load_model_with_fallback()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        stock_symbol = data.get('symbol', 'POWERGRID.NS')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        df = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=False)
        
        if df.empty:
            return jsonify({'error': 'No data found for this symbol'})
        
        df = df.reset_index()
        
        # Check if trained model exists
        model_path_keras = os.path.join(os.path.dirname(__file__), "model", "my_model.keras")
        model_path_h5 = os.path.join(os.path.dirname(__file__), "model", "my_model.h5")
        has_trained_model = os.path.exists(model_path_keras) or os.path.exists(model_path_h5)
        
        if not has_trained_model:
            print("Model is untrained. Training on current data...")
            train_model_on_data(df)
        
        # Ensure scaler is loaded
        global scaler
        if scaler is None:
            scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print("Scaler loaded in predict()")
            else:
                return jsonify({'error': 'Scaler not found. Please train the model first.'})
        
        # Prepare data
        split_index = int(len(df) * 0.70)
        data_training = pd.DataFrame(df['Close'][0:split_index])
        data_testing = pd.DataFrame(df['Close'][split_index:])
        
        data_training_array = scaler.transform(data_training)
        
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)
        
        x_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
        
        x_test = np.array(x_test)
        
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        actual_prices = final_df.iloc[100:].values.flatten()
        
        dates = df['Date'].iloc[split_index+100:].dt.strftime('%Y-%m-%d').tolist()
        
        min_length = min(len(dates), len(actual_prices), len(predictions))
        dates = dates[:min_length]
        actual_prices = actual_prices[:min_length].tolist()
        predictions = predictions[:min_length].flatten().tolist()
        
        current_price = float(df['Close'].iloc[-1])  # ensure float
        last_100_days = df['Close'].iloc[-100:].values.reshape(-1, 1)
        last_100_days_scaled = scaler.transform(last_100_days)
        
        next_day_prediction = model.predict(np.array([last_100_days_scaled]))
        next_day_prediction = float(scaler.inverse_transform(next_day_prediction)[0][0])
        
        rmse = float(np.sqrt(np.mean((np.array(actual_prices) - np.array(predictions))**2)))
        
        price_change_percent = float(((next_day_prediction - current_price) / current_price) * 100)
        trend = "up" if price_change_percent > 0 else "down"
        
        response = {
            'symbol': stock_symbol,
            'dates': dates,
            'actual': actual_prices,
            'predicted': predictions,
            'current_price': round(current_price, 2),
            'next_day_prediction': round(next_day_prediction, 2),
            'price_change_percent': round(price_change_percent, 2),
            'trend': trend,
            'accuracy': round(max(0, 100 - (rmse / current_price * 100)), 2),
            'model_status': 'trained' if has_trained_model else 'untrained'
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return jsonify({'error': str(e), 'trace': error_trace})

# stock_info, model_status, train_model remain unchanged...

if __name__ == '__main__':
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
    
    load_model_with_fallback()
    
    if model is not None:
        print(f"Model has {len(model.layers)} layers")
        try:
            model.summary()
        except:
            print("Model summary not available")
    else:
        print("ERROR: Model is None. Creating new model...")
        model = create_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
