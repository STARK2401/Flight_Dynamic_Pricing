from flask import Flask, request, jsonify
import pickle
import mysql.connector
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim 
from geopy.distance import great_circle
import os
from datetime import datetime
from mysql.connector import Error

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Flight Price Prediction API! Use /predict_price to get price predictions."

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="musounohitotachi",
            database="airline_project"
        )
        return conn
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return None

# importing model
model_path = os.path.join("D:", "Downloads", "flight_dynamic_pricing_model.pkl")
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# function to get coords
geolocator = Nominatim(user_agent="flight_dynamic_pricing")
def get_coords(location):
    try:
        loc = geolocator.geocode(location)
        if loc:
            return loc.latitude, loc.longitude
        else:
            return np.nan, np.nan
    except:
        return np.nan, np.nan

# function to get the distance
def get_distance(source, destination):
    source_lat, source_lon = get_coords(source)
    dest_lat, dest_lon = get_coords(destination)
    distance = great_circle((source_lat, source_lon), (dest_lat, dest_lon)).km if not (np.isnan(source_lat) or np.isnan(source_lon) or np.isnan(dest_lat) or np.isnan(dest_lon)) else np.nan
    return distance

# function to preprocess data
def preprocess(test_df):
    test_df = pd.DataFrame([test_df])
    final_columns = ['month', 'day', 'remaining_seats', 'booking_lead_time', 'distance',
                     'flight_name_AirAsia', 'flight_name_AirIndia', 'flight_name_Emirates',
                     'flight_name_IndiGo', 'flight_name_SpiceJet', 'seat_type_Business',
                     'seat_type_Economy', 'seat_type_First Class']
    df_test = pd.get_dummies(test_df, columns=['flight_name', 'seat_type'])
    
    for column in final_columns:
        if column not in df_test.columns:
            df_test[column] = False
    df_test = df_test[final_columns]
    
    return df_test

# fetch data
def fetch_data(flight_id, seat_type, remaining_seats):
    conn = get_db_connection()
    if conn is None:
        return None, "Connection Failed"
    try:
        cursor = conn.cursor(dictionary=True)
        query = """SELECT flight_name, source, destination, flight_date
                   FROM flight
                   WHERE unique_id=%s"""
        cursor.execute(query, (flight_id,))
        result = cursor.fetchone()

        if not result:
            return None, "Flight not Found"
        else:
            distance = get_distance(result['source'], result['destination'])
            flight_date = datetime.strptime(result['flight_date'], '%Y-%m-%d')
            current_date = datetime.now()
            booking_lead_time = (flight_date - current_date).days

            model_input = {
                'month': flight_date.month,
                'day': flight_date.day,
                'remaining_seats': remaining_seats,
                'booking_lead_time': booking_lead_time,
                'distance': distance,
                'flight_name': result['flight_name'],
                'seat_type': seat_type
            }

            return model_input, None
        
    except mysql.connector.Error as e:
        return None, str(e)
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()

# data prediction
def predict_price(model_input):
    preprocessed_data = preprocess(model_input)
    predicted_price = model.predict(preprocessed_data)[0]
    return float(predicted_price)

@app.route('/predict_price', methods=['POST'])
def price_prediction_api():
    try:
        data = request.json
        flight_id = data.get('flight_id')
        seat_type = data.get('seat_type')
        remaining_seats = data.get('remaining_seats')

        if not all([flight_id, seat_type, remaining_seats]):
            return jsonify({"error": "Missing required fields"}), 400
        
        
        if remaining_seats==0:
            return jsonify({"error": f"No seats left  {error}"}), 400


        model_input, error = fetch_data(flight_id, seat_type, remaining_seats)
        
        if model_input is None:
            return jsonify({"error": f"Cannot retrieve data: {error}"}), 500
        
        if remaining_seats==0:
            return jsonify({"error": f"No seats left  {error}"}), 400

        
        predicted_price = predict_price(model_input)
        return jsonify({"predicted_price": predicted_price})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=True)