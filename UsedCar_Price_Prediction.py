import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and preprocessing steps
model = joblib.load('car_price_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scalers = joblib.load('scalers.pkl')

# Load dataset for filtering and identifying similar data
data = pd.read_csv('car_dheko_cleaned_dataset_Raw.csv')

# Set pandas option to handle future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Features used for training
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats', 'car_age', 'brand_popularity', 'mileage_normalized']

# Function to filter data based on user selections
def filter_data(oem=None, model=None, body_type=None, fuel_type=None, seats=None):
    filtered_data = data.copy()
    if oem:
        filtered_data = filtered_data[filtered_data['oem'] == oem]
    if model:
        filtered_data = filtered_data[filtered_data['model'] == model]
    if body_type:
        filtered_data = filtered_data[filtered_data['bt'] == body_type]
    if fuel_type:
        filtered_data = filtered_data[filtered_data['ft'] == fuel_type]
    # if seats:
    #     filtered_data = filtered_data[filtered_data['Seats'] == seats]
    return filtered_data

# Preprocessing function for user input
def preprocess_input(df):
    df['car_age'] = 2024 - df['modelYear']
    brand_popularity = data.groupby('oem')['price'].mean().to_dict()
    df['brand_popularity'] = df['oem'].map(brand_popularity)
    df['mileage_normalized'] = df['mileage'] / df['car_age']

    # Apply label encoding
    for column in ['ft', 'bt', 'transmission', 'oem', 'model', 'variantName', 'City']:
        if column in df.columns and column in label_encoders:
            df[column] = df[column].apply(lambda x: label_encoders[column].transform([x])[0])

    # Apply min-max scaling
    for column in ['km', 'ownerNo', 'modelYear']:
        if column in df.columns and column in scalers:
            df[column] = scalers[column].transform(df[[column]])

    return df

# Streamlit Application
# Streamlit App Interface Design

st.image("generic-red-sports-car_image.jpg", use_container_width=True)

st.title("Used Car Price Prediction")

# Sidebar for user inputs
st.sidebar.header('Input Car Features')

 # Visual feedback function
def check_field_filled(field_value, field_name):
 if field_value:
    st.sidebar.success(f"{field_name} selected")
 else:
    st.sidebar.error(f"{field_name} is required")


# Get user inputs 

# Get user inputs
oems = data['oem'].unique()
selected_oem = st.sidebar.selectbox('Manufacturer (OEM)', list(oems))
check_field_filled(selected_oem, 'OEM')
filtered_data = filter_data(oem=selected_oem)
models = filtered_data['model'].unique()
selected_model = st.sidebar.selectbox('Car Model', list(models))
check_field_filled(selected_model, 'Model')
filtered_data = filter_data(oem=selected_oem, model=selected_model)
body_type = filtered_data['bt'].unique()
fuel_type = filtered_data['ft'].unique()
variants = filtered_data['variantName'].unique()
mileages = filtered_data['mileage'].unique()
seats = filtered_data['Seats'].unique()
cities = data['City'].unique()
body_type = st.sidebar.selectbox('Body Type', list(body_type))
check_field_filled(body_type, 'Body Type')
fuel_type = st.sidebar.selectbox('Fuel Type', list(fuel_type))
check_field_filled(fuel_type, 'Fuel Type')
transmission = st.sidebar.selectbox('Transmission Type', list(data['transmission'].unique()))
check_field_filled(transmission, 'Transmission')
city = st.sidebar.selectbox('City', list(cities))
check_field_filled(city, 'City')
mileage = st.sidebar.selectbox('Mileage (KMs)', sorted(list(mileages)))
check_field_filled(mileage, 'Mileage')
seat_count = st.sidebar.selectbox('Seat Count', sorted(list(seats)))
check_field_filled(seat_count, 'Seat Count')
km =st.sidebar.number_input('Kilometers Driven', min_value=0, max_value=500000,
value=10000)
check_field_filled(km, 'Kilometers Driven')
ownerNo = st.sidebar.number_input('Number of Previous Owners', min_value=0,
max_value=10, value=1)
check_field_filled(ownerNo, 'Number of Owners')
modelYear = st.sidebar.number_input('Year of Manufacture', min_value=1980,
max_value=2024, value=2015)
check_field_filled(modelYear, 'Year of Manufacture')
selected_variant = st.sidebar.selectbox('Variant Name', list(variants))
check_field_filled(selected_variant, 'Variant Name')

# Create a DataFrame for user input
user_input_data = {
 'ft': [fuel_type],
 'bt': [body_type],
 'km': [km],
 'transmission': [transmission],
 'ownerNo': [ownerNo],
 'oem': [selected_oem],
 'model': [selected_model],
 'modelYear': [modelYear],
 'variantName': [selected_variant],
 'City': [city],
 'mileage': [mileage],
 'Seats': [seat_count],
 'car_age': [2024- modelYear],
 'brand_popularity': [data.groupby('oem')['price'].mean().to_dict().get(selected_oem)],
'mileage_normalized': [mileage / (2024- modelYear)]
 }
user_df = pd.DataFrame(user_input_data)
 # Ensure the columns are in the correct order and match the trained model's features
user_df = user_df[features]
 # Preprocess user input data
user_df = preprocess_input(user_df)
# Check if all fields are filled before allowing prediction
fields_filled = user_df.notnull().all().all()
if fields_filled:
    st.sidebar.success("All fields filled! Press 'Predict' to see the estimated price.")
else:
    st.sidebar.error("Please fill all required fields.")

# Button to trigger prediction
if st.sidebar.button('Predict'):
    if fields_filled:
        try:
        # Make prediction
            prediction = model.predict(user_df)
            st.write(f'Predicted Price: ₹{prediction[0]:,.2f}')
            st.write(f'Car Age: {user_df["car_age"][0]} years')
            st.write(f'Mileage Normalized: {user_df["mileage_normalized"][0]:,.2f}')
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        missing_fields = [col for col in user_df.columns if user_df[col].isnull().any()]
        st.error(f"Missing fields: {', '.join(missing_fields)}. Please fill all required fields.")


































