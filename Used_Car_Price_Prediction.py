import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Step 1: Load and Merge Datasets
def load_data():
    file_paths = {
        "Bangalore": "bangalore_cars.xlsx",
        "Chennai": "chennai_cars.xlsx",
        "Delhi": "delhi_cars.xlsx",
        "Hyderabad": "hyderabad_cars.xlsx",
        "Jaipur": "jaipur_cars.xlsx",
        "Kolkata": "kolkata_cars.xlsx",
    }

    dfs = []
    for city, path in file_paths.items():
        df = pd.read_excel(path)
        df["City"] = city  # Add city column
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save progress
    with open("cleaned_data.pkl", "wb") as file:
        pickle.dump(merged_df, file)
    return merged_df

# Step 2: Data Preprocessing
def preprocess_data(df):
    def extract_features(row, key):
        try:
            data = eval(row)  # Convert string representation of dictionary to dictionary
            return data.get(key, None)
        except:
            return None

    df['fuel_type'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'ft'))
    df['body_type'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'bt'))
    df['kilometers'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'km'))
    df['transmission'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'transmission'))
    df['owner_number'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'ownerNo'))
    df['oem'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'oem'))
    df['model'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'model'))
    df['model_year'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'modelYear'))
    df['price'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'price'))

    df['kilometers'] = df['kilometers'].astype(str).str.replace(r'[^\d]', '', regex=True).astype(float)
    df['price'] = df['price'].astype(str).str.replace(r'[^\d]', '', regex=True).astype(float)

    df.drop(columns=['new_car_detail', 'new_car_overview', 'new_car_feature', 'new_car_specs', 'car_links'], inplace=True)

    df.fillna({'fuel_type': 'Unknown', 'body_type': 'Unknown', 'transmission': 'Unknown', 'oem': 'Unknown', 'model': 'Unknown'}, inplace=True)
    df.dropna(subset=['kilometers', 'price', 'model_year'], inplace=True)
    
    # Save progress
    with open("preprocessed_data.pkl", "wb") as file:
        pickle.dump(df, file)
    return df

# Step 3: Exploratory Data Analysis (EDA)
def perform_eda(df):
    # Exploratory Data Analysis (EDA)
    print("Basic Statistical Summary:\n", df.describe())
    print("\nData Types:\n", df.dtypes)

    plt.figure(figsize=(10, 5))
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title('Distribution of Used Car Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df['kilometers'], y=df['price'])
    plt.title('Kilometers Driven vs. Price')
    plt.xlabel('Kilometers Driven')
    plt.ylabel('Price')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['fuel_type'], y=df['price'])
    plt.title('Price Distribution by Fuel Type')
    plt.xlabel('Fuel Type')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

# Step 4: Model Training and Evaluation
def train_model(df):
    features = ['fuel_type', 'body_type', 'transmission', 'owner_number', 'oem', 'model', 'model_year', 'kilometers']
    target = 'price'

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[features[:6]])  # Categorical Features
    encoded_feature_names = encoder.get_feature_names_out(features[:6])

    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['model_year', 'kilometers', 'owner_number']])
    scaled_df = pd.DataFrame(scaled_features, columns=['model_year', 'kilometers', 'owner_number'])

    # Ensure consistent column order
    X = pd.concat([encoded_df, scaled_df], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")

    # Save model, encoder, scaler, and feature order
    with open("car_price_model.pkl", "wb") as file:
        pickle.dump(model, file)
    with open("encoder.pkl", "wb") as file:
        pickle.dump(encoder, file)
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    with open("feature_names.pkl", "wb") as file:
        pickle.dump(list(X.columns), file)  # Store feature order

    print("Model, Encoder, Scaler, and Feature Names saved successfully.")
    return model, encoder, scaler, list(X.columns)


# Step 5: Model Selection and Feature Engineering
def feature_engineering(df):
    # Convert categorical features to numerical using Label Encoding or One-Hot Encoding
    from sklearn.preprocessing import LabelEncoder

    # Identify non-numeric columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # Feature Selection
    correlation = df.corr()['price'].abs().sort_values(ascending=False)
    selected_features = correlation[correlation > 0.1].index.tolist()
    selected_features.remove('price')

    print("Selected Features:", selected_features)

    # Splitting data into train and test sets
    X = df[selected_features]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Development
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} Performance:")
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))
        print("-"*40)

    print("Feature selection and model development completed.")

# Step 6: Model Development with Hyperparameter Tuning
def hyper_tuning(df):    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Model Evaluation
    y_pred = best_model.predict(X_test)
    print("Optimized Random Forest Performance:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    print("Best Parameters:", grid_search.best_params_)

    print("Hyperparameter tuning and optimization completed.")

import os

def deploy_app():
    required_files = ["car_price_model.pkl", "encoder.pkl", "scaler.pkl", "feature_names.pkl"]
    if not all(os.path.exists(f) for f in required_files):
        st.error("Model files are missing. Retrain and save the model before deploying.")
        return

    # Load the model, encoder, scaler, and feature order
    model = pickle.load(open("car_price_model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))  # Load stored feature names

    st.title("Used Car Price Prediction")
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Unknown"])
    body_type = st.selectbox("Body Type", ["Hatchback", "Sedan", "SUV", "Unknown"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Unknown"])
    owner_number = st.slider("Number of Owners", 1, 5, 1)
    oem = st.text_input("OEM (Manufacturer)", "Unknown")  # Added OEM input
    model_name = st.text_input("Model Name", "Unknown")  # Added Model Name input
    model_year = st.number_input("Model Year", 2000, 2025, 2015)
    kilometers = st.number_input("Kilometers Driven", 0, 300000, 50000)

    if st.button("Predict Price"):
        input_data = pd.DataFrame([[fuel_type, body_type, transmission, owner_number, oem, model_name, model_year, kilometers]],
                                  columns=['fuel_type', 'body_type', 'transmission', 'owner_number', 'oem', 'model', 'model_year', 'kilometers'])

        # Ensure feature order consistency
        categorical_features = ['fuel_type', 'body_type', 'transmission', 'owner_number', 'oem', 'model']

        # Use the exact same feature order as used during training
        input_encoded = encoder.transform(input_data[categorical_features])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_features))


        # Scale numerical features
        input_scaled = scaler.transform(input_data[['model_year', 'kilometers', 'owner_number']])
        input_scaled_df = pd.DataFrame(input_scaled, columns=['model_year', 'kilometers', 'owner_number'])

        # Merge encoded and scaled features
        input_final = pd.concat([input_encoded_df, input_scaled_df], axis=1)

        # Ensure consistent feature order
        input_final = input_final.reindex(columns=feature_names, fill_value=0)


        # Reorder columns to match the order during training
        input_final = input_final[feature_names]

        # Make prediction
        prediction = model.predict(input_final)
        st.success(f"Estimated Price: {prediction[0]:,.2f} INR")

# Execute workflow
# Load and preprocess data
data = load_data()
data = preprocess_data(data)

# Train the model and ensure all outputs are captured
if st.button("Train Model"):
    model, encoder, scaler, feature_names = train_model(data)
    st.write("Model trained!")

# Deploy the app
deploy_app()
