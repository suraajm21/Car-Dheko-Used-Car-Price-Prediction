import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import os

# Step 1: Loading and Merging Datasets
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
        df["City"] = city  # Adding city column
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

# Step 2: Data Preprocessing
def preprocess_data(df):
    def extract_features(row, key):
        try:
            data = eval(row)  # Converting string representation of dictionary to dictionary
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

    # Cleaning up strings
    df['kilometers'] = df['kilometers'].astype(str).str.replace(r'[^\d]', '', regex=True).astype(float)
    df['price'] = df['price'].astype(str).str.replace(r'[^\d]', '', regex=True).astype(float)

    # Dropping unneeded columns
    df.drop(columns=['new_car_detail','new_car_overview','new_car_feature','new_car_specs','car_links'],
            errors='ignore', inplace=True)

    # Handling missing values
    df.fillna({
        'fuel_type': 'Unknown',
        'body_type': 'Unknown',
        'transmission': 'Unknown',
        'oem': 'Unknown',
        'model': 'Unknown'
    }, inplace=True)

    # Dropping rows if these crucial columns are missing
    df.dropna(subset=['kilometers', 'price', 'model_year'], inplace=True)

    # Removing outliers only on real numeric columns:
    numeric_cols = ['kilometers','price','model_year']
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    filter_ = ~(
        (df[numeric_cols] < (Q1 - 1.5 * IQR)) |
        (df[numeric_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
    df = df[filter_].copy()

    # Returning cleaned dataframe (no final encoding or scaling yet)
    return df

# Step 3: Train-Test Split and Encoding/Scaling
def split_and_transform(df):
    features = ['fuel_type','body_type','transmission','oem','model','model_year','kilometers','owner_number']
    target = 'price'

    X = df[features].copy()
    y = df[target].copy()

    # Converting all string columns to categories via OneHotEncoder
    cat_cols = ['fuel_type','body_type','transmission','oem','model']
    num_cols = ['model_year','kilometers','owner_number']

    # One-Hot encoding the categorical columns
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = ohe.fit_transform(X[cat_cols])
    cat_names = ohe.get_feature_names_out(cat_cols)
    X_cat = pd.DataFrame(X_cat, columns=cat_names, index=X.index)

    # Scaling numeric columns
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[num_cols])
    X_num = pd.DataFrame(X_num, columns=num_cols, index=X.index)

    # Merging back
    X_final = pd.concat([X_cat, X_num], axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, ohe, scaler, cat_names, num_cols

# Step 4: Model Training
def train_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae}, MSE: {mse}, R2: {r2}")

    return model

# Step 5: Saveing Artifacts
def save_artifacts(model, ohe, scaler, cat_names, num_cols):
    with open("car_price_model.pkl", "wb") as file:
        pickle.dump(model, file)
    with open("encoder.pkl", "wb") as file:
        pickle.dump(ohe, file)
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # The final feature order is all OHE cat columns + numeric columns
    feature_order = list(cat_names) + list(num_cols)
    with open("feature_names.pkl", "wb") as file:
        pickle.dump(feature_order, file)

    print("Saved model, encoder, scaler, and feature names successfully.")

# Step 6: Streamlit Deployment
def deploy_app():
    st.title("Used Car Price Prediction")
    # Checking for all required files
    required_files = ["car_price_model.pkl", "encoder.pkl", "scaler.pkl", "feature_names.pkl"]
    if not all(os.path.exists(f) for f in required_files):
        st.error("Model files are missing. Retrain and save the model before deploying.")
        return

    model = pickle.load(open("car_price_model.pkl","rb"))
    ohe = pickle.load(open("encoder.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    feature_names = pickle.load(open("feature_names.pkl","rb"))

    # Building the form
    fuel_type = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG","LPG","Electric","Unknown"])
    body_type = st.selectbox("Body Type", ["Hatchback","Sedan","SUV","Unknown"])
    transmission = st.selectbox("Transmission", ["Manual","Automatic","Unknown"])
    oem = st.text_input("OEM (Manufacturer)", "Unknown")
    model_name = st.text_input("Model Name", "Unknown")
    model_year = st.number_input("Model Year", 2000, 2025, 2015)
    kilometers = st.number_input("Kilometers Driven", 0, 300000, 50000)
    owner_number = st.slider("Number of Owners", 1, 5, 1)

    if st.button("Predict Price"):
        # Constructing a single-row dataframe with raw inputs
        input_dict = {
            'fuel_type': [fuel_type],
            'body_type': [body_type],
            'transmission': [transmission],
            'oem': [oem],
            'model': [model_name],
            'model_year': [model_year],
            'kilometers': [kilometers],
            'owner_number': [owner_number]
        }
        input_df = pd.DataFrame(input_dict)

        # Splitting columns into cat vs numeric, in the same way as training
        cat_cols = ['fuel_type','body_type','transmission','oem','model']
        num_cols = ['model_year','kilometers','owner_number']

        X_cat = ohe.transform(input_df[cat_cols])
        X_cat = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(cat_cols))

        X_num = scaler.transform(input_df[num_cols])
        X_num = pd.DataFrame(X_num, columns=num_cols)

        # Final input
        X_final = pd.concat([X_cat, X_num], axis=1)

        # Reindexing columns to match training order
        X_final = X_final.reindex(columns=feature_names, fill_value=0)

        # Predicting
        pred = model.predict(X_final)
        st.success(f"Estimated Price: {pred[0]:,.2f} INR")

# --------------------------
# Main pipeline
# --------------------------
def main():
    st.sidebar.title("Car Dheko - Used Car Price Prediction")
    option = st.sidebar.selectbox("Choose a step", ["Preprocess and Train", "Deploy App"])
    
    if option == "Preprocess and Train":
        st.write("### Step 1: Loading & Preprocessing Data")
        df = load_data()
        df = preprocess_data(df)
        st.write(f"Data Shape after Preprocessing: {df.shape}")

        st.write("### Step 2: Train-Test Split, Encoding & Scaling")
        X_train, X_test, y_train, y_test, ohe, scaler, cat_names, num_cols = split_and_transform(df)
        st.write("Training Shape:", X_train.shape, "Test Shape:", X_test.shape)

        st.write("### Step 3: Train Model")
        model = train_model(X_train, X_test, y_train, y_test)

        if st.button("Save Model"):
            save_artifacts(model, ohe, scaler, cat_names, num_cols)
            st.success("Model artifacts saved.")

    elif option == "Deploy App":
        deploy_app()

if __name__ == "__main__":
    main()
