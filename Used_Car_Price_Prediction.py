import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import re
import os
from scipy.stats import zscore

def load_data():
    # Paths to city-specific Excel files
    file_paths = {
        "Bangalore": "bangalore_cars.xlsx",
        "Chennai":   "chennai_cars.xlsx",
        "Delhi":     "delhi_cars.xlsx",
        "Hyderabad": "hyderabad_cars.xlsx",
        "Jaipur":    "jaipur_cars.xlsx",
        "Kolkata":   "kolkata_cars.xlsx",
    }

    # List to collect each city's DataFrame
    dfs = []
    for city, path in file_paths.items():
        df_temp = pd.read_excel(path)
        df_temp["City"] = city
        dfs.append(df_temp)

    # Concatenating all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

def extract_price(price_str):

    # Checking for 'Lakh'
    if isinstance(price_str, str):
        match_lakh = re.search(r'([\d,.]+)\s*Lakh', price_str)
        if match_lakh:
            return float(match_lakh.group(1).replace(',', '')) * 100000
        
        # Checking for 'Cr' or 'crore'
        match_cr = re.search(r'([\d,.]+)\s*(cr|crore)', price_str, flags=re.IGNORECASE)
        if match_cr:
            return float(match_cr.group(1).replace(',', '')) * 10000000

        # General numeric fallback
        numeric_match = re.search(r'([\d.]+)', price_str)
        if numeric_match:
            return float(numeric_match.group(1))
    
    # Returning None if no pattern is matched
    return None

def extract_features(row, key):
    try:
        data = eval(row)  
        return data.get(key, None)
    except:
        return None

# Nested data extraction from 'new_car_detail' column
def preprocess_data(df):
    def extract_features(row, key):
        try:
            # parsing the string as dictionary
            data = eval(row)
            return data.get(key, None)
        except:
            return None

    # Extracting columns from JSON
    df['fuel_type']    = df['new_car_detail'].apply(lambda x: extract_features(x, 'ft'))
    df['body_type']    = df['new_car_detail'].apply(lambda x: extract_features(x, 'bt'))
    df['kilometers']   = df['new_car_detail'].apply(lambda x: extract_features(x, 'km'))
    df['transmission'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'transmission'))
    df['owner_number'] = df['new_car_detail'].apply(lambda x: extract_features(x, 'ownerNo'))
    df['oem']          = df['new_car_detail'].apply(lambda x: extract_features(x, 'oem'))
    df['model']        = df['new_car_detail'].apply(lambda x: extract_features(x, 'model'))
    df['model_year']   = df['new_car_detail'].apply(lambda x: extract_features(x, 'modelYear'))
    df['price']        = df['new_car_detail'].apply(lambda x: extract_features(x, 'price'))

    # Convert 'price' strings to numeric
    df['price'] = df['price'].apply(extract_price)

    # Cleaning 'kilometers' to remove non-numeric characters
    df['kilometers'] = df['kilometers'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df['kilometers'] = pd.to_numeric(df['kilometers'], errors='coerce')

    # Creating 'car_age'
    df['car_age'] = 2025 - df['model_year']

    # Dropping irrelevant columns (if they exist)
    df.drop(columns=['new_car_detail','new_car_overview','new_car_feature','new_car_specs','car_links'],
            errors='ignore', inplace=True)

    # Imputing missing categorical columns with 'Unknown'
    df.fillna({'fuel_type': 'Unknown', 'body_type': 'Unknown', 'transmission': 'Unknown', 'oem': 'Unknown', 'model': 'Unknown'}, inplace=True)

    # Converting 'owner_number' to numeric, drop rows where it's invalid
    df['owner_number'] = pd.to_numeric(df['owner_number'], errors='coerce')

    # Dropping rows where essential numeric columns are missing
    df.dropna(subset=['kilometers', 'price', 'model_year'], inplace=True)

    # Dropping 'model_year' now that we have 'car_age'
    df.drop(columns=['model_year'], inplace=True)

    # Further data constraints: removing unrealistic values
    df = df[df['kilometers'] < 500000]     # cars with < 500,000 kms
    df = df[df['price'] < 1.5e7]           # cars with < 1.5 crore rupees

    # Outlier removal using IQR for 'price'
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    price_filter = (df["price"] >= (Q1 - 1.0 * IQR)) & (df["price"] <= (Q3 + 1.0 * IQR))
    df = df[price_filter]

    return df

# --------------------------
# EDA Plots to Display in Streamlit
# --------------------------
def show_eda_plots(df):
    # 1. Distribution of Price
    fig, ax = plt.subplots()
    ax.hist(df["price"], bins=30)
    ax.set_title("Distribution of Used Car Prices")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 2. Scatter: Price vs. Kilometers
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["kilometers"], df["price"])
    ax2.set_title("Price vs. Kilometers Driven")
    ax2.set_xlabel("Kilometers Driven")
    ax2.set_ylabel("Price")
    st.pyplot(fig2)

    # 3. Correlation Heatmap
    corr = df.corr(numeric_only=True)
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", center=0, linewidths=0.5, ax=ax3)
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)

    # 4. Boxplot: Car Age vs. Price
    fig4, ax4 = plt.subplots()
    sns.boxplot(x=df["car_age"], y=df["price"], ax=ax4)
    ax4.set_title("Boxplot: Car Age vs. Price")
    ax4.set_xlabel("Car Age (Years)")
    ax4.set_ylabel("Price")
    st.pyplot(fig4)


def split_and_transform(df):

    # Identifying categorical and numeric columns
    cat_cols = ['City','fuel_type','body_type','transmission','oem','model']
    num_cols = ['car_age','kilometers','owner_number']

    # Target variable
    target   = 'price'

    X = df[cat_cols + num_cols].copy()
    y = df[target].copy()
    
    # OneHotEncode categorical columns
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = ohe.fit_transform(X[cat_cols])

    # Retrieving the categorical feature names after encoding
    cat_names = ohe.get_feature_names_out(cat_cols) 

    # Saving the categorical feature names for later use in prediction
    with open("cat_names.pkl", "wb") as file:
        pickle.dump(cat_names, file)

    # Creating a DataFrame for the encoded cats
    X_cat = pd.DataFrame(X_cat, columns=cat_names, index=X.index)

    # Scaling numeric columns
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[num_cols])
    X_num = pd.DataFrame(X_num, columns=num_cols, index=X.index)

    # Combining encoded categorical + scaled numeric
    X_final = pd.concat([X_cat, X_num], axis=1)
    feature_names = list(X_final.columns)  # Save feature names

    # Saving the feature names for later use in prediction
    with open("feature_names.pkl", "wb") as file:
        pickle.dump(feature_names, file)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, ohe, scaler, cat_names, num_cols


def train_model(X_train, X_test, y_train, y_test):

    # Initializing candidate models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    performance = {}

    # Training each model and evaluating
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)

        performance[model_name] = {"MAE": mae, "MSE": mse, "R2": r2}

    # Converting performance dict to DataFrame
    perf_df = pd.DataFrame(performance).T

    # Selecting best baseline model by R2
    best_baseline = perf_df["R2"].idxmax()
    chosen_model  = models[best_baseline]

    # If the best baseline is a model that can be tuned, doing so
    best_model = chosen_model
    if best_baseline in ["Random Forest", "Gradient Boosting"]:
        if best_baseline == "Random Forest":
            param_grid = {
                "n_estimators":      [50, 100],
                "max_depth":         [None, 10],
                "min_samples_split": [2, 5],
            }
        else: 
            # Gradient Boosting
            param_grid = {
                "n_estimators":  [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth":     [3, 5]
            }

        # Grid Search for hyperparameter tuning
        grid_search = GridSearchCV(chosen_model, param_grid, cv=3,
                                   scoring='neg_mean_absolute_error',
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        perf_df.loc[best_baseline,"Best_Params"] = str(grid_search.best_params_)

    # Evaluating the best model (baseline or tuned) on the test set
    y_pred_best = best_model.predict(X_test)
    mae_best = mean_absolute_error(y_test, y_pred_best)
    mse_best = mean_squared_error(y_test, y_pred_best)
    r2_best  = r2_score(y_test, y_pred_best)

    final_metrics = {
        "Best Model": best_baseline,
        "MAE":        mae_best,
        "MSE":        mse_best,
        "R2":         r2_best
    }

    return best_model, perf_df, final_metrics

def save_artifacts(model, ohe, scaler, cat_names, num_cols):

    # Saving the model
    with open("car_price_model.pkl", "wb") as file:
        pickle.dump(model, file)

    # Saving the OneHotEncoder
    with open("encoder.pkl", "wb") as file:
        pickle.dump(ohe, file)

    # Saving the StandardScaler
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # Combining feature names for reindexing at prediction
    feature_order = list(cat_names) + list(num_cols)
    with open("feature_names.pkl", "wb") as file:
        pickle.dump(feature_order, file)
    print("Saved model, encoder, scaler, and feature names successfully.")

def deploy_app():
    st.title("Used Car Price Prediction")

    # Required files
    required_files = ["car_price_model.pkl", "encoder.pkl", "scaler.pkl", "feature_names.pkl"]
    if not all(os.path.exists(f) for f in required_files):
        st.error("Model files are missing. Retrain and save the model before deploying.")
        return
    
    # Loading the model and preprocessing objects
    with open("car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        ohe = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # Collecting user inputs
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Unknown"])
    body_type = st.selectbox("Body Type", ["Hatchback", "Sedan", "SUV", "Unknown"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Unknown"])
    oem = st.text_input("OEM (Manufacturer)", "Unknown")
    model_name = st.text_input("Model Name", "Unknown")
    city = st.selectbox("City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Unknown"])
    model_year = st.number_input("Model Year", 2000, 2025, 2015)
    kilometers = st.number_input("Kilometers Driven", 0, 300000, 50000)
    owner_num = st.selectbox("Number of Owners", [1, 2, 3, 4, 5])

    if st.button("Predict Price"):

        # Computing derived feature: car_age
        car_age = 2025 - model_year

        # Building input dictionary for inference
        input_dict = {
            "City": [city],
            "fuel_type": [fuel_type],
            "body_type": [body_type],
            "transmission": [transmission],
            "oem": [oem],
            "model": [model_name],
            "owner_number": [int(owner_num)], 
            "car_age": [car_age],
            "kilometers": [kilometers]
        }
        input_df = pd.DataFrame(input_dict)

        # Separating categorical and numeric columns
        cat_cols = ["City", "fuel_type", "body_type", "transmission", "oem", "model"]
        num_cols = ["car_age", "kilometers", "owner_number"] 

        # Loading the encoder's expected feature names
        with open("cat_names.pkl", "rb") as file:
            cat_names = pickle.load(file)  

        # Encoding categorical features
        X_cat = ohe.transform(input_df[cat_cols])
        X_cat = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(cat_cols))
        X_cat = X_cat.reindex(columns=cat_names, fill_value=0)  

        # Ensuring numeric columns are properly typed, then scaling
        input_df["owner_number"] = pd.to_numeric(input_df["owner_number"], errors="coerce")
        X_num = scaler.transform(input_df[num_cols])
        X_num = pd.DataFrame(X_num, columns=num_cols)

        # Combining processed categorical + numeric
        X_final = pd.concat([X_cat, X_num], axis=1)

        # Reindexing to ensure correct column order
        with open("feature_names.pkl", "rb") as file:
            feature_names = pickle.load(file) 

        X_final = X_final.reindex(columns=feature_names, fill_value=0)  

        # Predicting with the loaded model
        pred = model.predict(X_final)
        st.success(f"Estimated Price: ₹{pred[0]:,.2f}")

# Main Streamlit Workflow
def main():
    st.sidebar.title("Car Dheko - Used Car Price Prediction")
    option = st.sidebar.selectbox("Choose a step", ["Preprocess and Train", "Deploy App"])
    
    if option == "Preprocess and Train":
        st.write("### Step 1: Loading & Preprocessing Data")
        df = load_data()
        df = preprocess_data(df)
        st.write(f"Data Shape after Preprocessing: {df.shape}")

        if st.checkbox("Show EDA Plots"):
            show_eda_plots(df)

        st.write("### Step 2: Train-Test Split, Encoding & Scaling")
        X_train, X_test, y_train, y_test, ohe, scaler, cat_names, num_cols = split_and_transform(df)
        st.write("Training Shape:", X_train.shape, "Test Shape:", X_test.shape)

        st.write("### Step 3: Train Model")
        best_model, perf_df, final_metrics = train_model(X_train, X_test, y_train, y_test)

        st.write("#### Baseline Model Performance")
        st.table(perf_df[["MAE","MSE","R2"]])
        st.write(f"**Best baseline model based on R²**: {final_metrics['Best Model']}")

        st.write("#### Final Model Performance (Tuned)")
        st.write(f"**MAE**: {final_metrics['MAE']:.2f}")
        st.write(f"**MSE**: {final_metrics['MSE']:.2f}")
        st.write(f"**R²**:  {final_metrics['R2']:.2f}")

        if st.button("Save Model"):
            save_artifacts(best_model, ohe, scaler, cat_names, num_cols)
            st.success("Model artifacts saved.")

    elif option == "Deploy App":
        deploy_app()

if __name__ == "__main__":
    main()
