# Car-Dheko-Used-Car-Price-Prediction
The goal of this project is to build an accurate and interactive machine learning model for predicting the price of used cars. By leveraging historical data from multiple cities, the model is trained to estimate prices based on key car attributes.

1. Dataset:
```
The dataset consists of used car listings from multiple cities, containing features such as:
    * Make and Model: Manufacturer and model of the car.
    * Year of Manufacture: The year the car was manufactured.
    * Fuel Type: Petrol, Diesel, CNG, etc.
    * Transmission Type: Manual or Automatic.
    * Kilometers Driven: The total distance traveled by the car.
    * Ownership Details: Number of previous owners.
    * Body Type: Hatchback, Sedan, SUV, etc.
    * Price: Target variable representing the selling price of the used car.
```
2. Merging Datasets:
```
* Each dataset was loaded into a Pandas DataFrame.
* A new column named City was added to distinguish the source city.
* All DataFrames were concatenated into a single dataset to form the unified corpus of used car listings.
```
3. Methodology:
```
1. Data Preprocessing:
        i) Handling Unstructured Columns:
                * Some columns stored multiple features in nested JSON-like strings (new_car_detail).
                * We parsed these strings using eval() to extract individual keys: fuel_type, body_type, kilometers, transmission, oem, model, model_year, and price.
        ii) Data Cleaning:
                * String Cleanup: Removed non-numeric characters (e.g., “km”, “Rs.”) from kilometers and price.
                * Type Conversion: Converted numerical fields to float or int as appropriate.
                * Filled missing categorical fields with 'Unknown'.
                * Dropped rows missing critical numeric columns like kilometers, price, or model_year.
        iii) Encoding:
                * Categorical variables were converted into numerical values using Label Encoding.
        iv) Outlier Removal:
                * Used the Interquartile Range (IQR) method on kilometers, price, and model_year.
                * Removed rows with values outside 1.5 * IQR above the 75th percentile or below the 25th percentile.

2. Exploratory Data Analysis (EDA):
       * Descriptive Statistics: Mean, median, standard deviation for kilometers and price.
       * Visualizations: Histograms, boxplots, and scatter plots to check distributions and relationships.
       * Insights: Observed typical mileage thresholds, price distributions, and how features like fuel_type or transmission correlate with price.
```
3. Feature Engineering & Selection:
```
* Key Features: fuel_type, body_type, transmission, oem, model, model_year, kilometers, owner_number were retained based on EDA and domain knowledge.
* Transformations:
      - One-Hot Encoding for categorical columns (fuel_type, body_type, transmission, oem, model).
      - Standard Scaling for numeric columns (kilometers, model_year, owner_number).
```
4.  Model Development:
```
* Train-Test Split: We split the data into 80% training and 20% testing (or 70-30) to gauge generalization performance.
* Model Selection: Explored multiple regressor algorithms (Linear Regression, Random Forest, Gradient Boosting).
* Chosen Model: Random Forest Regressor showed robust performance in terms of MAE, MSE, and R-squared.
* Hyperparameters: We used n_estimators=100 and random_state=42. These were chosen after initial experimentation balancing speed and accuracy.

Models Used:
1. Linear Regression:
       * A simple and interpretable model, but performed poorly due to non-linearity in data.
2. Random Forest Regressor (Final Choice):
       * A powerful ensemble learning model chosen for its superior performance in handling non-linear relationships and feature importance evaluation.
3. Gradient Boosting Regressor:
       * Often yields high predictive accuracy, especially on structured tabular data. Requires careful tuning of learning rate and number of estimators to avoid overfitting.
```
5. Model Evaluation:
```
The models were evaluated using:
        * Mean Absolute Error (MAE): Measures the average magnitude of errors in predictions.
        * Mean Squared Error (MSE): Penalizes larger errors more heavily.
        * R-squared Score (R²): Represents the proportion of variance explained by the model.

Final Model Performance:
        * MAE: 14,700
        * MSE: 5,500,000,000
        * R² Score: 0.51 (indicating a good fit of the model to the data)
```
6. Deployment Using Streamlit:
```
A Streamlit web application was developed to enable real-time user interaction:
        * User-friendly interface:
              - Drop-down menus and input fields for car attributes.
        * Prediction Mechanism:
              - User inputs are transformed and passed to the trained Random Forest model.
              - The estimated price is displayed instantly to the user.
        * Scalability:
              - The app is ready for deployment on cloud platforms like AWS, Heroku, or Streamlit Sharing.
```
7. Results & Conclusion:
```
* The Random Forest Regressor outperformed other models in terms of accuracy and robustness.
* The Streamlit application provides a seamless experience for users to estimate used car prices based on historical data.
* The project successfully demonstrates the end-to-end process of data cleaning, feature selection, model development, hyperparameter tuning, and deployment.
```
8. Future Enhancements:
```
* Integration of location-based pricing to capture market trends.
* Deployment on cloud platforms for broader accessibility.
* Incorporation of advanced models like XGBoost or Neural Networks for further improvements in accuracy.
```







