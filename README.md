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
2. Methodology:
```
1. Data Preprocessing:
        i) Data Cleaning:
                * Missing values were handled using mode imputation for categorical variables and median imputation for numerical variables.
        ii) Standardization:
                * String units like 'kms' and 'Rs.' were removed to convert data into proper numerical format.
        iii) Encoding:
                * Categorical variables were converted into numerical values using Label Encoding.
        iv) Normalization:
                * Min-Max Scaling was applied to numerical features like kilometers driven and price.
        v) Outlier Removal:
                * The Interquartile Range (IQR) method was used to detect and remove extreme values.

2. Exploratory Data Analysis (EDA):
        - Visualizations:
                * Histograms to analyze the distribution of car prices and kilometers driven.
                * Correlation heatmap to determine relationships between features.
                * Boxplots to identify the impact of categorical features on price.
```
3. Feature Selection:
```
* Features with correlation greater than 0.1 with price were selected for training.
* Domain knowledge was applied to retain meaningful variables.
```
4.  Model Development:
```
1. Linear Regression:
       * A simple and interpretable model, but performed poorly due to non-linearity in data.
2. Random Forest Regressor:
       * A powerful ensemble learning model chosen for its superior performance in handling non-linear relationships and feature importance evaluation.
3. Hyperparameter Tuning:
       * GridSearchCV was used to optimize the following parameters:
                - Number of estimators (n_estimators): 50, 100, 200
                - Maximum depth (max_depth): None, 10, 20
                - Minimum samples per split (min_samples_split): 2, 5, 10
                - Minimum samples per leaf (min_samples_leaf): 1, 2, 4
       * The best model configuration was selected based on cross-validation performance.
```
5. Model Evaluation:
```
The models were evaluated using:
        * Mean Absolute Error (MAE): Measures the average magnitude of errors in predictions.
        * Mean Squared Error (MSE): Penalizes larger errors more heavily.
        * R-squared Score (R²): Represents the proportion of variance explained by the model.

Final Model Performance:
        * MAE: 22,500
        * MSE: 7,900,000,000
        * R² Score: 0.87 (indicating a good fit of the model to the data)
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







