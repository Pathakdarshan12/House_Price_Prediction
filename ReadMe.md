# House Price Prediction App

<img src="https://github.com/">

## Introduction
The House Price Prediction App is a machine learning-based web application developed to predict house prices in India. The purpose of this project is to provide users with a convenient tool to estimate house prices based on various factors such as the number of rooms, bathrooms, balconies, total area, flat type, and city.

## Background
Predicting house prices accurately is crucial for both buyers and sellers in the real estate market. Buyers can use price predictions to make informed decisions about purchasing properties, while sellers can use them to set competitive prices and maximize profits. Machine learning models like XGBoost regression offer a data-driven approach to estimate house prices, considering multiple factors that influence pricing.

## Technologies Used
- Streamlit
- Pandas
- NumPy
- XGBoost
- Matplotlib

## Project Components
### Data Preprocessing
The project begins with data preprocessing, including data cleaning and feature engineering. Features such as the number of rooms, bathrooms, balconies, and total area are scaled and processed to prepare the dataset for model training.

### Model Training
The XGBoost regression model is trained using the preprocessed dataset. Hyperparameters like learning rate, max depth, and number of estimators are tuned to optimize the model's performance. The trained model is saved using the pickle library for later use.

### Streamlit Application
The core of the project is the Streamlit web application, which provides an intuitive user interface for interacting with the house price prediction model. Users can input details about the house, such as the number of rooms, bathrooms, balconies, flat type, city, and total area, and the application predicts the house price based on these inputs.

## Demonstration
The application features a sidebar with options for selecting the input parameters. Users can adjust the sliders and dropdown menus to input their desired house features. Upon clicking the "Predict House Price" button, the application displays the predicted house price along with relevant visualizations, such as mean house prices for each city.

## Results
The House Price Prediction App accurately predicts house prices based on the input parameters provided by the user. The XGBoost regression model achieves high accuracy in estimating house prices, providing valuable insights for both buyers and sellers in the real estate market.

## Conclusion
The House Price Prediction App is a useful tool for estimating house prices in India. By leveraging machine learning techniques, the application provides valuable insights for individuals involved in the real estate market, facilitating informed decision-making and improving overall efficiency.

## References
- [Streamlit Documentation](https://streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## Credentials
Developed by Darshan Pathak

## How to Run the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/house-price-prediction-app.git
   cd house-price-prediction-app
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Interact with the application:**
   Open the provided local URL in your web browser, input the desired house features, and get the predicted house price.

## Directory Structure
- `app.py`: The main Streamlit application file.
- `data/`: Directory containing the dataset.
- `models/`: Directory containing the saved machine learning model.
- `requirements.txt`: File containing the list of dependencies.
- `README.md`: Project documentation file (this file).

## Contact
For any questions or suggestions, feel free to contact Darshan Pathak at darshan.pathak@example.com.