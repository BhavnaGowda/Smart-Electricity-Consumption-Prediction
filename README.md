# ⚡ Smart Electricity Consumption Prediction Using Machine Learning

A machine learning-based project that analyzes historical household electricity consumption and predicts future energy usage. The project also provides an interactive dashboard built using Streamlit for data visualization, model comparison, and consumption forecasting.

## 📌 Project Overview

Electricity consumption forecasting helps households understand their energy usage patterns and make informed decisions about energy management.

This project uses historical household electricity consumption data and machine learning algorithms to predict hourly electricity consumption.

The system includes:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model training
- Model performance evaluation
- Hourly electricity consumption prediction
- Interactive Streamlit dashboard

## 🎯 Objectives

- Analyze historical household electricity consumption.
- Identify electricity usage patterns and trends.
- Apply different machine learning algorithms for prediction.
- Compare models using standard evaluation metrics.
- Predict future hourly electricity consumption.
- Develop an interactive dashboard for visualization and prediction.

## 📊 Dataset

**Individual Household Electric Power Consumption Dataset**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

Dataset details:
- Approximately 2.07 million minute-level measurements.
- Data collected between December 2006 and November 2010.
- Hourly electricity consumption is derived from the original measurements.
- Missing values are handled during preprocessing.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost (optional in notebook) |
| Model Storage | Joblib |
| Dashboard | Streamlit |
| Development Environment | VS Code, Jupyter Notebook, Google Colab |

## 🤖 Machine Learning Models

The following models are evaluated:

1. Persistence Baseline
2. Linear Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Histogram Gradient Boosting Regressor
6. XGBoost Regressor (optional experiment)

## 📈 Model Performance

The following results were obtained during the notebook experiments:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Histogram Gradient Boosting | 0.3295 | 0.4760 | 0.5882 |
| XGBoost | 0.3296 | 0.4762 | 0.5877 |
| Random Forest | 0.3319 | 0.4825 | 0.5767 |
| Linear Regression | 0.3672 | 0.5160 | 0.5159 |
| Decision Tree | 0.3724 | 0.5524 | 0.4452 |
| Persistence Baseline | 0.6331 | 0.7477 | -0.0162 |

**Best model in the original notebook experiment:** Histogram Gradient Boosting Regressor.

*Note: These are the original notebook experiment results. Retraining locally may produce slightly different metrics.*

## 📂 Project Structure

```text
Smart-Electricity-Consumption-Prediction/
│
├── .streamlit/
│   └── config.toml
│
├── models/
│
├── dataset/                         # Local dataset (not uploaded)
│
├── dashboard.py                     # Streamlit dashboard
├── features.py                      # Feature engineering
├── train_model.py                   # Model training
├── best_model.joblib                # Trained model
├── hourly_consumption.csv           # Hourly dataset
├── actual_vs_predicted.csv          # Prediction results
├── model_comparison_results.csv     # Model evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/BhavnaGowda/Smart-Electricity-Consumption-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Smart-Electricity-Consumption-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```powershell
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit dashboard

```bash
python -m streamlit run dashboard.py
```

The dashboard will open in your browser.

Local URL:

http://localhost:8501

## 🌐 Live Dashboard

The application can be deployed using Streamlit Community Cloud.

**Live Demo:** Coming soon.

## 📊 Dashboard Features

The interactive dashboard includes:

- **Overview:** Summary of electricity consumption data and key metrics.
- **Consumption Analysis:** Explore hourly consumption trends and patterns.
- **Model Comparison:** Compare machine learning model performance.
- **Prediction:** Forecast future electricity consumption.
- **Energy Insights:** Explore useful insights from historical consumption data.

## 📏 Evaluation Metrics

The models are evaluated using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

## 🔮 Future Enhancements

- Integrate real-time electricity consumption data.
- Improve prediction accuracy through hyperparameter optimization.
- Add daily, weekly and monthly forecasting.
- Develop personalized energy-saving recommendations.
- Deploy the application for public access.

## 👩‍💻 Author

**Bhavana Gowda**

GitHub: [@BhavnaGowda](https://github.com/BhavnaGowda)

Project Repository: [Smart Electricity Consumption Prediction](https://github.com/BhavnaGowda/Smart-Electricity-Consumption-Prediction)

---

⭐ If you find this project useful, consider giving the repository a star!
