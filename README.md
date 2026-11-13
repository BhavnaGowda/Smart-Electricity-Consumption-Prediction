# ⚡ Smart Electricity Consumption Prediction Using Machine Learning

An end-to-end machine learning project that analyzes historical household electricity consumption, predicts future hourly energy usage, and provides an interactive dashboard for visualization and forecasting.

<p align="center">

  **🌐 Live Dashboard:** [Open Application](https://smart-electricity-consumption-prediction-9ykdvmvgtje5afqdj9ckh.streamlit.app/)

  **💻 GitHub Repository:** [Smart Electricity Consumption Prediction](https://github.com/BhavnaGowda/Smart-Electricity-Consumption-Prediction)

</p>

---

## 🌐 Live Application

Explore the deployed application:

**[⚡ Smart Electricity Consumption Prediction – Streamlit Dashboard](https://smart-electricity-consumption-prediction-9ykdvmvgtje5afqdj9ckh.streamlit.app/)**

The application is deployed using **Streamlit Community Cloud** and provides interactive data analysis, model comparisons, electricity consumption forecasting, and energy insights.

---

## 📌 Project Overview

Electricity consumption forecasting plays an important role in understanding household energy usage and supporting efficient energy management.

This project uses historical household electricity consumption data and machine learning algorithms to predict hourly electricity consumption.

The system follows an end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and deployment through an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze historical household electricity consumption.
- Identify electricity usage patterns and trends.
- Perform data preprocessing and feature engineering.
- Implement multiple machine learning regression algorithms.
- Evaluate and compare model performance.
- Predict future hourly electricity consumption.
- Develop an interactive dashboard for visualization and forecasting.

## 📊 Dataset

**Individual Household Electric Power Consumption Dataset**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

Dataset information:

- Original measurements: Approximately 2.07 million records.
- Collection period: December 2006 – November 2010.
- Original measurement frequency: One minute.
- Processed data: Hourly electricity consumption.
- Missing values are handled during preprocessing.

The dataset contains household electricity measurements, which are processed and transformed into hourly consumption data for machine learning.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Model Serialization | Joblib |
| Dashboard | Streamlit |
| Development Tools | VS Code, Jupyter Notebook, Google Colab |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## ⚙️ Project Workflow

1. **Data Collection:** Obtain the household electricity consumption dataset from UCI.
2. **Data Preprocessing:** Handle missing values and prepare the dataset.
3. **Exploratory Data Analysis:** Analyze consumption trends and patterns.
4. **Feature Engineering:** Generate time-based features and historical consumption features.
5. **Model Training:** Train multiple regression algorithms.
6. **Model Evaluation:** Compare models using MAE, MSE, RMSE, and R².
7. **Prediction:** Forecast future hourly electricity consumption.
8. **Deployment:** Build and deploy an interactive Streamlit dashboard.

## 🤖 Machine Learning Models

The following models are implemented and evaluated:

1. Persistence Baseline
2. Linear Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Histogram Gradient Boosting Regressor
6. XGBoost Regressor

## 📈 Model Performance

The following results were obtained during the original notebook experiments:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Histogram Gradient Boosting | 0.3295 | 0.4760 | 0.5882 |
| XGBoost | 0.3296 | 0.4762 | 0.5877 |
| Random Forest | 0.3319 | 0.4825 | 0.5767 |
| Linear Regression | 0.3672 | 0.5160 | 0.5159 |
| Decision Tree | 0.3724 | 0.5524 | 0.4452 |
| Persistence Baseline | 0.6331 | 0.7477 | -0.0162 |

**Best-performing model in the original notebook experiment:** Histogram Gradient Boosting Regressor.

*Note: Local retraining may produce slightly different results due to differences in library versions and training behavior.*

## 📏 Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error in the target variable's units.
- **R² Score:** Measures how well the model explains the variation in the target variable.

## 📂 Project Structure

```text
Smart-Electricity-Consumption-Prediction/
│
├── .streamlit/
│   └── config.toml
│
├── models/
│
├── dataset/                         # Local dataset (excluded from Git)
│
├── dashboard.py                     # Streamlit dashboard
├── features.py                      # Feature engineering
├── train_model.py                   # Model training script
├── best_model.joblib                # Trained ML model
├── hourly_consumption.csv           # Processed hourly dataset
├── actual_vs_predicted.csv          # Actual vs predicted values
├── model_comparison_results.csv     # Model evaluation results
├── requirements.txt                 # Project dependencies
├── .gitignore
└── README.md
```

## 📊 Dashboard Features

The interactive Streamlit dashboard provides the following sections:

### 1. Overview
- Key electricity consumption statistics.
- Dataset summary.
- Overall consumption information.

### 2. Consumption Analysis
- Hourly electricity consumption trends.
- Historical usage patterns.
- Interactive visualizations.

### 3. Model Comparison
- Comparison of different machine learning models.
- Performance evaluation metrics.
- Model performance visualization.

### 4. Prediction
- Future hourly electricity consumption forecasting.
- Prediction-related analysis.
- Feature inspection and model evaluation tools.

### 5. Energy Insights
- Electricity consumption patterns.
- Useful insights from historical usage.
- Data-driven energy analysis.

---

## 🚀 Run the Project Locally

### Prerequisites

- Python 3.10 or later (use a version compatible with the project dependencies).
- Git.
- VS Code or another Python IDE.

### Step 1: Clone the repository

```bash
git clone https://github.com/BhavnaGowda/Smart-Electricity-Consumption-Prediction.git
```

### Step 2: Navigate to the project directory

```bash
cd Smart-Electricity-Consumption-Prediction
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Run the Streamlit dashboard

```bash
python -m streamlit run dashboard.py
```

The dashboard will open in your browser.

**Local application URL:**

http://localhost:8501

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

**Live URL:**

https://smart-electricity-consumption-prediction-9ykdvmvgtje5afqdj9ckh.streamlit.app/

The deployment is connected to the GitHub repository.

## 🔮 Future Enhancements

- Integrate real-time electricity consumption data.
- Improve prediction accuracy through hyperparameter optimization.
- Extend forecasting to daily, weekly, and monthly consumption.
- Introduce personalized energy-saving recommendations.
- Explore advanced machine learning and deep learning models.
- Enhance deployment and scalability.

---

## 👩‍💻 Author

**Bhavana Gowda**

- GitHub: [@BhavnaGowda](https://github.com/BhavnaGowda)
- Project Repository: [Smart Electricity Consumption Prediction](https://github.com/BhavnaGowda/Smart-Electricity-Consumption-Prediction)
- Live Dashboard: [Streamlit Application](https://smart-electricity-consumption-prediction-9ykdvmvgtje5afqdj9ckh.streamlit.app/)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, Machine Learning, and Streamlit.**
