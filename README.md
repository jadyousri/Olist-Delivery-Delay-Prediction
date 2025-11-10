# 📦 Olist Delay Prediction Dashboard

A comprehensive Streamlit web application for analyzing delivery delays in Olist's Brazilian e-commerce marketplace and predicting order delays using machine learning.

## 🎯 Project Overview

This project analyzes Olist's e-commerce data to identify factors contributing to delivery delays and their impact on customer satisfaction. The application provides:

- **Exploratory Data Analysis (EDA)** with interactive visualizations
- **Statistical Hypothesis Testing** to validate assumptions
- **Machine Learning Predictions** using XGBoost to forecast delivery delays
- **Business Recommendations** based on data-driven insights

## ✨ Features

### 📊 Data Analysis
- **Customer Analysis**: Distribution of first-time vs. repeat customers
- **Payment Analysis**: Payment type and range distributions
- **Delivery Analysis**: Delay patterns and responsibility attribution
- **Regional Analysis**: Geographic patterns in orders and delays

### 📈 Interactive Visualizations
- **Temporal Analysis**: Monthly and daily trends in orders, delays, and sales
- **Product Categories**: Top categories by orders, sales, delays, and ratings
- **Seller Performance**: Top sellers analysis
- **Heatmaps**: Order patterns by hour/day and regional delay patterns

### 🔬 Statistical Testing
- **T-test**: Compare review scores between delayed and on-time orders
- **Chi-square Test**: Analyze association between payment methods and delays
- **ANOVA Test**: Examine delay differences across customer states

### 🔮 Machine Learning Predictions
- **XGBoost Model**: Predict whether an order will be delayed
- **Real-time Predictions**: Input order details and get instant predictions
- **Probability Scores**: View confidence levels for predictions

### 💡 Business Insights
- Actionable recommendations to reduce delivery delays
- Impact estimates for each recommendation
- Strategic insights for operational improvements

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Proj
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run "Data Ops.py"
```

The application will automatically:
- Download the Olist dataset from Kaggle (if not already available)
- Load and preprocess the data
- Train the prediction model (if not already trained)

## 📋 Requirements

The project requires the following Python packages:

- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Plotting library
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.14.0` - Interactive visualizations
- `scipy>=1.10.0` - Scientific computing and statistics
- `scikit-learn>=1.3.0` - Machine learning library
- `imbalanced-learn>=0.11.0` - Handling imbalanced datasets
- `xgboost>=2.0.0` - Gradient boosting framework
- `joblib>=1.3.0` - Model serialization
- `kagglehub>=0.2.0` - Kaggle dataset access

## 📁 Project Structure

```
Proj/
│
├── Data Ops.py                    # Main Streamlit application
├── delay_prediction_model.pkl      # Trained XGBoost model (auto-generated)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🎮 Usage Guide

### Navigation
The application has six main sections accessible via the sidebar:

1. **📋 Overview**: Project introduction and dataset summary
2. **📊 Data Analysis**: Exploratory data analysis with tabs for different aspects
3. **📈 Visualizations**: Interactive charts and graphs
4. **🔬 Hypothesis Testing**: Statistical tests with results
5. **🔮 Predictions**: Machine learning model for delay prediction
6. **💡 Recommendations**: Business insights and recommendations

### Making Predictions

1. Navigate to the **Predictions** page
2. Enter order details:
   - Price
   - Freight Value
   - Payment Value
   - Payment Type
   - Product Category
   - Customer State
   - Seller State
3. Click **"Predict Delay"** button
4. View the prediction result and probability scores

### Exploring Data

- Use the **Data Analysis** page to explore different aspects of the dataset
- Switch between tabs to view customer, payment, delivery, and regional analyses
- Check the **Visualizations** page for temporal trends and patterns
- Review **Hypothesis Testing** results to understand statistical significance

## 🔧 Technical Details

### Data Source
The application uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle.

### Data Processing
- Data is automatically downloaded from Kaggle using `kagglehub`
- Multiple CSV files are merged to create a comprehensive dataset
- Feature engineering includes:
  - Delay calculations
  - Customer type classification
  - Payment range categorization
  - Regional classifications
  - Delivery timing metrics

### Machine Learning Model
- **Algorithm**: XGBoost Classifier
- **Features**: Price, Freight Value, Payment Value, Payment Type, Product Category, Customer State, Seller State
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
- **Handling Imbalance**: RandomOverSampler
- **Model Performance**: ROC-AUC and PR-AUC metrics displayed after training

### Model Training
The model is automatically trained on first use if `delay_prediction_model.pkl` doesn't exist. Training uses:
- 75% training data, 25% test data
- Stratified split to maintain class distribution
- 500 estimators with learning rate 0.01

## 📊 Key Insights

Based on the analysis:

1. **93.5%** of deliveries are on-time, but **6.5%** delays significantly impact satisfaction
2. **Carriers cause 89%** of delays, highlighting logistical bottlenecks
3. **86.7%** of customers are first-time buyers, indicating retention challenges
4. High-delay regions like **Amazonas** and **Maranhão** need targeted interventions
5. **Credit cards** are the most common payment method but also have the most delays

## 🎨 Features Highlights

- **Modern UI**: Clean, professional interface with gradient styling
- **Interactive Charts**: Plotly visualizations with hover details
- **Responsive Design**: Works on different screen sizes
- **Real-time Updates**: Instant predictions and analysis
- **Caching**: Optimized performance with Streamlit caching

## 🔍 Troubleshooting

### Dataset Not Found
If you encounter dataset loading errors:
1. Ensure you have internet connection for Kaggle download
2. Check Kaggle API credentials if required
3. Verify the dataset path in the code

### Model Training Issues
- Ensure sufficient memory (dataset is ~117K rows)
- Training may take a few minutes on first run
- Model is saved automatically for future use

### Display Issues
- Clear browser cache if visualizations don't render
- Ensure all dependencies are installed correctly
- Check Python version compatibility

## 📝 Notes

- The model file (`delay_prediction_model.pkl`) is auto-generated and can be large
- First run may take longer due to data download and model training
- All data processing is cached for faster subsequent runs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational and analytical purposes.

## 👤 Author

Created as part of data analysis and machine learning project for Olist e-commerce delay prediction.

## 🙏 Acknowledgments

- Olist for providing the public dataset
- Kaggle for hosting the dataset
- Streamlit for the excellent framework
- All open-source contributors of the libraries used

---

**Note**: This application is designed for analysis and demonstration purposes. For production use, additional considerations such as data privacy, model validation, and deployment infrastructure should be addressed.

