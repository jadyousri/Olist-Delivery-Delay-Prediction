import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import joblib
import os
import kagglehub

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

# Set page config
st.set_page_config(
    page_title="Olist Delay Prediction Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        /* Main Header Styling */
        .main-header {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem 0;
            letter-spacing: -0.5px;
        }
        
        /* Section Headers */
        h1 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #1e3a8a !important;
            margin-top: 2rem !important;
            margin-bottom: 1.5rem !important;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #3b82f6;
        }
        
        h2 {
            font-size: 2rem !important;
            font-weight: 600 !important;
            color: #1e40af !important;
            margin-top: 1.5rem !important;
            margin-bottom: 1rem !important;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid #60a5fa;
        }
        
        h3 {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            color: #2563eb !important;
            margin-top: 1.2rem !important;
            margin-bottom: 0.8rem !important;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
            border-right: 2px solid #cbd5e1;
        }
        
        [data-testid="stSidebar"] [data-baseweb="base-input"] {
            background-color: white;
        }
        
        /* Radio Button Styling for Navigation */
        .stRadio > div {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 1rem;
        }
        
        .stRadio label {
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            color: #1e293b !important;
            padding: 0.75rem 1rem !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        .stRadio label:hover {
            background-color: #f1f5f9 !important;
            transform: translateX(5px);
        }
        
        .stRadio [data-baseweb="radio"] > div[aria-checked="true"] {
            background-color: #3b82f6 !important;
            border-color: #2563eb !important;
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8fafc;
            padding: 0.5rem;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: white;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            color: #475569;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #f1f5f9;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
        }
        
        /* Info/Alert Boxes */
        .stInfo {
            background-color: #dbeafe !important;
            border-left: 4px solid #3b82f6 !important;
            padding: 1rem !important;
            border-radius: 8px !important;
        }
        
        .stSuccess {
            background-color: #d1fae5 !important;
            border-left: 4px solid #10b981 !important;
            padding: 1rem !important;
            border-radius: 8px !important;
        }
        
        .stError {
            background-color: #fee2e2 !important;
            border-left: 4px solid #ef4444 !important;
            padding: 1rem !important;
            border-radius: 8px !important;
        }
        
        .stWarning {
            background-color: #fef3c7 !important;
            border-left: 4px solid #f59e0b !important;
            padding: 1rem !important;
            border-radius: 8px !important;
        }
        
        /* Dataframe Styling */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
        }
        
        /* Input Fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            border-radius: 8px;
            border: 2px solid #e2e8f0;
            padding: 0.5rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        /* Spacing Improvements */
        .main .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        
        /* Markdown Text Styling */
        .stMarkdown {
            line-height: 1.8;
        }
        
        .stMarkdown ul, .stMarkdown ol {
            padding-left: 2rem;
        }
        
        .stMarkdown li {
            margin-bottom: 0.5rem;
        }
        
        /* Plotly Chart Container */
        .js-plotly-plot {
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 1rem;
            background-color: white;
        }
        
        /* Sidebar Title */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #1e293b !important;
        }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        # Try to load from Kaggle
        path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
        base_path = path
    except:
        # Fallback: try to load from local path
        base_path = "/root/.cache/kagglehub/datasets/olistbr/brazilian-ecommerce/versions/2"
        if not os.path.exists(base_path):
            st.error("Dataset not found. Please ensure the dataset is available.")
            return None
    
    try:
        customers = pd.read_csv(f"{base_path}/olist_customers_dataset.csv")
        orders = pd.read_csv(f"{base_path}/olist_orders_dataset.csv")
        order_items = pd.read_csv(f"{base_path}/olist_order_items_dataset.csv")
        products = pd.read_csv(f"{base_path}/olist_products_dataset.csv")
        payments = pd.read_csv(f"{base_path}/olist_order_payments_dataset.csv")
        reviews = pd.read_csv(f"{base_path}/olist_order_reviews_dataset.csv")
        geolocation = pd.read_csv(f"{base_path}/olist_geolocation_dataset.csv")
        products_trans = pd.read_csv(f"{base_path}/product_category_name_translation.csv")
        sellers = pd.read_csv(f"{base_path}/olist_sellers_dataset.csv")
        
        # Create geolocation1
        geolocation["rnk"] = geolocation.groupby("geolocation_zip_code_prefix").cumcount() + 1
        geolocation1 = geolocation[geolocation["rnk"] == 1].drop(columns=["rnk"])
        
        # Merge product translations
        products = products.merge(products_trans, on="product_category_name", how="left")
        
        # Build full dataset
        df = order_items.merge(orders, on="order_id", how="inner")
        df = df.merge(products, on="product_id", how="inner")
        df = df.merge(customers, on="customer_id", how="inner")
        df = df.merge(reviews[["order_id", "review_score"]], on="order_id", how="inner")
        df = df.merge(payments[["order_id", "payment_type", "payment_value"]], on="order_id", how="inner")
        
        df = df.merge(
            geolocation1,
            left_on="customer_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="inner"
        )
        
        df = df.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
        
        # Select final columns
        full_dataset = df[
            [
                "order_id", "product_id", "product_category_name_english", "review_score",
                "seller_id", "price", "freight_value", "order_status", "customer_id",
                "customer_zip_code_prefix", "geolocation_lat", "geolocation_lng",
                "customer_state", "payment_type", "payment_value", "shipping_limit_date",
                "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                "order_delivered_customer_date", "order_estimated_delivery_date", "seller_state"
            ]
        ]
        
        full_dataset.rename(columns={
            "geolocation_lat": "geo_lat",
            "geolocation_lng": "geo_long"
        }, inplace=True)
        
        return full_dataset
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def preprocess_data(df):
    """Clean and preprocess the dataset"""
    # Clean product_category_name_english
    df['product_category_name_english'] = df['product_category_name_english'].apply(
        lambda x: x.replace("\r", "") if "\r" in str(x) else x
    )
    df['product_category_name_english'] = df['product_category_name_english'].replace(r"\N", "UNKNOWN")
    
    # Remove blank spaces
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    
    # Convert date columns
    date_cols = [
        'shipping_limit_date', 'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Sort by purchase timestamp
    df = df.sort_values('order_purchase_timestamp')
    df.reset_index(drop=True, inplace=True)
    
    return df

@st.cache_data
def engineer_features(df):
    """Create engineered features"""
    df = df.copy()
    
    # Delay features
    df['delay_in_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['delayed'] = df['delay_in_days'].apply(lambda x: 0 if x <= 0 else 1)
    
    # Payment range
    bins = [0, 100, 200, 300, 400, 500, 600, 50000]
    labels = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-600', 'more than 600']
    df['payment_range'] = pd.cut(df['payment_value'], bins=bins, labels=labels, right=False)
    
    # Customer type
    customer_counts = df['customer_id'].value_counts()
    df['customer_type'] = df['customer_id'].map(
        lambda x: 'Repeated' if customer_counts[x] > 1 else 'First-time'
    )
    
    # Customer region
    North = ['AC', 'AM', 'PA', 'RO', 'RR', 'TO']
    North_east = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE']
    Central_West = ['GO', 'MT', 'MS']
    South_east = ['ES', 'MG', 'RJ', 'SP']
    South = ['PR', 'RS', 'SC']
    
    df['cust_region'] = df['customer_state'].apply(
        lambda x: 'North' if x in North
        else 'North_east' if x in North_east
        else 'Central_West' if x in Central_West
        else 'South_east' if x in South_east
        else 'South'
    )
    
    # Delivery timing features
    df['days_taken_seller_to_carrier'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days
    df['estimated_days_for_delivery_by_carrier'] = (df['order_estimated_delivery_date'] - df['order_delivered_carrier_date']).dt.days
    df['days_taken_by_carrier'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.days
    
    # Reorder columns
    df = df[[
        'order_id', 'product_id', 'product_category_name_english', 'review_score',
        'seller_id', 'seller_state', 'price', 'freight_value', 'order_status',
        'customer_id', 'customer_type', 'customer_zip_code_prefix', 'geo_lat',
        'geo_long', 'customer_state', 'cust_region', 'payment_type', 'payment_value',
        'payment_range', 'shipping_limit_date', 'order_purchase_timestamp',
        'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date', 'delayed', 'delay_in_days',
        'days_taken_seller_to_carrier', 'estimated_days_for_delivery_by_carrier',
        'days_taken_by_carrier'
    ]]
    
    return df

# Main App
st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">📦 Olist Delay Prediction Dashboard</h1>
        <p style="font-size: 1.2rem; color: #64748b; margin-top: -1rem;">
            Analyze delivery delays and predict order delays with machine learning
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 1.5rem;">
        <h2 style="color: #1e293b; font-size: 1.5rem; font-weight: 700; margin: 0;">📊 Navigation</h2>
    </div>
""", unsafe_allow_html=True)

# Navigation with icons
navigation_options = {
    "📋 Overview": "Overview",
    "📊 Data Analysis": "Data Analysis",
    "📈 Visualizations": "Visualizations",
    "🔬 Hypothesis Testing": "Hypothesis Testing",
    "🔮 Predictions": "Predictions",
    "💡 Recommendations": "Recommendations"
}

selected_nav = st.sidebar.radio(
    "Select a section:",
    options=list(navigation_options.keys()),
    label_visibility="collapsed"
)

page = navigation_options[selected_nav]

# Add separator and info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="background-color: #f1f5f9; padding: 1rem; border-radius: 8px; margin-top: 2rem;">
        <p style="font-size: 0.9rem; color: #475569; margin: 0;">
            <strong>💡 Tip:</strong> Use the navigation above to explore different sections of the analysis.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
if 'df' not in st.session_state:
    with st.spinner("Loading and preprocessing data..."):
        raw_df = load_data()
        if raw_df is not None:
            processed_df = preprocess_data(raw_df)
            st.session_state['df'] = engineer_features(processed_df)
            st.session_state['data_loaded'] = True
        else:
            st.session_state['data_loaded'] = False
            st.error("Failed to load data. Please check your data source.")

if st.session_state.get('data_loaded', False):
    df = st.session_state['df']
    
    if page == "Overview":
        st.markdown("---")
        st.markdown("### 🎯 Business Problem")
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6; margin: 1rem 0;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li><strong>Olist</strong> is an online marketplace that connects multiple sellers to customers across Brazil.</li>
                <li>The company faces challenges with <strong>delivery delays</strong>, which negatively affect customer satisfaction and review scores.</li>
                <li>Late deliveries reduce customer trust, increase churn, and can hurt Olist's overall sales and brand reputation.</li>
                <li>The business wants to identify what factors are driving these delays and how much they impact customer satisfaction.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Project Objective")
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10b981; margin: 1rem 0;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li>Analyze Olist's e-commerce data to uncover factors that lead to delivery delays.</li>
                <li>Understand how delays impact customer satisfaction and review scores.</li>
                <li>Recommend operational and strategic improvements to reduce delays and enhance customer experience.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Orders", f"{len(df):,}")
        with col2:
            st.metric("Delayed Orders", f"{df['delayed'].sum():,}", f"{(df['delayed'].sum()/len(df)*100):.2f}%")
        with col3:
            st.metric("Total Sales", f"R$ {df['payment_value'].sum():,.2f}")
        with col4:
            st.metric("Avg Review Score", f"{df['review_score'].mean():.2f}")
        
        st.subheader("Dataset Info")
        st.dataframe(df.head(10))
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
    elif page == "Data Analysis":
        st.markdown("---")
        st.markdown("## 📊 Exploratory Data Analysis")
        st.markdown("""
        <div style="background-color: #eff6ff; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
            <p style="margin: 0; color: #1e40af; font-size: 1.1rem;">
                Explore different aspects of the dataset including customer behavior, payment patterns, delivery performance, and regional analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["👥 Customer Analysis", "💳 Payment Analysis", "🚚 Delivery Analysis", "🗺️ Regional Analysis"])
        
        with tab1:
            st.markdown("### 👥 Customer Type Distribution")
            customer_type_counts = df['customer_type'].value_counts()
            fig = px.pie(
                values=customer_type_counts.values,
                names=customer_type_counts.index,
                title="Repeated vs First-time Customers",
                height=500,
                hole=0.3
            )
            fig.update_traces(
                textposition='outside',
                textinfo='percent+label',
                textfont_size=14,
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig.update_layout(
                font=dict(size=14),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                ),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **Insights:**
            - Around 86.7% of customers are first-time buyers, showing Olist heavily depends on new customer acquisition.
            - Only 13.3% are repeat customers, indicating limited customer loyalty and retention challenges.
            - Enhancing post-purchase experience and delivery reliability could help convert more first-time buyers into repeat ones.
            """)
        
        with tab2:
            st.subheader("Payment Type Analysis")
            payment_dist = df.groupby(['payment_type']).agg(
                Total_orders=('order_id', 'count'),
                Delayed_orders=('delayed', 'sum')
            ).reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    payment_dist, x='payment_type', y='Total_orders',
                    title="Total Orders by Payment Type",
                    labels={'Total_orders': 'Total Orders', 'payment_type': 'Payment Type'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    payment_dist, x='payment_type', y='Delayed_orders',
                    title="Delayed Orders by Payment Type",
                    labels={'Delayed_orders': 'Delayed Orders', 'payment_type': 'Payment Type'},
                    color='Delayed_orders',
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Payment Range Distribution")
            pay_range = df.groupby(['payment_range']).agg(
                order_count=('order_id', 'count')
            ).reset_index()
            
            fig = px.bar(
                pay_range, x='payment_range', y='order_count',
                title="Order Distribution across Payment Range",
                labels={'order_count': 'Order Count', 'payment_range': 'Payment Range'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Delivery Status")
            Early_or_on_time = df[df['delayed'] == 0].shape[0]
            Delayed = df[df['delayed'] == 1].shape[0]
            
            # Responsibility analysis
            responsible = df.apply(
                lambda x: 'Seller' if (x['delayed'] == 1 and (
                    (x['order_delivered_carrier_date'] > x['order_estimated_delivery_date']) or
                    (x['order_estimated_delivery_date'] - x['order_delivered_carrier_date']).days < 3
                ))
                else 'Carrier' if (x['delayed'] == 1 and x['order_delivered_carrier_date'] < x['order_estimated_delivery_date'])
                else 'no_delay',
                axis=1
            )
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    values=[Early_or_on_time, Delayed],
                    names=['Early/On-time', 'Delayed'],
                    title="Delivery Status"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                resp_counts = responsible.value_counts()
                if 'no_delay' in resp_counts.index:
                    resp_counts = resp_counts.drop('no_delay')
                fig = px.pie(
                    values=resp_counts.values,
                    names=resp_counts.index,
                    title="Delay Responsibility"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Insights:**
            - 93.5% of deliveries are on-time, but the 6.5% delays still significantly impact satisfaction.
            - Carriers cause 89% of delays, highlighting logistical inefficiencies as the main bottleneck.
            - Improving courier partnerships or monitoring carrier SLAs could drastically enhance delivery performance.
            """)
        
        with tab4:
            st.subheader("Orders by Customer Region")
            region_counts = df['cust_region'].value_counts()
            fig = px.bar(
                x=region_counts.index, y=region_counts.values,
                title="Orders across Customer Region",
                labels={'x': 'Customer Region', 'y': 'Order Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Delay % by Seller State")
            state_delay_sumry = df.groupby(['seller_state']).agg(
                delayed_order=('delayed', 'sum'),
                total_order=('delayed', 'count')
            ).reset_index()
            state_delay_sumry['delay%'] = round(
                (state_delay_sumry['delayed_order'] / state_delay_sumry['total_order']) * 100, 2
            )
            
            fig = px.bar(
                state_delay_sumry, x='seller_state', y='delay%',
                title="Delay(%) across Seller's State",
                labels={'delay%': 'Delay %', 'seller_state': "Seller's State"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Visualizations":
        st.markdown("---")
        st.markdown("## 📈 Data Visualizations")
        st.markdown("""
        <div style="background-color: #eff6ff; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
            <p style="margin: 0; color: #1e40af; font-size: 1.1rem;">
                Interactive charts and visualizations to explore temporal trends, product categories, seller performance, and regional patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
            ["📅 Temporal Analysis", "📦 Product Categories", "👤 Seller Analysis", "🔥 Heatmaps"]
        )
        
        with viz_tab1:
            st.markdown("### 📅 Monthly Trends")
            monthly_orders = df.groupby(df['order_approved_at'].dt.month_name()).agg(
                total_order=("order_id", "count"),
                total_delays=("delayed", "sum"),
                total_sales=('payment_value', 'sum'),
                ave_sales=('payment_value', 'mean')
            ).reset_index()
            monthly_orders.rename(columns={'order_approved_at': "Month_name"}, inplace=True)
            
            # Reorder months
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_orders['Month_name'] = pd.Categorical(
                monthly_orders['Month_name'], categories=month_order, ordered=True
            )
            monthly_orders = monthly_orders.sort_values('Month_name')
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    monthly_orders, x='Month_name', y='total_order',
                    title="Total Orders over the Months",
                    height=400,
                    labels={'total_order': 'Total Orders', 'Month_name': 'Month'}
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
                fig.update_yaxes(tickfont=dict(size=11))
                fig.update_layout(
                    font=dict(size=12),
                    margin=dict(l=50, r=20, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    monthly_orders, x='Month_name', y='total_delays',
                    title="Total Delayed Orders over the Months",
                    height=400,
                    labels={'total_delays': 'Delayed Orders', 'Month_name': 'Month'},
                    color='total_delays',
                    color_continuous_scale='Oranges'
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
                fig.update_yaxes(tickfont=dict(size=11))
                fig.update_layout(
                    font=dict(size=12),
                    margin=dict(l=50, r=20, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            col3, col4 = st.columns(2)
            with col3:
                fig = px.bar(
                    monthly_orders, x='Month_name', y='total_sales',
                    title="Total Sales over the Months",
                    height=400,
                    labels={'total_sales': 'Total Sales (R$)', 'Month_name': 'Month'},
                    color='total_sales',
                    color_continuous_scale='Blues'
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
                fig.update_yaxes(tickfont=dict(size=11))
                fig.update_layout(
                    font=dict(size=12),
                    margin=dict(l=50, r=20, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                fig = px.bar(
                    monthly_orders, x='Month_name', y='ave_sales',
                    title="Average Sales over the Months",
                    height=400,
                    labels={'ave_sales': 'Average Sales (R$)', 'Month_name': 'Month'},
                    color='ave_sales',
                    color_continuous_scale='Greens'
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
                fig.update_yaxes(tickfont=dict(size=11))
                fig.update_layout(
                    font=dict(size=12),
                    margin=dict(l=50, r=20, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing before Daily Trends
            st.markdown("### 📅 Daily Trends")
            Day_orders = df.groupby(df['order_approved_at'].dt.day_name()).agg(
                total_order=("order_id", "count"),
                total_delay=("delayed", "sum"),
                total_sales=('payment_value', 'sum'),
                ave_sales=('payment_value', 'mean')
            ).reset_index()
            Day_orders.rename(columns={'order_approved_at': "Day_name"}, inplace=True)
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            Day_orders['Day_name'] = pd.Categorical(
                Day_orders['Day_name'], categories=day_order, ordered=True
            )
            Day_orders = Day_orders.sort_values('Day_name')
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    Day_orders, x='Day_name', y='total_order',
                    title="Total Orders across the Days",
                    height=400,
                    labels={'total_order': 'Total Orders', 'Day_name': 'Day of Week'}
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
                fig.update_yaxes(tickfont=dict(size=11))
                fig.update_layout(
                    font=dict(size=12),
                    margin=dict(l=50, r=20, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    Day_orders, x='Day_name', y='total_delay',
                    title="Total Delayed Orders across the Days",
                    height=400,
                    labels={'total_delay': 'Delayed Orders', 'Day_name': 'Day of Week'},
                    color='total_delay',
                    color_continuous_scale='Reds'
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
                fig.update_yaxes(tickfont=dict(size=11))
                fig.update_layout(
                    font=dict(size=12),
                    margin=dict(l=50, r=20, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            col3, col4 = st.columns(2)
            with col3:
                fig = px.bar(
                    Day_orders, x='Day_name', y='total_sales',
                    title="Total Sales across the Days",
                    height=400,
                    labels={'total_sales': 'Total Sales (R$)', 'Day_name': 'Day of Week'},
                    color='total_sales',
                    color_continuous_scale='Blues'
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
                fig.update_yaxes(tickfont=dict(size=11))
                fig.update_layout(
                    font=dict(size=12),
                    margin=dict(l=50, r=20, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                fig = px.bar(
                    Day_orders, x='Day_name', y='ave_sales',
                    title="Average Sales across the Days",
                    height=400,
                    labels={'ave_sales': 'Average Sales (R$)', 'Day_name': 'Day of Week'},
                    color='ave_sales',
                    color_continuous_scale='Greens'
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
                fig.update_yaxes(tickfont=dict(size=11))
                fig.update_layout(
                    font=dict(size=12),
                    margin=dict(l=50, r=20, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.subheader("Top Product Categories")
            prod_cat = df.groupby(['product_category_name_english']).agg(
                total_order=('order_id', 'count'),
                total_delays=('delayed', 'sum'),
                ave_rating=('review_score', 'mean'),
                total_sales=('payment_value', 'sum'),
                ave_price=('price', 'mean')
            ).reset_index()
            
            top_20 = prod_cat.nlargest(20, 'total_order')
            fig = px.bar(
                top_20, x='product_category_name_english', y='total_order',
                title="Top 20 Product Categories by Orders"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                top_sales = prod_cat.nlargest(20, 'total_sales')
                fig = px.bar(
                    top_sales, x='product_category_name_english', y='total_sales',
                    title="Top 20 Categories by Sales"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                top_delays = prod_cat.nlargest(20, 'total_delays')
                fig = px.bar(
                    top_delays, x='product_category_name_english', y='total_delays',
                    title="Top 20 Categories by Delays"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            st.subheader("Seller Performance")
            le = LabelEncoder()
            df['seller_id_en'] = le.fit_transform(df['seller_id']) + 1
            
            seller = df.groupby(['seller_id_en', 'seller_id']).agg(
                total_sales=('payment_value', 'sum'),
                total_delays=('delayed', 'sum'),
                rating=('review_score', 'mean'),
                total_orders=('order_id', 'count')
            ).reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                top_orders = seller.nlargest(20, 'total_orders')
                fig = px.bar(
                    top_orders, x='seller_id_en', y='total_orders',
                    title="Top 20 Sellers by Orders"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                top_sales = seller.nlargest(20, 'total_sales')
                fig = px.bar(
                    top_sales, x='seller_id_en', y='total_sales',
                    title="Top 20 Sellers by Sales"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab4:
            st.subheader("Orders by Hour and Day")
            total_order_pivot = pd.pivot_table(
                data=df,
                index=df['order_approved_at'].dt.hour,
                columns=df['order_approved_at'].dt.day_name(),
                values='order_id',
                aggfunc='count'
            )
            
            # Reorder columns
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            total_order_pivot = total_order_pivot.reindex(columns=day_order, fill_value=0)
            
            fig = px.imshow(
                total_order_pivot,
                labels=dict(x="Weekday", y="Hour", color="Order Count"),
                title="Total Orders over the Hour by Weekdays",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Delays by Region and Seller State")
            reg_delay = pd.pivot_table(
                data=df,
                index='cust_region',
                columns='seller_state',
                values='delayed',
                aggfunc='sum',
                fill_value=0
            )
            
            fig = px.imshow(
                reg_delay,
                labels=dict(x="Seller's State", y="Customer's Region", color="Delayed Orders"),
                title="Total Delayed Orders between Seller's State and Customer Region",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Hypothesis Testing":
        st.markdown("---")
        st.markdown("## 🔬 Statistical Hypothesis Testing")
        st.markdown("""
        <div style="background-color: #eff6ff; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
            <p style="margin: 0; color: #1e40af; font-size: 1.1rem;">
                Statistical tests to validate hypotheses about delivery delays, review scores, payment methods, and regional differences.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        test_tab1, test_tab2, test_tab3 = st.tabs(
            ["⭐ Test 1: Review Scores", "💳 Test 2: Payment Method", "🗺️ Test 3: Customer States"]
        )
        
        with test_tab1:
            st.subheader("Test 1: Are review scores different for delayed vs. on-time deliveries?")
            st.markdown("""
            **H₀:** The mean review score is the same for delayed and on-time orders.  
            **H₁:** They differ significantly.
            """)
            
            on_time_reviews = df.loc[df['delayed'] == 0, 'review_score']
            delayed_reviews = df.loc[df['delayed'] == 1, 'review_score']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Rating (On-time)", f"{on_time_reviews.mean():.5f}")
            with col2:
                st.metric("Avg Rating (Delayed)", f"{delayed_reviews.mean():.5f}")
            
            tstat, pvalue = stats.ttest_ind(
                on_time_reviews, delayed_reviews, equal_var=False, nan_policy='omit'
            )
            
            st.write(f"**t-statistic:** {tstat:.5f}")
            st.write(f"**p-value:** {pvalue:.10f}")
            
            if pvalue < 0.05:
                st.success("**Result:** We reject H₀. There is a significant difference between average review scores of delayed and on-time orders.")
            else:
                st.info("**Result:** We fail to reject H₀. No significant difference between average review scores.")
        
        with test_tab2:
            st.subheader("Test 2: Does Payment Method Affect Delivery Delays?")
            st.markdown("""
            **H₀:** Payment type and delivery delay are independent.  
            **H₁:** Payment type and delivery delay are associated with each other.
            """)
            
            contingency = pd.crosstab(df['payment_type'], df['delayed'])
            st.write("**Contingency Table (Observed Frequencies):**")
            st.dataframe(contingency)
            
            chi2, p, dof, expected = chi2_contingency(contingency)
            expected_df = pd.DataFrame(
                expected, index=contingency.index, columns=contingency.columns
            )
            
            st.write("**Expected Frequencies (Under H₀):**")
            st.dataframe(expected_df.round(0))
            
            st.write(f"**Chi² Statistic:** {chi2:.5f}")
            st.write(f"**Degrees of Freedom:** {dof}")
            st.write(f"**p-value:** {p:.5f}")
            
            if p < 0.05:
                st.success("**Result:** We reject H₀. There is a statistically significant association between payment type and delivery delay.")
            else:
                st.info("**Result:** We fail to reject H₀. No significant association between payment type and delivery delay.")
        
        with test_tab3:
            st.subheader("Test 3: Does average delivery delay differ across customer states?")
            st.markdown("""
            **H₀:** Average delay is the same across all customer states.  
            **H₁:** At least one state has a different average delay.
            """)
            
            df_anova = df[['customer_state', 'delay_in_days']]
            days = []
            state_names = []
            
            for state, d_days in df_anova.groupby('customer_state'):
                if len(d_days) > 5:
                    days.append(d_days['delay_in_days'].dropna())
                    state_names.append(state)
            
            st.write(f"**States included in ANOVA test:** {len(days)}")
            
            fstat, pval = f_oneway(*days)
            
            st.write(f"**F-statistic:** {fstat:.4f}")
            st.write(f"**p-value:** {pval:.6f}")
            
            if pval < 0.05:
                st.success("**Result:** We reject H₀. There is a statistically significant difference in average delivery delay across customer states.")
            else:
                st.info("**Result:** We fail to reject H₀. No significant difference in average delivery delay across customer states.")
    
    elif page == "Predictions":
        st.markdown("---")
        st.markdown("## 🔮 Delay Prediction Model")
        st.markdown("""
        <div style="background-color: #eff6ff; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
            <p style="margin: 0; color: #1e40af; font-size: 1.1rem;">
                Use machine learning to predict whether an order will be delayed based on order characteristics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if model exists, if not train it
        model_path = "delay_prediction_model.pkl"
        
        if not os.path.exists(model_path):
            st.info("Model not found. Training model...")
            with st.spinner("Training XGBoost model..."):
                features = ['price', 'freight_value', 'payment_value']
                cat_cols = ['payment_type', 'product_category_name_english', 'customer_state', 'seller_state']
                
                x = df[features + cat_cols].copy()
                y = df['delayed']
                
                num_transform = StandardScaler()
                cat_transform = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                
                preproc = ColumnTransformer([
                    ('num', num_transform, features),
                    ('cat', cat_transform, cat_cols)
                ])
                
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.25, random_state=42, stratify=y
                )
                
                xgb_model = XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.01,
                    max_depth=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=1,
                    n_jobs=-1
                )
                
                pipe = Pipeline([
                    ('preprocessor', preproc),
                    ('oversample', RandomOverSampler(random_state=42)),
                    ('model', xgb_model)
                ])
                
                pipe.fit(x_train, y_train)
                joblib.dump(pipe, model_path)
                
                pred = pipe.predict(x_test)
                pred_proba = pipe.predict_proba(x_test)[:, 1]
                
                st.success("Model trained successfully!")
                st.write("**Model Performance:**")
                st.text(classification_report(y_test, pred))
                st.write(f"**ROC-AUC:** {roc_auc_score(y_test, pred_proba):.4f}")
                st.write(f"**PR-AUC:** {average_precision_score(y_test, pred_proba):.4f}")
                st.write(f"**Train Score:** {pipe.score(x_train, y_train):.4f}")
                st.write(f"**Test Score:** {pipe.score(x_test, y_test):.4f}")
        else:
            pipe = joblib.load(model_path)
            st.success("Model loaded successfully!")
        
        st.markdown("---")
        st.markdown("### 🔮 Predict Single Order")
        st.markdown("""
        <div style="background-color: #f0fdf4; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #10b981;">
            <p style="margin: 0; color: #166534; font-size: 1rem;">
                Enter order details below to predict whether the order will be delayed.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            price = st.number_input("Price", min_value=0.0, value=100.0, step=0.01)
            freight_value = st.number_input("Freight Value", min_value=0.0, value=20.0, step=0.01)
            payment_value = st.number_input("Payment Value", min_value=0.0, value=120.0, step=0.01)
        
        with col2:
            payment_type = st.selectbox(
                "Payment Type",
                options=df['payment_type'].unique().tolist()
            )
            product_category = st.selectbox(
                "Product Category",
                options=sorted(df['product_category_name_english'].dropna().unique().tolist())
            )
            customer_state = st.selectbox(
                "Customer State",
                options=sorted(df['customer_state'].unique().tolist())
            )
            seller_state = st.selectbox(
                "Seller State",
                options=sorted(df['seller_state'].unique().tolist())
            )
        
        if st.button("Predict Delay", type="primary"):
            new_data = pd.DataFrame({
                'price': [price],
                'freight_value': [freight_value],
                'payment_value': [payment_value],
                'payment_type': [payment_type],
                'product_category_name_english': [product_category],
                'customer_state': [customer_state],
                'seller_state': [seller_state]
            })
            
            try:
                prediction = pipe.predict(new_data)
                prediction_proba = pipe.predict_proba(new_data)[0]
                
                if prediction[0] == 1:
                    st.error(f"⚠️ **Prediction: Likely to be Delayed**")
                    st.write(f"Probability: {prediction_proba[1]:.2%}")
                else:
                    st.success(f"✅ **Prediction: Likely to be On-Time or Early**")
                    st.write(f"Probability: {prediction_proba[0]:.2%}")
                
                # Show probabilities
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("On-Time Probability", f"{prediction_proba[0]:.2%}")
                with col2:
                    st.metric("Delayed Probability", f"{prediction_proba[1]:.2%}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    elif page == "Recommendations":
        st.markdown("---")
        st.markdown("## 💡 Business Recommendations for Olist")
        st.markdown("""
        <div style="background-color: #eff6ff; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
            <p style="margin: 0; color: #1e40af; font-size: 1.1rem;">
                Actionable insights and recommendations to improve delivery performance and customer satisfaction.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 1️⃣ Strengthen Carrier Performance")
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ef4444; margin: 1rem 0;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li><strong>Implement stricter SLAs</strong> and track on-time delivery KPIs for each carrier.</li>
                <li><strong>Collaborate with high-performing logistics partners</strong> and penalize underperformers.</li>
                <li><strong>Impact:</strong> Reduce overall delivery delays by 30–40% within 3–6 months.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 2️⃣ Improve Delivery in High-Delay Regions (Amazonas & Maranhão)")
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #f59e0b; margin: 1rem 0;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li><strong>Establish regional hubs</strong> or collaborate with local couriers to shorten transit times.</li>
                <li><strong>Prioritize faster delivery routes</strong> for high-delay zones.</li>
                <li><strong>Impact:</strong> Cut delay rates in these states by up to 50%, enhancing nationwide reliability.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 3️⃣ Enhance Customer Retention & Experience")
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10b981; margin: 1rem 0;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li><strong>Launch loyalty rewards</strong> and send proactive delay notifications with compensation options.</li>
                <li><strong>Focus on turning first-time buyers</strong> into repeat customers through better service.</li>
                <li><strong>Impact:</strong> Boost average review scores by +1 point and repeat purchase rate by 10–15%.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 4️⃣ Optimize Seller & Category Operations")
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #8b5cf6; margin: 1rem 0;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li><strong>Support top sellers</strong> and improve logistics for high-demand categories like Bed & Bath and Health & Beauty.</li>
                <li><strong>Introduce capacity planning</strong> to avoid seller overload during peak periods.</li>
                <li><strong>Impact:</strong> Reduce category-related delays by 20–25% and increase fulfillment efficiency.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 5️⃣ Leverage Predictive Analytics for Delay Prevention")
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6; margin: 1rem 0;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li><strong>Deploy machine learning</strong> to predict delay risk at order placement and flag high-risk shipments.</li>
                <li><strong>Use accurate data-driven delivery estimates</strong> at checkout to improve transparency.</li>
                <li><strong>Impact:</strong> Prevent up to 25% of potential delays before dispatch and strengthen customer trust.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("Please wait while data is being loaded...")

