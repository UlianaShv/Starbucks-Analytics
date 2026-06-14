import os
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

app = FastAPI(title="Starbucks Customer Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH = "starbucks_customer_ordering_patterns.csv"

df = None
scaler = None
kmeans = None
threshold_h1 = None
h1_stats = None

CLUSTER_NAMES = {
    0: "Fast & Standard",
    1: "Patient & Standard",
    2: "Dissatisfied",
    3: "Bulk Buyers",
    4: "Customization Kings"
}

class OrderCreate(BaseModel):
    order_channel: str = Field(..., description="Drive-Thru, Mobile App, In-Store Cashier, Kiosk")
    customer_age_group: str = Field(..., description="18-24, 25-34, 35-44, 45-54, 55+")
    customer_gender: str = Field(..., description="Female, Male, Non-binary, Prefer not to say")
    is_rewards_member: bool
    cart_size: int = Field(..., ge=1, le=20)
    num_customizations: int = Field(..., ge=0, le=20)
    total_spend: float = Field(..., ge=0.0)
    fulfillment_time_min: float = Field(..., ge=0.0)
    drink_category: str = Field(..., description="Refresher, Brewed Coffee, Frappuccino, Espresso, Tea, Other")
    has_food_item: bool
    order_ahead: bool
    customer_satisfaction: int = Field(..., ge=1, le=5)

def init_models_and_data():
    global df, scaler, kmeans, threshold_h1, h1_stats
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    
    bins = np.arange(0, df['fulfillment_time_min'].max() + 1.5, 1)
    df['fulfillment_bin'] = pd.cut(df['fulfillment_time_min'], bins=bins)
    
    df['is_dissatisfied'] = (df['customer_satisfaction'] <= 3).astype(int)
    X = df[['fulfillment_time_min']]
    y = df['is_dissatisfied']
    dt = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt.fit(X, y)
    threshold_h1 = float(dt.tree_.threshold[0])
    
    # K-Means Clustering
    print("Training K-Means (K=5) models...")
    features = ['total_spend', 'num_customizations', 'cart_size', 'fulfillment_time_min', 'customer_satisfaction']
    X_cls = df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cls)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    df['segment'] = df['cluster'].map(CLUSTER_NAMES)
    print("Models initialized successfully!")

@app.on_event("startup")
def startup_event():
    init_models_and_data()

@app.get("/")
def read_root():
    return {"message": "Welcome to Starbucks Customer Analytics API", "total_records": len(df)}

@app.get("/orders")
def get_orders(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    channel: str = Query(None),
    min_satisfaction: int = Query(None, ge=1, le=5)
):
    global df
    filtered_df = df.copy()
    
    if channel:
        filtered_df = filtered_df[filtered_df['order_channel'] == channel]
    if min_satisfaction:
        filtered_df = filtered_df[filtered_df['customer_satisfaction'] >= min_satisfaction]
        
    total_count = len(filtered_df)
    
    cols_to_drop = [c for c in ['fulfillment_bin', 'is_dissatisfied'] if c in filtered_df.columns]
    if cols_to_drop:
        filtered_df = filtered_df.drop(columns=cols_to_drop)
    
    chunk = filtered_df.iloc[offset:offset+limit].replace({np.nan: None}).to_dict(orient="records")
    
    return {
        "total": total_count, "limit": limit, "offset": offset, "data": chunk
    }

@app.post("/orders")
def create_order(order: OrderCreate):
    global df, scaler, kmeans
    
    now = datetime.now()
    new_order_id = f"ORD_{len(df) + 1:08d}"
    new_customer_id = f"CUST_{len(df) + 10000}"
    order_date = now.strftime("%Y-%m-%d")
    order_time = now.strftime("%H:%M")
    day_of_week = now.strftime("%a")
    
    store_id = "STR_999"
    store_location_type = "Urban"
    region = "West"
    
    row_dict = {
        "customer_id": new_customer_id,
        "order_id": new_order_id,
        "order_date": order_date,
        "order_time": order_time,
        "day_of_week": day_of_week,
        "order_channel": order.order_channel,
        "store_id": store_id,
        "store_location_type": store_location_type,
        "region": region,
        "customer_age_group": order.customer_age_group,
        "customer_gender": order.customer_gender,
        "is_rewards_member": order.is_rewards_member,
        "cart_size": order.cart_size,
        "num_customizations": order.num_customizations,
        "total_spend": order.total_spend,
        "fulfillment_time_min": order.fulfillment_time_min,
        "drink_category": order.drink_category,
        "has_food_item": order.has_food_item,
        "order_ahead": order.order_ahead,
        "customer_satisfaction": order.customer_satisfaction
    }
    
    features_input = np.array([[
        order.total_spend,
        order.num_customizations,
        order.cart_size,
        order.fulfillment_time_min,
        order.customer_satisfaction
    ]])
    scaled_input = scaler.transform(features_input)
    cluster_id = int(kmeans.predict(scaled_input)[0])
    
    row_dict["cluster"] = cluster_id
    row_dict["segment"] = CLUSTER_NAMES[cluster_id]
    
    bins = np.arange(0, max(df['fulfillment_time_min'].max(), order.fulfillment_time_min) + 2.5, 1)
    temp_bin = pd.cut([order.fulfillment_time_min], bins=bins)[0]
    row_dict["fulfillment_bin"] = temp_bin

    new_df_row = pd.DataFrame([row_dict])
    
    csv_columns = [
        "customer_id", "order_id", "order_date", "order_time", "day_of_week", 
        "order_channel", "store_id", "store_location_type", "region", 
        "customer_age_group", "customer_gender", "is_rewards_member", "cart_size", 
        "num_customizations", "total_spend", "fulfillment_time_min", 
        "drink_category", "has_food_item", "order_ahead", "customer_satisfaction"
    ]
    csv_row = new_df_row[csv_columns]
    csv_row.to_csv(CSV_PATH, mode='a', header=False, index=False)

    df = pd.concat([df, new_df_row], ignore_index=True)
    df['fulfillment_bin'] = pd.cut(df['fulfillment_time_min'], bins=bins)
    df['is_dissatisfied'] = (df['customer_satisfaction'] <= 3).astype(int)
    
    return {"status": "success", "order_id": new_order_id, "cluster": cluster_id, "segment": CLUSTER_NAMES[cluster_id]}

from scipy import stats

@app.get("/analytics/hypothesis1")
def get_hypothesis1():
    global df, threshold_h1
    
    satisfaction_by_bin = df.groupby('fulfillment_bin', observed=False)['customer_satisfaction'].agg(['mean', 'count']).reset_index()
    satisfaction_by_bin['fulfillment_bin'] = satisfaction_by_bin['fulfillment_bin'].astype(str)
    satisfaction_by_bin['bin_str'] = satisfaction_by_bin['fulfillment_bin']
    
    under_threshold_ratio = float(df[df['fulfillment_time_min'] <= threshold_h1]['is_dissatisfied'].mean())
    over_threshold_ratio = float(df[df['fulfillment_time_min'] > threshold_h1]['is_dissatisfied'].mean())
    
    df_10_11 = df[(df['fulfillment_time_min'] > 10) & (df['fulfillment_time_min'] <= 11)]
    df_9_10 = df[(df['fulfillment_time_min'] > 9) & (df['fulfillment_time_min'] <= 10)]
    
    t_stat, p_val = stats.ttest_ind(df_10_11['customer_satisfaction'], df_9_10['customer_satisfaction'], equal_var=False)
    
    df_long = df[df['fulfillment_time_min'] > 8][['fulfillment_time_min', 'customer_satisfaction']].copy()
    long_wait_data = df_long.to_dict(orient="records")
    
    return {
        "threshold": threshold_h1,
        "bins": satisfaction_by_bin.replace({np.nan: None}).to_dict(orient="records"),
        "under_threshold_dissatisfied_pct": under_threshold_ratio,
        "over_threshold_dissatisfied_pct": over_threshold_ratio,
        "anomaly_stats": {
            "count": len(df_10_11),
            "channels": df_10_11['order_channel'].value_counts().to_dict(),
            "rewards_pct": float(df_10_11['is_rewards_member'].mean()),
            "avg_spend": float(df_10_11['total_spend'].mean()),
            "avg_customizations": float(df_10_11['num_customizations'].mean()),
            "t_stat": float(t_stat),
            "p_val": float(p_val)
        },
        "long_wait_orders": long_wait_data
    }

@app.get("/analytics/hypothesis2")
def get_hypothesis2():
    global df
    features = ['total_spend', 'num_customizations', 'cart_size', 'fulfillment_time_min', 'customer_satisfaction']
    
    profiles = df.groupby('segment')[features].mean().reindex(
        ["Fast & Standard", "Patient & Standard", "Dissatisfied", "Bulk Buyers", "Customization Kings"]
    ).reset_index()
    
    counts = df['segment'].value_counts().to_dict()
    profiles['count'] = profiles['segment'].map(counts)
    
    channel_dist = pd.crosstab(df['segment'], df['order_channel']).reindex(
        ["Fast & Standard", "Patient & Standard", "Dissatisfied", "Bulk Buyers", "Customization Kings"]
    ).reset_index().to_dict(orient="records")
    
    rewards_ratio = df.groupby('segment')['is_rewards_member'].mean().reindex(
        ["Fast & Standard", "Patient & Standard", "Dissatisfied", "Bulk Buyers", "Customization Kings"]
    ).reset_index().to_dict(orient="records")
    

    
    return {
        "profiles": profiles.replace({np.nan: None}).to_dict(orient="records"),
        "channel_distribution": channel_dist,
        "rewards_distribution": rewards_ratio,
    }

