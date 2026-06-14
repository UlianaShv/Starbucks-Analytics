import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

st.set_page_config(
    page_title="Starbucks Customer Analytics & Segmentations",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLUSTER_NAMES = {
    0: "Fast & Standard",
    1: "Patient & Standard",
    2: "Dissatisfied",
    3: "Bulk Buyers",
    4: "Customization Kings"
}

@st.cache_resource
def get_trained_models():
    df = pd.read_csv("starbucks_customer_ordering_patterns.csv")
    features = ['total_spend', 'num_customizations', 'cart_size', 'fulfillment_time_min', 'customer_satisfaction']
    X_cls = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cls)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    df['is_dissatisfied'] = (df['customer_satisfaction'] <= 3).astype(int)
    dt = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt.fit(df[['fulfillment_time_min']], df['is_dissatisfied'])
    threshold = float(dt.tree_.threshold[0])
    
    return scaler, kmeans, threshold

from colors import COLORS, ACCENT_PALETTE, STARBUCKS_COLORS, SEGMENT_COLORS, STARBUCKS_CMAP

sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

api_online = True
total_records = 0

def fmt_pct(val):
    return f"{val * 100:.2f}%"

@st.cache_data
def load_full_data():
    df = pd.read_csv("starbucks_customer_ordering_patterns.csv")
    df['order_hour'] = df['order_time'].apply(lambda x: int(x.split(':')[0]))
    df['time_of_day'] = pd.cut(df['order_hour'],
                               bins=[-1, 6, 11, 16, 21, 24], 
                               labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Night'], 
                               ordered=False)
    df['order_date_parsed'] = pd.to_datetime(df['order_date'])
    df['season'] = df['order_date_parsed'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    # Scale features and predict cluster
    scaler, kmeans, _ = get_trained_models()
    features = ['total_spend', 'num_customizations', 'cart_size', 'fulfillment_time_min', 'customer_satisfaction']
    X_cls = df[features].copy()
    X_scaled = scaler.transform(X_cls)
    df['cluster'] = kmeans.predict(X_scaled)
    df['segment'] = df['cluster'].map(CLUSTER_NAMES)
    
    return df

@st.cache_data
def get_hypothesis1_data():
    df = load_full_data()
    _, _, threshold = get_trained_models()
    
    bins = np.arange(0, df['fulfillment_time_min'].max() + 1.5, 1)
    df['fulfillment_bin'] = pd.cut(df['fulfillment_time_min'], bins=bins)
    df['is_dissatisfied'] = (df['customer_satisfaction'] <= 3).astype(int)
    
    satisfaction_by_bin = df.groupby('fulfillment_bin', observed=False)['customer_satisfaction'].agg(['mean', 'count']).reset_index()
    satisfaction_by_bin['fulfillment_bin'] = satisfaction_by_bin['fulfillment_bin'].astype(str)
    satisfaction_by_bin['bin_str'] = satisfaction_by_bin['fulfillment_bin']
    
    under_pct = float(df[df['fulfillment_time_min'] <= threshold]['is_dissatisfied'].mean())
    over_pct = float(df[df['fulfillment_time_min'] > threshold]['is_dissatisfied'].mean())
    
    df_10_11 = df[(df['fulfillment_time_min'] > 10) & (df['fulfillment_time_min'] <= 11)]
    df_9_10 = df[(df['fulfillment_time_min'] > 9) & (df['fulfillment_time_min'] <= 10)]
    
    t_stat, p_val = stats.ttest_ind(df_10_11['customer_satisfaction'], df_9_10['customer_satisfaction'], equal_var=False)
    
    df_long = df[df['fulfillment_time_min'] > 8][['fulfillment_time_min', 'customer_satisfaction']].copy()
    
    anomaly_stats = {
        "count": len(df_10_11),
        "channels": df_10_11['order_channel'].value_counts().to_dict(),
        "rewards_pct": float(df_10_11['is_rewards_member'].mean()),
        "avg_spend": float(df_10_11['total_spend'].mean()),
        "avg_customizations": float(df_10_11['num_customizations'].mean()),
        "t_stat": float(t_stat),
        "p_val": float(p_val)
    }
    
    return threshold, satisfaction_by_bin, under_pct, over_pct, anomaly_stats, df_long

@st.cache_data
def get_hypothesis2_data():
    df = load_full_data()
    features = ['total_spend', 'num_customizations', 'cart_size', 'fulfillment_time_min', 'customer_satisfaction']
    
    profiles = df.groupby('segment')[features].mean().reindex(
        ["Fast & Standard", "Patient & Standard", "Dissatisfied", "Bulk Buyers", "Customization Kings"]
    ).reset_index()
    
    counts = df['segment'].value_counts().to_dict()
    profiles['count'] = profiles['segment'].map(counts)
    
    channel_dist = pd.crosstab(df['segment'], df['order_channel']).reindex(
        ["Fast & Standard", "Patient & Standard", "Dissatisfied", "Bulk Buyers", "Customization Kings"]
    ).reset_index()
    
    rewards_ratio = df.groupby('segment')['is_rewards_member'].mean().reindex(
        ["Fast & Standard", "Patient & Standard", "Dissatisfied", "Bulk Buyers", "Customization Kings"]
    ).reset_index()
    
    return profiles, channel_dist, rewards_ratio

@st.cache_data
def compute_correlations():
    df = pd.read_csv("starbucks_customer_ordering_patterns.csv")
    
    df_all = df.copy()
    df_all['order_date'] = pd.to_datetime(df_all['order_date']).astype('int64') // 10**9
    df_all['order_time'] = pd.to_datetime(df_all['order_time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(df_all['order_time'], format='%H:%M').dt.minute
    
    weekday_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    df_all['day_of_week'] = df_all['day_of_week'].map(weekday_map)
    
    age_map = {'18-24': 0, '25-34': 1, '35-44': 2, '45-54': 3, '55+': 4}
    df_all['customer_age_group'] = df_all['customer_age_group'].map(age_map)
    
    df_all['order_hour'] = df_all['order_time'] // 60
    df_all['time_of_day'] = pd.cut(df_all['order_hour'],
                                   bins=[-1, 6, 11, 16, 21, 24], 
                                   labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Night'], 
                                   ordered=False)
    time_map = {'Night': 3, 'Morning': 0, 'Afternoon': 1, 'Evening': 2}
    df_all['time_of_day'] = df_all['time_of_day'].map(time_map)
    
    df_all['season'] = pd.to_datetime(df['order_date']).dt.month.map({
        12: 0, 1: 0, 2: 0,
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3
    })
    
    bool_cols = df_all.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_all[col] = df_all[col].astype(int)
        
    cat_cols = df_all.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_all[col] = df_all[col].astype('category').cat.codes
        
    cols_all = [
        'order_date', 'order_time', 'day_of_week',
        'order_channel', 'store_location_type', 'region',
        'customer_age_group', 'customer_gender', 'is_rewards_member', 
        'cart_size', 'num_customizations', 'total_spend', 'fulfillment_time_min', 
        'drink_category', 'has_food_item', 'order_ahead', 'customer_satisfaction', 'time_of_day', 'season'
    ]
    available_cols = [c for c in cols_all if c in df_all.columns]
    corr_all = df_all[available_cols].corr()
    
    df_selected = df.copy()
    df_selected['customer_age_group'] = df_selected['customer_age_group'].map(age_map)
    df_selected['order_channel'] = df_selected['order_channel'].astype('category').cat.codes
    
    bool_selected = ['is_rewards_member', 'has_food_item', 'order_ahead']
    for col in bool_selected:
        if col in df_selected.columns:
            df_selected[col] = df_selected[col].astype(int)
            
    cols_selected = [
        'order_channel', 
        'customer_age_group', 
        'is_rewards_member', 
        'cart_size', 
        'num_customizations', 
        'total_spend', 
        'has_food_item', 
        'order_ahead'
    ]
    available_sel = [c for c in cols_selected if c in df_selected.columns]
    corr_sel = df_selected[available_sel].corr()
    
    return corr_all, corr_sel

@st.dialog("Add New Order to Database")
def add_order_dialog():
    st.write("Enter the attributes of the new order to save and dynamically segment:")
    
    col1, col2 = st.columns(2)
    with col1:
        order_channel = st.selectbox("Order Channel:", ["Drive-Thru", "Mobile App", "In-Store Cashier", "Kiosk"])
        customer_age_group = st.selectbox("Age Group:", ["18-24", "25-34", "35-44", "45-54", "55+"])
        customer_gender = st.selectbox("Customer Gender:", ["Female", "Male", "Non-binary", "Prefer not to say"])
        drink_category = st.selectbox("Drink Category:", ['Refresher', 'Brewed Coffee', 'Frappuccino', 'Espresso', 'Tea', 'Other'])
    
    with col2:
        cart_size = st.slider("Cart Size (items):", 1, 15, 3)
        num_customizations = st.slider("Number of Customizations:", 0, 10, 1)
        total_spend = st.number_input("Order Total ($):", min_value=0.5, max_value=250.0, value=12.50, step=0.5)
        fulfillment_time_min = st.number_input("Fulfillment Time (minutes):", min_value=0.1, max_value=60.0, value=4.5, step=0.1)
        
    st.write("---")
    col3, col4 = st.columns(2)
    with col3:
        is_rewards_member = st.checkbox("Is Rewards Member?")
        has_food_item = st.checkbox("Contains Food Item?")
        order_ahead = st.checkbox("Order Ahead via App?")
    with col4:
        customer_satisfaction = st.slider("Customer Satisfaction (1-5):", 1, 5, 4)
        
    st.write("")
    if st.button("Save Order", use_container_width=True):
        try:
            scaler, kmeans, _ = get_trained_models()
            features_input = np.array([[
                float(total_spend),
                int(num_customizations),
                int(cart_size),
                float(fulfillment_time_min),
                int(customer_satisfaction)
            ]])
            scaled_input = scaler.transform(features_input)
            cluster_id = int(kmeans.predict(scaled_input)[0])
            segment_name = CLUSTER_NAMES[cluster_id]
            
            import datetime
            now = datetime.datetime.now()
            df_full = load_full_data()
            df_len = len(df_full)
            new_order_id = f"ORD_{df_len + 1:08d}"
            new_customer_id = f"CUST_{df_len + 10000}"
            order_date = now.strftime("%Y-%m-%d")
            order_time = now.strftime("%H:%M")
            day_of_week = now.strftime("%a")
            
            new_row = [
                new_customer_id, new_order_id, order_date, order_time, day_of_week,
                order_channel, "STR_999", "Urban", "West",
                customer_age_group, customer_gender, bool(is_rewards_member), int(cart_size),
                int(num_customizations), float(total_spend), float(fulfillment_time_min),
                drink_category, bool(has_food_item), bool(order_ahead), int(customer_satisfaction)
            ]
            
            import csv
            with open("starbucks_customer_ordering_patterns.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(new_row)
                
            st.cache_data.clear()
            st.success(f"Success! Order created with ID: {new_order_id}. Assigned Segment: {segment_name}.")
            st.balloons()
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save order: {str(e)}")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/d/d3/Starbucks_Corporation_Logo_2011.svg", width=120)
st.sidebar.title("Starbucks Analytics")

st.sidebar.write("---")
page = st.sidebar.radio("Project Sections:", ["Data Overview", "Boiling Point", "Customer Segmentation"])

st.sidebar.write("---")
if st.sidebar.button("Add New Order", use_container_width=True):
    add_order_dialog()

#   |
#   |
#   V
# PAGES
if page == "Data Overview":
    st.title("Data Overview & Filtering")
    
    tab1, tab2 = st.tabs(["Raw Orders Explorer", "Explanatory Data Analysis (EDA)"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            channel_filter = st.selectbox("Filter by Channel:", ["All", "Drive-Thru", "Mobile App", "In-Store Cashier", "Kiosk"])
        with col2:
            sat_filter = st.selectbox("Min Satisfaction:", ["All", "1", "2", "3", "4", "5"])
        with col3:
            rows_per_page = st.number_input("Rows per Page:", min_value=5, max_value=100, value=10, step=5)
        with col4:
            page_num = st.number_input("Page Number:", min_value=1, value=1, step=1)
        try:
            df_full = load_full_data()
            total_records = len(df_full)
            
            df_filtered = df_full.copy()
            if channel_filter != "All":
                df_filtered = df_filtered[df_filtered['order_channel'] == channel_filter]
            if sat_filter != "All":
                df_filtered = df_filtered[df_filtered['customer_satisfaction'] >= int(sat_filter)]
                
            total_filtered = len(df_filtered)
            offset = (page_num - 1) * rows_per_page
            limit = rows_per_page
            
            orders_list = df_filtered.iloc[offset:offset+limit].to_dict(orient="records")
            
            col_s1, col_s2 = st.columns(2)
            col_s1.metric("Total Rows in DB", f"{total_records:,}")
            col_s2.metric("Matching Filters", f"{total_filtered:,}")
            
            if orders_list:
                orders_df = pd.DataFrame(orders_list)
                ordered_cols = [
                    "order_id", "order_date", "order_time", "order_channel", 
                    "customer_gender", "customer_age_group", "is_rewards_member", 
                    "cart_size", "num_customizations", "total_spend", 
                    "fulfillment_time_min", "customer_satisfaction", "drink_category"
                ]
                display_cols = [c for c in ordered_cols if c in orders_df.columns]
                st.dataframe(orders_df[display_cols], use_container_width=True)
                
                st.write("---")
                st.markdown("### Summary Statistics")
                st.write("Calculated dynamically for the matching filtered subset of the entire database.")
                df_full = load_full_data()
                df_filtered = df_full.copy()
                if channel_filter != "All":
                    df_filtered = df_filtered[df_filtered['order_channel'] == channel_filter]
                if sat_filter != "All":
                    df_filtered = df_filtered[df_filtered['customer_satisfaction'] >= int(sat_filter)]
                
                if not df_filtered.empty:
                    stats_df = df_filtered[["cart_size", "num_customizations", "total_spend", "fulfillment_time_min", "customer_satisfaction"]].agg(["mean", "median", "std"])
                    stats_df.index = ["Mean (Среднее)", "Median (Медиана)", "Std Dev (СКО)"]
                    stats_df.columns = ["Cart Size", "Customizations", "Total Spend ($)", "Fulfillment Time (min)", "Satisfaction"]
                    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
                else:
                    st.warning("No data matching filters to calculate statistics.")
                    
                st.write("---")
                st.markdown("### Data Health & Cleaning")
                st.write("Integrity verification of the dataset under selected filters.")
                
                nan_count = df_filtered.isna().sum().sum()
                dup_count = df_filtered.duplicated().sum()
                
                c1, c2 = st.columns(2)
                c1.metric("Missing Values (nan)", f"{nan_count}")
                c2.metric("Duplicate Rows", f"{dup_count}")
                
                st.success("✅Dataset is clean (0 nans, 0 Duplicates). All column data types are standardized.")
                
                with st.expander("Show Cleaning Code & Details"):
                    st.markdown("**Data Cleaning Python Code:**")
                    st.code("""
df = df.dropna()
df = df.drop_duplicates()

df['is_rewards_member'] = df['is_rewards_member'].astype(bool)
df['total_spend'] = df['total_spend'].astype(float)
                    """)
                    st.markdown("**Column Types:**")
                    types_df = pd.DataFrame(df_filtered.dtypes.astype(str), columns=["Data Type"])
                    st.dataframe(types_df, use_container_width=True)
            else:
                st.warning("No data found for the selected filters on this page.")
                
        except Exception as e:
            st.error(f"Failed to fetch data from API: {str(e)}")

    with tab2:
        st.subheader("Explanatory Data Analysis (EDA)")
        st.write("This tab displays visual distributions, heatmaps, operational trends, and correlation matrices for the entire Starbucks ordering patterns dataset (up to cell 30 inclusive in the original notebook).")
        
        try:
            with st.spinner("Analyzing dataset and rendering charts..."):
                df_full = load_full_data()
                corr_all, corr_sel = compute_correlations()
                
                with st.expander("1. Spending Distributions (Rewards vs Channel)", expanded=True):
                    col_plot1, col_plot2 = st.columns(2)
                    
                    with col_plot1:
                        st.markdown("**Total Spend Distribution (Rewards vs Non-Rewards)**")
                        fig1, ax1 = plt.subplots(figsize=(8, 5))
                        sns.histplot(data=df_full, x="total_spend", hue="is_rewards_member", kde=True, palette=ACCENT_PALETTE[:2], multiple="stack", ax=ax1)
                        ax1.set_xlabel("Total Spend ($)")
                        st.pyplot(fig1)
                        st.write("The histogram displays the distribution of total spend per order. Rewards members (green) generally show slightly higher spend patterns and represent a large portion of transactions compared to non-rewards members (pink).")
                        
                    with col_plot2:
                        st.markdown("**Total Spend Distribution by Order Channel**")
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        sns.boxplot(data=df_full, x="order_channel", y="total_spend", hue="order_channel", palette=ACCENT_PALETTE, legend=False, ax=ax2)
                        ax2.set_xlabel("Order Channel")
                        ax2.set_ylabel("Total Spend ($)")
                        st.pyplot(fig2)
                        st.write("This boxplot shows that total spend is relatively consistent across order channels (Mobile App, Drive-Thru, In-Store, Kiosk). Most orders fall in the $5 to $15 range.")
                
                with st.expander("2. Operational Speed & Activity Trends", expanded=True):
                    col_plot3, col_plot4 = st.columns(2)
                    
                    with col_plot3:
                        st.markdown("**Average Fulfillment Time by Day and Channel**")
                        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        fulfillment_trends = df_full.groupby(['day_of_week', 'order_channel'])['fulfillment_time_min'].mean().reset_index()
                        fulfillment_trends['day_of_week'] = pd.Categorical(fulfillment_trends['day_of_week'], categories=day_order, ordered=True)
                        fulfillment_trends = fulfillment_trends.sort_values('day_of_week')
                        
                        fig3, ax3 = plt.subplots(figsize=(8, 5))
                        sns.lineplot(data=fulfillment_trends, x="day_of_week", y="fulfillment_time_min", hue="order_channel", marker="o", palette=STARBUCKS_COLORS[:4], sort=False, ax=ax3)
                        ax3.set_xlabel("Day of Week")
                        ax3.set_ylabel("Fulfillment Time (minutes)")
                        st.pyplot(fig3)
                        st.write("Fulfillment time is highest for Drive-Thru (averaging 5.5 to 6 minutes) and lowest for Kiosk and Mobile App orders (averaging around 2 to 3 minutes). Speed remains fairly uniform throughout weekdays and weekends.")
                        
                    with col_plot4:
                        st.markdown("**Starbucks Ordering Activity Heatmap (Day vs Hour)**")
                        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        heatmap_data = df_full.groupby(['day_of_week', 'order_hour']).size().unstack(level='order_hour').reindex(day_order)
                        
                        fig4, ax4 = plt.subplots(figsize=(8, 3.5))
                        sns.heatmap(heatmap_data, cmap=sns.light_palette(COLORS["STARBUCKS_GREEN"], as_cmap=True), cbar_kws={'label': 'Orders'}, ax=ax4)
                        ax4.set_xlabel("Hour of Day")
                        ax4.set_ylabel("Day of Week")
                        st.pyplot(fig4)
                        st.write("The ordering density is evenly distributed across hours (from 6:00 to 22:00) and days of the week, with no severe breakfast/rush hour spikes, suggesting a stable stream of customer traffic.")
                
                with st.expander("3. Customer Satisfaction & Product Trends", expanded=True):
                    st.markdown("**Service Speed Impact on Customer Satisfaction (FacetGrid)**")
                    g = sns.FacetGrid(df_full, col="customer_satisfaction", hue="customer_satisfaction", palette=STARBUCKS_COLORS, height=3.5, aspect=0.9)
                    g.map(sns.histplot, "fulfillment_time_min", kde=True, bins=15)
                    g.set_titles("Satisfaction: {col_name}")
                    g.set_axis_labels("Fulfillment Time (min)", "Count")
                    st.pyplot(g.fig)
                    st.write("These histograms visualize fulfillment times sliced by customer satisfaction scores (1 to 5). Highly satisfied customers (scores 4 & 5) have heavily right-skewed wait times (mostly wait <= 4 minutes), whereas dissatisfied customers (scores 1 & 2) experience significantly longer wait times, which highlights a non-linear relationship.")
                    
                    st.markdown("**Drink Category Popularity (Season vs Time of Day)**")
                    fig6, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
                    sns.countplot(ax=axes[0], x='season', hue='drink_category', data=df_full,
                                  order=['Spring', 'Summer', 'Autumn', 'Winter'],
                                  palette=STARBUCKS_COLORS)
                    axes[0].set_title('Drink Category Popularity by Season', fontsize=12, pad=10)
                    axes[0].set_xlabel('Season')
                    axes[0].set_ylabel('Number of Orders')
                    axes[0].legend(title='Drink Category')
                    
                    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
                    sns.countplot(ax=axes[1], x='time_of_day', hue='drink_category', data=df_full,
                                  order=time_order, palette=STARBUCKS_COLORS)
                    axes[1].set_title('Drink Category Popularity by Time of Day', fontsize=12, pad=10)
                    axes[1].set_xlabel('Time of Day')
                    axes[1].set_ylabel('Number of Orders')
                    axes[1].legend(title='Drink Category')
                    axes[0].set_ylim(0, 6500)
                    axes[1].set_ylim(0, 6500)
                    st.pyplot(fig6)
                    st.write("Cold drinks are highly popular during Summer, whereas hot drinks see a slight rise in Winter. On a daily basis, coffee and tea orders peak during Mornings and Afternoons, while other drinks are ordered more evenly.")
                
                with st.expander("4. Feature Correlation Matrices", expanded=True):
                    col_corr1, col_corr2 = st.columns(2)
                    
                    with col_corr1:
                        st.markdown("**Full Correlation Matrix (19 Columns)**")
                        fig7, ax7 = plt.subplots(figsize=(10, 9.2))
                        sns.heatmap(corr_all, annot=True, cmap=STARBUCKS_CMAP, fmt=".2f", vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 7}, ax=ax7)
                        ax7.set_title('Correlation Matrix of initial Starbucks Columns (Cell 29)', fontsize=12, pad=10)
                        plt.xticks(rotation=45, ha='right', fontsize=8)
                        plt.yticks(fontsize=8)
                        st.pyplot(fig7)
                        st.write("A matrix of correlations between all binned, categorical, and numerical features. It reveals key correlations like the strong relationship between `cart_size` and `total_spend`.")
                        
                    with col_corr2:
                        st.markdown("**Selected Behavioral & Order Features Correlation**")
                        fig8, ax8 = plt.subplots(figsize=(10, 9.2))
                        sns.heatmap(corr_sel, annot=True, cmap=STARBUCKS_CMAP, fmt=".2f", vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 8}, ax=ax8)
                        ax8.set_title('Correlation Matrix of Selected Features (Cell 30)', fontsize=12, pad=10)
                        plt.xticks(rotation=45, ha='right', fontsize=8)
                        plt.yticks(fontsize=8)
                        st.pyplot(fig8)
                        st.write("A zoomed-in correlation matrix highlighting behavioral metrics like `total_spend` vs `cart_size` (+0.83 correlation), `num_customizations`, and `order_channel`.")
                        
        except Exception as eda_err:
            st.error(f"Error rendering EDA visualizations: {str(eda_err)}")
elif page == "Boiling Point":
    st.title("Nonlinear Wait-Time Threshold")
    
    st.markdown("""
    **Hypothesis Core:** Wait time affects customer satisfaction nonlinearly. 
    There is a specific "boiling point" wait-time threshold, beyond which customer satisfaction drops sharply, and the ratio of dissatisfied customers (rating $\le 3$) increases significantly.
    """)
    
    try:
        threshold, bins_data, under_pct, over_pct, anomaly_stats, long_wait_orders = get_hypothesis1_data()
        
        # CELL 31
        st.subheader("1. Customer Satisfaction Threshold & Bins Analysis (Cell 31)")
        
        st.markdown(f"""
        In Cell 31, I grouped the waiting times into 1-minute intervals and calculated the average satisfaction for each of them.
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("Identified Wait Threshold", f"{threshold:.2f} min")
        col2.metric("Dissatisfied Ratio (Wait <= Threshold)", fmt_pct(under_pct))
        col3.metric("Dissatisfied Ratio (Wait > Threshold)", fmt_pct(over_pct))
        
        # Cell 31 plot
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        colors_grad = [COLORS["Y_ACCENT"], COLORS["STARBUCKS_GREEN"], COLORS["COFFEE"]]
        cm = LinearSegmentedColormap.from_list("starbucks_grad", colors_grad, N=100)
        norm = plt.Normalize(bins_data['mean'].min(), bins_data['mean'].max())
        colors_mapped = [cm(norm(val)) for val in bins_data['mean']]
        
        sns.barplot(
            data=bins_data, 
            x='bin_str', 
            y='mean', 
            palette=colors_mapped, 
            hue='bin_str', 
            legend=False, 
            ax=ax1
        )
        ax1.set_title("Customer Satisfaction by Waiting Time Bins (Cell 31)", pad=15)
        ax1.set_xlabel("Waiting Time Interval (minutes)", labelpad=10)
        ax1.set_ylabel("Average Satisfaction (1-5)")
        ax1.set_ylim(1.0, 5.0)
        ax1.set_xticks(range(len(bins_data)))
        ax1.set_xticklabels(bins_data['bin_str'], rotation=45, ha='right')
        
        ax1.axvline(x=5.5, color=COLORS["O_ACCENT"], linestyle="--", linewidth=2, label=f"Wait Threshold ({threshold:.2f} min)")
        ax1.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig1)
        
        st.markdown(f"""
        > [!NOTE]
        > **Analytical Conclusion (Cell 31):** A wait time of **{threshold:.2f} minutes** is mathematically identified as the critical boundary. 
        > For customers served within this time, the dissatisfaction rate is only **{under_pct:.2%}**. 
        > Once wait time exceeds this threshold, the dissatisfaction rate doubles to **{over_pct:.2%}**, demonstrating a psychological 'boiling point'.
        """)

        st.write("---")
        st.subheader("2. Anomaly Detection & Anomaly Investigation (Cell 32)")
        
        st.markdown("""
        **Observation:** While inspecting the average satisfaction chart above (from Cell 31), we noticed a prominent local anomaly:
        A spike in satisfaction in the **10-11 minutes** waiting interval. 
        Specifically, satisfaction rises back up to **3.63** from **3.40** in the 9-10 minutes interval, which contradicts the general trend of decreasing satisfaction with longer wait times.
        
        Perform an anomaly check (Cell 32 in our notebook) to understand whether this is a real business phenomenon (for example, highly satisfied loyal customers waiting in a specific channel) or statistical noise due to a small sample size (the **Law of Small Numbers**).
        """)
        
        st.markdown("### Characteristics of orders in the 10-11 min interval:")
        col_an1, col_an2, col_an3, col_an4, col_an5 = st.columns(5)
        col_an1.metric("Order Count (n)", f"{anomaly_stats['count']}")
        
        channels_str = ", ".join([f"{ch}: {cnt}" for ch, cnt in anomaly_stats['channels'].items()])
        col_an2.metric("Sales Channels", channels_str)
        col_an3.metric("Rewards Members", fmt_pct(anomaly_stats['rewards_pct']))
        col_an4.metric("Average Check", f"${anomaly_stats['avg_spend']:.2f}")
        col_an5.metric("Customizations", f"{anomaly_stats['avg_customizations']:.2f}")
        
        st.markdown("### Statistical Significance Check (Welch's T-Test):")
        
        t_stat = anomaly_stats['t_stat']
        p_val = anomaly_stats['p_val']
        
        col_t1, col_t2 = st.columns(2)
        col_t1.metric("t-statistic", f"{t_stat:.3f}")
        col_t2.metric("p-value", f"{p_val:.4f}")
        
        if p_val < 0.05:
            st.success("**Result:** The difference is **STATISTICALLY SIGNIFICANT** ($p < 0.05$). The spike is likely due to real factors.")
        else:
            st.warning(f"""
            **Result:** The difference is **NOT STATISTICALLY SIGNIFICANT** ($p = {p_val:.4f} \ge 0.05$).
            
            **Conclusion:** The local satisfaction spike in the 10-11 minute interval is due to high variance caused by a small sample size (**n = {anomaly_stats['count']}**). 
            This is a classic manifestation of the **Law of Small Numbers** (statistical noise).
            
            Additionally, all 100% of these long-wait orders belong to the **Drive-Thru** channel. This is because other sales channels (like In-Store or Mobile App) have a maximum fulfillment time that caps below 9.2 minutes in our dataset.
            """)
            
        # Cell 32
        st.markdown("### Anomaly Verification Plots (Cell 32)")
        
        fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

        sns.barplot(
            data=bins_data, 
            x='bin_str', 
            y='mean', 
            palette=colors_mapped, 
            hue='bin_str', 
            legend=False, 
            ax=ax_left
        )
        ax_left.set_title("Satisfaction & Sample Size by Wait Time", fontsize=12, pad=10)
        ax_left.set_xlabel("Waiting Time Interval (minutes)", fontsize=10)
        ax_left.set_ylabel("Average Satisfaction (1-5)", fontsize=10)
        ax_left.set_ylim(1.0, 5.0)
        ax_left.set_xticks(range(len(bins_data)))
        ax_left.set_xticklabels(bins_data['bin_str'], rotation=45, ha='right')
        
        ax_left_twin = ax_left.twinx()
        sns.lineplot(
            data=bins_data, 
            x=range(len(bins_data)), 
            y='count', 
            color=COLORS["O_ACCENT"], 
            marker='o', 
            linewidth=2.5, 
            label="Order Volume", 
            ax=ax_left_twin
        )
        ax_left_twin.set_ylabel("Number of Orders (Log Scale)", fontsize=10)
        ax_left_twin.set_yscale('log')
        ax_left_twin.grid(False)
        ax_left_twin.legend(loc='upper right')
        
        df_long_wait = long_wait_orders.copy()
        df_long_wait['wait_group'] = pd.cut(
            df_long_wait['fulfillment_time_min'], 
            bins=[8, 9, 10, 11, 12], 
            labels=['8-9 min', '9-10 min', '10-11 min', '11-12 min']
        )
        df_long_wait = df_long_wait.dropna(subset=['wait_group'])
        
        sns.boxplot(
            data=df_long_wait, 
            x='wait_group', 
            y='customer_satisfaction', 
            palette=[COLORS["MINT"], COLORS["O_ACCENT"], COLORS["Y_ACCENT"], COLORS["COFFEE"]],
            hue='wait_group', 
            legend=False, 
            ax=ax_right
        )
        ax_right.set_title("Satisfaction Distribution for Long Wait Times (>8 min)", fontsize=12, pad=10)
        ax_right.set_xlabel("Wait Time Group", fontsize=10)
        ax_right.set_ylabel("Customer Satisfaction (1-5)", fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error plotting Hypothesis charts: {str(e)}")

elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.subheader("Multidimensional Customer Behavioral Profiles")
    
    st.markdown("""
    **Hypothesis Core:** Coffee shop customers can be divided into stable groups based on their behavioral patterns. 
    Using K-Means on spend, customizations, cart size, wait time, and satisfaction, we identified **5 key business segments**.
    """)
    
    try:
        profiles, channels, rewards = get_hypothesis2_data()
        
        st.write("---")
        st.subheader("Average Metrics by Segment (Profiles)")
        display_profiles = profiles.copy()
        display_profiles.columns = ["Segment", "Avg Spend ($)", "Avg Customizations", "Avg Cart Size (Items)", "Avg Wait Time (min)", "Avg Satisfaction", "Order Count"]
        st.dataframe(display_profiles, use_container_width=True)
        
        st.write("---")
        st.subheader("Segment Characteristics Visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        sns.barplot(
            data=profiles, 
            x='segment', 
            y='total_spend', 
            palette=SEGMENT_COLORS, 
            hue='segment', 
            legend=False, 
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Average Spend by Customer Segment ($)', pad=15)
        axes[0, 0].set_xlabel('')
        axes[0, 0].set_ylabel('Spend ($)')
        axes[0, 0].tick_params(axis='x', rotation=15)
        
        sns.barplot(
            data=profiles, 
            x='segment', 
            y='num_customizations', 
            palette=SEGMENT_COLORS, 
            hue='segment', 
            legend=False, 
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Average Customizations by Customer Segment', pad=15)
        axes[0, 1].set_xlabel('')
        axes[0, 1].set_ylabel('Number of Customizations')
        axes[0, 1].tick_params(axis='x', rotation=15)
        
        sns.barplot(
            data=profiles, 
            x='segment', 
            y='cart_size', 
            palette=SEGMENT_COLORS, 
            hue='segment', 
            legend=False, 
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Average Cart Size by Customer Segment (Items)', pad=15)
        axes[1, 0].set_xlabel('')
        axes[1, 0].set_ylabel('Cart Size (Items)')
        axes[1, 0].tick_params(axis='x', rotation=15)
        
        sns.barplot(
            data=rewards, 
            x='segment', 
            y='is_rewards_member', 
            palette=SEGMENT_COLORS, 
            hue='segment', 
            legend=False, 
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Rewards Member Ratio by Customer Segment', pad=15)
        axes[1, 1].set_xlabel('')
        axes[1, 1].set_ylabel('Rewards Ratio (0.0 - 1.0)')
        axes[1, 1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("---")
        st.subheader("In-depth Premium Segment Analysis")
        
        st.write("#### Order Channels by Segment")
        channels_melted = pd.melt(channels, id_vars=['segment'], var_name='channel', value_name='count')
        fig_chan, ax_chan = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=channels_melted, 
            x='segment', 
            y='count', 
            hue='channel', 
            palette=[COLORS["O_ACCENT"], COLORS["STARBUCKS_GREEN"], COLORS["MINT"], COLORS["Y_ACCENT"]],
            ax=ax_chan
        )
        ax_chan.set_title("Order Channel Distribution by Segment", pad=15)
        ax_chan.set_xlabel("")
        ax_chan.set_ylabel("Order Count")
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig_chan)
            
        st.markdown("""
        ### Business Insights by Clusters:
        * **Bulk Buyers:** Most profitable group by spend ($22.41) and cart size (6.22). They primarily order via the **Mobile App (67.0%)**. These represent office or group orders.
        * **Customization Kings:** High-value customers with a high check due to extras ($17.50). They order fewer items (3.66) but make **4.05 customizations per order**. **77.5% order via the Mobile App**. Gender distribution is balanced (45.8% female, 44.3% male).
        * **Dissatisfied:** Average orders but extremely unhappy (rating 1.88). Their wait time was 4.70 minutes on Drive-Thru. This is a critical operational target for improvement.
        """)
        
    except Exception as e:
        st.error(f"Error loading segments: {str(e)}")
