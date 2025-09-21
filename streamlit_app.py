import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re

# Page configuration
st.set_page_config(
    page_title="Tourism Analytics Dashboard", 
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1e88e5;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #e53935;
    margin: 1.5rem 0 1rem 0;
}
.insight-box {
    background-color: #e3f2fd;
    border-left: 5px solid #1e88e5;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
.metric-card {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
.context-box {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
.geographic-box {
    background-color: #f3e5f5;
    border-left: 5px solid #9c27b0;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Function to clean URLs (to extract the region name from the DBpedia links)
def clean_url(url):
    match = re.search(r'\/([^\/]+)', url)
    if match:
        return match.group(1).replace("_", " ").title()  # Remove underscores and title case
    return url

# Load and clean data
DEFAULT_CSV_URL = "https://linked.aub.edu.lb/pkgcube/data/551015b5649368dd2612f795c2a9c2d8_20240902_115953.csv"

@st.cache_data
def load_data_from_url(url):
    try:
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        return None, str(e)

# Load Data
df, err = load_data_from_url(DEFAULT_CSV_URL)
if err:
    st.sidebar.error(f"❌ Could not load from URL: {err}")

# Clean region columns if they are URLs
df["Area"] = df["Area"].apply(clean_url)
df["Governorate"] = df["Governorate"].apply(clean_url)

# Find columns for initiative and tourism index
col_initiative = "Existence of initiatives"
col_governorate = "Governorate"
col_tourism_index = "Tourism Index"

# Filtering options
governorate_choice = st.sidebar.multiselect(
    "🏛️ Select Regions",
    options=sorted(df[col_governorate].dropna().unique().tolist()),
    help="Filter analysis by specific regions",
)

# Filter the data based on the selected governorate
df_filtered = df.copy()
if governorate_choice:
    df_filtered = df_filtered[df_filtered[col_governorate].isin(governorate_choice)]

# Visualizing the initiatives by region
if col_initiative and col_governorate:
    initiatives_exist_df = df_filtered[df_filtered[col_initiative] == 1]
    
    if initiatives_exist_df.empty:
        st.warning("⚠️ No initiatives found in the selected data.")
    else:
        initiatives_by_region = initiatives_exist_df.groupby(col_governorate)[col_initiative].count().reset_index()
        initiatives_by_region.columns = [col_governorate, 'Number of Initiatives']
        
        # Bar chart for initiatives by region
        fig_bar = px.bar(
            initiatives_by_region,
            x=col_governorate,
            y='Number of Initiatives',
            title='Number of Initiatives by Region',
            labels={col_governorate: "Region", 'Number of Initiatives': 'Number of Initiatives'},
            color='Number of Initiatives',
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Pie chart for initiatives by region
        fig_pie = px.pie(
            initiatives_by_region,
            values='Number of Initiatives',
            names=col_governorate,
            title='Initiatives Distribution by Region'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
