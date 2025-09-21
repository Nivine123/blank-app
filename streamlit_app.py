import streamlit as st
import pandas as pd

# Default dataset URL
DEFAULT_CSV_URL = "https://linked.aub.edu.lb/pkgcube/data/551015b5649368dd2612f795c2a9c2d8_20240902_115953.csv"

# Function to load data from the URL
@st.cache_data
def load_data_from_url(url):
    try:
        # Load CSV from URL into a DataFrame
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        # Return the error message if loading fails
        return None, str(e)

# Attempt to load the data from the URL
df, err = load_data_from_url(DEFAULT_CSV_URL)

# If data fails to load, show an error message
if df is None:
    st.error(f"❌ Error loading dataset from URL: {err}")
    st.stop()  # Stop execution if no data is available

# If data is loaded successfully, show a success message
else:
    st.success("✅ Dataset loaded successfully!")

# Show the dataset preview for debugging
st.write("### Dataset Preview (First 5 Rows):")
st.dataframe(df.head())

# Function to find a column based on a list of candidate column names
def find_col(df, candidates):
    """Return first column in df whose name contains any of the candidate substrings (case-insensitive)."""
    if df is None:
        return None
    cols = df.columns.tolist()
    for cand in candidates:
        cand_l = cand.lower()
        for original in cols:
            if cand_l in original.lower():
                return original
    return None

# Try to find the 'Area' column
col_area = find_col(df, ["Area", "City", "Municipality", "District", "Caza", "area", "city"])

# Check if 'col_area' was found and display it
if col_area:
    st.info(f"✅ 'Area' column found: {col_area}")
else:
    st.warning("⚠️ 'Area' column not found in the dataset.")

