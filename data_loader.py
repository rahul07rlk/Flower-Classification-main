import pandas as pd
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_csv(csv_path="Flowers.csv"):
    """
    Load the CSV file containing flower details.
    Assumes the CSV has an index column named 'Index' and a column 'Cat_Name'.
    """
    try:
        df = pd.read_csv(csv_path, index_col=['Index'])
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def get_name(df, cat_num):
    """
    Retrieve the flower name for the given category number from the CSV DataFrame.
    """
    try:
        return df.loc[cat_num, 'Cat_Name']
    except Exception as e:
        st.error(f"Error retrieving name for category {cat_num}: {e}")
        return "Unknown"

def get_details(df, cat_num):
    """
    Retrieve additional details for the given category number.
    Returns a DataFrame with details.
    """
    try:
        temp_df = df[df.index == cat_num].T.reset_index()
        # Skip the first row (the index) and rename columns
        temp_df = temp_df.iloc[1:]
        temp_df.columns = ['Major', 'Description']
        return temp_df
    except Exception as e:
        st.error(f"Error retrieving details for category {cat_num}: {e}")
        return pd.DataFrame()
