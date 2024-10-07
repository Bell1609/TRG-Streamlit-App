import pandas as pd
import matplotlib.pyplot as plt
import sys
import streamlit as st
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fs.data_handling import Data_Handling
from fs.graph_drawing import Graph_Drawing

data_handling = Data_Handling()
graph_drawing = Graph_Drawing()

# Main preprocessing function
def preprocess_data(df):
    # Convert 'Deal : id' to string type
    df['Deal : id'] = df['Deal : id'].astype(str)

    # Clean and convert amount columns
    df = data_handling.clean_and_convert_amount_columns(df)

    # Define mixed columns to convert to strings (replace with actual columns if needed)
    df = data_handling.convert_mixed_columns_to_string(df)

    # Convert date columns to datetime format
    df = data_handling.convert_date_columns_to_date(df)

    return df


st.header('Sales Data Insights')
st.subheader('Data Load')

# File uploaders for Deals data and Accounts data
deals_file = st.file_uploader('Upload your Deals data file:', ['csv', 'xlsx'])

if 'stage' not in st.session_state:
    st.session_state.stage = 0

# Mandatory fields for deals data
mandatory_deals_fields = [
    'Deal : Name', 'Deal : Account name', 'Deal : Closed date', 'Deal : Expected close date',
    'Deal : Total Deal Value', 'Deal : Probability (%)', 'Deal : Deal stage', 'Deal : Owner',
    'Deal : Created at'  # Ensure the required fields include 'Deal : Created at'
]

if deals_file:
    deals_data = data_handling.get_raw(deals_file)
    st.success('Data file uploaded successfully')

    if not deals_data.empty:
        # Validate mandatory fields in Deals data
        if not data_handling.validate_columns(deals_data, mandatory_deals_fields, 'Deals'):
            st.stop()

        # Preprocess the data
        deals_data = preprocess_data(deals_data)

        # Ensure the 'Deal : Created at' column is in datetime format
        deals_data['Deal : Created at'] = pd.to_datetime(deals_data['Deal : Created at'], errors='coerce')

        # Extract the min and max date range from the 'Deal : Created at' column
        min_date = deals_data['Deal : Created at'].min()
        max_date = deals_data['Deal : Created at'].max()
        st.write('Max date: ', max_date)

        # Add sidebar date input for selecting the "End Date" only
        st.sidebar.write("Deals was created on or before the selected date and deals were still opened or closed after the selected date")
        end_date = st.sidebar.date_input('End Date:', min_value=min_date, max_value=max_date, value=max_date)

        # Filtering based on the selected end date
        filtered_deals_data = deals_data[
            deals_data['Deal : Created at'] <= pd.to_datetime(end_date)
        ]

        st.markdown('Filtered data')
        st.write('Total Rows: ', filtered_deals_data['Deal : id'].count())
        st.dataframe(filtered_deals_data)


        # Assuming you already have trend data and deal counts
        trend = graph_drawing.pipeline_trend(filtered_deals_data, min_date, end_date)


        # Ensure 'Month' is in datetime format
        trend['Month'] = pd.to_datetime(trend['Month'])

        
        # Assuming trend_df is already created and available
        if not trend.empty:
            # Calculate min and max values from the 'Month' column
            min_date = trend['Month'].min().date()
            max_date = trend['Month'].max().date()
        else:
            st.error("No data available to generate trends.")
            min_date = None
            max_date = None

        # Check if min and max dates are available
        if min_date and max_date:
            # User selects the month range
            start_month = st.date_input(
                "Select Start Month", 
                value=min_date,  # Default to min date from trend_df
                min_value=min_date, 
                max_value=max_date
            )
            
            end_month = st.date_input(
                "Select End Month", 
                value=max_date,  # Default to max date from trend_df
                min_value=start_month,  # Ensure end month is after start month
                max_value=max_date
            )
        else:
            st.warning("Please load trend data to select the month range.")



        # Filter trend_df for the selected month range
        filtered_trend_df = trend[
            (trend['Month'] >= pd.to_datetime(start_month)) & 
            (trend['Month'] <= pd.to_datetime(end_month))
        ]

        st.dataframe(filtered_trend_df)
        # Call the plotting function with the filtered trend DataFrame
        graph_drawing.plot_pipeline_trend(filtered_trend_df, start_month, end_month)
