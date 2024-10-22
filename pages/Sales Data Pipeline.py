from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import sys
import streamlit as st
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fs.data_handling import Data_Handling
from fs.graph_drawing import Graph_Drawing
from authentication import make_sidebar

make_sidebar()
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
        

        # Add sidebar date input for selecting the "End Date" only
        st.sidebar.write("Deals was created on or before the selected date and deals were still opened or closed after the selected date")
        end_date = st.sidebar.date_input('Select Date:', min_value=min_date, max_value=max_date, value=max_date)

        # Extend end_date to include the full day
        end_of_day = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        st.write('Data is calculated up to date: ', end_date)
        # Filtering based on the selected end date (including full end_date)
        filtered_deals_data = deals_data[
            deals_data['Deal : Created at'] <= end_of_day
        ]


        # st.markdown('Filtered data')
        # st.write('Total Rows: ', filtered_deals_data['Deal : id'].count())
        # st.dataframe(filtered_deals_data)


        # Assuming you already have trend data and deal counts
        trend = graph_drawing.pipeline_trend(filtered_deals_data, min_date, end_date)


        # Ensure 'Month' is in datetime format
        trend['As At Date'] = pd.to_datetime(trend['As At Date'])

        # Assuming trend_df is already created and available
        if not trend.empty:
            # Calculate min and max values from the 'Month' column
            min_date = trend['As At Date'].min().date()
            max_date = trend['As At Date'].max().date()

        else:
            st.error("No data available to generate trends.")
            min_date = None
            max_date = None

        # Check if min and max dates are available
        if min_date and max_date:
            # Get the current month and year
            current_month = datetime.now().strftime('%Y-%m')
            
            # Calculate the date 12 months back from the current month
            twelve_months_back_date = datetime.now() - timedelta(days=365)
            twelve_months_back = twelve_months_back_date.strftime('%Y-%m')
            
            # Convert min and max dates to 'MM-YYYY' format
            min_month = min_date.strftime('%Y-%m')
            max_month = max_date.strftime('%Y-%m')
            
            # Generate the list of months in 'MM-YYYY' format
            month_options = pd.date_range(start=min_date, end=max_date, freq='MS').strftime('%Y-%m').tolist()
            
            # User selects the month range
            start_month = st.selectbox(
                "Select Start Month", 
                options=month_options,
                index=month_options.index(twelve_months_back) if twelve_months_back in month_options else 0  # Default to 12 months back or min month
            )
            
            end_month = st.selectbox(
                "Select End Month", 
                options=month_options,
                index=month_options.index(current_month)  # Default to current month of current year
            )
            
        
        else:
            st.warning("Please load trend data to select the month range.")
        
        # Call the plotting function with the filtered trend DataFrame
        graph_drawing.plot_pipeline_trend(trend, start_month, end_month)
