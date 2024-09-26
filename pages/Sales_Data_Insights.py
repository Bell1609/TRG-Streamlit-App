from __future__ import division
from io import BytesIO
import time
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import sys
import os
import sweetviz as sv
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
import stat


# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fs.data_handling import Data_Handling
from fs.graph_drawing import Graph_Drawing

data_handling = Data_Handling()
graph_drawing = Graph_Drawing()

def convert_mixed_columns_to_string(df):
        for col in df.columns:
            if df[col].dtype == 'object' and pd.api.types.infer_dtype(df[col]) == 'mixed':
                try:
                    df[col] = df[col].astype(str)
                    st.warning(f"Column '{col}' contained mixed types. It has been converted to string.")
                except Exception as e:
                    st.error(f"Failed to convert column '{col}' to string: {e}")
        return df


def clean_and_convert_amount_columns(df):
    # Define the columns to process
    columns_to_process = [
        'Deal : Total Deal Value', 'Deal : Deal value in Base Currency',
        'Deal : Expected deal value', 'Deal : Total Cost', 'Deal : Gross Margin (GM)',
        'Deal : Software revenue: Product 1', 'Deal : Software revenue: Product 2', 'Deal : Software revenue: Product 3', 'Deal : Software revenue: Product 4',
        'Deal : Software cost: Product 1', 'Deal : Software cost: Product 2', 'Deal : Software cost: Product 3', 'Deal : Software cost: Product 4',
        'Deal : ASM revenue: Product 1', 'Deal : ASM revenue: Product 2', 'Deal : ASM revenue: Product 3', 'Deal : ASM revenue: Product 4',
        'Deal : ASM cost: Product 1', 'Deal : ASM cost: Product 2', 'Deal : ASM cost: Product 3', 'Deal : ASM cost: Product 4',
        'Deal : Service revenue: Product 1', 'Deal : Service revenue: Product 2', 'Deal : Service revenue: Product 3', 'Deal : Service revenue: Product 4',
        'Deal : Service cost: Product 1', 'Deal : Service cost: Product 2', 'Deal : Service cost: Product 3', 'Deal : Service cost: Product 4',
        'Deal : Cons days: Product 1', 'Deal : Cons days: Product 2', 'Deal : Cons days: Product 3', 'Deal : Cons days: Product 4',
        'Deal : Technical days: Product 1', 'Deal : Technical days: Product 2', 'Deal : Technical days: Product 3', 'Deal : Technical days: Product 4',
        'Deal : PM days: Product 1', 'Deal : PM days: Product 2', 'Deal : PM days: Product 3', 'Deal : PM days: Product 4',
        'Deal : PA days: Product 1', 'Deal : PA days: Product 2', 'Deal : PA days: Product 3', 'Deal : PA days: Product 4',
        'Deal : Hosting revenue: Product 1', 'Deal : Hosting revenue: Product 2', 'Deal : Hosting revenue: Product 3', 'Deal : Hosting revenue: Product 4',
        'Deal : Managed service revenue: Product 1', 'Deal : Managed service revenue: Product 2', 'Deal : Managed service revenue: Product 3', 'Deal : Managed service revenue: Product 4',
        'Deal : Managed service cost: Product 1', 'Deal : Managed service cost: Product 2', 'Deal : Managed service cost: Product 3', 'Deal : Managed service cost: Product 4'
    ]
    
    # Convert columns to numeric
    for col in columns_to_process:
        if col in df.columns:
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
    
    # Sum columns
    df['Deal : Software revenue'] = df[['Deal : Software revenue: Product 1', 'Deal : Software revenue: Product 2', 'Deal : Software revenue: Product 3', 'Deal : Software revenue: Product 4']].sum(axis=1)
    df['Deal : Software cost'] = df[['Deal : Software cost: Product 1', 'Deal : Software cost: Product 2', 'Deal : Software cost: Product 3', 'Deal : Software cost: Product 4']].sum(axis=1)
    df['Deal : ASM revenue'] = df[['Deal : ASM revenue: Product 1', 'Deal : ASM revenue: Product 2', 'Deal : ASM revenue: Product 3', 'Deal : ASM revenue: Product 4']].sum(axis=1)
    df['Deal : ASM cost'] = df[['Deal : ASM cost: Product 1', 'Deal : ASM cost: Product 2', 'Deal : ASM cost: Product 3', 'Deal : ASM cost: Product 4']].sum(axis=1)
    df['Deal : Service revenue'] = df[['Deal : Service revenue: Product 1', 'Deal : Service revenue: Product 2', 'Deal : Service revenue: Product 3', 'Deal : Service revenue: Product 4']].sum(axis=1)
    df['Deal : Service cost'] = df[['Deal : Service cost: Product 1', 'Deal : Service cost: Product 2', 'Deal : Service cost: Product 3', 'Deal : Service cost: Product 4']].sum(axis=1)
    df['Deal : Cons days'] = df[['Deal : Cons days: Product 1', 'Deal : Cons days: Product 2', 'Deal : Cons days: Product 3', 'Deal : Cons days: Product 4']].sum(axis=1)
    df['Deal : Technical days'] = df[['Deal : Technical days: Product 1', 'Deal : Technical days: Product 2', 'Deal : Technical days: Product 3', 'Deal : Technical days: Product 4']].sum(axis=1)
    df['Deal : PM days'] = df[['Deal : PM days: Product 1', 'Deal : PM days: Product 2', 'Deal : PM days: Product 3', 'Deal : PM days: Product 4']].sum(axis=1)
    df['Deal : PA days'] = df[['Deal : PA days: Product 1', 'Deal : PA days: Product 2', 'Deal : PA days: Product 3', 'Deal : PA days: Product 4']].sum(axis=1)
    df['Deal : Hosting revenue'] = df[['Deal : Hosting revenue: Product 1', 'Deal : Hosting revenue: Product 2', 'Deal : Hosting revenue: Product 3', 'Deal : Hosting revenue: Product 4']].sum(axis=1)
    df['Deal : Managed service revenue'] = df[['Deal : Managed service revenue: Product 1', 'Deal : Managed service revenue: Product 2', 'Deal : Managed service revenue: Product 3', 'Deal : Managed service revenue: Product 4']].sum(axis=1)
    df['Deal : Managed service cost'] = df[['Deal : Managed service cost: Product 1', 'Deal : Managed service cost: Product 2', 'Deal : Managed service cost: Product 3', 'Deal : Managed service cost: Product 4']].sum(axis=1)

    
    return df

    
# Function to convert date columns to datetime format
def convert_date_columns_to_date(df):
    date_columns = [
        'Deal : Closed date', 
        'Deal : Expected close date', 
        'Deal : Created at', 
        'Deal : Updated at', 
        'Deal : Last assigned at', 
        'Deal : First assigned at', 
        'Deal : Deal stage updated at', 
        'Deal : Last activity date', 
        'Deal : Expected go live date/MED', 
        'Deal : Tentative start date/MSD', 
        'Deal : Commitment Expiration Date'
    ]
    
    for col in date_columns:
        if col in df.columns:
            # Convert to datetime using the format YYYY-MM-DD
            df[col] = pd.to_datetime(df[col], dayfirst=True, format='mixed', errors='coerce')
    
    return df

# Main preprocessing function
def preprocess_data(df):
    # Clean and convert amount columns
    df = clean_and_convert_amount_columns(df)
    # Drop the original columns
    df.drop(columns=[
        'Deal : Software revenue: Product 1', 'Deal : Software revenue: Product 2', 'Deal : Software revenue: Product 3', 'Deal : Software revenue: Product 4',
        'Deal : Software cost: Product 1', 'Deal : Software cost: Product 2', 'Deal : Software cost: Product 3', 'Deal : Software cost: Product 4',
        'Deal : ASM revenue: Product 1', 'Deal : ASM revenue: Product 2', 'Deal : ASM revenue: Product 3', 'Deal : ASM revenue: Product 4',
        'Deal : ASM cost: Product 1', 'Deal : ASM cost: Product 2', 'Deal : ASM cost: Product 3', 'Deal : ASM cost: Product 4',
        'Deal : Service revenue: Product 1', 'Deal : Service revenue: Product 2', 'Deal : Service revenue: Product 3', 'Deal : Service revenue: Product 4',
        'Deal : Service cost: Product 1', 'Deal : Service cost: Product 2', 'Deal : Service cost: Product 3', 'Deal : Service cost: Product 4',
        'Deal : Cons days: Product 1', 'Deal : Cons days: Product 2', 'Deal : Cons days: Product 3', 'Deal : Cons days: Product 4',
        'Deal : Technical days: Product 1', 'Deal : Technical days: Product 2', 'Deal : Technical days: Product 3', 'Deal : Technical days: Product 4',
        'Deal : PM days: Product 1', 'Deal : PM days: Product 2', 'Deal : PM days: Product 3', 'Deal : PM days: Product 4',
        'Deal : PA days: Product 1', 'Deal : PA days: Product 2', 'Deal : PA days: Product 3', 'Deal : PA days: Product 4',
        'Deal : Hosting revenue: Product 1', 'Deal : Hosting revenue: Product 2', 'Deal : Hosting revenue: Product 3', 'Deal : Hosting revenue: Product 4',
        'Deal : Managed service revenue: Product 1', 'Deal : Managed service revenue: Product 2', 'Deal : Managed service revenue: Product 3', 'Deal : Managed service revenue: Product 4',
        'Deal : Managed service cost: Product 1', 'Deal : Managed service cost: Product 2', 'Deal : Managed service cost: Product 3', 'Deal : Managed service cost: Product 4'
    ], inplace=True)
    
    # Define mixed columns to convert to strings (replace with actual columns if needed)
    df = convert_mixed_columns_to_string(df)
    
    # Convert date columns to datetime format
    df = convert_date_columns_to_date(df)
    
    return df

# Function to generate ydata_profiling report and save it
def generate_ydata_profiling_report(df, title):
    report = ProfileReport(df, title=title)
    report_file = f"{title} Report.html"  # Specify the file name
    report.to_file(report_file)            # Save the report as an HTML file
    return report_file                     # Return the file path

# Display existing profiling report function
def display_ydata_profiling_report(report_file_path):
    try:
        with open(report_file_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        components.html(report_html, height=700, scrolling=True)

    except PermissionError:
        st.error(f"Permission denied when trying to access {report_file_path}. Please check file permissions.")
    except FileNotFoundError:
        st.error(f"The file {report_file_path} does not exist. Please generate the report first.")
    except OSError as e:
        st.error(f"OS error occurred: {e}")
    except UnicodeDecodeError:
        st.error("Error decoding the profiling report. The file might contain incompatible characters.")
        
def set_file_permissions(file_path):
    try:
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        print(f"Permissions set to 644 for file: {file_path}")
        # Check permissions after setting
        permissions = oct(os.stat(file_path).st_mode)[-3:]
        print(f"Current permissions: {permissions}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except PermissionError:
        print(f"Permission denied: {file_path}")
    except OSError as e:
        print(f"OS error occurred: {e}")

# def data_profiling(df, df_name):
#     st.markdown(f'**{df_name} Data Profiling**')
#     st.write(f"Data Types for {df_name} data:")
#     st.write(df.dtypes)
#     st.write(f"Missing Values in {df_name} data:")
#     st.write(df.isnull().sum())
#     st.write(f"Basic Statistics for {df_name} data:")
#     st.write(df.describe())
    
# def data_profiling(df, df_name):
#     st.markdown(f'**{df_name} Data Profiling**')
#     st.write(f"Basic Statistics for {df_name} data:")
    
#     # Select only numeric columns for statistics
#     numeric_df = df.select_dtypes(include=['number'])

#     # Get the descriptive statistics using describe()
#     desc = numeric_df.describe()

#     # Calculate the sum for each numeric column and append it as a new row
#     sum_row = pd.DataFrame(numeric_df.sum(), columns=['sum']).T

#     # Concatenate the sum row with the describe() output
#     desc_with_sum = pd.concat([desc, sum_row])

#     # Display the statistics in Streamlit
#     st.write(desc_with_sum)


def data_profiling(df, df_name):
    st.markdown(f'**{df_name} Data Profiling**')
    st.write(f"Basic Statistics for {df_name} data:")
    
    # Select only numeric columns for statistics
    numeric_df = df.select_dtypes(include=['number'])

    # Get the descriptive statistics using describe()
    desc = numeric_df.describe()

    # Calculate the sum for each numeric column and append it as a new row
    sum_row = pd.DataFrame(numeric_df.sum(), columns=['sum']).T

    # Concatenate the sum row with the describe() output
    desc_with_sum = pd.concat([desc, sum_row])

    # Display the statistics in Streamlit
    st.write(desc_with_sum)


# Function to generate and display Sweetviz report
def generate_sweetviz_report(df, df_name):
    report = sv.analyze(df)
    report_name = f"{df_name}_report.html"
    report.show_html(filepath=report_name, open_browser=False)
    return report_name

def display_sweetviz_report(report_name):
    try:
        with open(report_name, 'r', encoding='utf-8') as f:
            report_html = f.read()
        components.html(report_html, height=700, scrolling=True)
    except UnicodeDecodeError:
        st.error("Error decoding the Sweetviz report. The file might contain characters that are not compatible with the default encoding.")
        

st.sidebar.success('Select the ticket data or sales data')

st.header('Sales Data Segmenting')

st.subheader('Data Load')

# File uploaders for Deals data and Accounts data
deals_file = st.file_uploader('Upload your Deals data file:', ['csv', 'xlsx'])
#accounts_file = st.file_uploader('Upload your Accounts data file:', ['csv', 'xlsx'])

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def click_button(stage):
    st.session_state.stage = stage

def create_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    
    writer.close()
    processed_data = output.getvalue()

    return processed_data

def filter_data_by_ranking(download_data):
    unique_rankings = download_data['Ranking'].unique().tolist()
    
    # Ensure there are unique values to select
    if unique_rankings:
        selected_rankings = st.multiselect('Select Clusters to Filter:', unique_rankings)
        
        if selected_rankings:
            # Filter the data based on the selected rankings
            filtered_data = download_data[download_data['Ranking'].isin(selected_rankings)]
            
            # Count the number of records where 'TRG Customer' is 'Yes' and 'No'
            trg_customer_yes_count = filtered_data[filtered_data['Account : TRG Customer'] == 'Yes'].shape[0]
            trg_customer_no_count = filtered_data[filtered_data['Account : TRG Customer'] == 'No'].shape[0]
            
            # Display the counts
            st.markdown(f"**Total 'TRG Customer' Count:**")
            st.markdown(f"- **Yes:** {trg_customer_yes_count}")
            st.markdown(f"- **No:** {trg_customer_no_count}")
            
            st.markdown(f'**Filtered Data by Rankings: {", ".join(selected_rankings)}**')
            st.dataframe(filtered_data)
            
            return filtered_data
        else:
            st.warning("Please select at least one ranking value to filter.")
            return download_data
    else:
        st.warning("No unique 'Ranking' values found to filter.")
        return download_data



# Mandatory fields for deals data
# mandatory_deals_fields = ['Deal : Account name', 'Deal : Closed date','Deal : Expected close date', 'Deal : Total Deal Value', 'Deal : Probability (%)',
#                           'Deal : Deal stage','Deal : Owner','Deal : Project type','Deal : Source','Deal : Total Cost','Deal : Gross Margin (GM)',
#                           'Deal : Software revenue: Product 1','Deal : Software revenue: Product 2','Deal : Software revenue: Product 3','Deal : Software revenue: Product 4',
#                           'Deal : Software cost: Product 1','Deal : Software cost: Product 2','Deal : Software cost: Product 3','Deal : Software cost: Product 4',
#                           'Deal : ASM revenue: Product 1','Deal : ASM revenue: Product 2','Deal : ASM revenue: Product 3','Deal : ASM revenue: Product 4',
#                           'Deal : ASM cost: Product 1','Deal : ASM cost: Product 2','Deal : ASM cost: Product 3','Deal : ASM cost: Product 4',
#                           'Deal : Service revenue: Product 1','Deal : Service revenue: Product 2','Deal : Service revenue: Product 3','Deal : Service revenue: Product 4',
#                           'Deal : Service cost: Product 1','Deal : Service cost: Product 2','Deal : Service cost: Product 3','Deal : Service cost: Product 4',
#                           'Deal : Cons days: Product 1','Deal : Cons days: Product 2','Deal : Cons days: Product 3','Deal : Cons days: Product 4',
#                           'Deal : Technical days: Product 1','Deal : Technical days: Product 2','Deal : Technical days: Product 3','Deal : Technical days: Product 4',
#                           'Deal : PM days: Product 1','Deal : PM days: Product 2','Deal : PM days: Product 3','Deal : PM days: Product 4',
#                           'Deal : PA days: Product 1','Deal : PA days: Product 2','Deal : PA days: Product 3','Deal : PA days: Product 4',
#                           'Deal : Hosting revenue: Product 1','Deal : Hosting revenue: Product 2','Deal : Hosting revenue: Product 3','Deal : Hosting revenue: Product 4',
#                           'Deal : Managed service revenue: Product 1','Deal : Managed service revenue: Product 2','Deal : Managed service revenue: Product 3','Deal : Managed service revenue: Product 4',
#                           'Deal : Managed service cost: Product 1','Deal : Managed service cost: Product 2','Deal : Managed service cost: Product 3','Deal : Managed service cost: Product 4']
#mandatory_accounts_fields = ['SalesAccount : id','Account : Name', 'Account : TRG Customer']

# Validation for mandatory fields
def validate_columns(df, mandatory_fields, file_type):
    missing_fields = [field for field in mandatory_fields if field not in df.columns]
    if missing_fields:
        st.error(f'The {file_type} data is missing the following mandatory columns: {", ".join(missing_fields)}')
        return False
    return True

if deals_file:
    deals_data = data_handling.get_raw(deals_file)
    
    if not deals_data.empty:
        # Convert columns with mixed types to strings
        deals_data = preprocess_data(deals_data)
        
        # Validate mandatory fields in Deals and Accounts data
        # if not validate_columns(deals_data, mandatory_deals_fields, 'Deals'):
        #     st.stop()
            
        # # Keep only the columns that are in mandatory_deals_fields
        # deals_data = deals_data[mandatory_deals_fields]
        #deal_output = create_excel(deals_data)

               

        st.subheader('Data Exploration')
        
        # Extract unique years from 'Created time' column
        #deals_data['Deal : Expected close date'] = pd.to_datetime(deals_data['Deal : Expected close date'])
        unique_years = deals_data['Deal : Expected close date'].dt.year.unique()
        unique_years.sort()

        # Sidebar dropdown for selecting years
        year_options = st.sidebar.multiselect(
            'Select Expected Close Years',
            options=unique_years,
            default=unique_years
        )

        # Filter deals data by 'Deal : Deal Stage'
        stage_options = deals_data['Deal : Deal stage'].unique()
        selected_stages = st.sidebar.multiselect('Select Deal Stages', options=stage_options, default=['Won'])
        
        type_options = deals_data['Deal : Project type'].unique()
        selected_types = st.sidebar.multiselect('Select Product Type', options=type_options, default=type_options)
        
        # Add a sidebar selectbox for 'Deal : Type of Renewal' if it exists in the dataset
        selected_type_of_renewal = st.sidebar.multiselect('Select Type of Renewal:', deals_data['Deal : Type of Renewal'].unique())

        
        # Filtering based on sidebar selections
        deals_data_filtered = deals_data[
            (deals_data['Deal : Deal stage'].isin(selected_stages)) &
            (deals_data['Deal : Project type'].isin(selected_types)) &
            (deals_data['Deal : Type of Renewal'].isin(selected_type_of_renewal)) &
            (deals_data['Deal : Expected close date'].dt.year.isin(year_options))
        ]

        st.markdown('Processed and Filtered Deals Data')
        st.dataframe(deals_data_filtered)  
        
        # Display all columns and their data types
        column_types = deals_data_filtered.dtypes
        st.subheader("Deals Data Filtered: Column Data Types")
        st.write(column_types)
        
        #Data profiling before segmentation
        data_profiling(deals_data_filtered, 'Deals')
        
        # Set default report file paths (in the same folder as the application)
        deals_report_file_path = 'Deals Data Report.html'
        #accounts_report_file_path = 'Accounts Data Report.html'
        
        # Generate Profiling Report Button
        if st.button('Generate Deals Profiling Reports'):
            # Generate the reports
            st.markdown('**Generating Deals Data Profile Report...**')
            deals_report_file_path = generate_ydata_profiling_report(deals_data, 'Deals Data')
                
            st.success('Reports generated successfully!')

        if st.button('Display Deals Profiling Reports'):
            # Validate if the report files exist before displaying them
            st.markdown('**Deals Data Profile Report**')
            if os.path.exists(deals_report_file_path):
                set_file_permissions(deals_report_file_path)
                display_ydata_profiling_report(deals_report_file_path)
            else:
                st.error('Deals Data Report does not exist. Please generate the report first.')

            st.markdown('**Accounts Data Profile Report**')

        
        #Duong

