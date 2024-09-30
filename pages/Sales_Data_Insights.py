from __future__ import division
from io import BytesIO
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import sys
import os
import sweetviz as sv
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
import stat
import numpy as np


# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fs.data_handling import Data_Handling
from fs.graph_drawing import Graph_Drawing

data_handling = Data_Handling()
graph_drawing = Graph_Drawing()


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
        

st.header('Sales Data Insights')

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

# Main preprocessing function
def preprocess_data(df):
    
    # Convert 'Deal : id' to string type
    df['Deal : id'] = df['Deal : id'].astype(str)

    # Clean and convert amount columns
    df = data_handling.clean_and_convert_amount_columns(df)
    # Drop the original columns
    
    
    # Define mixed columns to convert to strings (replace with actual columns if needed)
    df = data_handling.convert_mixed_columns_to_string(df)
    
    # Convert date columns to datetime format
    df = data_handling.convert_date_columns_to_date(df)
    
    return df

# Mandatory fields for deals data
mandatory_deals_fields = ['Deal : Name', 'Deal : Account name', 'Deal : Closed date','Deal : Expected close date', 'Deal : Total Deal Value', 'Deal : Probability (%)',
                          'Deal : Deal stage','Deal : Owner','Deal : Project type','Deal : Source','Deal : Total Cost','Deal : Gross Margin (GM)', 'Deal : Age (in days)',
                          'Deal : Tentative start date/MSD', 'Deal : Expected go live date/MED', 'Deal : Type of Renewal',
                          'Deal : Product 1', 'Deal : Product 2', 'Deal : Product 3', 'Deal : Product 4',
                          'Deal : Software revenue: Product 1','Deal : Software revenue: Product 2','Deal : Software revenue: Product 3','Deal : Software revenue: Product 4',
                          'Deal : Software cost: Product 1','Deal : Software cost: Product 2','Deal : Software cost: Product 3','Deal : Software cost: Product 4',
                          'Deal : ASM revenue: Product 1','Deal : ASM revenue: Product 2','Deal : ASM revenue: Product 3','Deal : ASM revenue: Product 4',
                          'Deal : ASM cost: Product 1','Deal : ASM cost: Product 2','Deal : ASM cost: Product 3','Deal : ASM cost: Product 4',
                          'Deal : Service revenue: Product 1','Deal : Service revenue: Product 2','Deal : Service revenue: Product 3','Deal : Service revenue: Product 4',
                          'Deal : Service cost: Product 1','Deal : Service cost: Product 2','Deal : Service cost: Product 3','Deal : Service cost: Product 4',
                          'Deal : Cons days: Product 1','Deal : Cons days: Product 2','Deal : Cons days: Product 3','Deal : Cons days: Product 4',
                          'Deal : Technical days: Product 1','Deal : Technical days: Product 2','Deal : Technical days: Product 3','Deal : Technical days: Product 4',
                          'Deal : PM days: Product 1','Deal : PM days: Product 2','Deal : PM days: Product 3','Deal : PM days: Product 4',
                          'Deal : PA days: Product 1','Deal : PA days: Product 2','Deal : PA days: Product 3','Deal : PA days: Product 4',
                          'Deal : Hosting revenue: Product 1','Deal : Hosting revenue: Product 2','Deal : Hosting revenue: Product 3','Deal : Hosting revenue: Product 4',
                          'Deal : Managed service revenue: Product 1','Deal : Managed service revenue: Product 2','Deal : Managed service revenue: Product 3','Deal : Managed service revenue: Product 4',
                          'Deal : Managed service cost: Product 1','Deal : Managed service cost: Product 2','Deal : Managed service cost: Product 3','Deal : Managed service cost: Product 4']
#mandatory_accounts_fields = ['SalesAccount : id','Account : Name', 'Account : TRG Customer']

if deals_file:
    deals_data = data_handling.get_raw(deals_file)
    st.success('Data file uploaded successfully')
    
    if not deals_data.empty:
        # Validate mandatory fields in Deals and Accounts data
        if not data_handling.validate_columns(deals_data, mandatory_deals_fields, 'Deals'):
            st.stop()
            
        # Convert columns with mixed types to strings
        deals_data = preprocess_data(deals_data)

            
        # # Keep only the columns that are in mandatory_deals_fields
        # deals_data = deals_data[mandatory_deals_fields]
        #deal_output = create_excel(deals_data)


        st.subheader('Data Exploration')
    

        # Ensure the 'Deal : Expected close date' column is in datetime format
        deals_data['Deal : Expected close date'] = pd.to_datetime(deals_data['Deal : Expected close date'], errors='coerce')

        # Extract the min and max date range from the 'Deal : Expected close date' column
        min_date = deals_data['Deal : Expected close date'].min()
        max_date = deals_data['Deal : Expected close date'].max()

        # Generate a list of month-year options between the min and max dates
        month_year_range = pd.date_range(min_date, max_date, freq='MS').strftime('%m-%Y').tolist()

        # Sidebar dropdown for selecting the 'From Month'
        st.sidebar.write('Select Expected Close Month')
        from_month = st.sidebar.selectbox(
            'From:',
            options=month_year_range,
            index=0  # Default to the first month
        )

        # Sidebar dropdown for selecting the 'To Month'
        to_month = st.sidebar.selectbox(
            'To:',
            options=month_year_range,
            index=len(month_year_range) - 1  # Default to the last month
        )

        # Validate that 'To Month' is greater than or equal to 'From Month'
        from_month_index = month_year_range.index(from_month)
        to_month_index = month_year_range.index(to_month)

        if to_month_index < from_month_index:
            st.sidebar.error("'To Month' must be greater than or equal to 'From Month'. Please select valid options.")

        # Report options for user selection
        report_options = ['Won', 'Pipeline']
        selected_reports = st.sidebar.multiselect('Select Report Type', options=report_options, default=['Won'])

        # Validate 'Report Type' selection
        if not selected_reports:
            st.sidebar.error("Please select at least one type of report.")
            st.stop()  # Stop execution if no report type is selected

        # Deal stage filtering logic based on report type
        if 'Won' in selected_reports and 'Pipeline' in selected_reports:
            # If both "Won" and "Pipeline" are selected, include "Won" and all other stages except "Lost"
            stage_options = [stage for stage in deals_data['Deal : Deal stage'].unique() if stage != 'Lost']
        elif 'Won' in selected_reports:
            # If only "Won" is selected, restrict stages to "Won"
            stage_options = ['Won']
        elif 'Pipeline' in selected_reports:
            # If only "Pipeline" is selected, include all stages except "Won" and "Lost"
            stage_options = [stage for stage in deals_data['Deal : Deal stage'].unique() if stage not in ['Won', 'Lost']]


        # Display filtered deal stages for selection
        selected_stages = st.sidebar.multiselect('Select Deal Stages', options=stage_options, default=stage_options)

        # Validate 'Deal Stage' selection
        if not selected_stages:
            st.sidebar.error("Please select at least one Deal Stage.")
            st.stop()  # Stop execution if no deal stage is selected

        # Filtering based on sidebar selections, including the new product filter
        deals_data_filtered = deals_data[
            (deals_data['Deal : Deal stage'].isin(selected_stages)) &
            (deals_data['Deal : Expected close date'] >= pd.to_datetime(from_month, format='%m-%Y')) &
            (deals_data['Deal : Expected close date'] <= pd.to_datetime(to_month, format='%m-%Y')) 
        ]
        
        # Project category options: Recurring and Non-Recurring
        project_categories = ['Recurring Projects', 'Non-Recurring Projects']
        selected_categories = st.sidebar.multiselect('Select Project Category', options=project_categories)

        # Validate 'Project Category' selection
        if not selected_categories:
            st.sidebar.error("Please select at least one Project Category.")
            st.stop()  # Stop execution if no category is selected

        # Define the recurring and non-recurring project types
        recurring_project_types = ['ARR', 'Existing - Additional users (No services)']
        non_recurring_project_types = [ptype for ptype in deals_data_filtered['Deal : Project type'].unique() if ptype not in recurring_project_types]

        # Filter project types based on selected categories
        if 'Recurring Projects' in selected_categories and 'Non-Recurring Projects' in selected_categories:
            # If both categories are selected, show all project types
            type_options = deals_data_filtered['Deal : Project type'].unique()
        elif 'Recurring Projects' in selected_categories:
            # If only 'Recurring Projects' is selected, show only recurring project types
            type_options = recurring_project_types
        elif 'Non-Recurring Projects' in selected_categories:
            # If only 'Non-Recurring Projects' is selected, show only non-recurring project types
            type_options = non_recurring_project_types

        # Display filtered project types for selection
        selected_types = st.sidebar.multiselect('Select Project Type', options=type_options, default=type_options)

        # Validate 'Deal : Project Type' selection
        if not selected_types:
            st.sidebar.error("Please select at least one Project Type.")
            st.stop()  # Stop execution if no project type is selected

        # Filtering based on sidebar selections
        deals_data_filtered = deals_data_filtered[(deals_data_filtered['Deal : Project type'].isin(selected_types))]

        # Add a sidebar selectbox for 'Deal : Type of Renewal' if it exists in the dataset
        renewal_options = deals_data_filtered['Deal : Type of Renewal'].unique()
    
        selected_type_of_renewal = st.sidebar.multiselect(
            'Select Type of Renewal:',
            options=renewal_options,
            default=renewal_options if renewal_options.size > 0 else None
            )
        if not selected_type_of_renewal:
            st.sidebar.error("Please select at least one Type of Renewal.")
            st.stop()  # Stop execution if no project type is selected

        deals_data_filtered = deals_data_filtered[(deals_data_filtered['Deal : Type of Renewal'].isin(selected_type_of_renewal))]
        
        # Add new column 'Deal : Product' that will be used to filter
        # Ensure that the filtered DataFrame is not empty before processing
        if not deals_data_filtered.empty:
            # Combine 'Deal : Product 1' to 'Deal : Product 4' into a single 'Deal : Product' column
            deals_data_filtered['Deal : Product'] = deals_data_filtered[
                ['Deal : Product 1', 'Deal : Product 2', 'Deal : Product 3', 'Deal : Product 4']
            ].fillna('').apply(
                lambda x: ', '.join([item for item in x if item != '']),
                axis=1
            )

            # Optionally, remove any leading or trailing commas or spaces (if necessary)
            deals_data_filtered['Deal : Product'] = deals_data_filtered['Deal : Product'].str.strip(', ')
        else:
            st.error("No data available after filtering.")

        # Extract unique products from all 'Deal : Product 1' to 'Deal : Product 4' columns, excluding NaN values
        unique_products = pd.concat([
            deals_data_filtered['Deal : Product 1'],
            deals_data_filtered['Deal : Product 2'],
            deals_data_filtered['Deal : Product 3'],
            deals_data_filtered['Deal : Product 4']
        ]).dropna().unique()


        # Multi-selection for Product Vendor with options: "Infor", "TRG", and "Others"
        vendor_options = ['Infor', 'TRG', 'Others']
        selected_vendors = st.sidebar.multiselect('Select Product Vendor', options=vendor_options)

        # Validate that at least one vendor is selected
        if not selected_vendors:
            st.sidebar.error("Please select at least one Product Vendor.")
            st.stop()

        # Define product filtering logic based on selected vendors
        if 'Infor' in selected_vendors or 'TRG' in selected_vendors or 'Others' in selected_vendors:
            product_options = [
                product for product in unique_products
                if ('Infor' in product and 'Infor' in selected_vendors)
                or ('TRG' in product and 'TRG' in selected_vendors)
                or ('Infor' not in product and 'TRG' not in product and 'Others' in selected_vendors)
            ]
        else:
            # If all vendors are selected or no specific filtering is needed, include all products
            product_options = sorted(unique_products)

        # Sort the final product options for better UX (optional)
        product_options = sorted(product_options)

        # Multi-selection for Products based on the filtered product options
        selected_products = st.sidebar.multiselect(
            'Select Product(s)',
            product_options,
            default=product_options  # Default to all options based on vendor selection
        )

        # Validate that at least one product is selected
        if not selected_products:
            st.sidebar.error("Please select at least one Product.")
            st.stop()


        # Ensure that the 'Deal : Product' column exists before trying to filter
        if 'Deal : Product' in deals_data_filtered.columns:
            # Filter the deals based on selected filters
            deals_data_filtered = deals_data_filtered[(data_handling.product_filter(deals_data_filtered['Deal : Product'], selected_products))]
        else:
            st.error("'Deal : Product' column does not exist for filtering.")

        # Filter account manager by 'Deal : Owner'
        owner_options = deals_data_filtered['Deal : Owner'].dropna().unique()

        selected_owners = st.sidebar.multiselect(
            'Select Deal Owners',
            options=owner_options,
            default=owner_options if owner_options.size > 0 else None
        )
        if not selected_owners:
            st.sidebar.error("Please select at least one Owner.")
            st.stop()  # Stop execution if no project type is selected

        deals_data_filtered = deals_data_filtered[(deals_data_filtered['Deal : Owner'].isin(selected_owners))]

        # Initialize a dictionary to hold the aggregated values for all selected products
        total_values = {
            'Deal Software revenue': 0,
            'Deal Software cost': 0,
            'Deal ASM revenue': 0,
            'Deal ASM cost': 0,
            'Deal Service revenue': 0,
            'Deal Service cost': 0,
            'Deal Cons days': 0,
            'Deal PM days': 0,
            'Deal PA days': 0,
            'Deal Technical days': 0,
            'Deal Hosting revenue': 0,
            'Deal Hosting cost': 0,
            'Deal Managed service revenue': 0,
            'Deal Managed service cost': 0
        }

        # Loop through all selected products and aggregate the values
        for product in selected_products:
            product_values = data_handling.get_product_values(deals_data_filtered, product, total_values)

            # Add the product-specific values to the total values
            for key in total_values:
                total_values[key] += product_values[key]

        # Now, let's insert the total values as new columns into deals_data_filtered
        for key in total_values:
            # Add the total value as a new column to the filtered deals DataFrame
            deals_data_filtered[key] = total_values[key]

           
        st.markdown('Processed and Filtered Deals Data')
        # Drop columns where all rows are NaN
        deals_data_filtered = deals_data_filtered.dropna(axis=1, how='all')
        st.dataframe(deals_data_filtered)  
    
        
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

        st.subheader('Data Visualization')

        # Drop-down box to allow the user to select a revenue/cost type
        selected_metric = st.selectbox(
            "Select Revenue or Cost Type to Visualize",
            options=list(total_values.keys())
        )

        # Check if the selected column exists in the DataFrame
        if selected_metric in deals_data_filtered.columns:
            # Ensure 'Date' column is in datetime format
            deals_data_filtered['Date'] = pd.to_datetime(deals_data_filtered['Deal : Expected close date'])

            # Sort by date to ensure proper ordering
            deals_data_filtered = deals_data_filtered.sort_values('Date')

            # Group by month and sum the selected metric
            trend_data = deals_data_filtered.resample('M', on='Date')[selected_metric].sum().reset_index()

            # Calculate the mean and sum value for the metric across all months
            mean_value = deals_data_filtered[selected_metric].mean()
            sum_value = deals_data_filtered[selected_metric].sum()

            # Calculate the total number of months between the min and max date
            date_range_in_months = (trend_data['Date'].dt.year - trend_data['Date'].dt.year.min()) * 12 + trend_data['Date'].dt.month - trend_data['Date'].dt.month.min()
            total_months = date_range_in_months.max()

            # Set the x-tick frequency: every 3 months if range > 36 months, else every month
            if total_months > 36:
                xtick_freq = 3
            else:
                xtick_freq = 1

            # Plot the bar chart for each month's value and the mean as a line
            plt.figure(figsize=(12, 6))  # Increase figure size for readability

            # Bar chart for the monthly values
            plt.bar(trend_data['Date'].dt.strftime('%Y-%m'), trend_data[selected_metric], label=f'{selected_metric} (Monthly)', color='skyblue')

            # Line plot for the mean value across all months
            plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean {selected_metric}')

            # Rotate the x-ticks to show them as 'yyyy-mm' with appropriate jump (1 or 3 months)
            plt.xticks(ticks=trend_data['Date'][::xtick_freq].dt.strftime('%Y-%m'), rotation=45, ha='right')

            # Enhance the plot with titles and labels
            plt.title(f'{selected_metric} Over Time (Monthly and Mean)', fontsize=16)
            plt.xlabel('Month', fontsize=14)
            plt.ylabel(selected_metric, fontsize=14)

            # Add a legend
            plt.legend()

            # Add gridlines for better readability
            plt.grid(visible=True, which='both', linestyle='--', linewidth=0.7)

            # Format sum and mean with commas and currency (USD)
            formatted_sum_value = f"${sum_value:,.2f}"  # Formats with commas and adds a $ symbol
            formatted_mean_value = f"${mean_value:,.2f}"

            # Display the sum and mean values outside the chart as a label with currency and grouping
            plt.text(1.01, 0.95, f"Total: {formatted_sum_value}\nMean: {formatted_mean_value}", 
                    transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))



            # Adjust the layout to prevent clipping of tick labels
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(plt)

        else:
            st.error(f"The column '{selected_metric}' does not exist in the data.")


