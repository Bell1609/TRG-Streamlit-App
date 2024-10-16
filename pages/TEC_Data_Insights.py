import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from calendar import monthrange
import sys
import os
from authentication import make_sidebar # authentication instance

make_sidebar() # authentication instance

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fs.data_handling import Data_Handling
data_handling = Data_Handling()

# Function to load Excel file and display dataframe
def load_data(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

# Function for data profiling and sum statistics
def data_profiling(df):
    st.subheader('Data Profiling: Basic Statistics')

    # Get the descriptive statistics
    desc_stats = df.describe(include='all')

    # Select numeric columns for sum calculation
    numeric_df = df.select_dtypes(include=['number'])

    # Calculate sum for numeric columns
    sum_stats = numeric_df.sum()

    # Convert sum stats to a DataFrame and transpose to match the shape of `desc_stats`
    sum_stats_df = pd.DataFrame(sum_stats, columns=['sum']).transpose()

    # Append the sum row to the describe DataFrame
    desc_stats_with_sum = pd.concat([desc_stats, sum_stats_df])

    # Display the combined DataFrame with sum row
    st.write(desc_stats_with_sum)


# Function to filter data based on Name and Month
def filter_data(df):
    names = df['Name'].unique().tolist()
    months = df['Month'].unique().tolist()

    # Move the multiselect options to the sidebar
    selected_names = st.sidebar.multiselect('Select Name(s)', names)
    selected_months = st.sidebar.multiselect('Select Month(s)', months)

    # You can then use these selected values in the rest of your code
    df = df[(df['Name'].isin(selected_names)) & (df['Month'].isin(selected_months))]


    st.subheader('Filtered Data')

    return df


def plot_pie_chart(df, average_working_days):
    task_columns = ['Billable in contract', 'CMS', 'CR FOC', 'Under Estimation', 'Implementation Issue', 
                    'Presales', 'Shadow', 'Support Task', 'Internal Tasks']
    
    # Ensure that the 'Total' column exists
    if 'Total' not in df.columns:
        st.error("The 'Total' column is not found in the dataset.")
        return
    
    available_task_columns = [col for col in task_columns if col in df.columns]

    if available_task_columns:
        # Replace NaN values with 0 in task columns and 'Total' column
        df[available_task_columns] = df[available_task_columns].fillna(0)
        df['Total'] = df['Total'].fillna(0)

        # Calculate the sum of each task column relative to the 'Total' column
        task_sums = df[available_task_columns].sum()
        total_sum = df['Total'].sum()

        # If no tasks or total are present, display an error
        if total_sum == 0 or task_sums.sum() == 0:
            st.error("No task data or 'Total' available for selected categories.")
            return

        # Calculate percentages of each task relative to the total days
        task_percentages = (task_sums / total_sum) * 100

        # Group tasks with percentage < 5% into "Others" for pie chart only
        threshold = 5
        small_tasks = task_percentages[task_percentages < threshold]
        large_tasks = task_percentages[task_percentages >= threshold]

        # For the pie chart, group small tasks into "Others"
        pie_task_percentages = pd.concat([large_tasks, pd.Series([small_tasks.sum()], index=['Others'])]) if not small_tasks.empty else task_percentages

        # Create a custom autopct function to hide percentages < 5%
        def autopct_format(pct):
            return ('%1.1f%%' % pct) if pct >= threshold else ''

        # Create the pie chart
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            pie_task_percentages, 
            labels=pie_task_percentages.index, 
            autopct=autopct_format, 
            startangle=90, 
            pctdistance=0.85,  # This pushes the percentages inside the slices
            labeldistance=1.1  # Adjust label distance for better readability
        )

        # Adjust text sizes to avoid overlap
        for text in autotexts:
            text.set_fontsize(10)
        
        for text in texts:
            text.set_fontsize(10)

        # Add legend to the right of the pie chart, adjust bbox_to_anchor to prevent overlap
        ax.legend(wedges, pie_task_percentages.index, title="Task Categories", loc="center left", bbox_to_anchor=(1.3, 0.5), fontsize=10)

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')

        # Use tight layout to prevent overlap
        plt.tight_layout()

        # Display the pie chart in Streamlit
        st.pyplot(fig)
        return task_percentages, task_sums

    else:
        st.error("The required task columns are not found in the dataset.")




def add_columns(df):
    # Ensure the 'Month' column is in datetime format
    df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

    # Add a 'Total' column by summing all numeric columns row-wise
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df['Bill and Non-Bill'] = df[numeric_cols].sum(axis=1)
    
    # Function to get the number of working days in a given month (excluding weekends)
    def get_working_days(month):
        year = month.year
        month_num = month.month
        # Get the number of days in the month
        num_days = monthrange(year, month_num)[1]
        # Create a date range for the month
        date_range = pd.date_range(start=f'{year}-{month_num:02d}-01', end=f'{year}-{month_num:02d}-{num_days}')
        # Filter out weekends (Saturday=5, Sunday=6)
        working_days = date_range[~date_range.weekday.isin([5, 6])].size
        return working_days

    # Apply the working days calculation and compute the 'Others' column
    df['Working_Days'] = df['Month'].apply(get_working_days)
    working_days = df['Working_Days'].mean()
    df['Internal Tasks'] = df.apply(lambda row: row['Working_Days'] - row['Bill and Non-Bill'] if row['Bill and Non-Bill'] < row['Working_Days'] else 0, axis=1)
    df['Total'] = df['Bill and Non-Bill'] + df['Internal Tasks']
    # Drop the 'Working_Days' column as it is intermediate data
    df.drop(columns=['Working_Days'], inplace=True)

    return df, working_days

def generate_forecast_report(df, from_month, to_month):
    # Convert the 'Deal : Tentative start date/MSD' to a datetime object if not already
    df['Deal : Tentative start date/MSD'] = pd.to_datetime(df['Deal : Tentative start date/MSD'], errors='coerce')

    # Filter rows where 'Deal : Tentative start date/MSD' is within the selected range
    mask = (df['Deal : Tentative start date/MSD'] >= from_month) & (df['Deal : Tentative start date/MSD'] <= to_month)
    filtered_df = df[mask]

    # Create a new column 'Month' in 'YYYY-MM' format
    filtered_df['Month'] = filtered_df['Deal : Tentative start date/MSD'].dt.to_period('M').astype(str)


    # Group by 'Month' and calculate the sum of 'Deal Technical days' and 'Probability Technical Days'
    forecast_data = filtered_df.groupby('Month').agg(
        {'Deal Technical Days': 'sum', 'Probability Technical Days': 'sum'}
    ).reset_index()

    # Format the sums to two decimal places
    forecast_data['Deal Technical Days'] = forecast_data['Deal Technical Days'].round(2)
    forecast_data['Probability Technical Days'] = forecast_data['Probability Technical Days'].round(2)

    # Calculate the total sums
    total_technical_days = forecast_data['Deal Technical Days'].sum()
    total_probability_days = forecast_data['Probability Technical Days'].sum()

    # Append the total row to the DataFrame
    total_row = pd.DataFrame([{'Month': 'Total', 'Deal Technical Days': total_technical_days, 'Probability Technical Days': total_probability_days}])
    forecast_data = pd.concat([forecast_data, total_row], ignore_index=True)

    # Generate the HTML content
    html_content = f"""
    <html>
    <head>
        <title>Forecast Report</title>
        <style>
            table {{
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: right; /* Align text to the right for numbers */
            }}
            th {{
                background-color: #f2f2f2;
                text-align: left;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
        </style>
    </head>
    <body>
        <h2>Forecast Report</h2>
        <p>Period: {from_month.strftime('%Y-%m-%d')} to {to_month.strftime('%Y-%m-%d')}</p>
        <table>
            <tr>
                <th>Month</th>
                <th>Deal Technical Days</th>
                <th>Probability Technical Days</th>
            </tr>
    """

    # Append rows with the data
    for _, row in forecast_data.iterrows():
        html_content += f"<tr><td>{row['Month']}</td><td>{row['Deal Technical Days']:.2f}</td><td>{row['Probability Technical Days']:.2f}</td></tr>"

    # Close the table and HTML document
    html_content += """
        </table>
    </body>
    </html>
    """

    # Write the report to an HTML file
    with open('forecast_report.html', 'w') as file:
        file.write(html_content)

    return 'forecast_report.html'





def show_forecast_report(html_file_path):
    # Read the HTML file content
    with open(html_file_path, 'r') as file:
        html_content = file.read()

    # Display the HTML content in Streamlit using an iframe
    st.markdown(html_content, unsafe_allow_html=True)

#----------------------
# Main app
#----------------------
# Streamlit app setup
st.title('Task Data Analysis App')

# Upload Excel file
file = st.file_uploader('Upload booking data file', type=['csv','xlsx'])

if file:
    df = load_data(file)
    
    if df is not None:
        df, avg_working_days = add_columns(df)
               
        # # Get unique months from the dataframe
        # months = df['Month'].unique()

        # # Convert to a list of strings or dates (whichever is applicable in your case)
        # months = list(pd.to_datetime(months).strftime('%Y-%m'))  # Convert to 'YYYY-MM' format strings if they are datetime

        # # Use Streamlit's sidebar multiselect
        # selected_months = st.sidebar.multiselect('Select Month(s)', months, default=months)
        
        # Ensure the 'Deal : Expected close date' column is in datetime format
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')

        # Extract the min and max date range from the 'Deal : Expected close date' column
        min_date = df['Month'].min()
        max_date = df['Month'].max()

        # Generate a list of month-year options between the min and max dates
        month_year_range = pd.date_range(min_date, max_date, freq='MS').strftime('%m-%Y').tolist()

        # Sidebar dropdown for selecting the 'From Month'
        st.sidebar.write('Select Month Range')
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


        # You can also apply the same logic to 'Name' if needed
        names = df['Name'].unique()
        selected_names = st.sidebar.multiselect('Select Name(s)', names, names)

        # Filter the dataframe based on the selected names and months        
        filtered_df = df[
            (df['Name'].isin(selected_names)) &
            (df['Month'] >= pd.to_datetime(from_month, format='%m-%Y')) &
            (df['Month'] <= pd.to_datetime(to_month, format='%m-%Y')) 
        ]


        # Calculate total capacity
        total_capacity = avg_working_days * len(selected_names)

        # Display a summary of the total value and percentages
        st.subheader(f"TEC Capacity Allocation per Month: {int(total_capacity)}")
        
        # Plot pie chart for task percentages
        task_percentages, task_sums = plot_pie_chart(filtered_df, avg_working_days)
        
       
        # Calculate 'Days' using average working days and percentages
        calculated_days = (task_percentages / 100) * total_capacity

        # Display all tasks in the table, even the small ones not shown in pie chart
        task_details = pd.DataFrame({
            'Task': task_sums.index,
            'Days': calculated_days,  # Calculate Days based on percentages
            'Percentage of Total (%)': task_percentages.values
        })

        # Sort the task_details DataFrame by 'Days' in descending order
        task_details = task_details.sort_values(by='Days', ascending=False)

        # Format 'Days' column for better readability after sorting
        task_details['Days'] = task_details['Days'].apply(lambda x: f"{x:,.1f}")
        task_details['Percentage of Total (%)'] = task_details['Percentage of Total (%)'].apply(lambda x: f"{x:,.1f}")

        # Show the task breakdown table without the index column
        st.table(task_details.reset_index(drop=True))
        
# Load deals data file to create forecast report for TEC
        
deals_file = st.file_uploader('Upload deals data file', type=['csv','xlsx'])

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
            
        st.subheader('Data Exploration')

        # Ensure the 'Deal : Expected close date' column is in datetime format
        deals_data['Deal : Tentative start date/MSD'] = pd.to_datetime(deals_data['Deal : Tentative start date/MSD'], errors='coerce')

        # Extract the min and max date range from the 'Deal : Expected close date' column
        min_date = deals_data['Deal : Tentative start date/MSD'].min()
        max_date = deals_data['Deal : Tentative start date/MSD'].max()

        # Generate a list of month-year options between the min and max dates
        month_year_range = pd.date_range(min_date, max_date, freq='MS').strftime('%m-%Y').tolist()

        # Sidebar dropdown for selecting the 'From Month'
        st.sidebar.write('Select Tentative Start Month')
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
        selected_reports = st.sidebar.multiselect('Select Report Type', options=report_options, default=['Won', 'Pipeline'])

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
            (deals_data['Deal : Tentative start date/MSD'] >= pd.to_datetime(from_month, format='%m-%Y')) &
            (deals_data['Deal : Tentative start date/MSD'] <= pd.to_datetime(to_month, format='%m-%Y')) 
        ]
        
        # Define the recurring and non-recurring project types
        type_options = [ptype for ptype in deals_data_filtered['Deal : Project type'].unique() if ptype not in ['ARR', 'Existing - Additional users (No services)']]

        # Display filtered project types for selection
        selected_types = st.sidebar.multiselect('Select Project Type', options=type_options, default=type_options)

        # Validate 'Deal : Project Type' selection
        if not selected_types:
            st.sidebar.error("Please select at least one Project Type.")
            st.stop()  # Stop execution if no project type is selected

        # Filtering based on sidebar selections
        deals_data_filtered = deals_data_filtered[(deals_data_filtered['Deal : Project type'].isin(selected_types))]

        
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
        selected_vendors = st.sidebar.multiselect('Select Product Vendor', options=vendor_options, default=vendor_options)

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
            deals_data_filtered = data_handling.filter_by_products(deals_data_filtered, selected_products)
        else:
            st.error("'Deal : Product' column does not exist for filtering.")

        # Step 1: Add new columns to the deals_data_filtered with initial values of 0
        new_columns = [
            'Deal Software Revenue',
            'Deal Software Cost',
            'Deal ASM Revenue',
            'Deal ASM Cost',
            'Deal Service Revenue',
            'Deal Service Cost',
            'Deal Cons Days',
            'Deal PM Days',
            'Deal PA Days',
            'Deal Technical Days',
            'Deal Hosting Revenue',
            'Deal Hosting Cost',
            'Deal Managed Service Revenue',
            'Deal Managed Service Cost'
        ]

        # Initialize the new columns in the dataframe with 0 for each row
        for col in new_columns:
            deals_data_filtered[col] = 0


        # Step 3: Call the function for each selected product
        deals_data_filtered = data_handling.get_product_values(deals_data_filtered, selected_products)

        # Now, deals_data_filtered will have the new columns populated with accumulated values for the selected products.

        deals_data_filtered['Probability Technical Days'] = deals_data_filtered['Deal Technical Days'] * deals_data_filtered['Deal : Probability (%)']/100

        st.markdown('Forecast Technical Days Based on Probability and Tentative Start Date')
        # Drop columns where all rows are NaN
        deals_data_filtered = deals_data_filtered.dropna(axis=1, how='all')
        forecast_data = deals_data_filtered[['Deal : id','Deal : Tentative start date/MSD','Deal : Name', 'Deal : Product', 'Deal : Probability (%)','Deal Technical Days', 'Probability Technical Days']]
        st.dataframe(forecast_data)    
        
        # #Data profiling before segmentation
        data_handling.data_profiling(forecast_data, 'Forecast')
        
        # Add button to generate the report
        if st.button("Generate Report"):
            html_file = generate_forecast_report(deals_data_filtered, pd.to_datetime(from_month), pd.to_datetime(to_month))
            st.success(f"Report generated: {html_file}")

        # Add button to display the report
        if st.button("Display Report"):
            try:
                show_forecast_report('forecast_report.html')
            except FileNotFoundError:
                st.error("Report not found. Please generate the report first.")