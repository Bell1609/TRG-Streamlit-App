from datetime import datetime
import sys
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
# Add the parent directory to the system path
import os

import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fs.data_handling import Data_Handling
from fs.graph_drawing import Graph_Drawing

data_handling = Data_Handling()
graph_drawing = Graph_Drawing()

# Function to load data
def load_data(file):
    if file is not None:
        try:
            # Check if the file is CSV or Excel
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                st.error("Please upload a CSV or Excel file.")
                return None

            return df
        except Exception as e:
            st.error(f"Error loading the file: {e}")
            return None
    else:
        return None

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

# Tab: Won Deals
def won_deals_tab(dataframe, selected_year):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Won")
    won_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Closed date')

    # Filter the deals further to include only those with 'Deal : Deal stage' == 'Won'
    won_deals = won_deals[won_deals['Deal : Deal stage'] == 'Won']

    st.subheader("Sales Actual Dashboard")
    
    # Group 1: Overall metrics
    row1_cols = st.columns(6, gap='medium')
    with row1_cols[0]:
        total_deal_value = graph_drawing.format_number(won_deals['Deal : Total Deal Value'].sum())
        st.metric('Total Deal Value', total_deal_value)
    with row1_cols[1]:
        total_service_revenue = graph_drawing.format_number(won_deals['Deal Service Revenue'].sum())
        st.metric('Total Service Revenue', total_service_revenue)
    with row1_cols[2]:
        total_software_revenue = graph_drawing.format_number(won_deals['Deal Software Revenue'].sum())
        st.metric('Total Software Revenue', total_software_revenue)
    with row1_cols[3]:
        total_asm_revenue = graph_drawing.format_number(won_deals['Deal ASM Revenue'].sum())
        st.metric('Total ASM Revenue', total_asm_revenue)
    with row1_cols[4]:
        total_managed_service_revenue = graph_drawing.format_number(won_deals['Deal Managed Service Revenue'].sum())
        st.metric('Total Managed Service Revenue', total_managed_service_revenue)
    with row1_cols[5]:
        avg_deal_value = graph_drawing.format_number(won_deals['Deal : Total Deal Value'].mean())
        st.metric('Average Deal Value', avg_deal_value)
    
    # Group 2: Revenue progress donut charts
    row2_cols = st.columns((1.5, 4.5), gap='medium', vertical_alignment="center")
    with row2_cols[0]:
        # Calculate the revenue data
        revenue_data = graph_drawing.calculate_revenue_data(won_deals, sales_targets)
        
        # Create individual donut charts for progress
        total_chart = graph_drawing.create_donut_chart(revenue_data['total_progress'], 'Total Revenue Progress', ['#27AE60', '#12783D'])
        recurring_chart = graph_drawing.create_donut_chart(revenue_data['recurring_progress'], 'Recurring Revenue Progress', ['#29b5e8', '#155F7A'])
        non_recurring_chart = graph_drawing.create_donut_chart(revenue_data['non_recurring_progress'], 'Non-Recurring Revenue Progress', ['#E74C3C', '#781F16'])
        
        # Display charts aligned properly
        st.write('Total Deals')
        st.altair_chart(total_chart)
        st.write('Recurring Deals')
        st.altair_chart(recurring_chart)
        st.write('Non-Recurring Deals')
        st.altair_chart(non_recurring_chart)
        
    with row2_cols[1]:
        # Group 3: Sales trend visualization
        metrics = [
            'Deal : Total Deal Value', 
            'Deal : Total Cost',
            'Deal : Total Gross Margin (GM)',
            'Deal Software Revenue',
            'Deal Software Cost',
            'Deal Retained Software Revenue',
            'Deal ASM Revenue',
            'Deal ASM Cost',
            'Deal Retained ASM Revenue',
            'Deal Service Revenue',
            'Deal Service Cost',
            'Deal Retained Service Revenue',
            'Deal Cons Days',
            'Deal PM Days',
            'Deal PA Days',
            'Deal Technical Days',
            'Deal Hosting Revenue',
            'Deal Hosting Cost',
            'Deal Managed Service Revenue',
            'Deal Managed Service Cost'
        ]
        
        st.markdown("### Sales Trend")
        fig = graph_drawing.visualize_metric_over_time(won_deals, metrics)
        st.plotly_chart(fig, use_container_width=True)
        # st.markdown("### Total Revenue Growth Rate")
        # graph_drawing.plot_deal_value_growth_rate(won_deals)
        
    st.markdown("### Sales Leaderboard")
    graph_drawing.visualize_actual_vs_target_sales(won_deals, sales_targets)

    # row3_cols = st.columns((2), gap='medium')
    # with row3_cols[0]:
    #     # Group 4: Actual vs Target by Owner
    #     st.markdown("### Sales Leaderboard")
    #     graph_drawing.visualize_actual_vs_target_sales(won_deals, sales_targets)
    #     #st.plotly_chart(fig2, use_container_width=True)
    # with row3_cols[1]:
    #     st.markdown("### Average Days to Close")
    #     graph_drawing.plot_avg_days_to_close(won_deals)

    
    
    
    
    
    
    
    

# # Tab: Open Deals
# def open_deals_tab(dataframe, selected_year):
#     st.header(f"Open Deals in {selected_year}")
#     from_month, to_month = calculate_date_range(selected_year, dataframe, "Open")
#     open_deals = filter_deals(dataframe, from_month, to_month, 'Deal : Expected close date')
    
#     # Group 1: Overall metrics
#     st.subheader("Overall Metrics")
#     display_overall_metrics(open_deals)
    
#     # Group 2: Top 10 Accounts by Deal Value
#     st.subheader("Top 10 Accounts by Deal Value")
#     top_10_accounts = open_deals.nlargest(10, 'Deal : Total Deal Value')
#     st.table(top_10_accounts[['Deal : Account Name', 'Deal : Total Deal Value']])
    
#     # Group 3: Total Deal Value by Stage
#     st.subheader("Deal Value by Stage")
#     deal_value_by_stage = open_deals.groupby('Deal : Deal stage')['Deal : Total Deal Value'].sum()
#     st.bar_chart(deal_value_by_stage)
    
#     # Group 4: Pipeline Growth Trend
#     st.subheader("Pipeline Growth Trend")
#     view_by = st.selectbox("View By", ["Month", "Quarter"])
#     # Use pipeline_trend function for growth trend
#     pipeline_trend(open_deals, view_by=view_by)

# Load sales target data
with open("config/sales_targets.yaml", 'r') as file:
    sales_targets = yaml.safe_load(file)

# Mandatory columns
required_columns = ['Deal : Name', 'Deal : Account name', 'Deal : Closed date','Deal : Expected close date', 'Deal : Total Deal Value', 'Deal : Probability (%)',
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

st.set_page_config(
    page_title="TRG Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# File uploaderƒ
st.sidebar.header("Configuration")
file = st.sidebar.file_uploader("Upload Deals Data File", type=['csv', 'xlsx'])

# Load and validate data
if file:
    deals_data = load_data(file)
    
    if deals_data is not None:
        if data_handling.validate_columns(deals_data, required_columns, "Deals"):
            # Proceed with processing once columns are validated
            # Convert columns with mixed types to strings
            deals_data = preprocess_data(deals_data)
            

            # Sidebar Filters
            st.sidebar.header("Filters")

            # # Ensure the 'Deal : Closed date' column is in datetime format
            # deals_data['Deal : Closed date'] = pd.to_datetime(deals_data['Deal : Closed date'], errors='coerce')

            # # Get the current year
            # current_year = datetime.today().year

            # # Default date range: January 1st to December 31st of the current year
            # default_min_date = pd.Timestamp(f'{current_year}-01-01')
            # default_max_date = pd.Timestamp(f'{current_year}-12-31')

            # # Extract the min and max date range from the 'Deal : Closed date' column
            # min_date = deals_data['Deal : Closed date'].min() if not deals_data['Deal : Closed date'].isna().all() else default_min_date
            # max_date = deals_data['Deal : Closed date'].max() if not deals_data['Deal : Closed date'].isna().all() else default_max_date

            # # Ensure the min_date and max_date do not exceed the default current year boundaries
            # min_date = max(min_date, default_min_date)
            # max_date = min(max_date, default_max_date)

            # # Generate a list of month-year options between the min and max dates
            # month_year_range = pd.date_range(min_date, max_date, freq='MS').strftime('%m-%Y').tolist()

            # # Find the index of January (01 of current year) and December (12 of current year) for the default values
            # default_from_index = next((i for i, date in enumerate(month_year_range) if date.endswith(f'{current_year}')), 0)
            # default_to_index = len(month_year_range) - 1 if max_date.year == current_year else default_from_index

            # # Sidebar dropdown for selecting the 'From Month'
  
            # from_month = st.sidebar.selectbox(
            #     'From Month:',
            #     options=month_year_range,
            #     index=default_from_index  # Default to January of the current year
            # )

            # # Sidebar dropdown for selecting the 'To Month'
            # to_month = st.sidebar.selectbox(
            #     'To Month:',
            #     options=month_year_range,
            #     index=default_to_index  # Default to December of the current year
            # )

            # # Validate that 'To Month' is greater than or equal to 'From Month'
            # from_month_index = month_year_range.index(from_month)
            # to_month_index = month_year_range.index(to_month)

            # if to_month_index < from_month_index:
            #     st.sidebar.error("'To Month' must be greater than or equal to 'From Month'. Please select valid options.")

            # else:
            #     # Convert selected from/to months into actual date objects
            #     from_date = pd.to_datetime(f'01-{from_month}', format='%d-%m-%Y')
            #     to_date = pd.to_datetime(f'01-{to_month}', format='%d-%m-%Y') + pd.offsets.MonthEnd(1)
                
            # Filter section on the sidebar
            selected_year = data_handling.year_selection(deals_data)
            
            # Project category options: Recurring and Non-Recurring
            project_categories = ['Recurring Projects', 'Non-Recurring Projects']
            selected_categories = st.sidebar.multiselect('Select Project Category', options=project_categories, default=project_categories)

            # Validate 'Project Category' selection
            if not selected_categories:
                st.sidebar.error("Please select at least one Project Category.")
                st.stop()  # Stop execution if no category is selected

            # Define the recurring and non-recurring project types
            recurring_project_types = ['ARR', 'Existing - Additional users (No services)']
            non_recurring_project_types = [ptype for ptype in deals_data['Deal : Project type'].unique() if ptype not in recurring_project_types]

            # Filter project types based on selected categories
            if 'Recurring Projects' in selected_categories and 'Non-Recurring Projects' in selected_categories:
                # If both categories are selected, show all project types
                type_options = deals_data['Deal : Project type'].unique()
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
            deals_data = deals_data[(deals_data['Deal : Project type'].isin(selected_types))]
            
            # Add new column 'Deal : Product' that will be used to filter
            # Ensure that the filtered DataFrame is not empty before processing
            if not deals_data.empty:
                # Combine 'Deal : Product 1' to 'Deal : Product 4' into a single 'Deal : Product' column
                deals_data['Deal : Product'] = deals_data[
                    ['Deal : Product 1', 'Deal : Product 2', 'Deal : Product 3', 'Deal : Product 4']
                ].fillna('').apply(
                    lambda x: ', '.join([item for item in x if item != '']),
                    axis=1
                )

                # Optionally, remove any leading or trailing commas or spaces (if necessary)
                deals_data['Deal : Product'] = deals_data['Deal : Product'].str.strip(', ')
            else:
                st.error("No data available after filtering.")
                
            # Extract unique products from all 'Deal : Product 1' to 'Deal : Product 4' columns, excluding NaN values
            unique_products = pd.concat([
                deals_data['Deal : Product 1'],
                deals_data['Deal : Product 2'],
                deals_data['Deal : Product 3'],
                deals_data['Deal : Product 4']
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

    
            if 'Deal : Product' in deals_data.columns:
                # Filter the deals based on selected filters
                deals_data_filtered = data_handling.filter_by_products(deals_data, product_options)
            else:
                st.error("'Deal : Product' column does not exist for filtering.")

            # Create total columns of all products
            new_columns = [
                'Deal Software Revenue',
                'Deal Software Cost',
                'Deal Retained Software Revenue', # equal value of 'Deal Software Revenue' - value of 'Deal Software Cost'
                'Deal ASM Revenue',
                'Deal ASM Cost',
                'Deal Retained ASM Revenue', # equal value of 'Deal ASM Revenue' - value of 'Deal ASM Cost' 
                'Deal Service Revenue',
                'Deal Service Cost',
                'Deal Retained Service Revenue', # equal value of 'Deal Service Revenue' - value of 'Deal Service Cost'
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
            new_deals_data_filtered = data_handling.get_product_values(deals_data_filtered, product_options)

            
            
            #Main dashboard GUI
            tab1, tab2 = st.tabs(["Won Deals", "Open Deals"])
    
            with tab1:
                won_deals_tab(new_deals_data_filtered, selected_year)
            
            with tab2:
                #open_deals_tab(deals_data, selected_year)
                st.write("Tab Open Deal")
            
            # # Conditional filtering based on deal stage
            # deals_data_filtered = deals_data[
            #     (deals_data['Deal : Closed date'] >= from_date) & 
            #     (deals_data['Deal : Closed date'] <= to_date) &
    
            #     # Filter by selected stages
            #     (deals_data['Deal : Deal stage'] == 'Won')
            #]
            #time_period = st.sidebar.selectbox("Time Period", ["Monthly", "Quarterly", "Yearly"])
            
            

            # Now, deals_data_filtered will have the new columns populated with accumulated values for the selected products.
        

            # Main dashboard
            #st.title("Sales Actual Dashboard")
            # Rename 'Deal : Owner' to 'Owner'
            #new_deals_data_filtered = new_deals_data_filtered.rename(columns={'Deal : Owner': 'Owner'})

            
            
            # KPIs Row

            # total_deal_value = sums_df.loc[sums_df['Column'] == 'Deal : Total Deal Value', 'Sum'].values[0]
            # total_service_rev = sums_df.loc[sums_df['Column'] == 'Deal Service Revenue', 'Sum'].values[0]
            # total_software_rev = sums_df.loc[sums_df['Column'] == 'Deal Software Revenue', 'Sum'].values[0]
            # total_ASM_rev = sums_df.loc[sums_df['Column'] == 'Deal ASM Revenue', 'Sum'].values[0]
            # total_AMS_rev = sums_df.loc[sums_df['Column'] == 'Deal Managed Service Revenue', 'Sum'].values[0]
            # total_AMS_rev = sums_df.loc[sums_df['Column'] == 'Deal Managed Service Revenue', 'Sum'].values[0]
            # total_cons_days = sums_df.loc[sums_df['Column'] == 'Deal Cons Days', 'Sum'].values[0]
            # total_pm_days = sums_df.loc[sums_df['Column'] == 'Deal PM Days', 'Sum'].values[0]
            # total_pa_days = sums_df.loc[sums_df['Column'] == 'Deal PA Days', 'Sum'].values[0]
            # total_tech_days = sums_df.loc[sums_df['Column'] == 'Deal Technical Days', 'Sum'].values[0]
            # num_deals_closed = len(new_deals_data_filtered[new_deals_data_filtered['Deal : Deal stage'] == 'Won'])
            # lost_data = deals_data[
            #         (deals_data['Deal : Deal stage'] == 'Lost') & 
            #         (deals_data['Deal : Closed date'] >= from_date) & 
            #         (deals_data['Deal : Closed date'] <= to_date)
            #     ]
            # num_deals_lost = lost_data['Deal : Deal stage'].count()
            # col = st.columns((1.5, 4.5, 2), gap='medium')
            
        
            # col1, col2, col3, col4 = st.columns(4)


            # col1.metric("Total Deal Value Revenue", f"${total_deal_value:,.2f}")
            # col2.metric("Total Service Revenue", f"${total_service_rev:,.2f}")
            # col3.metric("Number of Deals Closed", num_deals_closed)
            # col4.metric("Number of Deals Lost", num_deals_lost)

            

            # # Total Deal Value by Owner (Horizontal Bar Chart)
            # st.header("Total Deal Value by Owner")
            # owner_deal_value = new_deals_data_filtered.groupby('Owner', as_index=False)['Deal : Total Deal Value'].sum()
            # owner_deal_value = owner_deal_value.sort_values('Deal : Total Deal Value', ascending=False)
            # fig = px.bar(owner_deal_value, x='Deal : Total Deal Value', y='Owner', orientation='h', title="Total Deal Value by Owner")  # Adjusted column names
            # st.plotly_chart(fig)
                # Example usage in the main app
            # Display the comparison for each owner
        



            # # Recurring Revenue (Bar Chart)
            # st.header("Recurring Revenue")
            # recurring_revenue = new_deals_data_filtered.groupby('Deal : Owner', as_index=False)['Deal ASM Revenue'].sum()
            # fig = px.bar(recurring_revenue, x='Deal : Owner', y='Deal ASM Revenue', title="Annual Recurring Revenue (ARR)")  # Adjusted column names
            # st.plotly_chart(fig)

            
         

        else:
            st.error("The uploaded file does not contain all the mandatory columns.")
else:
    st.info("Please upload a Deals Data File (CSV or Excel).")
