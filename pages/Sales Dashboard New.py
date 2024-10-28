import datetime as dt
import sys
import streamlit as st
import pandas as pd
import altair as alt
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

def filter_won_deals(dataframe, selected_year):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Won")
    #st.write("Won Deals")
    #st.write(f"From month: {from_month} - To Month: {to_month}")
    won_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Closed date')

    # Filter the deals further to include only those with 'Deal : Deal stage' == 'Won'
    won_deals = won_deals[won_deals['Deal : Deal stage'] == 'Won']
    return won_deals

def filter_open_deals(dataframe, selected_year):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Open")
    #st.write("Open Deals")
    #st.write(f"From month: {from_month} - To Month: {to_month}")
    open_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Expected close date')

    # Filter the deals further to include only those with 'Deal : Deal stage' == 'Won'
    open_deals = open_deals[
        (open_deals['Deal : Deal stage'] != "Lost")
        ]

    return open_deals


# Filter all new deals created within the selected_year
def filter_new_deals(dataframe, selected_year):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Pipeline")
    new_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Created at')

    return new_deals

# Filter all pipeline deals expected closed within the selected_year
def filter_pipeline_deals(dataframe, selected_year, probability):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Open")
    pipeline_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Expected close date')
    
     # Filter the deals further to include only those with 'Deal : Deal probability (%)'
    pipeline_deals = pipeline_deals[
        (pipeline_deals['Deal : Probability (%)'] >= probability) &
        (~pipeline_deals['Deal : Deal stage'].isin(['Won', 'Lost']))
    ]
    return pipeline_deals

def filter_pipeline_backdate_deals(dataframe, selected_year):
    """Generate the trend of total deal value and deal count in the pipeline backdate grouped by month."""
    
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Pipeline")
    # st.write("Pipeline Deals")
    # st.write(f"From month: {from_month} - To Month: {to_month}")
    
    # Generate a range of month-ends from start to end
    date_range = pd.date_range(start=from_month, end=to_month, freq='M')

    # Convert DatetimeIndex to a list to allow appending
    date_range_list = date_range.tolist()

    # Adjust the time to 23:59:59 for each date in the list
    date_range_list = [date.replace(hour=23, minute=59, second=59) for date in date_range_list]

    # Convert end_date to a pandas Timestamp if it is not already
    end_date_ts = pd.Timestamp(to_month).replace(hour=23, minute=59, second=59)

    # If the exact end_date is not already in the date range, add it
    if end_date_ts not in date_range_list:
        date_range_list.append(end_date_ts)

    # Sort the list of dates to maintain chronological order
    date_range_list = sorted(date_range_list)

    # Convert the list back to a DatetimeIndex
    date_range = pd.DatetimeIndex(date_range_list)

    
    def pipeline_value_and_count_at_month(df, month_end):
        """Calculate total deal value and count of deals in the pipeline as of the end of a given month."""
        # Extend end_date to include the full day
        #end_of_month = pd.to_datetime(month_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        
        # Calculate the start of the month based on month_end
        month_start = month_end.replace(day=1)

        # Filter deals that were in the pipeline during the given month
        pipeline_deals = df[
            (df['Deal : Created at'] <= month_end) &  # Deal was created on or before the month end
            ((df['Deal : Closed date'].isna()) | (df['Deal : Closed date'] > month_end))  # Deal is still open or closed after the month end
        ]
        # st.write(f'Start: {month_start} - End: {end_of_month}')
        # st.write(f'Rows: {pipeline_deals["Deal : id"].count()}')
        # st.dataframe(pipeline_deals[['Deal : Name','Deal : Total Deal Value','Deal : Owner','Deal : Project type']])
        # Sum the total deal value for the filtered deals
        total_value = pipeline_deals['Deal : Total Deal Value'].sum()

        # Count deals created in the current month (between month_start and month_end)
        deals_created = df[
            (df['Deal : Created at'] >= month_start) &  
            (df['Deal : Created at'] <= month_end)
        ]
        deal_created_count = deals_created['Deal : id'].nunique()
        
        deals_closed = df[
            (df['Deal : Closed date'] >= month_start) &  
            (df['Deal : Closed date'] <= month_end) &
            (df['Deal : Deal stage'] == 'Won')
        ]
        deal_closed_count = deals_closed['Deal : id'].nunique()

        return total_value, deal_created_count, deal_closed_count

    # Initialize lists to store results
    months = []
    as_at_date = []
    total_values = []
    deal_created_counts = []
    deal_closed_counts = []

    # Calculate total deal value and deal count for each month in the date range
    for month_end in date_range:
        total_value, deal_created_count, deal_closed_count = pipeline_value_and_count_at_month(dataframe, month_end)
        months.append(month_end.strftime('%Y-%m'))
        as_at_date.append(month_end)
        total_values.append(total_value)  # Store total value
        deal_created_counts.append(deal_created_count)  # Store deal count
        deal_closed_counts.append(deal_closed_count)  # Store deal count

    
    # Create a DataFrame to return
    trend_df = pd.DataFrame({
        'Month': months,
        'As At Date': as_at_date,
        'Total Deal Value': total_values,
        'Deals Created Count': deal_created_counts,
        'Deals Closed Count': deal_closed_counts
    })

    return trend_df

# Add last year columns to pipeline value backdate by months
def add_last_year_columns(pipeline_deals, pre_pipeline_deals):
    # Convert 'Month' columns to datetime format to make comparison easier
    pipeline_deals['Month'] = pd.to_datetime(pipeline_deals['Month'], format='%Y-%m')
    pre_pipeline_deals['Month'] = pd.to_datetime(pre_pipeline_deals['Month'], format='%Y-%m')

    # Initialize the new columns in pipeline_deals with NaN (numeric type, float)
    pre_pipeline_deals['Current Year Total Deal Value'] = pd.Series(dtype='float64')
    pre_pipeline_deals['Current Year Deals Created Count'] = pd.Series(dtype='float64')
    pre_pipeline_deals['Current Year Deals Closed Count'] = pd.Series(dtype='float64')

    # Loop through each row in pipeline_deals
    for index, row in pre_pipeline_deals.iterrows():
        # Calculate the current year month
        current_year_month = row['Month'] + pd.DateOffset(years=1)
        
        # Find the corresponding row in pipeline_deals with the same month as current_year_month
        current_year_data = pipeline_deals[pipeline_deals['Month'] == current_year_month]
        
        # If matching row exists in pre_pipeline_deals, update the columns
        if not current_year_data.empty:
            pre_pipeline_deals.at[index, 'Current Year Total Deal Value'] = current_year_data['Total Deal Value'].values[0]
            pre_pipeline_deals.at[index, 'Current Year Deals Created Count'] = current_year_data['Deals Created Count'].values[0]
            pre_pipeline_deals.at[index, 'Current Year Deals Closed Count'] = current_year_data['Deals Closed Count'].values[0]
    
    # Ensure the new columns are numeric
    pre_pipeline_deals['Current Year Total Deal Value'] = pd.to_numeric(pre_pipeline_deals['Current Year Total Deal Value'], errors='coerce')
    pre_pipeline_deals['Current Year Deals Created Count'] = pd.to_numeric(pre_pipeline_deals['Current Year Deals Created Count'], errors='coerce')
    pre_pipeline_deals['Current Year Deals Closed Count'] = pd.to_numeric(pre_pipeline_deals['Current Year Deals Closed Count'], errors='coerce')
    
    return pre_pipeline_deals

def won_deals_tab(dataframe, selected_year):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Won")
    won_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Closed date')

    # Filter the deals further to include only those with 'Deal : Deal stage' == 'Won'
    won_deals = won_deals[won_deals['Deal : Deal stage'] == 'Won']
    with st.container(border= True):
        #st.subheader(f"Sales Performance Dashboard {selected_year}")
        st.markdown(f"## Sales Overview {selected_year}")

        # Row 1: Overall metrics within a bordered container
        row1_cols = st.columns(6, gap='medium')
        with row1_cols[0]:
            with st.container(border=True):
                total_deal_value = graph_drawing.format_number(won_deals['Deal : Total Deal Value'].sum())
                st.metric('Total Deal Value', total_deal_value)
        with row1_cols[1]:
            with st.container(border=True):
                total_service_revenue = graph_drawing.format_number(won_deals['Deal Service Revenue'].sum())
                st.metric('Total Service Revenue', total_service_revenue)
        with row1_cols[2]:
            with st.container(border=True):
                total_software_revenue = graph_drawing.format_number(won_deals['Deal Software Revenue'].sum())
                st.metric('Total Software Revenue', total_software_revenue)
        with row1_cols[3]:
            with st.container(border=True):
                total_asm_revenue = graph_drawing.format_number(won_deals['Deal ASM Revenue'].sum())
                st.metric('Total ASM Revenue', total_asm_revenue)
        with row1_cols[4]:
            with st.container(border=True):
                total_managed_service_revenue = graph_drawing.format_number(won_deals['Deal Managed Service Revenue'].sum())
                st.metric('Total AMS Revenue', total_managed_service_revenue)
        with row1_cols[5]:
            with st.container(border=True):
                avg_deal_value = graph_drawing.format_number(won_deals['Deal : Total Deal Value'].mean())
                st.metric('Average Deal Value', avg_deal_value)
    
        st.markdown('<div> </div>', unsafe_allow_html=True)
        # Row 2: Revenue progress donut charts and sales trend visualization within a bordered container
        
        # CSS for centering elements inside row2_cols[0]
        alignment_css = """
            <style>
            .center-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                height: 680px;
                border: 1px solid #d3d3d3;
                border-radius: 10px;
                padding: 15px;
            }
            </style>
        """
        st.markdown(alignment_css, unsafe_allow_html=True)
        
        row2_cols = st.columns((1, 4), gap='medium', vertical_alignment="center")
        with row2_cols[0]:
            with st.container(height=680,border=True):
            
                # Calculate the revenue data
                revenue_data = graph_drawing.calculate_revenue_data(won_deals, sales_targets)
                
                # Create individual donut charts for progress
                total_chart = graph_drawing.create_donut_chart(revenue_data['total_progress'], 'Total Revenue Progress', ['#27AE60', '#12783D'])
                recurring_chart = graph_drawing.create_donut_chart(revenue_data['recurring_progress'], 'Recurring Revenue Progress', ['#29b5e8', '#155F7A'])
                non_recurring_chart = graph_drawing.create_donut_chart(revenue_data['non_recurring_progress'], 'Non-Recurring Revenue Progress', ['#E74C3C', '#781F16'])
                st.markdown("##### Actual vs Target")
                # Display charts aligned properly
                st.write('Total Deals')
                st.altair_chart(total_chart)
                st.write('Recurring Deals')
                st.altair_chart(recurring_chart)
                st.write('Non-Recurring Deals')
                st.altair_chart(non_recurring_chart)
       
            
        with row2_cols[1]:
            # Sales trend visualization
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
            with st.container(height=680,border=True):
                st.markdown("### Sales Trend")
                fig = graph_drawing.visualize_metric_over_time(won_deals, metrics)
                st.plotly_chart(fig, use_container_width=True)
    

        # Sales Leaderboard
        st.markdown("### Sales Leaderboard")
        graph_drawing.visualize_actual_vs_target_sales(won_deals, sales_targets)

    
st.set_page_config(
    page_title="TRG Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Load sales target data
with open("config/sales_targets.yaml", 'r') as file:
    sales_targets = yaml.safe_load(file)
    
# Load css style
# with open("config/style.css") as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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



# File uploaderƒ
st.sidebar.header("Configuration")
file_deals = st.sidebar.file_uploader("Upload Deals Data File", type=['csv', 'xlsx'])
#file_acct = st.sidebar.file_uploader("Upload Accounting Data File", type=['csv','xlsx'])
# Add filter View By on the sidebar
view_by = st.sidebar.selectbox(
    "View Trend By",
    options=["Monthly", "Quarterly"],
    help="Select option to view the chart over time by month or quarter"
)

probability = st.sidebar.number_input(
    "Deal Probability (%)",
    min_value=10,
    max_value=100,
    value=80,
    step=10,
    help="Enter deal probability % to view forecast deals"
)
# Load and validate data
if file_deals:
    deals_data = load_data(file_deals)
    #acct_data = load_data(file_acct)
    
    if deals_data is not None:
        deals_data = preprocess_data(deals_data)
        # Sidebar Filters
        st.sidebar.header("Filters")
    
        # Filter section on the sidebar
        selected_year = data_handling.year_selection(deals_data)
        
        
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
        # Filter the deals based on selected filters
        deals_data_filtered = data_handling.filter_by_products(deals_data, product_options)


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
        # Get the current month
        current_month = dt.datetime.now().month
        current_year = dt.datetime.now().year
        
        # Styling for containers using Streamlit's "st.container" and "st.columns"
        with tab1:
            #st.write(current_month)
            #sales actual dashboard
            won_deals_tab(new_deals_data_filtered, selected_year)
            #sale over view dashboard
            won_deals = filter_won_deals(new_deals_data_filtered, selected_year)
            pre_won_deals = filter_won_deals(new_deals_data_filtered, selected_year - 1)
            open_deals = filter_open_deals(new_deals_data_filtered, selected_year)
            pre_open_deals = filter_open_deals(new_deals_data_filtered, selected_year - 1)
            pipeline_deals = filter_pipeline_backdate_deals(new_deals_data_filtered, selected_year)
            pre_pipeline_deals = filter_pipeline_backdate_deals(new_deals_data_filtered, selected_year - 1)
            pre_pipeline_deals = add_last_year_columns(pipeline_deals, pre_pipeline_deals)
            new_deals_year = filter_new_deals(new_deals_data_filtered, selected_year)
            opps_deals_probability = filter_pipeline_deals(new_deals_data_filtered, selected_year, probability)
            
            # Calculate total deal values
            cur_won_value = won_deals['Deal : Total Deal Value'].sum()
            pre_won_value = pre_won_deals['Deal : Total Deal Value'].sum()
            delta_won = cur_won_value - pre_won_value

            cur_open_value = open_deals['Deal : Total Deal Value'].sum()
            pre_open_value = pre_open_deals['Deal : Total Deal Value'].sum()
            delta_open = cur_open_value - pre_open_value

            
            # Get the count of deals created in the specified range
            new_deals_year_count = new_deals_year.shape[0]
            # Get the total deal value for the specified range
            new_deals_year_value = new_deals_year['Deal : Total Deal Value'].sum()
            
            # Calculate conversion rate (Won Deals / Open Deals * 100)
            won_deals_count = won_deals.shape[0]
            open_deals_count = open_deals.shape[0]
            conversion_rate = (won_deals_count / new_deals_year_count * 100) if new_deals_year_count > 0 else 0
            
            
            # Combine them to form 'yyyy-mm' format
            current_year_month = f"{selected_year}-{current_month}"
        

            # Add a 'Created Month' column to the dataframe in 'yyyy-mm' format
            new_deals_year['Created Month'] = new_deals_year['Deal : Created at'].dt.strftime('%Y-%m')

            # Further filter new_deals_year by current year and month
            new_deals_month = new_deals_year[new_deals_year['Created Month'] == current_year_month]
            
            # # Add a 'Created Month' column to the dataframe in 'yyyy-mm' format
            # new_deals_year['Created Month'] = new_deals_year['Deal : Created at'].dt.strftime('%Y-%m')
            # # Further filter new_deals_year by current month and probability
            # new_deals_month = new_deals_year[new_deals_year['Created Month'] == current_month]

            # Get the count and total deal value for the filtered deals
            deals_created_month_count = new_deals_month.shape[0]
            total_deal_month_value = new_deals_month['Deal : Total Deal Value'].sum()
            
            # Further filter new_deals_year by probability
            opps_deals_probability_count = opps_deals_probability.shape[0]
            # Get the total deal value for the specified range
            opps_deals_probability_value = opps_deals_probability['Deal : Total Deal Value'].sum()
            
            

            # Create main row of three columns
            row2_cols = st.columns((1, 3, 2), gap="small", vertical_alignment="center")

            # Custom CSS to add space between containers
            st.markdown(
                """
                <style>
                .metric-container {
                    margin-bottom: 30px;  /* Adjust the margin as needed */
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Use st.markdown to apply the custom CSS
            with row2_cols[0]:
                with st.container(height=450, border=True):  # No class_ argument here
                    # Custom CSS for font size
                    st.markdown(
                        """
                        <style>
                        .header {
                            font-size: 20px;
                            font-weight: bold;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    # Applying the custom CSS to the header
                    st.markdown('<p class="header">Total Deal Value</p>', unsafe_allow_html=True)

                    
                    # Add metrics with a custom class applied via markdown
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("All Deals", f"$ {graph_drawing.format_number(cur_open_value)}", graph_drawing.format_number(delta_open), delta_color="normal",help="YTD total value of created deals which have probability > 0")
                    #st.markdown('</div>', unsafe_allow_html=True)

                    #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Won Deals", f"$ {graph_drawing.format_number(cur_won_value)}", graph_drawing.format_number(delta_won), delta_color="normal", help="YTD won deals total value")
                    #st.markdown('</div>', unsafe_allow_html=True)

                    # Add Conversion Rate metric
                    #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Conversion Rate", f"{conversion_rate:.2f}%", help="Ratio of won deals vs all created deals within the year")
                    st.markdown('</div>', unsafe_allow_html=True)


            # Column [1]: Container for the chart
            with row2_cols[1]:
                with st.container(height=450,border=True):
                    # Plot the graph
                    fig = graph_drawing.visualize_pipeline_deal_values_current_last_year(pre_pipeline_deals, view_by)
                    st.plotly_chart(fig, use_container_width=True)
        

            # Column [2]: Container for three additional metrics
            with row2_cols[2]:
                with st.container(height=450,border=True):
                    with st.container(height=130, border=True):
                        # visualize deals created count and total deal value in current month and by probability
                        cols =  st.columns(2)
                 
                        cols[0].metric(f"Deals in pipeline from {probability}%", opps_deals_probability_count, help=f"Deals expected to close in the selected year which have probability >= {probability}")
                        cols[1].metric(" ", f"$ {graph_drawing.format_number(opps_deals_probability_value)}")

                    with st.container(height=130, border=True):
                        cols =  st.columns(2)
                        # visualize deals created count and total deal value of current month
                        cols[0].metric("Deals created MTD", deals_created_month_count, help="MTD created deals, including lost deals")
                        cols[1].metric(" ", f"$ {graph_drawing.format_number(total_deal_month_value)}")
                    

                    with st.container(height=130, border=True):
                        # visualize deals created count and total deal value of selected year
                        cols =  st.columns(2)
                        cols[0].metric("Deals created YTD", new_deals_year_count, help="YTD created deals, including lost deals")
                        cols[1].metric(" ", f"$ {graph_drawing.format_number(new_deals_year_value)}")




        with tab2:
            #open_deals_tab(deals_data, selected_year)
            st.write("Tab Open Deal")
        
        
else:
    st.info("Please upload Data Files (CSV or Excel).")
