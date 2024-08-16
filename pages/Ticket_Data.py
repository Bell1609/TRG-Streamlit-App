from __future__ import division
from io import BytesIO
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pages.fd.ticket_data_graph_drawing import Ticket_Graph_Drawing
from pages.fd.ticket_data_handling import Ticket_Data

st.set_page_config(page_title='Home Page')

st.header('Ticket Data Segmenting')

ticket_data = Ticket_Data()
ticket_graph_drawing = Ticket_Graph_Drawing()

st.sidebar.success('Select the ticket data or sales data')

st.title('Customer Segmenting App')

file = st.file_uploader('Upload your file:', ['csv', 'xlsx'])
if 'stage' not in st.session_state:
    st.session_state.stage = 0

def click_button(stage):
    st.session_state.stage = stage
    
def create_excel(df):
    output = BytesIO()
    writer =  pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    
    writer.close()
    processed_data = output.getvalue()

    return processed_data

if file:
    raw_data = ticket_data.get_raw(file)
    if not raw_data.empty:
        st.dataframe(raw_data)
        try:
            df = ticket_data.create_dataframe(raw_data)
            st.success('Dataframe created successfully.')
        except KeyError as ke:
            st.error(f'You need columns with such names: Group Company, Brand, Client code, Customer, Closed time')
            st.stop()
        except Exception as e:
            st.error(f'Error creating dataframe: {e}')
            st.stop()
            
        if st.button('Run FD Segmentation'):
            click_button(1)
        
        if st.session_state.stage >= 1:
            # Creates RFM dataframe for the segmentation
            df_ticket = ticket_data.create_df_with_relevant_info(df)

            # Creates dataframe with clusters from kmeans
            kmeans_data, cluster_centers, silhouette_score = ticket_data.create_kmeans_dataframe(df_ticket, df)
            st.header('Silhouette Score: {:0.2f}'.format(silhouette_score))
            download_data = ticket_data.create_dataframe_to_download(kmeans_data, raw_data)
            output = create_excel(download_data) # Initializes the Excel sheet

            # Creates graphs 
            figure = ticket_graph_drawing.recency_graph(df_ticket)
            st.pyplot(figure)
            plt.close()
            
            figure = ticket_graph_drawing.tickets_graph(df_ticket)
            st.pyplot(figure)
            plt.close()
            
            figure = ticket_graph_drawing.interactions_graph(df_ticket)
            st.pyplot(figure)
            plt.close()
                
            if st.button('Show treemap'):
                if st.session_state.stage < 2:
                    click_button(2)
            
            if st.session_state.stage >= 2:
                # Creates treemaps
                tree_figure = ticket_graph_drawing.treemap_drawing(cluster_centers)
                st.pyplot(tree_figure)
                plt.close()
            
            if st.button('Show scatterplot'):
                if st.session_state.stage < 3:
                    click_button(3)
            
            if st.session_state.stage >= 3:
                scatter_figure = ticket_graph_drawing.scatter_3d_drawing(kmeans_data)
                st.plotly_chart(scatter_figure)
                plt.close()
                
            if st.download_button('Download CSV', download_data.to_csv().encode('utf-8'), 'tickets-cluster.csv', 'text/csv'):
                st.write('CSV file downloaded!')
            
            if st.download_button('Download Excel', data=output, file_name='tickets-cluster.xlsx',mime='application/vnd.ms-excel'):
                st.write('Excel file downloaded!')