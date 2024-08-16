from __future__ import division
from io import BytesIO
import streamlit as st 
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter

from pages.fs.data_handling import Data_Handling
from pages.fs.graph_drawing import Graph_Drawing

data_handling = Data_Handling()
graph_drawing = Graph_Drawing()

st.sidebar.success('Select the ticket data or sales data')

st.header('Sales Data Segmenting')

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
    raw_data = data_handling.get_raw(file)
    if not raw_data.empty:
        st.dataframe(raw_data)
        try:
            df = data_handling.create_dataframe(raw_data)
            st.success('Dataframe created successfully.')
        except KeyError as ke:
            st.error(f'You need columns with such names: AccountID, CloseDate, DealValue, DealStage')
            st.stop()
        except Exception as e:
            st.error(f'Error creating dataframe: {type(e)}')
            st.stop()
            
        if st.button('Run RFM Segmentation'):
            click_button(1)
        
        if st.session_state.stage >= 1:
            print(df.describe())
            # Creates RFM dataframe for the segmentation
            rfm_data = data_handling.create_rfm_dataframe(df)

            # Creates dataframe with clusters from kmeans
            kmeans_data, cluster_centers, silhouette_score = data_handling.create_kmeans_dataframe(rfm_data)
            download_data = data_handling.create_dataframe_to_download(kmeans_data, raw_data)
            st.header('Silhouette Score: {:0.2f}'.format(silhouette_score))
            output = create_excel(download_data) # Initializes the Excel sheet

            # Creates graphs 
            for component, color in zip(['Recency', 'Frequency', 'Monetary'], ['blue', 'green', 'orange']):
                figure = graph_drawing.rfm_component_graph(rfm_data, component, color)
                st.pyplot(figure)
                plt.close()
                
            if st.button('Show treemap'):
                click_button(2)
            
            if st.session_state.stage >= 2:
                # Creates treemaps
                tree_figure = graph_drawing.treemap_drawing(cluster_centers)
                st.pyplot(tree_figure)
            
            if st.button('Show scatterplot'):
                click_button(3)
            
            if st.session_state.stage >= 3:
                scatter_figure = graph_drawing.scatter_3d_drawing(kmeans_data)
                st.plotly_chart(scatter_figure)
                
            if st.download_button('Download CSV', download_data.to_csv().encode('utf-8'), 'sales-cluster.csv', 'text/csv'):
                st.write('CSV file downloaded!')
            
            if st.download_button('Download Excel', data=output, file_name='sales-cluster.xlsx',mime='application/vnd.ms-excel'):
                st.write('Excel file downloaded!')