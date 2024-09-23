# TRG Customer Segmentation App

## Overview
This application is intended for segmenting TRG Customer data extracted from FreshSales and FreshDesk. However, any sales and tickets data will work if there is enough fields required for analysis.

## Required fields
### For FreshSales analysis
- AccountID: The ID of the account who did the purchase
- CloseDate: The date that the deal was closed on
- DealValue: The value of deals (in a consistent currency)
- DealStage: The status of the deal (Won for deals won, and Lost for deals lost, and others)
### For FreshDesk analysis
- Group Company: The group that the client's company belongs to
- Brand: The brand of the client's company
- Client code: The client code provided by FreshDesk
- TRG Customer: Whether or not a TRG Customer raised that ticket (Yes or No)
- Closed time: The time that the ticket was closed

## Installation
To run this app locally, you need to have Python installed along with the necessary packages. Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/TRG-International/TRG-Streamlit_App.git
    cd TRG-Streamlit-App
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run Instructions.py
    ```
