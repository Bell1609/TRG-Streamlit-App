from __future__ import division
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from pathlib import Path
from authentication import make_sidebar

# Load user credentials from a YAML file
def load_credentials():
    config_file = Path(__file__).parent / "config.yaml"
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def main():
    make_sidebar()

    st.write("Please log in to continue.")

    # Load user configuration
    config = load_credentials()

    # Pre-hashing all plain text passwords once
    stauth.Hasher.hash_passwords(config['credentials'])
    
    # Create an authenticator instance
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # Create the login form
    authenticator.login()
    name = ''

    # Check if login was successful
    if st.session_state.get('authentication_status'):
        st.success(f"Logged in as {st.session_state['name']}!")
        st.session_state.logged_in = True  # Set login status in session state
        st.session_state.authenticator = authenticator  # Store authenticator in session state
        st.session_state['username'] = name  # Store username for future reference
        st.switch_page("pages/Instructions.py")  # Switch to the main page
    elif st.session_state.get('authentication_status') is False:
        st.error("Username or password is incorrect")
    elif st.session_state.get('authentication_status') is None:
        st.warning("Please enter your username and password")

    # Logout button
    if st.session_state.get("logged_in", True):
        authenticator.logout('Logout')  # Adjust the button name and location as needed

if __name__ == "__main__":
    main()