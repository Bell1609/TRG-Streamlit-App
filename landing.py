import streamlit as st
import streamlit_authenticator as stauth
from pathlib import Path
import yaml
from authentication import make_sidebar, register_user, reset_password

def load_credentials():
    config_file = Path(__file__).parent / "config/config.yaml"
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

# Callback to switch to the login form
def switch_to_login():
    st.session_state['form'] = 'login'

# Callback to switch to the registration form
def switch_to_register():
    st.session_state['form'] = 'register'

# Callback to switch to the reset password form
def switch_to_reset_password():
    st.session_state['form'] = 'reset_password'

def main():
    make_sidebar()

    # Initialize session state for controlling the view
    if 'form' not in st.session_state:
        st.session_state['form'] = 'login'  # Default view is login form

    config = load_credentials()
    stauth.Hasher.hash_passwords(config['credentials'])

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        auto_hash=False # pass is pre-hashed during registration already
    )

    if st.session_state['form'] == 'login':
        st.write("Please log in or register to continue.")
        authenticator.login()

        if st.session_state.get('authentication_status'):
            st.success(f"Logged in as {st.session_state['name']}!")
            st.session_state.logged_in = True
            st.session_state.authenticator = authenticator
            st.switch_page("pages/Instructions.py")
        elif st.session_state.get('authentication_status') is False:
            st.error("Username or password is incorrect")
        elif st.session_state.get('authentication_status') is None:
            st.warning("Please enter your username and password")

        # Use on_click to ensure session state is updated immediately
        st.button("Register", on_click=switch_to_register)
        st.button("Reset Password", on_click=switch_to_reset_password)

    elif st.session_state['form'] == 'register':
        register_user()

        # Use on_click to return to login without a double-click
        st.button("Back to login", on_click=switch_to_login)

    elif st.session_state['form'] == 'reset_password':
        reset_password()
        st.button("Back to login", on_click=switch_to_login)

    if st.session_state.get("logged_in", False):
        authenticator.logout('Logout')

if __name__ == "__main__":
    main()