import streamlit as st
import streamlit_authenticator as stauth
from pathlib import Path
import yaml
from authentication import make_sidebar, register_user, reset_password, verify_user

def load_credentials():
    config_file = Path(__file__).parent / "config/config.yaml"
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

# Callback functions to switch forms
def switch_to_login():
    st.session_state['form'] = 'login'

def switch_to_register():
    st.session_state['form'] = 'register'

def switch_to_reset_password():
    st.session_state['form'] = 'reset_password'

def switch_to_activation():
    st.session_state['form'] = 'activation'

def main():
    make_sidebar()
        
    # Initialize session state variables if not already set
    if 'form' not in st.session_state:
        st.session_state['form'] = 'login'  # Default view is login form
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    
    # Debugging print statement to track the form state
    st.write("Current form state:", st.session_state['form'])  # Helps track reruns
        
    config = load_credentials()
    stauth.Hasher.hash_passwords(config['credentials'])

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        auto_hash=False  # passwords are pre-hashed during registration
    )

    # Handle login form
    if st.session_state['form'] == 'login':
        st.write("Please log in or register to continue.")
        
        authenticator.login()
    
        if st.session_state.get('authentication_status'):
            # Check if the user is verified
            username = st.session_state.get('username')
            user_data = config['credentials']['usernames'].get(username)
            if user_data and user_data.get('verified', False):
                st.success(f"Logged in as {username}!")
                st.session_state.update({
                    'logged_in': True,
                    'username': username,
                    'authenticator': authenticator
                })
                st.switch_page("pages/Instructions.py")
            else:
                st.error("Your account is not verified. Please check your email for the activation link.")
                st.session_state['authentication_status'] = None  # Reset to prevent access
                
        elif st.session_state.get('authentication_status') is False:
            st.error("Username or password is incorrect")
        elif st.session_state.get('authentication_status') is None:
            st.warning("Please enter your username and password")

        # UI Buttons for Register, Reset Password, and Activate Account
        st.button("Register", on_click=switch_to_register)
        st.button("Reset Password", on_click=switch_to_reset_password)
        st.button("Activate Account", on_click=switch_to_activation)

    # Registration Form
    elif st.session_state['form'] == 'register':
        register_user()
        st.button("Back to login", on_click=switch_to_login)

    # Reset Password Form
    elif st.session_state['form'] == 'reset_password':
        reset_password()
        st.button("Back to login", on_click=switch_to_login)
    
    # Activation Form
    elif st.session_state['form'] == 'activation':
        st.write("Enter the activation code sent to your email.")
        with st.form("activation_form"):
            email = st.text_input("Enter your email")
            code = st.text_input("Enter your activation code")
            submit_button = st.form_submit_button("Activate Account")

            if submit_button:
                if verify_user(email, code):
                    st.success("Account activated! You can now log in.")
                    st.session_state['form'] = 'login'
                    st.rerun()  # Refresh to load the login form
                else:
                    st.error("Activation failed. Please check your code and try again.")
                    
        # Back to Login button outside the form
        st.button("Back to login", on_click=switch_to_login)
        
    # Logout Button in Sidebar
    if st.session_state.get("logged_in", False):
        authenticator.logout('Log out', 'sidebar')

if __name__ == "__main__":
    main()
