import streamlit as st
import time
time.sleep(0.1)
st.set_page_config(layout="centered")

USERNAME = "xxxxxxxxxxxxxx" # TODO: change to the username
PASSWORD = "xxxxxxxxxxxxxx" # TODO: change to the password


st.header("Login to CrowdAgent System")
st.divider()

username = st.text_input("Username", value=USERNAME, max_chars=20)
password = st.text_input("Password", type="password")

if st.button("Login"):
    if username == USERNAME and password == PASSWORD:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success("Login successful!")
        time.sleep(0.5)
        st.switch_page("home.py")
    else:
        st.error("Invalid username or password. Please try again.")


st.info("Please refer to **Appendix D.1 Access Details** for the username and password.")