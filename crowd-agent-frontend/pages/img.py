import streamlit as st
from util.path_encryption import decrypt_path
import os

st.set_page_config(layout="wide")

if "token" in st.query_params:
    if st.query_params["token"] != "xxxxxxxxxxxxxxxxx":  # TODO: change to the token
        st.switch_page('pages/login.py')
else:
    st.switch_page('pages/login.py')


if "pth" in st.query_params:
    pth = decrypt_path(st.query_params["pth"], "xxxxxxxxxxxxxxxxx")  # TODO: change to the secret key
    pth = st.query_params["pth"].replace(" ", "+")

    raw_pth = decrypt_path(pth, "xxxxxxxxxxxxxxxxx")  # TODO: change to the secret key
    if os.path.exists(raw_pth):
        st.image(raw_pth, use_container_width =True)
    else:
        st.write("Image not found")
