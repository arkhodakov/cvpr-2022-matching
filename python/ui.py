import streamlit as st

from gui.landing import landing
from gui.loader import read_uploaded_files
from gui.sidebar import sidebar
from gui.viewer import viewer

st.set_page_config(
    page_title="CVPR 2022 - Models matching",
    layout="wide"
)
st.title("Models matching")

ground_files, target_files, model_pattern = sidebar()

gtdata = read_uploaded_files(ground_files, model_pattern) if ground_files else None
tgdata = read_uploaded_files(target_files, model_pattern) if target_files else None

if gtdata and tgdata:
    viewer(gtdata, tgdata)
else:
    landing()
