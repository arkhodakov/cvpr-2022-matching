import streamlit as st

# Demo: https://regex101.com/r/NX1dl1/1
default_model_regex: str = r"(?P<model>.*)_(?P<classname>columns|doors|walls).json"

def sidebar():
    with st.sidebar:
        groundfiles = st.file_uploader(
            label="Ground truth data files",
            type=["json"],
            accept_multiple_files=True,
            help="Choose which data files to use as ground-truth models."
        )

        augmentation = st.button(
            label="Augmentation data ✨",
            help="Apply augmentation to the gt data and use it as the predictions",
            disabled=(not groundfiles)
        )

        targetfiles = st.file_uploader(
            label="Target (predictions) data files",
            type=["json"],
            accept_multiple_files=True
        )

        model_pattern: str = st.text_input(
            label="Model name regular expression",
            value=default_model_regex
        )
        st.markdown("[Check demo or edit](https://regex101.com/r/NX1dl1/1)")

        return groundfiles, targetfiles, model_pattern
