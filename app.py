import streamlit as st

from streamlit_pages.train_app import training_page_slm, training_page_bert
from streamlit_pages.dataset_app import generate_datasets, view_datasets
from streamlit_pages.inference_app import inference_page


def home():
    st.title("Training small language models!")

    st.write("This is a simple app to train small language models!")


def dataset_page():
    st.title("Dataset")
    tab1, tab2 = st.tabs(["Generate Dataset", "View Dataset"])
    with tab1:
        st.write("In this tab, you can generate a dataset for training SLMs.")
        generate_datasets()
    with tab2:
        st.write("In this tab, you can view the datasets you have generated.")
        view_datasets()


pg = st.navigation(
    pages={
        "Home": [
            st.Page(title="Home", page=home),
            st.Page(title="Dataset", page=dataset_page),
        ],
        "Train": [
            st.Page(title="Train SLMs", page=training_page_slm),
            st.Page(title="Train BERT", page=training_page_bert),
        ],
        "Inference": [st.Page(title="Inference", page=inference_page)],
    }
)
pg.run()
