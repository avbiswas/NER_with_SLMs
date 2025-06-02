import os
import json
import pandas as pd
import asyncio
import streamlit as st
from small_lm.generate_dataset import main as datagen_main


def generate_datasets():

    if "generation_status" not in st.session_state:
        st.session_state.generation_status = None
    if "status_placeholder" not in st.session_state:
        st.session_state.status_placeholder = None

    def dataset_generation_callback(system_prompt, model_name, total_examples):
        st.session_state.generation_status = "Generating dataset..."
        st.session_state.status_placeholder.info(st.session_state.generation_status)

        # Run the async operation
        filename = asyncio.run(
            datagen_main(
                system_prompt=system_prompt,
                model=model_name,
                total_examples=total_examples,
            )
        )

        # Update the status
        st.session_state.generation_status = (
            f"Dataset generated successfully! Filename: **{filename}**!"
        )
        st.session_state.status_placeholder.success(st.session_state.generation_status)

    list_of_prompts = os.listdir("configs/prompts")
    list_of_prompts = ["None"] + [
        f"configs/prompts/{prompt}" for prompt in list_of_prompts
    ]
    selected_prompt = st.selectbox("Select a prompt", list_of_prompts, index=1)

    if "datagen_prompt" not in st.session_state:
        st.session_state.datagen_prompt = "You are a helpful assistant."

    if selected_prompt != "None":
        with open(selected_prompt, "r") as f:
            st.session_state.datagen_prompt = f.read()

    st.text_area("System Prompt", value=st.session_state.datagen_prompt, height=400)

    model_name = st.text_input("Model Name", value="gpt-4.1-mini-2025-04-14")

    total_examples = st.number_input(
        "Total Examples", value=5, min_value=5, step=50, max_value=10000
    )

    st.button(
        "Generate Dataset",
        on_click=dataset_generation_callback,
        kwargs={
            "system_prompt": st.session_state.datagen_prompt,
            "model_name": model_name,
            "total_examples": total_examples,
        },
    )

    st.session_state.status_placeholder = st.empty()
    if st.session_state.generation_status:
        if "successfully" in st.session_state.generation_status:
            st.session_state.status_placeholder.success(
                st.session_state.generation_status
            )
        else:
            st.session_state.status_placeholder.info(st.session_state.generation_status)


def view_datasets():
    dataset_files = os.listdir("datasets")
    dataset_files = [
        f"datasets/{file}" for file in dataset_files if file.endswith(".json")
    ]
    selected_dataset = st.selectbox("Select a dataset", dataset_files, index=1)

    with open(selected_dataset, "r") as f:
        dataset = json.load(f)
    df = pd.DataFrame(dataset["dataset"])
    for col in df.columns:
        df[col] = df[col].astype(str)

    st.metric("Number of examples", len(df))

    st.dataframe(df)
    st.dataframe(df[df["labels"].isna()])
