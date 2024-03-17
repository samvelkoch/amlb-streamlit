from pathlib import Path
import re
import logging

import pandas as pd
import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer

logger = logging.getLogger(__file__)

def initialize_data(data_dir: Path):
    if "raw_data" not in st.session_state:
        _load_default_results(data_dir)
        if "raw_data" not in st.session_state:
            logger.warning("No default dataset loaded.")
    if "metadataset" not in st.session_state:
        _load_default_metadata(data_dir)
        if "metadataset" not in st.session_state:
            logger.warning("No default metadata loaded.")


def _load_default_results(data_directory: Path):
    """Loads the 2023 results file."""
    filepath = data_directory / "amlb_all.csv"
    if filepath.exists():
        categorical_features = {
            feature: "category"
            for feature in [
                "id", "task", "framework", "constraint", "type", "metric", "mode", "app_version"
            ]
        }
        st.session_state.raw_data = pd.read_csv(
            filepath,
            dtype=categorical_features,
        )
        st.session_state.filtered_results = st.session_state.raw_data


def _name_with_space(name: str) -> str:
    words = re.findall(r"([A-Z][a-z]+)", name)
    return ' '.join(
        word if word not in ['Of', 'With'] else word.lower() for word in words) if words else name


def _determine_task_type(number_of_classes: int) -> str:
    if number_of_classes == 2:
        return "Binary Classification"
    if number_of_classes > 2:
        return "Multiclass Classification"
    return "Regression"


def _load_default_metadata(data_directory: Path):
    filepath = data_directory / "metadata.csv"
    if filepath.exists():
        categorical_features = {
            feature: "category"
            for feature in [
                "name", "status", "format",
            ]
        }
        metadataset = pd.read_csv(
            filepath,
            dtype=categorical_features,
        )

        features = ["NumberOfClasses", "NumberOfFeatures", "NumberOfInstances",
                    "NumberOfInstancesWithMissingValues", "NumberOfMissingValues",
                    "NumberOfSymbolicFeatures", "name"]
        datasets = metadataset[features].rename(
            columns={feature: _name_with_space(feature) for feature in features}
        )

        datasets["type"] = datasets["Number of Classes"].apply(_determine_task_type).astype("category")
        datasets["Percentage of Categorical Features"] = (datasets[
                                                              "Number of Symbolic Features"] /
                                                          datasets[
                                                              "Number of Features"]) * 100
        datasets["Percentage of Missing Values"] = (datasets[
                                                        "Number of Missing Values"] / (
                                                            datasets[
                                                                "Number of Instances"] *
                                                            datasets[
                                                                "Number of Features"])) * 100
        datasets["name"] = datasets["name"].str.replace(".","_")
        st.session_state.metadataset = datasets
        st.session_state.filtered_metadataset = datasets


def create_file_input():
    """Creates a file input which may store a dataframe under session_state.raw_data."""
    with st.sidebar:
        # streamlit side panel (input forms)
        raw_data = st.file_uploader(
            label='Select a results file:',
            # label_visibility='collapsed',
            accept_multiple_files=False,
            type="csv",
            help="Any results file produced by AMLB 2.1 or later.",
        )

        if raw_data:
            # TODO: infer file type: results vs metadata?
            st.session_state.raw_data = pd.read_csv(raw_data)


def show_tables(expanded: bool = False):
    """Generates content for the overview page."""
    if "raw_data" not in st.session_state:
        st.text("Please upload a result file from the sidebar on the left.")
        return

    with st.expander("Filter Results", expanded=expanded):
        df = st.session_state.raw_data
        df['framework'] = df['framework'].astype('category')
        filtered_results = dataframe_explorer(df, case=False)
        st.dataframe(filtered_results, use_container_width=True)
        st.session_state.filtered_results = filtered_results

    with st.expander("Filter Datasets", expanded=expanded):
        filtered_datasets = dataframe_explorer(st.session_state.metadataset, case=False)
        st.dataframe(filtered_datasets, use_container_width=True)
        st.session_state.filtered_metadataset = filtered_datasets


def get_filtered_results():
    """Combines the filtered datasets with the filtered results."""
    selected_tasks = st.session_state.filtered_results["task"].isin(st.session_state.filtered_metadataset["name"])
    return st.session_state.filtered_results[selected_tasks]
