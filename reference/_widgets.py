import pandas as pd
import streamlit as st

def _add_persistent_selectbox(label: str, key: str, options: list):
    """Add a selectbox which is persistent even if it is not always rendered."""
    index_selected = 0
    if selected_value := st.session_state.get(key):
        index_selected = list(options).index(selected_value)
    return st.selectbox(
        label=label,
        options=options,
        key=key,
        index=index_selected,
    )



def _add_axis_control(dataset: pd.DataFrame, axis_name: str, container_name: str):
    """Adds controls to select, crop, and scale data along an axis."""
    suffix = f"{axis_name}_{container_name}"

    left, right = st.columns([0.8, 0.2])
    with left:
        column_name = _add_persistent_selectbox(
            label=f"{axis_name.upper()}-axis",
            key=f"column_{suffix}",
            options=dataset.select_dtypes(include="number").columns,
        )

    with right:
        _add_persistent_selectbox(
            "Log",
            key=f"log_{suffix}",
            options=[None, 2, 10],
        )

    _, middle, _ = st.columns([0.02, 0.88, 0.1])
    with middle:
        st.slider(
            "Range",
            value=(dataset[column_name].min(), dataset[column_name].max()),
            min_value=dataset[column_name].min(),
            max_value=dataset[column_name].max(),
            key=f"range_{suffix}",
        )
