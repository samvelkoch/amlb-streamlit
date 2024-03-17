import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from ._plotting import with_large_text, task_type_colormap
from ._widgets import _add_persistent_selectbox, _add_axis_control


def histplot(data: pd.DataFrame, column: str, hue: str | None = None, log_scale: int = 10):
    multiple_types = data["type"].nunique() > 1
    hue_order = None if hue != "type" else ["Regression", "Multiclass Classification", "Binary Classification"]

    ax = seaborn.histplot(
        data,
        x=column,
        hue=hue,
        multiple="stack",
        log_scale=log_scale,
        hue_order=hue_order,
        palette=task_type_colormap if hue_order else None,
        shrink=0.8,  # allow some horizontal space between bars
        linewidth=0,  # don't draw lines on edges
        bins=10,
    )
    with_large_text(ax, xlabel=column, ylabel="Datasets", title=f"Datasets by {column}")

    if legend := ax.get_legend():
        if multiple_types:
            legend.set_title("")
        else:
            legend.remove()

    return ax


def histogram_option_controls(dataset: pd.DataFrame, name: str):
    _add_axis_control(dataset, axis_name="x", container_name=name)

    _add_persistent_selectbox(
        label="Hue",
        options=[None] + list(dataset.select_dtypes(include="category").columns),
        key=f"hue_{name}",
    )


def _histogram_option_controls(dataset: pd.DataFrame, name: str):
    left, right = st.columns(2)
    with left:
        st.session_state.datasets_x_axis = st.selectbox(
            'X-axis',
            dataset.select_dtypes(include="number").columns,
            key=f"x_axis_{name}",
        )
    with right:
        st.session_state.datasets_y_axis = st.selectbox(
            'Y-axis',
            dataset.select_dtypes(include="number").columns,
            key=f"y_axis_{name}",
        )


def show_figure(data, container):
    fig = plt.figure()
    histplot(
        data,
        column=st.session_state[f"column_x_{container.name}"],
        hue=st.session_state[f"hue_{container.name}"],
        log_scale=st.session_state[f"log_x_{container.name}"],
    )
    container.window.pyplot(fig)
