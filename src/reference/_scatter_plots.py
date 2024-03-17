import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import streamlit as st

from ._plotting import with_large_text, task_type_colormap
from ._widgets import _add_persistent_selectbox, _add_axis_control


def scatter_plot(data, x, y, hue: str | None = None, hue_order: list[str] | None = None):
    ax = seaborn.scatterplot(
        data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        palette=task_type_colormap,
        s=60,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Datasets by Data Dimensions", size='xx-large')
    ax.legend().set_title("")
    with_large_text(
        ax,
        xlabel=x,
        ylabel=y,
        title=f"Datasets by {x} and {y}",
    )


def scatterplot_option_controls(dataset: pd.DataFrame, name: str):
    _add_axis_control(dataset, axis_name="x", container_name=name)
    _add_axis_control(dataset, axis_name="y", container_name=name)

    _add_persistent_selectbox(
        label="Hue",
        options=[None] + list(dataset.select_dtypes(include="category").columns),
        key=f"hue_{name}",
    )


def show_scatterplot(data, container):
    fig = plt.figure()
    scatter_plot(
        data,
        x=st.session_state[f"column_x_{container.name}"],
        y=st.session_state[f"column_y_{container.name}"],
        hue=st.session_state[f"hue_{container.name}"],
    )
    container.window.pyplot(fig)
