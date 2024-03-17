from pathlib import Path
from typing import NamedTuple
import streamlit as st
from data_input import show_tables, initialize_data
from sidebar import create_sidebar
from pages._plotting import presets
from pages._histogram import show_figure, histogram_option_controls
from pages._scatter_plots import scatterplot_option_controls, show_scatterplot

__version__ = "0.2"
_repository = "https://github.com/Coorsaa/automlbenchmark_shiny_app/"


class Container(NamedTuple):
    window: "st.DeltaGenerator"
    name: str


def configure_streamlit():
    """Sets the streamlit page configuration."""
    st.set_page_config(
        page_title=f"AutoML-Benchmark Analysis App - v{__version__}",
        menu_items={
            "Get help": f"{_repository}",
            "Report a bug": f"{_repository}/issues/new",
        },
        layout="wide",
    )


def create_visualization_container(column: int):
    # the column is needed to ensure consistent but unique keys, which are required for
    # the statefulness of the widgets.
    def _set_visualization_type(from_widget: str):
        st.session_state[f"vis_{from_widget}"] = st.session_state[from_widget]

    container = st.container()
    with container:
        left, right, _ = st.columns(3)
        with left:
            kind = st.selectbox(
                label="Kind",
                options=["Histogram", "Bar", "Scatter", "Preset"],
                key=f"kind_{column}",
                on_change=_set_visualization_type,
                args=(f"kind_{column}",),
            )
            st.session_state[f"vis_kind_{column}"] = kind
        with right:
            if kind.casefold() == "preset":
                _ = st.selectbox(
                    label="Figure",
                    options=presets,
                    key=f"preset_{column}",
                    # help="hello\n**markdown**\nbullet list\n * hello\n * boo\n\ngoobye",
                )
            else:
                _ = st.selectbox(
                    label="Source",
                    options=["Datasets", "Results"],
                    key=f"source_{column}",
                    # help="hello\n**markdown**\nbullet list\n * hello\n * boo\n\ngoobye",
                )
    return container


if __name__ == "__main__":
    configure_streamlit()
    create_sidebar()

    columns = st.columns(2, gap="medium")

    initialize_data(Path("~/Documents/data/amlb/").expanduser())
    show_tables(expanded=True)

    containers = []
    for i, column in enumerate(columns):
        with column:
            container = create_visualization_container(i)
            containers.append(Container(window=container, name=f"container_{i}"))
            # visualization_function = ... if ... if ...
            data = st.session_state.filtered_metadataset if st.session_state.get(f"source_{i}") == "Datasets" else st.session_state.filtered_results
            plot_container = st.container()

            if st.session_state[f"kind_{i}"].casefold() == "preset":
                settings = presets[st.session_state[f"preset_{i}"]]
                data = st.session_state[settings["data"]]
                st.session_state[f"column_x_container_{i}"] = settings["x"]
                if "y" in settings:
                    st.session_state[f"column_y_container_{i}"] = settings["y"]
                st.session_state[f"hue_container_{i}"] = settings["hue"]
                st.session_state[f"vis_kind_{i}"] = settings["kind"]
                if "log_x" in settings:
                    st.session_state[f"log_x_container_{i}"] = settings["log_x"]

            with st.expander("Plot Options", expanded=True):
                if st.session_state[f"vis_kind_{i}"].casefold() == "histogram":
                    histogram_option_controls(data, name=f"container_{i}")
                if st.session_state[f"vis_kind_{i}"].casefold() == "scatter":
                    scatterplot_option_controls(data, name=f"container_{i}")

            if st.session_state[f"vis_kind_{i}"].casefold() == "histogram":
                show_figure(
                    data,
                    Container(window=container, name=f"container_{i}"),
                )
            if st.session_state[f"vis_kind_{i}"].casefold() == "scatter":
                show_scatterplot(
                    data,
                    Container(window=container, name=f"container_{i}")
                )

