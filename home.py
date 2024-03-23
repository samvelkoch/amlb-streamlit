from pathlib import Path

import streamlit as st


from data_input import initialize_data
from core.ui import write_card, create_file_input


__version__ = "0.2"
_repository = "https://github.com/Coorsaa/automlbenchmark_shiny_app/"

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

st.write("# AutoML Benchmark")

st.write("""

This app allows you to further inspect the results presented in the JMLR paper
"AMLB: an AutoML Benchmark" by Gijsbers et al. (2024). The paper discusses the benchmark design and its limitations,
and contextualizes the results. While we provide some additional context in this app, we strongly encourage you to 
read the paper before drawing any conclusions. 
""")

write_card(
    body= """This app was not part of the JMLR peer review process. If you find any mistakes,
please open a <a style="color:cyan;" href="#">Github issue</a>. We welcome contributions.""",
    header="Important",
)

st.write("This app includes all the figures of the paper, with some"
         " additional controls which let you look at specific aspects of the data:")
st.page_link("home.py", label=" * Figure 1: Benchmarking Suite")
st.page_link("pages/cd_diagram.py", label=" * Figure 2: Critical Difference Diagrams")
st.page_link("pages/performance.py", label=" * Figure 3 and 4: Scaled Performance Boxplots")
st.page_link("home.py", label=" * Figure 5: Bradley-Terry Trees")
st.page_link("pages/inference.py", label=" * Figure 6 and 7: Model Inference Time")
st.page_link("pages/stability.py", label=" * Figure 8 and 9: Errors and training duration.")

st.write(
    """
    The code and data of this app is available at https://github.com/pgijsbers/amlb-streamlit.
    The raw data is available on our minio server S3 bucket https://openml1.win.tue.nl/automlbenchmark 
    and the preprocessing notebooks are available at https://github.com/pgijsbers/amlb-results. 
    """)

write_card(
    body= "Click the three stacked dots in the top right, then navigate to"
          "`Settings` and select `Wide Mode`.",
    header="Plots too small?",
    icon="💡"
)

st.write("""
## Code and Contributions

The code for this app is hosted in our Github repository. The code for the visualizations
is more-or-less directly copied from the notebooks we used to generate the figures for our paper.

Right now, the code for the visualization app is a mess with lots of duplication.
We hope to clean it up in the future, and welcome any contributions in that effort.
If you plan to customize the visualizations, the notebook may be an easier starting point.

While we think the interactive visualizations are important, unfortunately we do not have
the time to dedicate on-going support and improvement of this app right now. However,
we will commit to correcting any factual mistakes or adding clarifications where necessary.
If you are interested in contributing, please let us know (on Github), we would love to help get you started.
"""
)

create_file_input()

if __name__ == "__main__":
    initialize_data(Path("data/").expanduser())
