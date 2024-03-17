import streamlit as st
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from core.data import get_print_friendly_name, preprocess_data, impute_results, is_old
from core.visualization import FRAMEWORK_TO_COLOR, box_plot
from core.ui import write_card, filters, create_file_input

create_file_input()


def plot_inference_barplot(results, constraint, col, title:str):
    data = results[results["constraint"] == constraint].copy()
    # data = data[~data["framework"].isin(["constantpredictor", "TPOT"])]

    data["timeout"] = data["info"].apply(lambda msg: isinstance(msg, str) and "Interrupting thread MainThread" in msg)
    data["row/s"] = 10_000 / data[col]

    fig, ax = box_plot(
        data,
        metric="row/s",
        title=title,
        figsize=(8, 4),
        add_counts=False,#timeout_counts,
        with_framework_names=True, # ttype == "regression",
    )
    ax.set_ylabel("rows per second")
    ax.set_yscale("log")
    st.pyplot(fig)


def calculate_pareto(xs, ys) -> list[tuple[float, float]]:
    pairs = set(zip(xs, ys))
    return [
        (x, y)
        for x, y in pairs
        # check below is only correct because `pairs` is a set, so never x==x2 *and* y==y2
        if not any((x2>=x and y2 >=y) and (x!=x2 or y!=y2) for x2, y2 in pairs)
    ]

def plot_pareto(data, x, y, ax, color="#cccccc"):
    pareto = sorted(calculate_pareto(data[x], data[y]))
    for opt, next_opt in zip(pareto, pareto[1:]):
        ax.plot([opt[0], opt[0], next_opt[0]], [opt[1],next_opt[1], next_opt[1]], color=color, zorder=0)

def plot_scatter(data: pd.DataFrame, x:str,  title: str):
    color_map = {k: v for k, v in FRAMEWORK_TO_COLOR.items() if k not in exclude}

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax = seaborn.scatterplot(
        data,
        x=x,
        y="scaled",
        hue="framework",
        palette=color_map,
        s=70,  # marker size
        ax=ax,
    )
    plot_pareto(data, x=x, y="scaled", ax=ax)
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_xlabel('median rows per second')
    ax.set_ylabel('median scaled performance')
    seaborn.move_legend(ax, "upper right", bbox_to_anchor=(1.6, 1))
    st.pyplot(fig)

st.write("# Inference Time")
st.write("Inference time denotes the time the model built by the AutoML framework takes to generate predictions.")
with st.expander("More on inference time and how we measure it..."):
    st.write(
        """
        Inference time denotes the time the model built by the AutoML framework takes to generate predictions.
        How long inference takes depends on a lot of different factors, such as complexity of the models
        (as a rule, large ensembles take more time than a linear model) as well as the underlying machine learning
        libraries the different AutoML frameworks use.
        
        In the benchmark, we measure both in-memory and from disk inference times. For in-memory inference times
        we pass a pandas dataframe to the framework, and for loading from disk we provide them with the location
        on disk. We measured single-row inference speed for both from-disk and in-memory measurements, and
        measured various batch sizes for from-disk inference speed.
        
        For batched inference speed, we only measured batch sizes that did not exceed dataset size, 
        because many AutoML frameworks would not provide models stable or fast enough to reliably perform inference on 
        batches much larger than training size. Data is sampled from the test set.
        """
    )
    write_card(
        header="Many frameworks support optimizing models for inference speed before deployment.",
        body=" This functionality is not used in this benchmark. "
             "Results are meant as a proxy and to demonstrate differences in models beyond predictive performance. "
             "Other important differences include factors like e.g., interpretability."
    )

filter_ = filters(
    constraints={"1 hour": "1h8c_gp3"},
)

results = st.session_state.filtered_results.copy()
results["framework"] = results["framework"].apply(get_print_friendly_name)
results = results[~results["framework"].isin(["TPOT", "NaiveAutoML", "constantpredictor", "TunedRandomForest"])]
# The framework category still remembers the frameworks were once there in the category metadata
results["framework"] = results["framework"].astype('string').astype('category')
results = results[results["task"].isin(filter_.task_names)]
left, right = st.columns([0.5,0.5])
with left:
    plot_inference_barplot(results, constraint="1h8c_gp3", col="infer_batch_size_file_10000",
                           title="From-Disk Batch Inference Speed")
with right:
    plot_inference_barplot(results, constraint="1h8c_gp3", col="infer_batch_size_df_1", title="In-Memory Inference Speed")


st.write("## Inference and Performance")
st.write("Amongst the best models, there is a trade-off between predictive performance and inference times."
         " Note that this plot updates according to the dataset selection above.")

results = st.session_state.filtered_results.copy()
results["framework"] = results["framework"].apply(get_print_friendly_name)
mr = preprocess_data(results)
exclude = ["constantpredictor", "RandomForest", "TunedRandomForest", "TPOT", "NaiveAutoML"]
if "neg_rmse" in filter_.metrics:
    exclude += ["autosklearn2"]

data = mr[~mr["framework"].isin(exclude)]
data = data[(data["constraint"].isin(filter_.constraints)) & (data["metric"].isin(filter_.metrics))]
data = data[data["task"].isin(filter_.task_names)]

data = data.groupby(["framework", "constraint"], as_index=False)[
    ["infer_batch_size_file_10000", "scaled", "infer_batch_size_df_1"]].median()
# st.write("ADD SWITCH DISK/MEMORY")
data["disk_row_per_s"] = 10_000. / data["infer_batch_size_file_10000"]
data["ram_row_per_s"] = 1. / data["infer_batch_size_df_1"]

inference_measure = st.selectbox(
    "Inference Type",
    options=["10,000 rows from disk", "single row in memory"],
)
inference_measure = {
    "10,000 rows from disk": "disk_row_per_s",
    "single row in memory": "ram_row_per_s"
}[inference_measure]
if inference_measure == "ram_row_per_s":
    # H2O doesn't have in-memory inference, it's always serialized to disk in our setup
    data = data[~data["framework"].isin(["H2OAutoML"])]
budgets = [get_print_friendly_name(c) for c in filter_.constraints]
plot_scatter(data, x=inference_measure, title=f"{','.join(budgets)}")
