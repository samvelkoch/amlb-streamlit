""" "Adapted" from https://github.com/PGijsbers/amlb-results/blob/main/notebooks/error_visualization.ipynb"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from core.data import get_print_friendly_name, is_old
from core.visualization import FRAMEWORK_TO_COLOR, add_horizontal_lines, box_plot
from core.ui import create_file_input

create_file_input()

def generate_error_table():
    import itertools
    import re

    results = st.session_state.filtered_results
    results["framework"] = results["framework"].apply(get_print_friendly_name)
    results = results[results["framework"] != "NaiveAutoML"]

    with_errors = results[~results["info"].isna()][
        ["framework", "task", "fold", "constraint", "info", "metric", "duration"]]

    TIMEOUT_PATTERN = re.compile("Interrupting thread MainThread \[ident=\d+\] after \d+s timeout.")

    def is_timeout(message: str) -> bool:
        if re.search(TIMEOUT_PATTERN, message):
            return True
        return False

    def is_memory(message: str) -> bool:
        if "Cannot allocate memory" in message:
            return True
        if "exit status 134" in message:
            return True
        if "exit status 137" in message:
            return True
        if "exit status 139" in message:
            return True
        if "exit status 143" in message:
            return True
        if "std::bad_alloc" in message:
            return True
        if "Dummy prediction failed with run state StatusType.MEMOUT" in message:
            return True  # autosklearn
        if "This could be caused by a segmentation fault while calling the function or by an excessive memory usage" in message:
            return True  # lightautoml
        if "OutOfMemoryError: GC overhead limit exceeded" in message:
            return True  # H2O
        return False

    def is_data(message: str) -> bool:
        if "NoResultError: y_true and y_pred contain different number of classes" in message:
            return True
        if "The least populated class in y has only 1 member, which is too few." in message:
            return True  # GAMA
        return False

    def is_implementation(message: str) -> bool:
        if "Unsupported metric `auc` for regression problems" in message:
            return True  # FLAML produces NaN predictions
        if "A pipeline has not yet been optimized. Please call fit() first." in message:
            return True  # TPOT
        if message == "NoResultError: probability estimates are not available for loss='hinge'":
            return True  # TPOT
        if "object has no attribute 'predict_proba'" in message:
            return True  # TPOT
        if "'NoneType' object is not iterable" in message:
            return True  # GAMA
        if message == "NoResultError: ":
            return True  # GAMA
        if "Ran out of input" in message:
            return True  # GAMA
        if "Python int too large to convert to C ssize_t" in message:
            return True  # GAMA
        if "invalid load key, " in message:
            return True  # GAMA
        if "Pipeline finished with 0 models for some reason." in message:
            return True  # Light AutoML
        if "No models produced. \nPlease check your data or submit" in message:
            return True  # MLJar
        if "Object of type float32 is not JSON serializable" in message:
            return True  # MLJar
        if "The feature names should match those that were passed during fit" in message:
            return True  # MLJar
        if re.search("At position \d+ should be feature with name", message):
            return True  # MLJar
        if "Ensemble_prediction_0_for_" in message:
            return True  # MLJar
        if "NeuralNetFastAI_BAG_L1'" in message:
            return True  # AutoGluon
        if "No learner was chosen in the initial phase." in message:
            return True  # NaiveAutoML
        return False

    def confirmed_fixed(message: str) -> bool:
        if "'NoneType' object has no attribute 'name'" in message:
            return True  # bug with infer_limit set in 0.8.0, fixed in 0.8.3.
        return False

    checks = dict(
        timeout=is_timeout,
        memory=is_memory,
        data=is_data,
        implementation=is_implementation,
        fixed=confirmed_fixed,
    )

    def classify_error(message: str):
        for type_, check in checks.items():
            if check(message):
                return type_
        return "unknown"

    with_errors["error_type"] = with_errors["info"].apply(classify_error)
    error_counts = with_errors.groupby(["framework", "error_type"], as_index=False).count()
    frameworks = list(with_errors.groupby("framework").count().task.sort_values(ascending=False).index)
    error_types = error_counts["error_type"].unique()

    all_combinations = pd.DataFrame(itertools.product(error_types, frameworks), columns=["error_type", "framework"])
    error_counts = pd.concat([error_counts, all_combinations]).drop_duplicates(subset=["error_type", "framework"],
                                                                               keep='first')
    error_counts = error_counts.fillna(0)
    return error_counts, with_errors

def plot_error_by_task(with_errors, at_least:int):
    errors_by_task = with_errors[~with_errors["error_type"].isin(["investigate", "implementation"])].groupby("task").count()["info"]
    metadata = st.session_state.filtered_metadataset
    metadata = metadata.drop("type", axis=1)
    all_results = metadata.set_index("name").join(errors_by_task)
    all_results = all_results.fillna(0)
    all_results = all_results.rename(columns=dict(info="count"))
    errors = all_results[all_results["count"] > 0].copy()
    no_errors = all_results[all_results["count"] == 0].copy()

    fig, ax = plt.subplots()

    errors["marker_size"] = errors["count"].apply(
        lambda c: 10 if c < 3 else (30 if c < 11 else (60 if c < 51 else 100)))
    classification = errors[errors["Number of Classes"] > 0]
    classification = classification[classification["count"]>=at_least]
    regression = errors[errors["Number of Classes"] == 0]
    regression = regression[regression["count"]>=at_least]

    if at_least == 0:
        ax.scatter(no_errors["Number of Instances"], no_errors["Number of Features"], color="#555555", s=1, label="No Error")
    ax.scatter(regression["Number of Instances"], regression["Number of Features"], color="#32a02d",
               s=regression["marker_size"], label="Regression", edgecolors="white", linewidths=.5)
    ax.scatter(classification["Number of Instances"], classification["Number of Features"], color="#2078b4",
               s=classification["marker_size"], label="Classification", edgecolors="white", linewidths=.5)

    # ax.scatter(0, 0, color="#ffffff", s=60, label="$\hspace{2}$ Error Counts:")
    ax.scatter(0, 0, color="#000000", s=10, label="$\leq$ 2")
    ax.scatter(0, 0, color="#000000", s=30, label="in [3, 10]")
    ax.scatter(0, 0, color="#000000", s=60, label="in [11, 50]")
    ax.scatter(0, 0, color="#000000", s=100, label="$\geq$50")

    ax.set_xscale("log")
    ax.set_xlabel("Instances")
    # We want to explicitly set limits based on the whole dataset,
    # moving axes are very misleading
    ax.set_xlim([1e2,1e8])
    ax.set_ylim([1, 1e5])
    ax.set_yscale("log")
    ax.set_ylabel("Features")
    ax.set_title("Number of Errors by Data Dimensions")

    # lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
    # lgnd.legend_handles[0]._sizes = [20]

    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles=handles[:3], labels=labels[:3], bbox_to_anchor=(0.9, 0.88), title="Type (color)")
    leg._legend_box.align = "left"
    leg = fig.legend(handles=handles[3:], labels=labels[3:], bbox_to_anchor=(0.9, 0.68), title="Count (size)")
    leg._legend_box.align = "left"
    st.pyplot(fig)

def plot_error_type_by_framework(error_counts, include_types: list[str]):
    frameworks = error_counts.groupby("framework").sum()["info"].sort_values(ascending=False).index
    error_types = error_counts["error_type"].unique()
    color_by_error_type = {
        "data": "#a6cee3",  # light blue
        "implementation": "#2078b4",  # dark blue
        "memory": "#A7CE85",  # light green
        "timeout": "#32a02d",  # dark green
        "rerun": "#999999",
        "investigate": "#cccccc",
        "fixed": "#fe9900",
    }
    assert all(error in color_by_error_type for error in error_types)

    color_by_error_type = {k: v for k, v in color_by_error_type.items() if k in error_types}

    errors_by_framework = {
        error_type: [
            error_counts[(error_counts["error_type"] == error_type) & (error_counts["framework"] == framework)][
                "info"].iloc[0]
            for framework in frameworks
        ]
        for error_type in color_by_error_type
    }

    fig, ax = plt.subplots()

    bottoms = np.zeros(len(frameworks))
    for error_type, counts in errors_by_framework.items():
        ax.bar(frameworks, counts, label=error_type, bottom=bottoms, width=.6, color=color_by_error_type[error_type])
        bottoms += counts

    ax.set_ylim([0, max(bottoms) + 20])
    ax.set_ylabel("count")
    ax.tick_params(axis="x", which="major", rotation=-90)
    ax.legend(loc="upper right")
    ax.set_title("Error types by framework")
    st.pyplot(fig)


def plot_train_duration_for_constraint(constraint: str, results):
    time_budget = get_print_friendly_name(constraint)
    data = results[results["constraint"] == constraint]
    data = data[~data["framework"].isin(["constantpredictor"])]

    data["timeout"] = data["info"].apply(lambda msg: isinstance(msg, str) and "Interrupting thread MainThread" in msg)
    timeout_counts = dict(data[["framework", "timeout"]].groupby("framework").sum()["timeout"])

    time_limit = 3600 if constraint == "1h8c_gp3" else 3600 * 4
    fig, ax = box_plot(
        data,
        metric="training_duration",
        title=f"Training Duration {time_budget}",
        # ylim=[-3 if metric != "auc" or constraint !="4h8c_gp3" else -7, 0],
        figsize=(6, 3),
        add_counts=timeout_counts,
        with_framework_names=True,  # ttype == "regression",
    )
    add_horizontal_lines(ax, ((time_limit, "grey"), (time_limit + 3600 + 1200, "red")))
    ax.set_ylabel("Training Duration")
    if constraint == "1h8c_gp3":
        ax.set_yticks([3600, 7200], ["1H", "2H"])
    if constraint == "4h8c_gp3":
        ax.set_yticks([3600 * 4, 3600 * 5 + 1200], ["4H", "5H"])
    st.pyplot(fig)

if __name__ == "__main__":
    st.write(
        """
        # Stability
        
        In this page, we take a look at the stability of the AutoML frameworks.
        This corresponds to Section 6.4 from our paper.
        We look at the frequency and type of errors encountered, but also
        evaluate how well the AutoML frameworks observe the provided time constraint.
        """
    )

    st.write(
        """
        ## Errors
        
        Here, we look at the errors we encounter while running our experiments.
        The logs reveal that the most errors happen during training time, and rarely during inference time.
        """
    )
    with st.expander("What do the categories mean?"):
        st.write("""
        We categorize the errors into the following categories:
        
         * **Memory**: The framework crashed due to exceeding available memory or encountering other memory-related
         errors, such as segmentation faults.
         * **Time**: The framework exceeded the time limit past the leniency period.
         * **Data**: Errors due to specific data characteristics (such as imbalanced data) occurred.
         * **Implementation**: Any errors caused by bugs in the AutoML framework code occurred.
         
        These categories are a bit crude and ultimately subjective, since from a reductive viewpoint, 
        all errors are implementation errors. However, they serve for a quick overview. We also introduce a ‘fixed’ 
        category to denote errors from a specific bug in `AUTOGLUON(HQIL)` which is already fixed in newer releases. 
        Additional details on this, and other errors encountered, can be found in Appendix D of our paper. 
        """)
    with st.expander("Why so many errors?"):
        st.write(
            """
            Engineering a well-behaved AutoML framework is hard for several reasons.
            First, it needs to handle all kind of different input data.
            Whether that's dealing with different data types, data dimensions varying orders of magnitude,
            or very skewed class distributions -- different approaches need to be correctly identified for 
            different data, and it's easy to miss an edge case.
            
            Second, AutoML frameworks almost always use ML algorithm implementations developed by third parties.
            These may have been developed with different design goals, assumptions, or philosophies. For example, while 
            an AutoML framework shouldn't crash on large data, individual algorithms may have been developed with the 
            understanding that large datasets may make the algorithm unstable (and e.g., raise memory errors). This 
            means that AutoML framework developers have to account for all the weird ways underlying components may
            behave, whether by design or not.
            
            Third, additional complexities like custom data preprocessing pipelines add a lot of complexity to the
            framework as a whole. We see many AutoML frameworks add some kind of pre- and/or post-processing steps
            to their AutoML framework, and they all have to work well with the very different ML pipelines which
            may be generated during the phase in which the AutoML framework finds and optimizes models.
            """
        )
    with st.expander("Why are most errors during training?"):
        st.write(
            """
            See "Why so many errors?" for the many reasons making a well-behaved AutoML framework is hard.
            Most of that complexity is encountered during the training phase.
            If an AutoML framework successfully finishes training, it likely already found a pipeline which
            can handle inference on the data: that was already part of the pipeline evaluation process!
            
            Additionally, if an AutoML framework crashes during training it can not also crash during inference, 
            because that is no longer run. So, even if a hypothetical framework would crash 50% of the time during
            training, if it reaches the inference stage crash 50% of the time during inference, you would expect 
            only half as many failures during inference time as during training.
            """
        )

    with st.expander("Does training time stability matter?"):
        st.write(
            """
            Yes, in some cases having a stable well-behaved AutoML framework matters, in others, not so much.
            For example, if you simply want to build a model once, then it may not matter to you.
            You can invoke the training procedure multiple times, until it runs to completion, or
            evaluate the use of other AutoML framework.

            On the other hand, if you want to use the AutoML framework regularly, e.g., to retrain models 
            on a daily basis, or tackle many different problems with them, then it may be very disruptive to
            have a system which is not well-behaved.
            """
        )
    r, l = generate_error_table()
    left, right = st.columns([0.7, 0.3])
    with right:
        error_types_bar = st.multiselect(
            label="Include Error Types:",
            options=r["error_type"].unique(),
            default=r["error_type"].unique(),
        )
        frameworks_bar = st.multiselect(
            label="Include Frameworks:",
            options=r["framework"].unique(),
            default=r["framework"].unique(),
        )
    with left:
        r = r[r["error_type"].isin(error_types_bar)]
        r = r[r["framework"].isin(frameworks_bar)]
        plot_error_type_by_framework(r, include_types=error_types_bar)

    left, right = st.columns([0.4, 0.6])
    with left:
        error_types_scatter = st.multiselect(
            label="Include Error Types:",
            options=r["error_type"].unique(),
            default=r["error_type"].unique(),
            key="error_types_scatter",
        )
        frameworks_scatter = st.multiselect(
            label="Include Frameworks:",
            options=r["framework"].unique(),
            default=r["framework"].unique(),
            key="framework_scatter",
        )
        minimal_error_count = st.slider(
            "Minimal error count:",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
        )
    with right:
        l = l[l["error_type"].isin(error_types_scatter)]
        l = l[l["framework"].isin(frameworks_scatter)]
        plot_error_by_task(l, minimal_error_count)
    st.write(
        """
        ## Training Duration
        
        Here, we look at time the AutoML framework used to build a model.
        Each framework was configured to make the most use of the given time.
        We tested the frameworks at two different time constraints: 1 hour and 4 hours.
        """
    )
    left, right = st.columns([0.8, 0.2])
    with right:
        const = st.selectbox(
            label="Time Constraint",
            options=["1 hour", "4 hours"],
            index=0,
        )
        types = st.multiselect(
            label="Task Types:",
            options=list(st.session_state.filtered_results["type"].unique()),
            default=list(st.session_state.filtered_results["type"].unique()),
        )
    with left:
        ree= st.session_state.filtered_results[st.session_state.filtered_results["type"].isin(types)]
        const_map = {"1 hour": "1h8c_gp3", "4 hours":"4h8c_gp3"}
        plot_train_duration_for_constraint(const_map[const], ree)