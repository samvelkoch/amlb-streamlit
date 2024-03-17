import itertools
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Errors", page_icon="😪", layout="wide")


def get_print_friendly_name(name: str, extras: dict[str, str] = None) -> str:
    if extras is None:
        extras = {}

    frameworks = {
        "AutoGluon_benchmark": "AutoGluon(B)",
        "AutoGluon_hq": "AutoGluon(HQ)",
        "AutoGluon_hq_il001": "AutoGluon(HQIL)",
        "GAMA_benchmark": "GAMA(B)",
        "mljarsupervised_benchmark": "MLJAR(B)",
        "mljarsupervised_perform": "MLJAR(P)",
    }
    budgets = {
        "1h8c_gp3": "1 hour",
        "4h8c_gp3": "4 hours",
    }
    print_friendly_names = (frameworks | budgets | extras)
    return print_friendly_names.get(name, name)


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
    if  "object has no attribute 'predict_proba'" in message:
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

def plot_errors(results: pd.DataFrame):
    print("naness:", results.isna().any())
    results = results.copy()
    results["framework"] = results["framework"].apply(get_print_friendly_name)
    results = results[results["framework"] != "NaiveAutoML"]
    with_errors = results[~results["info"].isna()][
        ["framework", "task", "fold", "constraint", "info", "metric", "duration"]]
    with_errors["error_type"] = with_errors["info"].apply(classify_error)

    error_counts = with_errors.groupby(["framework", "error_type"],
                                       as_index=False).count()

    frameworks = list(with_errors.groupby("framework").count().task.sort_values(
        ascending=False).index)
    error_types = error_counts["error_type"].unique()

    all_combinations = pd.DataFrame(itertools.product(error_types, frameworks),
                                    columns=["error_type", "framework"])
    error_counts = pd.concat([error_counts, all_combinations]).drop_duplicates(
        subset=["error_type", "framework"], keep='first')
    error_counts = error_counts.fillna(0)

    color_by_error_type = {
        "data": "#a6cee3",  # light blue
        "implementation": "#2078b4",  # dark blue
        "memory": "#A7CE85",  # light green
        "timeout": "#32a02d",  # dark green
        "rerun": "#999999",
        "investigate": "#cccccc",
        "fixed": "#fe9900",
        "unknown": "#000000",
    }
    colors = set(error_types) - set(color_by_error_type)
    print( "colors",colors)
    assert not colors, f"{colors}"
    color_by_error_type = {k: v for k, v in color_by_error_type.items() if
                           k in error_types}

    errors_by_framework = {
        error_type: [
            error_counts[(error_counts["error_type"] == error_type) & (
                        error_counts["framework"] == framework)]["info"].iloc[0]
            for framework in frameworks
        ]
        for error_type in color_by_error_type
    }

    fig, ax = plt.subplots()

    bottoms = np.zeros(len(frameworks))
    for error_type, counts in errors_by_framework.items():
        ax.bar(frameworks, counts, label=error_type, bottom=bottoms, width=.6,
               color=color_by_error_type[error_type])
        bottoms += counts

    ax.set_ylim([0, max(bottoms) + 20])
    ax.set_ylabel("count")
    ax.tick_params(axis="x", which="major", rotation=-90)
    ax.legend(loc="upper right")
    ax.set_title("Error types by framework")
    return fig

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data_input import get_filtered_results, show_tables
fig = plot_errors(get_filtered_results())
st.pyplot(fig)
show_tables()