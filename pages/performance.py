import streamlit as st
import seaborn
import matplotlib.pyplot as plt

from core.data import get_print_friendly_name, is_old, preprocess_data, filter_results
from core.visualization import FRAMEWORK_TO_COLOR, add_horizontal_lines
from core.ui import create_file_input, filters

create_file_input()
st.write("# Performance")
st.write(
    """
    In our paper, instead of reporting AUC, log loss, or RMSE directly, we often use a "scaled performance" score instead.
    Scaled performance for framework A on dataset D is determined by scaling its predictive score (log loss, rmse, auc)
    between the Random Forest performance on D, at 0, and the best observed performance on D across all frameworks, at 1.
    If a framework's score is lower than 0, this means it performed worse than our baseline Random Forest method.
    """
)

with st.expander("Why scaled performance?"):
    st.write(
    """
    We do this for three reasons:
     
     * The metrics are hard to evaluate at a glance. Is an AUC of 0.8 good? Maybe. 
     What is good, and what is not, depends on the task. 
     
     * The metrics are not generally commensurable across different datasets. Averaging RMSE, which can differ orders of
    magnitude across different tasks, may result in the average performance being dominated by the contribution of a single dataset.
    
     * Because we are comparing AutoML frameworks, we want something that is interpretable in the context of this comparison.
       Scaling from Random Forest (0) to best observed (1) performance immediately lets you gauge how it compares relative
       to the other frameworks.
    """
)

filter_ = filters()

mean_results = st.session_state.filtered_results.copy()
mean_results["framework"] = mean_results["framework"].apply(get_print_friendly_name)
mean_results = preprocess_data(mean_results)
frameworks_to_exclude = ["RandomForest", "NaiveAutoML"]
mean_results = mean_results[~mean_results["framework"].isin(frameworks_to_exclude)]
mean_results = mean_results[(mean_results["constraint"] == filter_.constraints[0]) & (mean_results["metric"].isin(filter_.metrics))]
mean_results = filter_results(st.session_state.metadataset, mean_results, *filter_.instance_range, *filter_.feature_range)


def box_plot(data, metric=None, ylog=False, title="", ylim=None, figsize=(16, 9), with_framework_names=True,
             add_counts=None, color_map=None):
    """Creates a boxplot with data["frameworks"] on the x-axis and data[`metric`] on the y-axis

    The figure's y-axis may be limited by `ylim` and the number of values outside this limit may be shown in the tick labels.
    """
    if add_counts and (add_counts != "outliers" and not isinstance(add_counts, dict)):
        raise ValueError("`add_counts` must be 'outliers' or a dictionary mapping each framework to a number.")

    color_map = color_map or FRAMEWORK_TO_COLOR
    color_map = {k: v for k, v in color_map.items() if
                 k in data["framework"].unique() or k in ["autosklearn2", "AutoGluon(HQIL)"]}

    metric = metric or data.metric.unique()[0]
    if metric.startswith("neg_"):
        pos_metric = metric[len("neg_"):]
        data[pos_metric], metric = -data[metric], pos_metric

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    seaborn.boxplot(
        data=data,
        x="framework",
        y=metric,
        order=color_map,
        hue="constraint" if data.constraint.nunique() > 1 else None,
        palette=color_map if data.constraint.nunique() == 1 else None,
        ax=ax,
        fliersize=1,
    )

    if ylog:
        ax.set_yscale("log")

    ax.set_ylabel(metric, size='xx-large')
    ax.set_xlabel("")
    ax.tick_params(axis='both', which='both', labelsize=18)

    if title:
        ax.set_title(title, fontsize=18)

    # Dirty hack for displaying outliers, we overlap minor and major tick labels, where
    # minor labels are used to display the number of outliers, and major tick labels may
    # be used to display the framework names.
    constraint = data.constraint.unique()[0]
    smetric = data.metric.unique()[0]
    frameworks = color_map.keys()
    frameworks = [
        f"{fw if with_framework_names else ''}*" if is_old(fw, constraint, smetric) else (
            fw if with_framework_names else '')
        for fw in frameworks
    ]
    if add_counts:
        # There will be minor tick labels displayed for outliers,
        # to avoid rendering on top of each other, we offset the label location
        # with a dirty hack of using leading spaces :-)
        frameworks = [f"   {fw}" for fw in frameworks]
    ax.tick_params(axis="x", which="major", rotation=-90)
    ax.set_xticks(*zip(*enumerate(frameworks)))

    if ylim:
        ax.set_ylim(ylim)
        if add_counts != "outliers":
            print("Warning! Ylim is set but outliers are not reported.")

    counts = []
    if add_counts:
        if add_counts == "outliers":
            add_counts = {}
            for framework in color_map:
                framework_outliers = data[(data["framework"] == framework) & (data[metric] < ylim[0])]
                add_counts[framework] = f"{len(framework_outliers)}"

        # We need to offset the minor tick labels, otherwise they won't render.
        ax.set_xticks(
            ticks=[i - 0.01 for i in range(len(color_map))],
            labels=[f"[{add_counts.get(f, 'x')}]" for f in color_map],
            minor=True
        )

    return fig, ax


data = mean_results.copy()
time_budget = get_print_friendly_name(filter_.constraints[0])

periwinkle_blue = "#6f7cc8"
fig, ax = box_plot(
    data,
    metric="scaled",
    title=f"{filter_.task_type.capitalize()}, {time_budget}",
    ylim=[-2, 1],
    figsize=(6, 3),
    add_counts="outliers"
)
ax.set_ylabel("Scaled Performance")
add_horizontal_lines(ax, ((0, periwinkle_blue),))
st.pyplot(fig)


mean_results = st.session_state.filtered_results.copy()
mean_results = mean_results[~mean_results["framework"].isin(["NaiveAutoML"])]
mean_results["task"] = mean_results["task"].astype('object')
mean_results["metric"] = mean_results["metric"].astype('object')
mean_results["framework"] = mean_results["framework"].astype('object')
mean_results["constraint"] = mean_results["constraint"].astype('object')
mean_results = mean_results[
    ["framework", "task", "constraint", "metric", "result"]
].groupby(["framework", "task", "constraint", "metric"], as_index=False).agg(
    {"result": "mean"}
)

lookup = mean_results.set_index(["framework", "task", "constraint"])
for index, row in mean_results.iterrows():
    lower = lookup.loc[("RandomForest", row["task"], "1h8c_gp3"), "result"]
    upper = lookup.loc[(slice(None), row["task"], slice(None)), "result"].max()
    if lower == upper:
        mean_results.loc[index, "scaled"] = float("nan")
    else:
        mean_results.loc[index, "scaled"] = (row["result"] - lower) / (upper - lower)

st.write("## The Effect of the Time Constraint \n"
         "In the plot below, you can see that there is generally only marginal improvement when allowing for more time."
         " In this case, the 'scaled performance' is slightly modified: 0 is the RF performance with a 4 hour budget, "
         "and 1 is the best observed performance across any budget."
" Note that this plot updates according to the dataset selection above."
         )

fig, ax = plt.subplots(1, 1, figsize=(6,4))
# The frameworks below are excluded because they do not have recent results for all of the benchmark and time constraints,
# except constantpredictor which we exclude because it will predict the same regardless of constraint and so is not interesting.
mean_results = mean_results[~mean_results.framework.str.lower().isin(
    ["autogluon(hqil)", "tpot", "gama(b)", "constantpredictor", "randomforest", "mljar(p)"])
]
baselines = ["constantpredictor", "RandomForest", "TunedRandomForest"]
mean_results = mean_results[mean_results["task"].isin(filter_.task_names)]

seaborn.boxplot(
    data=mean_results,
    y="framework",
    x="scaled",
    hue="constraint",
    ax=ax,
    fliersize=1,
    order=[
        *sorted((f for f in mean_results.framework.unique() if f not in baselines), key=lambda f: f.lower()),
        *sorted(b for b in baselines if b in mean_results.framework.unique()),
    ]
)
ax.set_xlim([-2, 1])

ax.set_xlabel("Scaled Performance", size='xx-large')
ax.set_ylabel("")
#ax.tick_params(axis="x", which="major", rotation=-70)
ax.tick_params(axis='both', which = 'both', labelsize = 18)
ax.set_title("Scaled performance after 1 and 4 hours", fontsize=18)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=["1 hour", "4 hours"], title="Time Constraint")
seaborn.move_legend(ax, "upper right", bbox_to_anchor=(1.3, 1.02))

st.pyplot(fig)