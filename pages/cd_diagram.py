from autorank import autorank, plot_stats
import streamlit as st
import matplotlib.pyplot as plt

from core.data import preprocess_data, get_print_friendly_name
from core.ui import filters, create_file_input

create_file_input()

st.write("# Critical Difference Diagrams")
st.write("""
    This page shows the critical difference (cd) diagrams [[Demšar, 2006]](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)
    comparing the average rank of the evaluated AutoML frameworks across the benchmarking suites.
    """
)

with st.expander("More about CD diagrams"):
    st.write(
        """
        Critical Difference (CD) diagrams show the average rank of each framework,
        and which framework achieve statistically significantly different ranks.
        
        ### Calculating Ranks 
        When comparing frameworks to each other on a single dataset, you could
        order them by their predictive performance and rank them: the most
        accurate model gets rank 1, then the next rank 2, and so on.
        When comparing the frameworks across multiple datasets, we can aggregate
        those ranks, for example by taking the mean to compute the average ranks.
        In the CD diagram below, for each framework you will find a line connecting
        it to the horizontal axis to denote the average rank obtained by each framework.
        
        ### Critical Differences
        Sometimes, the difference in average ranks are not statistically significant:
        how big the difference in rank needs to be depends also on the number of observations
        we have, and the number of frameworks we are comparing. How big the difference needs
        to be is called the "critical difference" and is also visually demonstrated by a bar
        on the top left of the plot. Differences between frameworks whose average ranks differ 
        by less than the critical difference are not considered to be statistically significant.
        This is visually indicated by a horizontal bar connecting the lines of two or more frameworks.
        
        ### Criticism
        CD diagrams have their own problems. In a 2017 paper Demšar, with multiple collaborators,
        point out multiple problems with statistical hypothesis testing in the setting of 
        comparing multiple algorithms [[Benavoli, 2017]](https://www.jmlr.org/papers/volume18/16-305/16-305.pdf).
        In short, statistical significant differences may be artifacts of how many observations are obtained,
        and do not necessarily reflect a difference that is meaningful in a practical sense.
        
        While they propose a different method (that we hope to include in the future), it is for 1-on-1 
        comparisons of algorithms and not equipped to visualize well the differences between multiple classifiers.
        Additionally, it requires the definition of what a "meaningful difference" is in practice (which they refer
        to as a "region of practical equivalence"). This depends per task and is highly subjective, so it would
        be a very laborious task to define this for each task in the benchmark - if you can even find a consensus.
        """

    )


filter_ = filters()
mean_results = st.session_state.filtered_results.copy()
mean_results["framework"] = mean_results["framework"].apply(get_print_friendly_name)
mean_results = preprocess_data(mean_results)
frameworks_to_exclude = ["autosklearn2", "NaiveAutoML"]
if "neg_rmse" in filter_.metrics and "4h8c_gp3" in filter_.constraints:
    frameworks_to_exclude.extend(["MLJAR(P)", "AutoGluon(HQ)", "AutoGluon(HQIL)"])
mean_results = mean_results[~mean_results["framework"].isin(frameworks_to_exclude)]
mean_results = mean_results[(mean_results["constraint"].isin(filter_.constraints)) & (mean_results["metric"].isin(filter_.metrics))]
mean_results = mean_results[["framework", "task", "result"]]
mean_results = mean_results[mean_results["task"].isin(filter_.task_names)]
mean_results = mean_results.pivot(index="task", columns="framework", values="result")
try:
    result = autorank(
        mean_results,
        force_mode="nonparametric"
    )
except ValueError as e:
    if "requires at least five" not in str(e):
        raise
    st.write(f"Creating the CD diagram requires including at least five datasets to be included.")
else:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_stats(result, ax=ax)
    st.pyplot(fig)
