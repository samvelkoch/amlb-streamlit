import pandas as pd


def get_print_friendly_name(name: str, extras: dict[str, str] = None) -> str:
    # Copied from https://github.com/PGijsbers/amlb-results/blob/main/notebooks/data_processing.py
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

def impute_results(results: pd.DataFrame, where: pd.Series, with_: str = "constantpredictor",
                   indicator_column: str = "imputed") -> pd.DataFrame:
    """Impute result column of `results`, `where_` a condition holds true, `with_` the result of another framework.

    results: pd.DataFrame
      Regular AMLB results dataframe, must have columns "framework", "task", "fold", "constraint", and "result".
    where: pd.Series
      A logical index into `results` that defines the row where "result" should be imputed.
    with_: str
      The name of the "framework" which should be used to determine the value to impute with.
    indicator_column: str, optional
      The name of the column where a boolean will mark whether or not the "result" value of the row was imputed.

    Returns a copy of the original dataframe with imputed results.
    """
    if with_ not in results["framework"].unique():
        raise ValueError(f"{with_=} is not in `results`")
    results = results.copy()

    if indicator_column and indicator_column not in results.columns:
        results[indicator_column] = False

    lookup_table = results.set_index(["framework", "task", "fold", "constraint"])
    for index, row in results[where].iterrows():
        task, fold, constraint = row[["task", "fold", "constraint"]]
        results.loc[index, "result"] = lookup_table.loc[(with_, task, fold, constraint)].result
        if indicator_column:
            results.loc[index, indicator_column] = True
    return results

def preprocess_data(results):
    results = impute_results(
        results,
        where=results["result"].isna(),
        with_="constantpredictor",
    )
    autogluon_bug = results["info"].apply(
        lambda v: isinstance(v, str) and "'NoneType' object has no attribute 'name'" in v)
    results = impute_results(
        results,
        where=autogluon_bug,
        with_="AutoGluon(HQ)",
    )
    mean_results = (results[
        ["framework", "task", "constraint", "metric", "result", "imputed", "infer_batch_size_file_10000", "infer_batch_size_df_1"]])
    mean_results["task"] = mean_results["task"].astype('object')
    mean_results["metric"] = mean_results["metric"].astype('object')
    mean_results["framework"] = mean_results["framework"].astype('object')
    mean_results["constraint"] = mean_results["constraint"].astype('object')
    mean_results = mean_results.groupby(
        ["framework", "task", "constraint", "metric"], as_index=False).agg(
        {
            "result": "mean",
            "infer_batch_size_file_10000": "mean",
            "infer_batch_size_df_1": "mean",
            "imputed": "sum"
        }
    )
    lookup = mean_results.set_index(["framework", "task", "constraint"])
    for index, row in mean_results.iterrows():
        lower = lookup.loc[("RandomForest", row["task"], row["constraint"]), "result"]
        upper = lookup.loc[(slice(None), row["task"], row["constraint"]), "result"].max()
        if lower == upper:
            mean_results.loc[index, "scaled"] = float("nan")
        else:
            mean_results.loc[index, "scaled"] = (row["result"] - lower) / (upper - lower)
    return mean_results

def is_old(framework: str, constraint: str, metric: str) -> bool:
    """Encodes the table in `raw_to_clean.ipynb`"""
    if framework == "TunedRandomForest":
        return True
    if constraint == "1h8c_gp3":
        return False
    if framework in ["autosklearn2", "GAMA(B)", "TPOT"]:
        return True
    return framework == "MLJAR(B)" and metric != "neg_rmse"

def filter_results(metadata: pd.DataFrame, results: pd.DataFrame, min_n: int = 0, max_n: int | None = None, min_p: int = 0, max_p: int | None = None):
    # For whatever reason, directly indexing by the string name gives a KeyError
    col_name ="Number of Instances"
    filtered_meta = metadata[metadata[col_name] > min_n]
    if max_n:
        filtered_meta = filtered_meta[filtered_meta[col_name] <= max_n]
    col_name = "Number of Features"
    filtered_meta = filtered_meta[filtered_meta[col_name] > min_p]
    if max_p:
        filtered_meta = filtered_meta[filtered_meta[col_name] <= max_p]

    return results[results.task.isin(filtered_meta.name)]


