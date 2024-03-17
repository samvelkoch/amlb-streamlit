task_type_colormap = {
    "Regression": "#2ba02b",
    "Multiclass Classification": "#fa7f12",
    "Binary Classification": "#1f77b4"
}


def with_large_text(ax, xlabel: str, ylabel: str, title: str):
    ax.set_title(title, size='xx-large')
    ax.set_ylabel(ylabel, size='xx-large')
    ax.set_xlabel(xlabel, size='xx-large')
    ax.tick_params(axis='both', which='both', labelsize=18)


presets = {
    "1a: Datasets by Number of Features": {
        "kind": "histogram",
        "x": "Number of Features",
        "log_x": 10,
        "hue": "type",
        "data": "filtered_metadataset",
    },
    "1b: Datasets by Number of Instances": {
        "kind": "histogram",
        "x": "Number of Instances",
        "log_x": 10,
        "hue": "type",
        "data": "filtered_metadataset",
    },
    "1c: Datasets by Data Dimensions": {
        "kind": "scatter",
        "x": "Number of Instances",
        "y": "Number of Features",
        "hue": "type",
        "data": "filtered_metadataset",
    },
    "1d: Datasets by Percentage of Missing Values": {
        "kind": "histogram",
        "x": "Percentage of Missing Values",
        "log_x": None,
        "hue": "type",
        "data": "filtered_metadataset",
    },
    "1e: Datasets by Percentage of Categorical Features": {
        "kind": "histogram",
        "x": "Percentage of Categorical Features",
        "log_x": None,
        "hue": "type",
        "data": "filtered_metadataset",
    },
    "1f: Datasets by Number of Classes": {
        "kind": "histogram",
        "x": "Number of Classes",
        "log_x": 2,
        "hue": "type",
        "data": "filtered_metadataset",
    },
    "2": {

    },
    # Whisper-plot of inference speed (a: disk, b: in-memory)
    "6": {

    },
    # Whisper-plot of training duration (a: 1 hour, b: 4 hour)
    "9": {

    },
}
