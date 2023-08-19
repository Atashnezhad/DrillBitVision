import json
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class Metrics(BaseModel):
    precision: float = None
    recall: float = None
    f1_score: float = Field(alias="f1-score")
    support: float = None


class History(BaseModel):
    loss: List[float]
    categorical_accuracy: List[float]
    val_loss: List[float]
    val_categorical_accuracy: List[float]


class Categories(BaseModel):
    MildDemented: Metrics
    ModerateDemented: Metrics
    NonDemented: Metrics
    VeryMildDemented: Metrics
    accuracy: float


class CaseData(BaseModel):
    report: Categories
    history: History


class ParsedData(BaseModel):
    case: CaseData = None

    @staticmethod
    def parse_data(_data: Dict, name: str) -> "ParsedData":
        accepted_format = dict(case=_data.get(name))
        # print(accepted_format)
        return ParsedData(**accepted_format)


def plot_1():
    # make a plot and compare all cases accuracy in one plot
    # Create a list of colors and line styles for each case
    line_styles = ["-", "--", "-.", ":"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    plt.figure(figsize=(10, 6))
    plt.title("History of Val Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)

    # Iterate through cases and apply customizations
    for idx, case in enumerate(cases_list):
        history = cases_dict.get(case).history
        plt.plot(
            history.val_loss,
            label=case,
            linestyle=line_styles[idx % len(line_styles)],
            color=colors[idx % len(colors)],
            linewidth=2,
            marker="o",  # Add markers at data points
            markersize=5,
            markerfacecolor=colors[idx % len(colors)],
            markeredgecolor="k",  # Marker edge color
        )

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add legend outside the plot
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Customize tick marks on the axes
    plt.xticks(
        np.arange(0, len(history.categorical_accuracy), step=5)
    )  # Adjust step as needed
    # plt.yticks(np.arange(0, 1.1, step=0.1))

    # change x and y axis labels font size
    plt.tick_params(axis="both", which="major", labelsize=16)
    # change legend font size
    plt.rcParams["legend.fontsize"] = 16
    # x and y axis font size
    plt.rcParams["axes.labelsize"] = 16

    # Add a grid with dashed lines
    plt.grid(True, linestyle="--", alpha=0.7)
    # Add a background color to the plot
    plt.gca().set_facecolor("#f9f9f9")
    # Add a title to the legend
    leg = plt.legend()
    leg.set_title("Cases")
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_2():
    plt.figure(figsize=(12, 6))
    # Set the whole font size
    plt.rcParams.update({"font.size": 16})

    bars = plt.barh(
        cases_list,
        [cases_dict.get(case).report.accuracy for case in cases_list],
        color=["b", "g", "r", "c", "m", "y", "k"],
        height=0.5,
        align="center",
        alpha=0.7,
        tick_label=cases_list,
    )

    # dont show y axis labels
    plt.yticks([])
    # set y axis label cases
    plt.ylabel("Cases", fontsize=18)

    # Add legend with proper labels
    plt.legend(bars, cases_list, loc="upper left", bbox_to_anchor=(1, 1))

    # Set the title of the plot
    plt.title(
        "Accuracy of Cases", fontsize=20, pad=20
    )  # Increase font size and add padding

    # X-axis label
    plt.xlabel("Accuracy", fontsize=18)

    # Invert y-axis for horizontal bar plot
    plt.gca().invert_yaxis()

    # Add grid with dashed lines
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set x-axis limits to 0 and 1
    plt.xlim(0, 1)

    # Customize y-axis tick labels font size
    plt.tick_params(axis="y", labelsize=14)

    # Add background color to the plot
    plt.gca().set_facecolor("#f9f9f9")

    # Add some padding around the plot
    plt.tight_layout(pad=1.0)

    # Display the plot
    plt.show()


def plot_3():
    import matplotlib.pyplot as plt

    # Set the whole font size
    plt.rcParams.update({"font.size": 22})
    metric_names = [
        "precision",
        "recall",
        "f1_score",
        "support",
    ]  # Add other metrics if needed

    metrics_dict = {}
    for metric in metric_names:
        case_metrics = {}
        for idx, case in enumerate(cases_list):
            value = getattr(parsed_data[idx].case.report.VeryMildDemented, metric)
            case_metrics[case] = value
        metrics_dict[metric] = case_metrics

    print(metrics_dict)

    # Create a list of metric names to plot
    metric_names_to_plot = ["precision", "recall", "f1_score"]

    import matplotlib.pyplot as plt

    # Create subplots
    fig, axes = plt.subplots(len(metric_names_to_plot), 1, figsize=(16, 14))
    # set figure title
    fig.suptitle("Very MildDemented Comparison", fontsize=30)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    colors = ["b", "g", "r", "c", "m", "y", "k"]

    for idx, metric_name in enumerate(metric_names_to_plot):
        ax = axes[idx]
        ax.set_title(f"{metric_name.capitalize()} Comparison")

        for i, case in enumerate(cases_list):
            metric_values = metrics_dict[metric_name]
            metric_value = metric_values.get(
                case
            )  # Get the metric value for the current case
            if metric_value is not None:
                ax.bar(
                    i,
                    metric_value,
                    color=colors[i % len(colors)],
                    alpha=0.7,
                    label=case,
                )

        ax.set_xlabel("Cases")
        ax.set_ylabel(metric_name.capitalize())
        # set y-axis limits
        ax.set_ylim(0, 1)
        # rotate x-axis labels
        ax.set_xticks([])
        ax.legend(
            loc="upper left", bbox_to_anchor=(1, 1)
        )  # Move the legend outside of the figure

    # tighten layout
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # read json data from file
    json_filename = "results.json"
    data_path = Path(__file__).parent
    data_address = data_path / json_filename

    with open(data_address, "r") as f:
        data_list = f.read()
    # Parse the JSON string into a dictionary
    data_list = json.loads(data_list)

    cases_list = [
        "original",
        "Hessian_filter_seismic",
        "LBP_filter_grey",
        "Segmented_3_viridis",
        "Segmented_5_seismic",
        "Edited_seismic",
    ]
    # parse data
    parsed_data = [
        ParsedData.parse_data(_data, name) for _data, name in zip(data_list, cases_list)
    ]

    # make a dictionary using cases_list as keys and parsed_data cases as values
    cases_dict = {cases_list[i]: parsed_data[i].case for i in range(len(parsed_data))}
    pprint(cases_dict.get("original"))

    # plot_1()
    # plot_2()
    plot_3()
