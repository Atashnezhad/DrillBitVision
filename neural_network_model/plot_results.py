import json
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict, List
from pathlib import Path


class Metrics(BaseModel):
    precision: float = None
    recall: float = None
    f1_score: float = None
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
    def parse_data(_data: Dict, name: str) -> 'ParsedData':
        accepted_format = dict(
            case=_data.get(name)
        )
        # print(accepted_format)
        return ParsedData(**accepted_format)


def plot_1(cases_dict):
    # make a plot and compare all cases accuracy in one plot
    # Create a list of colors and line styles for each case
    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

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
            marker='o',  # Add markers at data points
            markersize=5,
            markerfacecolor=colors[idx % len(colors)],
            markeredgecolor='k'  # Marker edge color
        )

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Customize tick marks on the axes
    plt.xticks(np.arange(0, len(history.categorical_accuracy), step=5))  # Adjust step as needed
    # plt.yticks(np.arange(0, 1.1, step=0.1))

    # change x and y axis labels font size
    plt.tick_params(axis='both', which='major', labelsize=16)
    # change legend font size
    plt.rcParams['legend.fontsize'] = 16
    # x and y axis font size
    plt.rcParams['axes.labelsize'] = 16

    # Add a grid with dashed lines
    plt.grid(True, linestyle='--', alpha=0.7)
    # Add a background color to the plot
    plt.gca().set_facecolor('#f9f9f9')
    # Add a title to the legend
    leg = plt.legend()
    leg.set_title("Cases")
    plt.tight_layout()

    # Display the plot
    plt.show()


if __name__ == '__main__':
    # read json data from file
    json_filename = "results.json"
    data_path = Path(__file__).parent
    data_address = data_path / json_filename

    with open(data_address, 'r') as f:
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
        ParsedData.parse_data(_data, name)
        for _data, name in zip(data_list, cases_list)
    ]

    # make a dictionary using cases_list as keys and parsed_data cases as values
    cases_dict = {
        cases_list[i]: parsed_data[i].case
        for i in range(len(parsed_data))
    }
    pprint(cases_dict.get("original"))

    # plot_1(cases_dict)

    plt.figure(figsize=(10, 6))
    # set the whole fond size
    plt.rcParams.update({'font.size': 16})

    plt.bar(
        cases_list,
        [cases_dict.get(case).report.accuracy for case in cases_list],
        color=['b', 'g', 'r', 'c', 'm', 'y', 'k'],
        width=0.5,
        align='center',
        alpha=0.7
    )
    # set the title of the plot
    plt.title("Accuracy of Cases", fontsize=16)
    # y axis label
    plt.ylabel("Accuracy", fontsize=16)

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    # set y axis limits to 0 and 1
    plt.ylim(0, 1)

    # x axis labels font size
    # plt.tick_params(axis='x', which='major', labelsize=16)
    plt.tight_layout()
    plt.show()




