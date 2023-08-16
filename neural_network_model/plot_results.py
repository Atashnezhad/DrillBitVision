import json

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

    # print(parsed_data[0].case.report.MildDemented.precision)
    for i in range(len(parsed_data)):
        print(parsed_data[i].case.report.MildDemented.precision)
