from collections import defaultdict
import json
import typer
import dotenv
import logging
import numpy as np

from pathlib import Path

import config
import loader
import matching
import utils

dotenv.load_dotenv()
utils.load_logger()


app = typer.Typer()


@app.command(
    name="match",
    help="Match two models"
)
def match(
    ref_data: Path = typer.Argument(..., help="Path to a ground-truth (reference) file or directory", exists=True, file_okay=True, dir_okay=True),
    user_data: Path = typer.Argument(..., help="Path to a target (user prediction) file or directory", exists=True, file_okay=True, dir_okay=True),
    output: Path = typer.Option("output", help="Path to the output directory", file_okay=False, dir_okay=True)
):
    output.mkdir(parents=True, exist_ok=True)
    logging.info("Reading the models..")

    ground_files = loader.read_source(ref_data.resolve())
    logging.info(f"Found {len(ground_files.keys())} GT models: {list(ground_files.keys())}")
    target_files = loader.read_source(user_data.resolve())
    logging.info(f"Found {len(target_files.keys())} Target models: {list(target_files.keys())}")

    for model in target_files.keys():
        gtdata = ground_files.get(model)
        if gtdata is None:
            logging.error(f"Cannot find gt model for '{model}'.")
            continue
        tgdata = target_files[model]

        data = defaultdict(dict)
        for floor in tgdata.keys():
            gtfloor = gtdata.get(floor)
            if gtfloor is None:
                logging.error(f"Cannot find '{gtfloor}' floor in gt model '{model}'.")
            else:
                logging.info(f"Matching '{model}', floor '{floor}'...")
            tgfloor = tgdata[floor]

            gtstructures = loader.read_structures(gtfloor)
            tgstructures = loader.read_structures(tgfloor)

            data["floors"][floor] = matching.match(gtstructures, tgstructures, output, model, floor)

        # TODO: Encapsulate it & refactor
        iou, precision, recall, f1 = [], [], [], []
        iou_cls = defaultdict(list)
        precision_cls = defaultdict(list)
        recall_cls = defaultdict(list)
        f1_cls = defaultdict(list)
        for floor in data["floors"].values():
            for classname, ious in floor["ious"]["general"].items():
                iou_cls[classname].append(ious["mean"])
                iou.append(ious["mean"])
            for classname, metrics in floor["metrics"].items():
                threshold: dict = metrics["thresholds"][config.metrics_thresholds[0]]
                precision_cls[classname].append(threshold["precision"])
                precision.append(threshold["precision"])
                recall_cls[classname].append(threshold["recall"])
                recall.append(threshold["recall"])
                f1_cls[classname].append(threshold["f1"])
                f1.append(threshold["f1"])
        iou: float = np.mean(iou)
        iou_cls: dict = {key: np.mean(value) for key, value in iou_cls.items()}
        precision: float = np.mean(precision)
        precision_cls: dict = {key: np.mean(value) for key, value in precision_cls.items()}
        recall: float = np.mean(recall)
        recall_cls: dict = {key: np.mean(value) for key, value in recall_cls.items()}
        f1: float = np.mean(f1)
        f1_cls: dict = {key: np.mean(value) for key, value in f1_cls.items()}

        data["iou"]["global"] = iou
        data["iou"]["classes"] = iou_cls
        data["precision"]["global"] = precision
        data["precision"]["classes"] = precision_cls
        data["recall"]["global"] = recall
        data["recall"]["classes"] = recall_cls
        data["f1"]["global"] = f1
        data["f1"]["classes"] = f1_cls

        logging.info("Matching data export...")
        with open(output.joinpath(f"{model}.json"), "w+", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4, cls=utils.NumpyArrayEncoder)

if __name__ == "__main__":
    app()
