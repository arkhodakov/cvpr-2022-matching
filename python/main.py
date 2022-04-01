import cv2
import json
import typer
import logging

import ezdxf
from ezdxf.document import Drawing

import vertices
from pathlib import Path
from typing import Optional

from utils import load_logger
load_logger()


app = typer.Typer()

@app.command(
    name="export-json",
    help="Parse and export DXF data in seravee08 JSON format."
)
def export_json(
    path: Path = typer.Argument(..., help="Path to a dxf file", exists=True, file_okay=True, dir_okay=False),
    outputpath: Path = typer.Argument(None, help="Output file path"),
    # Optional
    remove_empty_layers: Optional[bool] = typer.Option(True, help="Remove layers with 0 structures."),
    remove_empty_vertices: Optional[bool] = typer.Option(True, help="Remove vertices with (0, 0, 0) coordinate."),
):
    logging.info("Reading input file...")
    document: Drawing = ezdxf.readfile(path)

    logging.info("Extracting vertices to a dictionary format.")
    extraction = vertices.extract_json(document, remove_empty_layers, remove_empty_vertices)

    outputpath = outputpath or path.parent.joinpath(f"{path.name.split('.')[0]}-output.json")
    with open(outputpath, "w+", encoding="utf-8") as out:
        json.dump(extraction, out, ensure_ascii=False, indent=4)
    logging.info(f"Output JSON file has been saved to: {outputpath.resolve()}")


@app.command(
    name="plot",
    help="Parse and plot DXF data."
)
def plot(
    path: Path = typer.Argument(..., help="Path to a dxf file", exists=True, file_okay=True, dir_okay=False),
):
    logging.info("Reading input file...")
    document: Drawing = ezdxf.readfile(path)

    logging.info("Plotting vertices to a numpy image.")
    layerlist = ["A-WALL", "A-GLAZ", "A-DOOR", "A-DOOR-FRAM"]
    structures = vertices.extract(document, normalize=True, layerlist=layerlist)
    origin = vertices.plot(structures, document)
    cv2.imshow("Preview", origin)
    cv2.waitKey(0)


@app.command(
    name="match",
    help="Match two CAD models"
)
def match(
    gtpath: Path = typer.Argument(..., help="Path to a GT dxf file", exists=True, file_okay=True, dir_okay=False),
    tgpach: Path = typer.Argument(..., help="Path to a GT dxf file", exists=True, file_okay=True, dir_okay=False),
):
    logging.info("Reading the models..")
    gtdoc: Drawing = ezdxf.readfile(gtpath)
    tgdoc: Drawing = ezdxf.readfile(tgpach)

    logging.info("Matching models...")
    layerlist = ["I-WALL", "A-WALL", "A-GLAZ", "A-DOOR", "A-DOOR-FRAM"]
    vertices.match(gtdoc, tgdoc, layerlist=layerlist)


if __name__ == "__main__":
    app()
