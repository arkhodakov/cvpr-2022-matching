import cv2
import json
import typer
import logging

import ezdxf
from ezdxf.document import Drawing
from pathlib import Path
from typing import Optional

import config
import endpoints
import matching
import utils

utils.load_logger()


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
    extraction = endpoints.exportJSON(document, config.layerslist, remove_empty_layers, remove_empty_vertices)

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
    structures = endpoints.getStructures(document, layerslist=config.layerslist)
    origin = utils.plotStructures(structures, document)
    cv2.imshow("Preview", origin)
    cv2.waitKey(0)


@app.command(
    name="match",
    help="Match two CAD models"
)
def match(
    gtpath: Path = typer.Argument(..., help="Path to a GT dxf file", exists=True, file_okay=True, dir_okay=False),
    tgpach: Path = typer.Argument(..., help="Path to a GT dxf file", exists=True, file_okay=True, dir_okay=False),
    apply_matrix: Optional[bool] = typer.Option(True, help="Whether apply transformation matrix to the target")
):
    logging.info("Reading the models..")
    gtdoc: Drawing = ezdxf.readfile(gtpath)
    tgdoc: Drawing = ezdxf.readfile(tgpach)

    logging.info("Matching models...")
    matching.match(gtdoc, tgdoc, config.layerslist, apply_matrix)


if __name__ == "__main__":
    app()
