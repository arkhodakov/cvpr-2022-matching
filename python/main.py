import typer
import dotenv
import logging

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
    gtpath: Path = typer.Argument(..., help="Path to a ground-truth file", exists=True, file_okay=True, dir_okay=False),
    tgpath: Path = typer.Argument(..., help="Path to a target file", exists=True, file_okay=True, dir_okay=False),
):
    logging.info("Reading the models..")

    gtdata = loader.read_json(gtpath)
    gtstructures = loader.read_structures(gtdata)

    tgdata = loader.read_json(tgpath)
    tgstructures = loader.read_structures(tgdata)

    matching.match(gtstructures, tgstructures)


if __name__ == "__main__":
    app()
