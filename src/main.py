import json
import typer
import dotenv
import logging

from pathlib import Path

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
    ref_data: Path = typer.Argument(..., help="Path to a ground-truth (reference) file", exists=True, file_okay=True, dir_okay=True),
    user_data: Path = typer.Argument(..., help="Path to a target (user prediction) file", exists=True, file_okay=True, dir_okay=True),
    output: Path = typer.Option("match.json", help="Path to the output file", file_okay=True, dir_okay=False)
):
    logging.info("Reading the models..")

    gtdata = loader.read_json(ref_data)
    gtstructures = loader.read_structures(gtdata)

    tgdata = loader.read_json(user_data)
    tgstructures = loader.read_structures(tgdata)

    match = matching.match(gtstructures, tgstructures)
    with open(output, "w+", encoding="utf-8") as file:
        json.dump(match, file, ensure_ascii=False, indent=4, cls=utils.NumpyArrayEncoder)

if __name__ == "__main__":
    app()
