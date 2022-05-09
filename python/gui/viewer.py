import loader
import matching
import utils

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from typing import Dict, List

pd.set_option("display.precision", 3)
render_width, render_height = 1024, 1024

def show(origin: np.ndarray) -> None:
    fig = plt.figure()

    plt.imshow(origin)
    plt.axis("off")
    st.pyplot(fig)

def dfstyle(styler):
    styler.format(precision=2)
    # styler.format_index(precision=2)
    return styler

def viewer(
    gtdata: Dict[str, np.ndarray],
    tgdata: Dict[str, np.ndarray]
) -> None:
    with st.container():
        ground_selection = st.selectbox(
            label="Ground",
            options=gtdata.keys()
        )

        gtstructures: np.ndarray = gtdata[ground_selection]
        gtindex, gtendpoints = loader.read_endpoints(gtstructures)
        ground = gtendpoints.reshape(-1, 3)

        preview_column, data_column = st.columns([2, 4])
        with preview_column:
            origin = utils.plot([ground], [gtstructures], render_width, render_height)
            show(origin)
        with data_column:
            pass

    with st.container():
        target_selection = st.multiselect(
            label="Predictions to compare",
            options=tgdata.keys(),
        )

        for target_model in target_selection:
            tgstructures: np.ndarray = tgdata[target_model]
            tgindex, tgendpoints = loader.read_endpoints(tgstructures)
            target = tgendpoints.reshape(-1, 3)

            with st.container():
                preview_column, metrics_column, iou_column = st.columns([2, 2, 2])
                with preview_column:
                    origin = utils.plot([target], [tgstructures], render_width, render_height)
                    show(origin)
                with metrics_column:
                    with st.spinner("Calculating precision/recall metrics"):
                        metrics = matching.calculate_metrics(gtindex, ground, tgindex, target)
                        st.subheader("Metrics:")
                        for classname, metrics in metrics.items():
                            st.write(f"**{classname.capitalize()}**: found {metrics['predicted']} of {metrics['total']}")
                            df = pd.DataFrame.from_dict(metrics["thresholds"], orient="index")
                            st.table(df.style.pipe(dfstyle))

                with iou_column:
                    with st.spinner("Calculating 3D IoU"):
                        ious = matching.calculate_iou(gtindex, ground.reshape(-1, 8, 3), tgindex, target.reshape(-1, 8, 3)).get("general", dict())
                        st.subheader("IoU matching:")
                        for i, (classname, iou) in enumerate(ious.items()):
                            st.write(f"{classname.capitalize()}: ", iou.get("mean", .0))
