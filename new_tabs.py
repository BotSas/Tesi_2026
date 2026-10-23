import io

import holoviews as hv
import numpy as np
import panel as pn
import tifffile
import xarray as xr
import traceback

import lib.phasorfunctions as phf
import lib.phasorgraphs as phg


hv.extension("bokeh")


file_input_tiff = pn.widgets.FileInput(accept=".tif,.tiff", name="Upload TIFF")
roi_width_input = pn.widgets.IntSlider(
    name="ROI width (pixel)", start=1, end=101, value=20, step=1
)
median_size_input = pn.widgets.IntSlider(
    name="Median filter size", start=1, end=9, value=3, step=2
)
threshold_input = pn.widgets.FloatSlider(
    name="Intensity threshold", start=0.0, end=1.0, value=0.05, step=0.01
)
irf_g_input = pn.widgets.FloatInput(name="IRF G", value=1.0, step=0.01)
irf_s_input = pn.widgets.FloatInput(name="IRF S", value=0.0, step=0.01)


tab4_card_ext = pn.Card(
    file_input_tiff,
    pn.layout.Divider(),
    pn.pane.Markdown("**ROI and phasor processing**"),
    roi_width_input,
    median_size_input,
    threshold_input,
    pn.pane.Markdown("**IRF phasor**"),
    irf_g_input,
    irf_s_input,
    header=pn.pane.HTML(
        (
            '<div style="width: 100%; text-align: center; '
            'font-size: 1.17em; font-weight: bold; line-height: 1.7; '
            'margin: 0;">4) TIFF Image Stack</div>'
        ),
        styles={"width": "100%", "margin": "5px 0"},
    ),
    collapsed=True,
    width_policy="max",
    sizing_mode="stretch_width",
)


def _read_stack(file_bytes):
    with tifffile.TiffFile(io.BytesIO(file_bytes)) as tif:
        stack = tif.asarray()

    if stack.ndim == 2:
        stack = np.expand_dims(stack, axis=0)
    if stack.ndim != 3:
        raise ValueError(
            "Il TIFF deve contenere uno stack 3D nel formato (tempo, altezza, larghezza)."
        )

    # x/y are deliberately the spatial dimensions expected by phasors_from_region.
    return xr.DataArray(
        stack.astype(np.float64, copy=False),
        dims=("time", "x", "y"),
    )


def _roi_mask(nx, ny, x_center, y_center, width):
    width = max(1, min(int(width), nx, ny))
    half = width // 2
    x0 = max(0, int(round(x_center)) - half)
    x1 = min(nx, x0 + width)
    y0 = max(0, int(round(y_center)) - half)
    y1 = min(ny, y0 + width)
    mask = np.zeros((nx, ny), dtype=bool)
    mask[x0:x1, y0:y1] = True
    return mask, (x0, x1, y0, y1)


def _phasor_plot(df, mean, std, irf_g, irf_s):
    cloud = hv.Points(
        (df["G"].to_numpy(), df["S"].to_numpy()), kdims=["G", "S"], label="ROI cloud"
    ).opts(color="purple", size=4, alpha=0.35)
    mean_point = hv.Points(
        [(mean[0], mean[1])], kdims=["G", "S"], label="ROI mean"
    ).opts(color="red", size=10, marker="star")
    irf_point = hv.Points(
        [(irf_g, irf_s)], kdims=["G", "S"], label="IRF"
    ).opts(color="black", size=8, marker="x")

    theta = np.linspace(0, np.pi, 300)
    circle = hv.Curve(
        (0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta)), kdims=["G"], vdims=["S"]
    ).opts(color="gray", line_width=1.5)
    return (circle * cloud * mean_point * irf_point).opts(
        title=f"ROI phasors (N={len(df)}, mean +/- std: "
        f"({mean[0]:.3f} +/- {std[0]:.3f}, {mean[1]:.3f} +/- {std[1]:.3f})",
        width=600,
        height=500,
        xlim=(-0.1, 1.1),
        ylim=(-0.1, 0.7),
        xlabel="g",
        ylabel="s",
        data_aspect=1,
        legend_position="bottom",
        tools=["hover"],
    )

#_Light cache - makes param alteration easier ______________________________________________________________

_cached_tiff = {"bytes_id": None, "stack": None}

def _get_stack(file_bytes):
    if _cached_tiff["bytes_id"] != id(file_bytes):
        _cached_tiff["stack"] = _read_stack(file_bytes)
        _cached_tiff["bytes_id"] = id(file_bytes)
    return _cached_tiff["stack"]

#_______________________________________________________________________________________________________

def _render_tab4(file_bytes, roi_width, median_size, threshold_pct, irf_g, irf_s):
    if not file_bytes:
        return pn.pane.Markdown("Upload a TIFF file from sidebar.", align="center")

    try:
        image_stack = _read_stack(file_bytes)
        nx, ny = image_stack.sizes["x"], image_stack.sizes["y"]
        flat_image = image_stack.sum(dim="time").values
        
        image = hv.Image(
            (np.arange(nx), np.arange(ny), flat_image.T),
            kdims=["x", "y"],
            vdims=["counts"],
        ).opts(
            cmap="magma",
            colorbar=True,
            width=600,
            height=600,
            tools=["tap", "hover"],
            active_tools=["tap"],
            xlabel="x",
            ylabel="y",
            title="Flattened image - click to select ROI",
        )
        tap = hv.streams.Tap(source=image, x=float(nx / 2), y=float(ny / 2))

        def selected_roi(x, y):
            x_val = nx / 2 if x is None else x
            y_val = ny / 2 if y is None else y
            return _roi_mask(nx, ny, x_val, y_val, roi_width)[1]

        def image_with_roi(x, y):
            x0, x1, y0, y1 = selected_roi(x, y)
            # Bounds usa (left, bottom, right, top) senza conflitti di kdims!
            bounds = hv.Bounds((x0, y0, x1, y1)).opts(
                color="cyan", line_width=3
            )
            return image * bounds

        def roi_decay(x, y):
            x0, x1, y0, y1 = selected_roi(x, y)
            decay = image_stack.isel(x=slice(x0, x1), y=slice(y0, y1)).sum(
                dim=("x", "y")
            )
            time_ns = np.linspace(
                0.0, 12.5, image_stack.sizes["time"], endpoint=False
            )
            return hv.Curve((time_ns, decay.values), "time (ns)", "counts").opts(
                title=f"ROI decay: x={x0}:{x1}, y={y0}:{y1}",
                width=600,
                height=300,
                line_width=2,
                color="purple",
                tools=["hover"],
            )

        def roi_phasors(x, y):
            x_val = nx / 2 if x is None else x
            y_val = ny / 2 if y is None else y
            mask, _ = _roi_mask(nx, ny, x_val, y_val, roi_width)
            df = phf.phasors_from_region(
                image_stack,
                roi_mask=mask,
                median_size=int(median_size),
                threshold_pct=float(threshold_pct),
            )
            g_dec, s_dec = phf.phasor_deconv(
                df["G"].to_numpy(), df["S"].to_numpy(), float(irf_g), float(irf_s)
            )
            mean, _, std, _, _, _ = phf.cloud_statistics(g_dec, s_dec)
            df_deconvolved = df.assign(G=g_dec, S=s_dec)
            return _phasor_plot(
                df_deconvolved, mean, std, float(irf_g), float(irf_s)
            )

        # Test
        _ = roi_phasors(nx / 2, ny / 2)

        return pn.Column(
            hv.DynamicMap(image_with_roi, streams=[tap]),
            pn.Row(
                hv.DynamicMap(roi_decay, streams=[tap]),
                hv.DynamicMap(roi_phasors, streams=[tap]),
            ),
            sizing_mode="stretch_width",
        )
    except Exception as exc:
        return pn.pane.Alert(
            f"**There was an error uploading tiff file :( **\n```python\n{traceback.format_exc()}\n```",
            alert_type="danger",
        )

#__exporting to graphical_engine_______________________________________________________________-

tab4_view_ext = pn.panel(
    pn.bind(
        _render_tab4,
        file_bytes=file_input_tiff.param.value,
        roi_width=roi_width_input.param.value,
        median_size=median_size_input.param.value,
        threshold_pct=threshold_input.param.value,
        irf_g=irf_g_input.param.value,
        irf_s=irf_s_input.param.value,
    ),
    sizing_mode="stretch_width",
)