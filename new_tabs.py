import numpy as np
import tifffile
import io
import xarray as xr
import panel as pn
from matplotlib.figure import Figure

# program specific
import lib.phasorpyplot as phplt
import lib.phasorfunctions as phf
import lib.phasorgraphs as phg
 
 
# Widget
file_input_tiff = pn.widgets.FileInput(accept = '.tif,.tiff', name = 'Upload TIFF')

tab4_card_ext = pn.Card(
    file_input_tiff,
    header=pn.pane.HTML(
        (
            '<div style="width: 100%; text-align: center; '
            'font-size: 1.17em; font-weight: bold; line-height: 1.7; '
            'margin: 0;">4) TIFF Image Stack</div>'
        ),
        styles = {"width": "100%", "margin": "5px 0"},
    ),
    collapsed = True,
    width_policy = "max",
    sizing_mode = "stretch_width",
)

@pn.depends(file_input_tiff)
def tab4_view_ext(file_bytes):
    # Message shown if no file is uploaded
    if file_bytes is None:
        return pn.pane.Markdown("Upload a TIFF file from sidebar.", align = "center")

    try:    # reading file
        with tifffile.TiffFile(io.BytesIO(file_bytes)) as tif:
            image_stack_np = tif.asarray()
        
        if image_stack_np.ndim == 2:
            image_stack_np = np.expand_dims(image_stack_np, axis=0)
        if image_stack_np.ndim != 3:
            raise ValueError(
                "Il TIFF deve contenere uno stack 3D nel formato "
                "(tempo, altezza, larghezza)."
            )

        image_stack = xr.DataArray(
            image_stack_np.astype(np.float64, copy=False),
            dims = ("time", "y", "x"),
        )

        # Total intensity image and global decay.
        fig_img = phg.image_flattener_pn(image_stack)
        fig_decay = Figure(figsize = (5, 5))
        ax_decay = fig_decay.add_subplot(111)
        global_decay = image_stack.sum(dim = ("y", "x")).values
        time_ns = np.linspace(0.0, 12.5, image_stack.sizes["time"], endpoint = False)
        ax_decay.step(time_ns, global_decay, where = "mid", color = "purple")
        ax_decay.set_xlabel("time (ns)")
        ax_decay.set_ylabel("counts")
        ax_decay.set_title("Global temporal decay")
        ax_decay.grid(True, alpha = 0.3)

        if image_stack.sizes["time"] < 2:
            raise ValueError(
                "Per calcolare il fasore servono almeno due immagini temporali "
                "nello stack TIFF."
            )

        # Calculate one phasor per spatial pixel from the temporal stack.
        g_img, s_img, intensity = phf.image_phasor_calc(image_stack)
        valid = np.isfinite(g_img.values) & np.isfinite(s_img.values) & (intensity.values > 0)

        fig_phasor = Figure(figsize=(5, 5))
        ax_phasor = fig_phasor.add_subplot(111)
        
        theta = np.linspace(0, np.pi, 100)
        u_circle = 0.5 + 0.5 * np.cos(theta)
        v_circle = 0.5 * np.sin(theta)
        ax_phasor.plot(u_circle, v_circle, 'k-', linewidth = 1.5, label = 'Universal Circle')
        ax_phasor.scatter(
            g_img.values[valid],
            s_img.values[valid],
            s = 3,
            alpha = 0.25,
            color = "tab:purple",
            edgecolors = "none",
            label = "Image pixels",
        )
        
        ax_phasor.set_xlim(-0.1, 1.1)
        ax_phasor.set_ylim(-0.1, 0.6)
        ax_phasor.set_xlabel('g') 
        ax_phasor.set_ylabel('s') 
        ax_phasor.set_title("Phasor Plot")
        ax_phasor.set_aspect('equal')
        ax_phasor.grid(True, linestyle = '--', alpha = 0.5)

        # Layout finale
        return pn.Row(
            pn.Column(
                pn.pane.Matplotlib(fig_img, tight = True, sizing_mode = 'stretch_width'),
                pn.pane.Matplotlib(fig_decay, tight = True, sizing_mode = 'stretch_width'),
            ),
            pn.pane.Matplotlib(fig_phasor, tight = True, sizing_mode = 'stretch_width')
        )

    except Exception as e:
        return pn.pane.Alert(f"**Errore nel caricamento del TIFF:**\n{e}", alert_type = 'danger')