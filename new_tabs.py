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
file_input_tiff = pn.widgets.FileInput(accept = '.tif,.tiff', name='Upload TIFF')

tab4_card_ext = pn.Card(            # non riesco a farla vedereeeeeee
    file_input_tiff,
    title = "4) TIFF Image Stack",
    collapsed=True
)

@pn.depends(file_input_tiff)
def tab4_view_ext(file_bytes):
    # Message shown if no file is uploaded
    if file_bytes is None:
        return pn.pane.Markdown("Upload a TIFF file from sidebar.", align="center")

    try:    # reading file
        with tifffile.TiffFile(io.BytesIO(file_bytes)) as tif:
            image_stack_np = tif.asarray()
        
        if image_stack_np.ndim >= 3:
            image_stack = xr.DataArray(image_stack_np, dims=("time", "y", "x"))
        else:
            image_stack = xr.DataArray(np.expand_dims(image_stack_np, axis = 0), dims = ("time", "y", "x"))

        fig_img = phg.image_flattener(image_stack)

        # PHASOR PLANE (needs fixing)
        fig_phasor = Figure(figsize=(5, 5))
        ax_phasor = fig_phasor.add_subplot(111)
        
        theta = np.linspace(0, np.pi, 100)
        u_circle = 0.5 + 0.5 * np.cos(theta)
        v_circle = 0.5 * np.sin(theta)
        ax_phasor.plot(u_circle, v_circle, 'k-', linewidth=1.5, label='Universal Circle')
        
        ax_phasor.set_xlim(-0.1, 1.1)
        ax_phasor.set_ylim(-0.1, 0.6)
        ax_phasor.set_xlabel('g') 
        ax_phasor.set_ylabel('s') 
        ax_phasor.set_title("Phasor Plot")
        ax_phasor.set_aspect('equal')
        ax_phasor.grid(True, linestyle='--', alpha=0.5)

        # Layout finale
        return pn.Row(
            pn.pane.Matplotlib(fig_img, tight=True, sizing_mode='stretch_width'),
            pn.pane.Matplotlib(fig_phasor, tight=True, sizing_mode='stretch_width')
        )

    except Exception as e:
        return pn.pane.Alert(f"**alèjgm adfji gnadfòlk n:**\n{e}", alert_type='danger')