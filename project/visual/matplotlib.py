import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets

from ..core import utils


class SliceViewer:

    def __init__(
        self,
        array=None,
        index=None,
        vox_spacing=None,
        ax_height=2,
        ax_width=2,
        spacing=(0.5, 0.75), # hw
        padding=(0.75, 0.75, 0.5, 0.5), # lrbt
        cbar_width=0.25,
        cbar_spacing=0.25,
        line_color='rgb',
        line_width=1.5,
        title=None,
        **imshow_kws
    ):
        self._initialized = False

        self.vox_spacing = vox_spacing or (1, 1, 1)
        self.vox_size = tuple(abs(v) for v in self.vox_spacing)
        self.ax_signs = tuple(1 if v >= 0 else -1 for v in self.vox_spacing)

        self.subplot_kws = dict(
            ax_height=ax_height,
            ax_width=ax_width,
            spacing=spacing,
            padding=padding,
            cbar_width=cbar_width,
            cbar_spacing=cbar_spacing
        )

        # style helpers
        self.imshow_kws = utils.update_defaults(imshow_kws, interpolation='none')
        color_dict = get_color_dict(brightness=0.5, saturation=1.0)
        colors = utils.as_iterable(line_color, length=3, string_ok=True)
        colors = [color_dict.get(c, c) for c in colors]
        self.i_line_kws = dict(color=colors[0], linewidth=line_width)
        self.j_line_kws = dict(color=colors[1], linewidth=line_width)
        self.k_line_kws = dict(color=colors[2], linewidth=line_width)
    
        self.init_figure(title)

        if array is not None:
            self.update_array(array, index)

    def init_figure(self, title=None):
        fig, axes, cbar_ax = subplot_grid(1, 3, **self.subplot_kws)

        self.fig = fig
        self.axes = axes
        self.cbar_ax = cbar_ax

        self.ax_i = axes[0,0]
        self.ax_j = axes[0,1]
        self.ax_k = axes[0,2]

        self.fig.suptitle(title)

    def update_array(self, array, index=None):
        array = np.asarray(array)
    
        if array.ndim == 4:
            if array.shape[0] != 3 and array.shape[3] == 3:
                array = array.transpose(3,0,1,2)
            if array.shape[0] != 3:
                raise ValueError(array.shape)
            _, I, J, K = array.shape
            self.rgb = True
    
        elif array.ndim == 3:
            I, J, K = array.shape
            self.rgb = False
    
        else:
            raise ValueError(array.shape)

        if index is None:
            i, j, k = I//2, J//2, K//2
        else:
            i, j, k = map(int, index)

        self.array = array
        self.shape = (I, J, K)
        self.index = (i, j, k)

        self.update_images()

    def _display_slice_i(self):
        _, sy, sz = self.ax_signs
        if self.rgb:
            return self.array[:, self.index[0], ::sy, ::sz].T
        return self.array[self.index[0], ::sy, ::sz].T

    def _display_slice_j(self):
        sx, _, sz = self.ax_signs
        if self.rgb:
            return self.array[:, ::sx, self.index[1], ::sz].T
        return self.array[::sx, self.index[1], ::sz].T

    def _display_slice_k(self):
        sx, sy, _ = self.ax_signs
        if self.rgb:
            return self.array[:, ::sx, ::sy, self.index[2]].T
        return self.array[::sx, ::sy, self.index[2]].T

    def _display_coords(self, i, j, k):
        I, J, K = self.shape
        sx, sy, sz = self.ax_signs

        # map voxel indices to displayed coordinates
        di = i if sx > 0 else I - 1 - i
        dj = j if sy > 0 else J - 1 - j
        dk = k if sz > 0 else K - 1 - k

        return di, dj, dk

    def init_images(self, interact=True):
        assert not self._initialized
        i, j, k = self.index

        dx, dy, dz = self.vox_size
        aspect_i = dz / dy
        aspect_j = dz / dx
        aspect_k = dy / dx

        self.im_i = self.ax_i.imshow(
            self._display_slice_i(),
            origin='lower',
            aspect=aspect_i,
            **self.imshow_kws
        )
        self.im_j = self.ax_j.imshow(
            self._display_slice_j(),
            origin='lower',
            aspect=aspect_j,
            **self.imshow_kws
        )
        self.im_k = self.ax_k.imshow(
            self._display_slice_k(),
            origin='lower',
            aspect=aspect_k,
            **self.imshow_kws
        )
        if not self.rgb:
            plt.colorbar(self.im_k, cax=self.cbar_ax)

        # crosshair lines
        di, dj, dk = self._display_coords(i, j, k)
        self.v_i_j = self.ax_i.axvline(dj, **self.j_line_kws)
        self.h_i_k = self.ax_i.axhline(dk, **self.k_line_kws)
        self.v_j_i = self.ax_j.axvline(di, **self.i_line_kws)
        self.h_j_k = self.ax_j.axhline(dk, **self.k_line_kws)
        self.v_k_i = self.ax_k.axvline(di, **self.i_line_kws)
        self.h_k_j = self.ax_k.axhline(dj, **self.j_line_kws)

        self.ax_i.set_title(f'i = {i}')
        self.ax_j.set_title(f'j = {j}')
        self.ax_k.set_title(f'k = {k}')

        if interact:
            self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self._initialized = True

    def update_images(self):
        if not self._initialized:
            return self.init_images()

        i, j, k = self.index

        self.im_i.set_data(self._display_slice_i())
        self.im_j.set_data(self._display_slice_j())
        self.im_k.set_data(self._display_slice_k())

        di, dj, dk = self._display_coords(i, j, k)
        self.v_j_i.set_xdata([di, di])
        self.v_k_i.set_xdata([di, di])
        self.v_i_j.set_xdata([dj, dj])
        self.h_k_j.set_ydata([dj, dj])
        self.h_i_k.set_ydata([dk, dk])
        self.h_j_k.set_ydata([dk, dk])

        self.ax_i.set_title(f'i = {i}')
        self.ax_j.set_title(f'j = {j}')
        self.ax_k.set_title(f'k = {k}')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_i(self, new_i):
        i, j, k = self.index
        I, _, _ = self.shape
        i = int(np.clip(np.round(new_i), 0, I - 1))
        self.index = (i, j, k)
        self.update_images()

    def update_j(self, new_j):
        i, j, k = self.index
        _, J, _ = self.shape
        j = int(np.clip(np.round(new_j), 0, J - 1))
        self.index = (i, j, k)
        self.update_images()

    def update_k(self, new_k):
        i, j, k = self.index
        _, _, K = self.shape
        k = int(np.clip(np.round(new_k), 0, K - 1))
        self.index = (i, j, k)
        self.update_images()

    def on_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        I, J, K = self.shape
        sx, sy, sz = self.ax_signs

        def inv_x(coord, n, s):
            coord = int(np.round(coord))
            coord = int(np.clip(coord, 0, n - 1))
            return coord if s > 0 else (n - 1 - coord)
    
        ax = event.inaxes
        if ax == self.ax_i:
            j = inv_x(event.xdata, J, sy)
            k = inv_x(event.ydata, K, sz)
            self.index = (self.index[0], j, k)
            self.update_images()

        elif ax == self.ax_j:
            i = inv_x(event.xdata, I, sx)
            k = inv_x(event.ydata, K, sz)
            self.index = (i, self.index[1], k)
            self.update_images()

        elif ax == self.ax_k:
            i = inv_x(event.xdata, I, sx)
            j = inv_x(event.ydata, J, sy)
            self.index = (i, j, self.index[2])
            self.update_images()


def subplot_grid(
    n_rows,
    n_cols,
    ax_height,
    ax_width,
    spacing=0.3, # hw
    padding=0.0, # lrbt
    cbar_width=0.0,
    cbar_spacing=0.0,
    **subplot_kws
):
    ax_heights = utils.as_iterable(ax_height, length=n_rows)
    ax_widths  = utils.as_iterable(ax_width,  length=n_cols)

    h_spacing, w_spacing = utils.as_iterable(spacing, length=2)
    l_pad, r_pad, b_pad, t_pad = utils.as_iterable(padding, length=4)

    assert len(ax_heights) == n_rows, (len(ax_heights), n_rows)
    assert len(ax_widths)  == n_cols, (len(ax_widths), n_cols)

    total_ax_height = sum(ax_heights)
    total_ax_width  = sum(ax_widths)

    total_h_spacing = (n_rows - 1) * h_spacing 
    total_w_spacing = (n_cols - 1) * w_spacing

    fig_height = total_ax_height + total_h_spacing + b_pad + t_pad
    fig_width  = total_ax_width  + total_w_spacing + l_pad + r_pad

    extra_width = 0
    if cbar_width > 0:
        extra_width = cbar_width + cbar_spacing
        fig_width += extra_width

    mean_ax_height = total_ax_height / n_rows
    mean_ax_width  = total_ax_width  / n_cols

    gridspec_kws = {
        'height_ratios': ax_heights,
        'width_ratios':  ax_widths,
        'hspace': h_spacing / mean_ax_height,
        'wspace': w_spacing / mean_ax_width,
        'left': l_pad / fig_width,
        'right': 1.0 - (r_pad + extra_width) / fig_width,
        'bottom': b_pad / fig_height,
        'top': 1.0 - t_pad / fig_height
    }
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(fig_width, fig_height),
        gridspec_kw=gridspec_kws,
        **subplot_kws
    )

    if cbar_width > 0:
        cbar_left = total_ax_width + total_w_spacing + l_pad + cbar_spacing
        cbar_bottom = b_pad
        cbar_height = total_ax_height + total_h_spacing
        cbar_ax = fig.add_axes([
            cbar_left   / fig_width,
            cbar_bottom / fig_height,
            cbar_width  / fig_width,
            cbar_height / fig_height
        ])
        return fig, axes, cbar_ax
    else:
        return fig, axes


def set_ax_spine_props(ax, color, linewidth):
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(linewidth)


def get_color_dict(brightness=0.5, saturation=1.0):
    l = max(brightness - saturation / 2, 0)
    h = min(brightness + saturation / 2, 1)
    m = h * 0.9 + l * 0.1
    return {
        'r': (h,l,l),
        'g': (l,m,l),
        'b': (l,l,h),
        'c': (l,m,m),
        'm': (h,l,h),
        'y': (m,m,l),
        'w': (h,h,h),
        'k': (l,l,l),
    }


def get_label_cmap(
    n_labels: int = 5,
    base: str = 'tab10',
    select = (0, 9, 2, 8, 7),
    weights = (0.2, 0.7, 0.1),
    grayscale: bool = False,
    colorblind: bool = False,
    include_background: bool = True
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    c = plt.get_cmap(base).colors
    if select:
        c = [c[idx] for idx in select]
    w_r, w_g, w_b = weights
    if colorblind:
        cb = lambda r,g: (r*w_r + g*w_g) / (w_r + w_g)
        c = [(cb(r,g), cb(r,g), b) for r,g,b in c]
    if grayscale:
        gs = lambda r,g,b: (r*w_r + g*w_g + b*w_b)
        c = [(gs(r,g,b),)*3 for r,g,b in c]
    assert n_labels <= len(c), (n_labels, len(c))
    if include_background:
        return ListedColormap(['white'] + c[:n_labels])
    return ListedColormap(c[:n_labels])


def get_color_kws(key: str, n_labels: int = 5):
    k = key.lower()
    if k in {'image', 'img_true', 'img_pred'}:
        return dict(cmap='gray', clim=(-1, 1))
    elif k in {'E', 'E_true', 'E_pred'}:
        return dict(cmap='jet', clim=(0, 1e4), line_color='cmy')
    elif k in {'logE', 'logE_true', 'logE_pred'}:
        return dict(cmap='jet', clim=(2, 6), line_color='cmy')
    elif k in {'nu', 'nu_true', 'nu_pred'}:
        return dict(cmap='gray', clim=(0, 0.5), line_color='cmy')
    elif k in {'rho', 'rho_true', 'rho_pred'}:
        return dict(cmap='gray', clim=(0, 2e3), line_color='cmy')
    elif k in {'material', 'mat_true'} or k.startswith('mat_pred'):
        return dict(cmap=get_label_cmap(n_labels), clim=(1, n_labels))
    return dict(cmap='seismic', clim=(-3, 3))

