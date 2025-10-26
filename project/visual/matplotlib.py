import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets


def show_image_slices(
    array,
    idx=None,
    ax_height=2,
    ax_width=2,
    spacing=(0.5, 0.75), # hw
    padding=(0.75, 0.75, 0.5, 0.25), # lrbt
    cbar_width=0.25,
    cbar_spacing=0.25,
    title=None,
    colors='rgb',
    linewidth=1.5,
    interact=True,
    **imshow_kws
):
    if array.ndim == 4:
        C, I, J, K = array.shape
        assert C == 3
        rgb = True
    elif array.ndim == 3:
        I, J, K = array.shape
        rgb = False
    else:
        raise ValueError(array.shape)

    if idx is None:
        idx = (I//2, J//2, K//2)
    i, j, k = map(int, idx)

    imshow_kws = with_defaults(imshow_kws, interpolation='none')
    color_dict = get_color_dict(brightness=0.5, saturation=1.0)
    colors = as_iterable(colors, length=3, string_ok=True)
    colors = [color_dict.get(c, c) for c in colors]
    i_line_kws = dict(color=colors[0], linewidth=linewidth)
    j_line_kws = dict(color=colors[1], linewidth=linewidth)
    k_line_kws = dict(color=colors[2], linewidth=linewidth)
  
    fig, axes, cbar_ax = subplot_grid(1, 3, ax_height, ax_width, spacing, padding, cbar_width, cbar_spacing)
    if title:
        fig.suptitle(title)

    ax_i = axes[0,0]
    ax_j = axes[0,1]
    ax_k = axes[0,2]

    if rgb:
        im_i = ax_i.imshow(array[:,i,:,:].T, origin='lower', **imshow_kws)
        im_j = ax_j.imshow(array[:,:,j,:].T, origin='lower', **imshow_kws)
        im_k = ax_k.imshow(array[:,:,:,k].T, origin='lower', **imshow_kws)
    else:
        im_i = ax_i.imshow(array[i,:,:].T, origin='lower', **imshow_kws)
        im_j = ax_j.imshow(array[:,j,:].T, origin='lower', **imshow_kws)
        im_k = ax_k.imshow(array[:,:,k].T, origin='lower', **imshow_kws)

    plt.colorbar(im_k, cax=cbar_ax)
    #cbar_ax.yaxis.set_ticks_position('left')
    #cbar_ax.yaxis.set_label_position('left')

    set_ax_spine_props(ax_i, **i_line_kws)
    set_ax_spine_props(ax_j, **j_line_kws)
    set_ax_spine_props(ax_k, **k_line_kws)

    ax_i.set_title(f'i = {i}')
    ax_j.set_title(f'j = {j}')
    ax_k.set_title(f'k = {k}')

    ax_i.set_xlabel('j'); ax_i.set_ylabel('k')
    ax_j.set_xlabel('i'); ax_j.set_ylabel('k')
    ax_k.set_xlabel('i'); ax_k.set_ylabel('j')

    v_i_j = ax_i.axvline(j, **j_line_kws)
    h_i_k = ax_i.axhline(k, **k_line_kws)
    v_j_i = ax_j.axvline(i, **i_line_kws)
    h_j_k = ax_j.axhline(k, **k_line_kws)
    v_k_i = ax_k.axvline(i, **i_line_kws)
    h_k_j = ax_k.axhline(j, **j_line_kws)

    def update_i(new_i):
        nonlocal i
        if new_i != i:
            i = int(new_i)
            im_i.set_data(array[i,:,:].T)
            v_j_i.set_xdata([i, i])
            v_k_i.set_xdata([i, i])
            ax_i.set_title(f'i = {i}')
            fig.canvas.draw_idle()

    def update_j(new_j):
        nonlocal j
        if new_j != j:
            j = int(new_j)
            im_j.set_data(array[:,j,:].T)
            v_i_j.set_xdata([j, j])
            h_k_j.set_ydata([j, j])
            ax_j.set_title(f'j = {j}')
            fig.canvas.draw_idle()

    def update_k(new_k):
        nonlocal k
        if new_k != k:
            k = int(new_k)
            im_k.set_data(array[:,:,k].T)
            h_i_k.set_ydata([k, k])
            h_j_k.set_ydata([k, k])
            ax_k.set_title(f'k = {k}')
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        if ax is ax_i:
            update_j(event.xdata)
            update_k(event.ydata)
        elif ax is ax_j:
            update_i(event.xdata)
            update_k(event.ydata)
        elif ax is ax_k:
            update_i(event.xdata)
            update_j(event.ydata)

    if interact:
        fig.canvas.mpl_connect('button_press_event', on_click)

    return fig


def subplot_grid(
    n_rows,
    n_cols,
    ax_height,
    ax_width,
    spacing=0.3,
    padding=0.0,
    cbar_width=0.0,
    cbar_spacing=0.0,
    **subplot_kws
):
    ax_heights = as_iterable(ax_height, length=n_rows)
    ax_widths  = as_iterable(ax_width,  length=n_cols)

    h_spacing, w_spacing = as_iterable(spacing, length=2)
    l_pad, r_pad, b_pad, t_pad = as_iterable(padding, length=4)

    assert len(ax_heights) == n_rows
    assert len(ax_widths)  == n_cols

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
        cbar_height = total_ax_height
        cbar_ax = fig.add_axes([
            cbar_left   / fig_width,
            cbar_bottom / fig_height,
            cbar_width  / fig_width,
            cbar_height / fig_height
        ])
        return fig, axes, cbar_ax
    else:
        return fig, axes


def as_iterable(obj, length=1, string_ok=False):
    if not is_iterable(obj, string_ok):
        return [obj] * length
    return obj


def is_iterable(obj, string_ok=False):
    if isinstance(obj, str):
        return string_ok
    return hasattr(obj, '__iter__')


def fractional_index(idx, length):
    if isinstance(idx, float):
        return int(round(idx * (length - 1)))
    return idx


def with_defaults(overrides=None, **defaults):
    return defaults | (overrides or {})


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

