import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def warpPerspectiveAndAdjust(img, M):
    '''
    Warp `img` using transform matrix `M` and rescale the image accordingly
    '''
    h, w = img.shape[:2]
    pts = np.array([[0, 0, 1],
                    [0, h, 1],
                    [w, 0, 1],
                    [w, h, 1]])

    warped_points = np.array([M @ p for p in pts])

    minX = np.min(warped_points[:,0])
    minY = np.min(warped_points[:,1])
    maxX = np.max(warped_points[:,0])
    maxY = np.max(warped_points[:,1])
    newShape = (int(maxX - minX) + 2, int(maxY - minY + 1) + 2)

    T = np.array([[1, 0, int(-minX)],
                  [0, 1, int(-minY)],
                  [0, 0,          1]])

    return cv.warpPerspective(img, T @ M, newShape)


def warpPerspective(img, pts_src, pts_dst, shape):
    '''
    Warp `img` based on a homography that maps from `pts_src` to `pts_dst`
    '''
    h, _ = cv.findHomography(np.array(pts_src), np.array(pts_dst))
    return cv.warpPerspective(img, h, shape[1::-1])


def safeLoad(pathToFile, datatype = np.uint8, divisor = 1, flag = cv.IMREAD_COLOR):
    '''
    OpenCV does no validation checks due to performance reasons.
    Therefore, this function checks if the image could be loaded
    '''
    img = cv.imread(pathToFile, flag)
    if img is None:
        sys.exit("Image could not be loaded.")

    if datatype != np.uint8:
        img = img.astype(dtype = datatype) / divisor

    return img


def imageStats(img):
    '''
    Returns a few image statistics
    '''
    s = img.shape
    return f'Width: {s[1]}, Height: {s[0]}, Channels: {s[2] if len(s) > 2 else 1}, Datatype: {img.dtype}'


def contrastStretching(img, outlierCropFraction = 0):
    '''
    Applies the contrast stretching to an image
    '''
    imgF = img.astype(np.float32)
    cv.normalize(imgF, imgF, 0 - outlierCropFraction, 1 + outlierCropFraction, cv.NORM_MINMAX)
    if outlierCropFraction == 0:
        return imgF
    return np.clip(imgF, 0, 1)


def showImage(title, original_img, normalize = False, show_window_now = True, max_window_size = (12, 6), convert_BGR_to_RGB = True, grayscale_cmap = 'gray', ax = None):
    '''
    Displays an image and waits till the window is closed
    '''
    print(imageStats(original_img))

    img = original_img.copy()
    cmap = vmin = vmax = None
    if len(img.shape) < 3 or img.shape[2] < 3: # grayscale
        cmap = grayscale_cmap
        if cmap == 'gray':
            vmin = 0
            vmax = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0 # float
    else:
        if convert_BGR_to_RGB:
            img = img[:,:,::-1]

    if normalize:
        img = contrastStretching(img)

    def imshow():
        return ax.imshow(img, interpolation='antialiased', cmap=cmap, vmin=vmin, vmax=vmax)

    if ax is None:
        fig = plt.figure(title)
        ax = fig.subplots()
        plt_img = imshow()
        ax.axis('off')

        if max_window_size is not None:
            s = np.array(img.shape[:2]) / max_window_size[::-1]
            s *= max_window_size[::-1] / s.max()
            s = np.maximum(3, s)
            fig.set_figheight(s[0])
            fig.set_figwidth(s[1])

        fig.tight_layout()
    else:
        plt_img = imshow()
        ax.axis('off')
        fig = ax.get_figure()

    if show_window_now:
        plt.show()
    return plt_img, ax, fig


def showImageList(title, imgs, num_cols = None, normalize = False, show_window_now = True, convert_BGR_to_RGB = True, width_ratios = None, height_ratios = None, spacing = None, padding = None, max_window_size = (12, 6)):
    '''
    Displays multiple images and waits till the window is closed

    title:
        string

        Window title

    imgs:
        [None | image | ['caption', image] | ['caption', image, [h span, v span]], ...]

        List of images. Blank gap if None

    num_cols:
        int | None

        Number of images per line. All in one line if None

    normalize:
        True | False | [True | False, ...]

        Contrast streching. Single value applies to all images, array to the corresponding entry

    show_window_now:
        True | False

        If False, `plt.show()` has to be called later manually

    convert_BGR_to_RGB:
        True | False | [True | False, ...]

    width_ratios:
        [float, ...] | None

        Ratios of column widths

    height_ratios:
        [float, ...] | None

        Ratios of column heights

    spacing:
        [int, int] | None | False

        Horizontal and vertical spacing between images.
        `None` will use default values.
        `False` will not set spacing

    padding:
        [int, int, int, int] | None | False

        Space between window frame and images for all sides in the order: left, bottom, right, top.
        `None` will use default values.
        `False` will not set padding

    max_window_size:
        [int, int] | None

        Max window size (width, height), fitted to content aspect ratio
    '''
    tmp_imgs = []
    tmp_num_cols = 0 if num_cols is None else num_cols

    for img in imgs:
        tmp = type('', (), {})()
        tmp.pos = (0, 0)
        tmp.img = img
        tmp.title = None
        tmp.span = (1, 1)
        tmp.normalize = normalize[len(tmp_imgs)] if type(normalize) in (tuple, list) else normalize
        if type(img) in (tuple, list):
            tmp.img = img[1]
            tmp.title = img[0]
            if len(img) > 2:
                tmp.span = img[2]
        tmp_imgs.append(tmp)
        if num_cols is None:
            tmp_num_cols += tmp.span[0]

    num_cols = tmp_num_cols

    i = 0
    grid_blocked = []
    for tmp in tmp_imgs:
        while i in grid_blocked:
            i += 1
        tmp.pos = (i // num_cols, i % num_cols)
        for x in range(tmp.span[0]):
            for y in range(tmp.span[1]):
                grid_blocked.append(i + y * num_cols + x)

    num_grid_cells = max(grid_blocked) + 1
    num_rows = (num_grid_cells - 1) // num_cols + 1
    grid = (num_rows, num_cols)

    if width_ratios is not None:
        if len(width_ratios) > num_cols:
            width_ratios = width_ratios[:num_cols]
        elif len(width_ratios) < num_cols:
            width_ratios = width_ratios + ([1] * (num_cols - len(width_ratios)))
    if height_ratios is not None:
        if len(height_ratios) > num_rows:
            height_ratios = height_ratios[:num_rows]
        elif len(height_ratios) < num_rows:
            height_ratios = height_ratios + ([1] * (num_rows - len(height_ratios)))

    fig = plt.figure(title) if title is not None else plt.gcf()
    gs = gridspec.GridSpec(grid[0], grid[1], width_ratios=width_ratios, height_ratios=height_ratios)

    plt_imgs = []
    axs = []
    i = -1
    with_titles = False
    for tmp in tmp_imgs:
        i += 1
        rowStart = tmp.pos[0]
        rowStop = tmp.pos[0] + tmp.span[1]
        colStart = tmp.pos[1]
        colStop = tmp.pos[1] + tmp.span[0]
        ax = plt.subplot(gs[rowStart : rowStop, colStart : colStop])
        axs.append(ax)
        if tmp.img is not None:
            plt_img, _, _ = showImage(None, tmp.img, tmp.normalize, False, convert_BGR_to_RGB=convert_BGR_to_RGB, ax=ax)
            plt_imgs.append(plt_img)
        if tmp.title is not None:
            ax.set_title(tmp.title)
            with_titles = True

    if max_window_size is not None:
        window_size = list(grid)
        if width_ratios is not None:
            window_size[1] = 0
            for i in range(grid[1]):
                window_size[1] += width_ratios[i]
        if height_ratios is not None:
            window_size[0] = 0
            for i in range(grid[0]):
                window_size[0] += height_ratios[i]
        s = np.array(window_size) / max_window_size[::-1]
        s *= max_window_size[::-1] / s.max()
        s = np.maximum(3, s)
        if with_titles:
            s[0] += .5
        fig.set_figheight(s[0])
        fig.set_figwidth(s[1])

    fig.tight_layout()
    if spacing is not False:
        if spacing is None:
            spacing = (.01,) * 2
        fig.subplots_adjust(wspace=spacing[0], hspace=spacing[1])
    if padding is not False:
        if padding is None:
            padding = [0] * 4
            if max_window_size is not None and with_titles:
                padding[3] += .1
        fig.subplots_adjust(left=padding[0], bottom=padding[1], right=1 - padding[2], top=1 - padding[3])

    if show_window_now:
        plt.show()
    return plt_imgs, axs, fig


def addMouseButtonEvent(plt_fig, callback):
    return plt_fig.canvas.mpl_connect('button_press_event', callback)

def addMouseMoveEvent(plt_fig, callback):
    return plt_fig.canvas.mpl_connect('motion_notify_event', callback)

def removeMouseEvent(plt_fig, event_id):
    plt_fig.canvas.mpl_disconnect(event_id)
