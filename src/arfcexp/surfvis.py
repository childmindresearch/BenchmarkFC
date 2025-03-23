import math
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, NamedTuple

import nibabel as nib
import numpy as np
import pyvista as pv
from matplotlib import image
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import IdentityTransform
from PIL import Image, ImageOps

pv.start_xvfb()

CameraPos = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

VIEW_CAMERA_POS_MAP = {
    ("lh", "lateral"): ((-500, 0, 0), (0, 0, 0), (0, 0, 1)),
    ("lh", "medial"): ((500, 0, 0), (0, 0, 0), (0, 0, 1)),
    ("lh", "posterior"): ((0, -500, 0), (0, 0, 0), (0, 0, 1)),
    ("lh", "anterior"): ((0, 500, 0), (0, 0, 0), (0, 0, 1)),
    ("lh", "inferior"): ((0, 0, -500), (0, 0, 0), (-1, 0, 0)),
    ("lh", "superior"): ((0, 0, 500), (0, 0, 0), (1, 0, 0)),
    ("rh", "lateral"): ((500, 0, 0), (0, 0, 0), (0, 0, 1)),
    ("rh", "medial"): ((-500, 0, 0), (0, 0, 0), (0, 0, 1)),
    ("rh", "posterior"): ((0, -500, 0), (0, 0, 0), (0, 0, 1)),
    ("rh", "anterior"): ((0, 500, 0), (0, 0, 0), (0, 0, 1)),
    ("rh", "inferior"): ((0, 0, -500), (0, 0, 0), (1, 0, 0)),
    ("rh", "superior"): ((0, 0, 500), (0, 0, 0), (-1, 0, 0)),
}

# This is the width of the left lateral view / window size after cropping.
# We scale window size by this so that the cropped images are about the requested size.
WINDOW_SCALE = 5 / 3


class View(StrEnum):
    LATERAL = "lateral"
    MEDIAL = "medial"
    POSTERIOR = "posterior"
    ANTERIOR = "anterior"
    INFERIOR = "inferior"
    SUPERIOR = "superior"


class Surface(NamedTuple):
    points: np.ndarray
    faces: np.ndarray


class Overlay(ScalarMappable):
    """Color mapped overlay.

    See also:
        `matplotlib.image.imsave`
        `matplotlib.colorizer.Colorizer`
    """

    def __init__(
        self,
        values: np.ndarray,
        cmap: str | Colormap | None = None,
        norm: str | Normalize | None = None,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float | None = None,
    ):
        super().__init__(norm=norm, cmap=cmap)
        self.set_array(values)
        self.set_clim(vmin=vmin, vmax=vmax)
        self._alpha = alpha

    def pixel_values(self) -> np.ndarray:
        return self.to_rgba(self.get_array(), alpha=self._alpha)


class Plotter:
    """Surface plotter using pyvista."""

    def __init__(
        self,
        surf: Path | Surface,
        hemi: Literal["lh", "rh"] = "lh",
        sulc: Path | np.ndarray | None = None,
        color: Any = (0.6, 0.6, 0.6),
        width: int = 512,
    ):
        if isinstance(surf, (str, Path)):
            surf = _read_surface(surf)
        self._surf = Surface(*surf)
        self._hemi = hemi
        self._color = color
        self._width = width

        n_points = len(self._surf.points)
        if sulc is not None:
            if isinstance(sulc, (str, Path)):
                sulc = _read_shape(sulc)

            if sulc.shape != (n_points,):
                raise ValueError(
                    f"sulc data doesn't match surface; expected shape ({n_points},)."
                )

            self._base_layer = Overlay(
                np.where(sulc < 0, 0.4, 0.6), cmap="gray", vmin=0.0, vmax=1.0
            ).pixel_values()
        else:
            self._base_layer = _constant_layer(n_points, color)

        self._poly = _surface_to_polydata(surf)
        self._plotter = pv.Plotter(
            window_size=(int(WINDOW_SCALE * width), int(WINDOW_SCALE * 0.75 * width)),
            off_screen=True,
        )
        self._overlays: list[Overlay] = []

    def overlay(
        self,
        values: np.ndarray | Overlay,
        cmap: str | Colormap | None = None,
        norm: str | Normalize | None = None,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float | None = None,
    ) -> Overlay:
        # Invalidate the plotter. Note, plotter.clear() also clears shading properties.
        self._plotter.actors.clear()
        if isinstance(values, Overlay):
            overlay = values
        else:
            overlay = Overlay(
                values=values, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha
            )
        self._overlays.append(overlay)
        return overlay

    def _render(self) -> np.ndarray:
        layers = [self._base_layer]
        layers += [overlay.pixel_values() for overlay in self._overlays]
        composite = _alpha_composite(layers)

        self._plotter.actors.clear()

        self._plotter.add_mesh(
            self._poly.copy(),
            scalars=composite,
            rgb=True,
            show_scalar_bar=False,
        )

    def screenshot(self, view: View | CameraPos = View.LATERAL) -> Image.Image:
        if isinstance(view, (View, str)):
            camera_pos = VIEW_CAMERA_POS_MAP[(self._hemi, View(view).value)]
        else:
            camera_pos = view

        if len(self._plotter.renderer.actors) == 0:
            self._render()

        self._plotter.camera_position = camera_pos
        self._plotter.render()
        img = self._plotter.screenshot(return_img=True, transparent_background=True)

        img = _crop_transparent_background(img)
        return Image.fromarray(img)

    def imshow(self, view: View | CameraPos = View.LATERAL) -> Overlay | None:
        img = self.screenshot(view)
        plt.imshow(img, aspect="equal")
        if len(self._overlays) > 0:
            return self._overlays[-1]
        return None

    def clear(self):
        self._overlays.clear()
        self._plotter.actors.clear()


def _read_surface(path: Path) -> Surface:
    path = Path(path)
    match path.suffix:
        case ".gii":
            surf = nib.load(path)
            points = surf.darrays[0].data
            faces = surf.darrays[1].data
            surf = Surface(points, faces)
        case _:
            raise ValueError(
                f"Unsupported surface format: {path}. Only .gii supported."
            )
    return surf


def _read_shape(path: Path) -> np.ndarray:
    path = Path(path)
    match path.suffix:
        case ".gii":
            shape = nib.load(path).darrays[0].data
        case _:
            raise ValueError(
                f"Unsupported surface format: {path}. Only .gii supported."
            )
    return shape


def _surface_to_polydata(surf: Surface) -> pv.PolyData:
    points, faces = surf
    if not points.ndim == faces.ndim == 2 and points.shape[1] == faces.shape[1] == 3:
        raise ValueError("Invalid surface points/faces. Expected two Nx3 arrays.")

    # prepend number of points and flatten, pyvista format
    # https://docs.pyvista.org/examples/00-load/create-poly#sphx-glr-examples-00-load-create-poly-py
    faces = np.concatenate(
        [np.full((len(faces), 1), 3, dtype=faces.dtype), faces], axis=1
    )
    poly = pv.PolyData(points, faces.flatten())
    return poly


def _constant_layer(n_points: int, color: Any) -> np.ndarray:
    rgba = to_rgba(color)
    layer = np.tile(np.asarray(rgba), (n_points, 1))
    return layer


def _alpha_composite(layers: list[np.ndarray]) -> np.ndarray:
    """Make alpha blend of a stack of RGBA layers.

    Args:
        layers: array of layers, each shape (size, 4)

    Returns:
        output: composite of layers, shape (size, 4)
    """
    assert len(layers) > 0
    assert all(
        layer.ndim == 1 or layer.ndim == 2 and layer.shape[1] in {3, 4}
        for layer in layers
    )

    sizes = set(len(layer) for layer in layers)
    assert len(sizes) == 1
    size = sizes.pop()

    # Make all layers appear like image array, shape (1, width).
    layers = [layer[None] for layer in layers]

    # Make composite.
    output = np.zeros((1, size, 4), dtype=layers[0].dtype)
    transform = IdentityTransform()
    for layer in layers:
        image.resample(layer, output, transform=transform, interpolation=image.NEAREST)

    output = output.squeeze(0)
    return output


def _crop_transparent_background(image: np.ndarray) -> np.ndarray:
    assert image.ndim == 3 and image.shape[-1] == 4
    bg_mask = image[..., 3] == 0
    row_ind, col_ind = np.where(~bg_mask)
    y1, y2 = row_ind.min(), row_ind.max()
    x1, x2 = col_ind.min(), col_ind.max()
    cropped = image[y1 : y2 + 1][:, x1 : x2 + 1]
    return cropped


def montage(
    images: list[Image.Image | None] | list[list[Image.Image | None]],
    pad: int | None = None,
    color: str | tuple[int, ...] | None = None,
    ha: Literal["left", "center", "right"] = "center",
    va: Literal["top", "center", "bottom"] = "center",
    shareh: bool = False,
    sharew: bool = False,
) -> Image.Image:
    """Make a montage of images."""
    if not isinstance(images[0], list):
        images = [images]

    # Get background color of first image and set as background.
    first_img: Image.Image = next(
        img for row in images for img in row if img is not None
    )
    if color is None:
        color = tuple(np.asarray(first_img)[0, 0])
    mode = first_img.mode

    # Convert to centering argument for padding.
    hc = {"left": 0.0, "center": 0.5, "right": 1.0}[ha]
    vc = {"top": 0.0, "center": 0.5, "bottom": 1.0}[va]
    centering = (hc, vc)

    # Pad each row with None to make a ragged grid.
    ncol = max(len(row) for row in images)
    images = [row + (ncol - len(row)) * [None] for row in images]

    # Pad each image on all sides.
    if pad:
        images = [
            [
                ImageOps.expand(img, pad, fill=color) if img is not None else None
                for img in row
            ]
            for row in images
        ]

    # Get max widths of each column and max heights of each row.
    # Then, each image is resized/padded to the aligned size of its row/column.
    sizes = np.array(
        [[img.size if img is not None else (0, 0) for img in row] for row in images]
    )
    widths = np.max(sizes[:, :, 0], axis=0)
    heights = np.max(sizes[:, :, 1], axis=1)

    if sharew:
        widths = np.full_like(widths, widths.max())
    if shareh:
        heights = np.full_like(heights, heights.max())

    # Resize/pad each image to the appropriate size.
    pad_images = []
    for ii, row in enumerate(images):
        for jj, img in enumerate(row):
            size = widths[jj], heights[ii]
            if img is None:
                # Fill with blank background image.
                img = Image.new(mode, size, color=color)
            else:
                # Resize/pad to target size.
                img = ImageOps.pad(img, size, color=color, centering=centering)
            pad_images.append(img)

    # Finally, make the image grid. This is a simple function that just pastes the
    # image and doesn't handle padding or alignment at all.
    grid = image_grid(pad_images, ncol=ncol, color=color)
    return grid


def image_grid(
    images: list[Image.Image],
    ncol: int,
    color: str | tuple[int, ...] | None = None,
) -> Image.Image:
    """Paste images into a simple grid."""
    if color is None:
        color = tuple(np.asarray(images[0])[0, 0])

    widths, heights = zip(*(img.size for img in images))
    width = max(widths)
    height = max(heights)
    nrow = math.ceil(len(images) / ncol)

    left, upper, right, lower = 0, 0, 0, 0
    grid = Image.new(images[0].mode, size=(ncol * width, nrow * height), color=color)

    for ii, img in enumerate(images):
        grid.paste(img, (left, upper))
        right = max(right, left + img.width)
        lower = max(lower, upper + img.height)
        if (ii + 1) % ncol == 0:
            left, upper = 0, lower
        else:
            left, upper = left + img.width, upper

    grid = grid.crop((0, 0, right, lower))
    return grid
