"""
"""

import abc
import hashlib
import os
from pathlib import Path
from matplotlib import colors
from matplotlib.colors import ListedColormap
import requests
from io import BytesIO
from zipfile import ZipFile
from typing import Any, Callable, Dict, Optional, Union, Tuple, List

import ee
import torch
import numpy
import pystac
import geemap
import pandas as pd
import geopandas
from rasterio import merge
from rasterio import MemoryFile
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox
import torchvision.transforms.functional as tvF
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from forestsegnet2.datasets.cloudgeo import CloudRasterDataset


class GEEMSImage:
    """Wrapper class to fetch Google Earth Engine (GEE) images."""

    nodata: int = 0

    _id: str = None

    _params: Dict = {}

    _vis_params: Dict = {}

    _rgb_bands: List[str] = []

    def __init__(
        self,
        image: ee.Image,
        scale: int,
        bounds: Tuple[float, float, float, float] = None,
        epsg: Union[str, int] = 4326,
        bands: List[str] = None,
        dimensions: Tuple[int, int] = None,
    ):
        """Initialize a new GEEMSImage instance.

        Args:

            image : ee.Image
                GEE image to fetch.
            scale : int
                Scale refers to the pixel resolution to fetch GEE images.
            bounds : tuple
                Bounding box coordinates in the format (xMin, yMin, xMax, yMax). Default is None.
                If None, will use ee.Image.geometry() for the region.
            epsg : str | int
                EPSG CRS code to request the image from GEE. Default is None.
            bands : list
                List of bands to fetch from the image. Default is None. If None, will fetch all bands
                from ee.Image.bandNames.
            dimensions : tuple
                Dimensions of the image to fetch. Default is None. If None, will fetch the full image.
        """
        if not isinstance(image, ee.Image):
            raise ValueError("Input image must be an ee.Image object")

        self._all_bands = image.bandNames().getInfo()
        if not self._all_bands:
            raise ValueError("Image must have at least one band")

        self.image = image
        self.dimensions: Tuple[int, int] = dimensions
        self.crs = f"EPSG:{epsg}"
        self.scale = scale
        self.bounds = bounds
        self._bands: List[str] = bands

    @property
    def id(self):
        if not self._id:
            self._id = self.image.id().getInfo()
        return self._id

    @id.setter
    def id(self, value):
        assert value, "Image ID cannot be empty"
        self._id = value

    @property
    def bands(self):
        if not self._bands:
            self._bands = self._all_bands
        return self._bands

    @bands.setter
    def bands(self, value: List[str]):
        invalid_bands = set(value) - set(self._all_bands)
        if invalid_bands:
            raise ValueError(f"Invalid bands: {invalid_bands}")
        self._bands: List[str] = value

    @property
    def region(self):
        if self.bounds:
            return ee.Geometry.Rectangle(
                self.bounds, proj=self.crs, evenOdd=True, geodesic=False
            )
        else:
            return self.image.geometry()

    @property
    def stats(self):
        return pd.DataFrame(
            {
                "band": self.bands,
                "min": self._reduce_image(ee.Reducer.min()),
                "max": self._reduce_image(ee.Reducer.max()),
                "mean": self._reduce_image(ee.Reducer.mean()),
                "std": self._reduce_image(ee.Reducer.stdDev()),
            }
        ).sort_values("band")

    @property
    def params(self):
        if not self._params:
            _params = {
                "name": self.id,
                "crs": self.crs,
                "region": self.region,
                "filePerBand": False,
                "formatOptions": {"cloudOptimized": True, "noData": self.nodata},
            }
            # GEE will throw an error if both dimensions and scale are provided
            if self.dimensions:
                _params.update(dimensions=self.dimensions)
            else:
                _params.update(scale=self.scale)
            self._params = _params

        return self._params

    @params.setter
    def params(self, kwargs: Dict[str, Any]):
        if kwargs.get("dimensions") and self._params.get("scale"):
            raise ValueError("Cannot set both dimensions and scale.")
        elif kwargs.get("scale") and self._params.get("dimensions"):
            raise ValueError("Cannot set both dimensions and scale.")
        self._params.update(**kwargs)

    @property
    def vis_params(self):
        if not self._vis_params:
            self._vis_params = {
                "min": self.stats["min"].min().astype(float),
                "max": self.stats["max"].max().astype(float),
                "gamma": [0.95, 1.1, 1],
            }
        return self._vis_params

    @vis_params.setter
    def vis_params(self, kwargs: Dict[str, Any]):
        self._vis_params.update(**kwargs)

    @property
    def properties(self):
        return geemap.image_props(self.image).getInfo()

    def _reduce_image(self, reducer):
        return (
            self.image.select(self.bands)
            .reduceRegion(reducer, self.region, self.scale)
            .values()
            .getInfo()
        )

    def _get_url(
        self,
        preview: bool = False,
    ):
        """Get URL to fetch image.

        Parameters
        ----------
        params : dict or None (default None)
            Parameters to pass to the ee.Image.getDownloadURL. If None, will use the default parameters.
            Options include: name, scale, crs, crs_transform, region, format, dimensions, filePerBand and others.
            See https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl
        viz_params : dict or None (default None)
            Parameters to pass to ee.Image.visualize. Required if preview = True. For more information see
            https://developers.google.com/earth-engine/apidocs/ee-image-visualize
        """
        from copy import copy

        params = copy(self.params)
        if preview:
            params["format"] = "png"
            return self.image.visualize(**self.vis_params).getThumbURL(params)
        else:
            return self.image.getDownloadURL(params)

    def to_array(self):
        """Fetch image as a numpy array."""

        url = self._get_url()
        with requests.get(url, stream=True) as response:
            if response.status_code != 200:
                print(f"Request failed with status code: {response.status_code}")
                return

            try:
                zip = ZipFile(BytesIO(response.content))
                imgfile = zip.infolist()[0]
                with MemoryFile(zip.read(imgfile.filename)) as memfile:
                    with memfile.open() as dataset:
                        data = dataset.read()
                        profile = dataset.profile

            except Exception as e:  # downloaded zip is corrupt/failed
                print(
                    f"An error occurred while processing download: {response.content}"
                )
                raise e
            else:
                return data, profile

    def to_file(
        self,
        path: str,
        filename: str = None,
        preview: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Save image to disk.

        Args:
            path : str
                Directory to save the downloaded image.
            preview : bool
                If True, will download a preview of the image in PNG format. Default is False.
            filename : str
                Name of the file to save. If None, will use the image ID.
            overwrite : bool
                If True, will overwrite the file if it already exists. Default is False.
        """

        def write_replace(path: str, filename: str) -> bool:
            if os.path.exists(os.path.join(path, filename)) and not overwrite:
                print(f"File {filename} already exists. Set overwrite=True to replace.")
                return False
            return True

        if preview:
            url = self._get_url(preview=True)
            response = requests.get(url, stream=True)
            _filename = f"{self.id or 'image'}-preview.png"
            if filename:
                _filename = filename
            if write_replace(path, _filename):
                with open(os.path.join(path, _filename), "wb") as f:
                    f.write(response.content)

        else:
            url = self._get_url()
            response = requests.get(url, stream=True)
            try:
                zip = ZipFile(BytesIO(response.content))
                _filename = zip.infolist()[0]

                if filename:
                    _filename.filename = filename
                if write_replace(path, _filename.filename):
                    zip.extract(_filename, path=path)

            except Exception as e:  # downloaded zip is corrupt/failed
                print(f"Download failed: {response.content}")
                # print_message(msg, progressbar)
                raise e

    def show(self):
        """Display the image on a map."""

        assert self._rgb_bands, "RGB bands not set."

        Map = geemap.Map()
        Map.centerObject(self.region)
        region = ee.FeatureCollection(self.region)
        style = {"color": "ffff00ff", "fillColor": "00000000"}

        Map.addLayer(
            self.image.select(self._rgb_bands), self.vis_params, self.id or "Image"
        )
        Map.addLayer(region.style(**style), {}, "Bounds")

        return Map


class GEERasterDataset(CloudRasterDataset):
    """Abstract class to fetch multispectral images from Earth Engine."""

    nodata: int

    gee_asset_id: Union[Dict, str] = None

    instrument: str

    date_start: str = None

    date_end: str = None

    _cmap: dict = None

    filename_suffix: str = ""

    def __init__(
        self,
        tiles: Optional[Union[geopandas.GeoDataFrame, pystac.Collection]] = None,
        roi: Optional[BoundingBox] = None,
        path: Optional[str] = None,
        res: Union[int, None] = None,
        crs: Optional[CRS] = CRS.from_epsg(5070),
        transforms: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        download: bool = False,
        overwrite: bool = False,
        cache: bool = True,
    ):
        """
        Args:

            tiles : geopandas.GeoDataFrame
                GeoDataFrame containing the tile collection.
            roi : BoundingBox
                Region of interest to fetch data from.
            path : str
                Directory where Sentinel-2 data are stored or will be stored if download option
                is set to True. If path is provided and a matching file exists, the image will be
                loaded from that file unless overwrite = True. Default is None.
            res : int
                Pixel resolution of the image to fetch. Default is None.
            crs : Optional[CRS]
                Images will be fetched from Earth Engine using this Coordinate Reference System.
                Default is None.
            transform : Optional[Callable]
                A function/transform that takes in a sample and returns a transformed version
            download : bool
                If True, download the dataset to the path directory. Default is False.
            overwrite : bool
                If True, overwrite the dataset if it already exists. Default is False.
            cache : bool
                If True, cache the dataset in memory. Default is True.
        """
        super().__init__(
            path=path,
            tiles=tiles,
            roi=roi,
            transforms=transforms,
            crs=crs,
            download=download,
            cache=cache,
        )
        self.overwrite = overwrite

    def _get_cmap(self):

        if not self._cmap:
            raise ValueError("Property `self._cmap` not set")

        k, v = list(self._cmap.keys()), list(self._cmap.values())
        cmap = ListedColormap(v)
        norm = colors.BoundaryNorm(k, cmap.N)

        return cmap, norm

    @property
    @abc.abstractmethod
    def _collection(self) -> ee.ImageCollection:
        """Initialize GEE image collection and apply filters."""

    @abc.abstractmethod
    def _preprocess(self, image: ee.Image) -> ee.Image:
        """Preprocess image before fetching."""

    @abc.abstractmethod
    def _reducer(self, collection: ee.ImageCollection) -> ee.Image:
        """Reduce image collection to a single image."""

    def _get_pixels(self, query: BoundingBox) -> Tuple[numpy.array, dict]:
        """Fetch data from Earth Engine or local file."""

        minx, maxx, miny, maxy, _, _ = query
        dimensions = ((maxx - minx) / self.res, (maxy - miny) / self.res)
        tile_id = hashlib.md5(f"({minx}, {miny}, {maxx}, {maxy})".encode()).hexdigest()

        hits = [
            hit.object for hit in self.index.intersection(tuple(query), objects=True)
        ]
        # Check if the query matches a unique tile
        if len([hit for hit in hits if hit == tile_id]) != 1 and not self.roi:
            raise ValueError(
                f"Number of hits in {self.__class__.__name__} index shoud be exactly 1"
            )
        # Check if the query intersects the ROI
        if len(hits) != 1 and self.roi:
            raise ValueError(f"{query} outside of ROI {self.roi}")

        load_from_file = False
        if self.paths:
            filepath = os.path.join(
                self.paths,
                f"{tile_id}_{self.__class__.__name__}{self.filename_suffix}.tif",
            )

            if filepath in eval(self.files.__repr__()) and not self.overwrite:
                load_from_file = True

        if load_from_file:
            src = self._load_warp_file(filepath)
            data, _ = merge.merge([src], (minx, miny, maxx, maxy), self.res)
            # data = src.read()
            transform = src.profile["transform"]
            del src
        else:
            data = GEEMSImage(
                image=self._reducer(self._collection),
                scale=self.res,
                bounds=[minx, miny, maxx, maxy],
                epsg=self.crs.to_epsg(),
                bands=self.bands,
                dimensions=dimensions,
            )
            if self._download:
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                data.to_file(self.paths, Path(filepath).name, overwrite=self.overwrite)
                src = self._load_warp_file(filepath)
                data = src.read()
                transform = src.profile["transform"]
                del src
            else:
                data, profile = data.to_array()
                transform = profile["transform"]

        return data, transform

    def _minmax_scaling(self, data: torch.Tensor) -> torch.Tensor:
        """
        Min-Max scaling of multi-dimensional tensor of shape CxHxW.

        Assumes bands are stacked along the first dimension.
        """
        min_val = [data[i].min() for i in range(data.shape[0])]
        max_val = [data[i].max() for i in range(data.shape[0])]

        return torch.stack(
            [
                (data[i] - min_val[i]) / (max_val[i] - min_val[i])
                for i in range(data.shape[0])
            ]
        )

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
        contrast: float = 1,
        brightness: float = 1,
        denormalizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        cmap = None
        norm = None
        if self._cmap:
            cmap, norm = self._get_cmap()

        k = "image" if self.is_image else "mask"
        image = sample[k].squeeze()
        if self.rgb_bands and self.bands:
            if denormalizer:
                image = denormalizer(image)

            image = self._minmax_scaling(image)
            rgb_bands_idx = [self.bands.index(b) for b in self.rgb_bands]
            image = image[rgb_bands_idx]
            image = tvF.to_pil_image(image)
            image = tvF.adjust_contrast(image, contrast)
            image = tvF.adjust_brightness(image, brightness)

        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze()
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(image, cmap=cmap, norm=norm)
            axs[0].axis("off")
            axs[1].imshow(pred, cmap=cmap, norm=norm)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title(self.instrument)
                axs[1].set_title("Prediction")
        else:
            axs.imshow(image, cmap=cmap, norm=norm)
            axs.axis("off")
            if show_titles:
                axs.set_title(self.instrument)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
