import os
import abc
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from typing import Any, Callable, Dict, Optional, Union, Tuple, List
import requests
from requests.exceptions import HTTPError
import math
from copy import copy
from retry import retry
from PIL import Image

from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import ee
import torch
import numpy
import pandas as pd
from rasterio import merge
from rasterio import MemoryFile
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox
import torchvision.transforms.functional as tvF

from .cloudgeo import CloudRasterDataset
from .utils import minmax_scaling


class GEEMSImage:
    """Wrapper class to fetch Google Earth Engine (GEE) images.

    This class provides a convenient interface to fetch, process, and save
    images from Google Earth Engine with various configuration options.

    Attributes:
        _id (str): Internal image identifier.
        _params (Dict): Parameters for GEE image fetching.
        _vis_params (Dict): Visualization parameters.
        _rgb_bands (List[str]): RGB band names for visualization.
        _zip: Zip file object for downloaded data.
    """

    _id: str = None

    _params: Dict = {}

    _vis_params: Dict = {}

    _rgb_bands: List[str] = []

    _zip = None

    def __init__(
        self,
        image: ee.Image,
        scale: int,
        bounds: Tuple[float, float, float, float] = None,
        epsg: Union[str, int] = 4326,
        bands: List[str] = None,
        dimensions: Tuple[int, int] = None,
        nodata: int = 0,
    ):
        """Initialize new GEEMSImage instance.

        Args:
            image (ee.Image): GEE image to fetch.
            scale (int): Pixel resolution to fetch GEE images (meters per pixel).
            bounds (Tuple[float, float, float, float], optional): Bounding box coordinates
                in the format (xMin, yMin, xMax, yMax). If None, uses ee.Image.geometry().
            epsg (Union[str, int], optional): EPSG CRS code to request the image from GEE.
                Defaults to 4326.
            bands (List[str], optional): List of bands to fetch from the image. If None,
                fetches all bands from ee.Image.bandNames.
            dimensions (Tuple[int, int], optional): Dimensions of the image to fetch
                (width, height). If None, fetches the full image.
            nodata (int, optional): NoData value for the image. Defaults to 0.

        Raises:
            ValueError: If input image is not an ee.Image object or has no bands.
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
        self.nodata = nodata

    @property
    def id(self) -> str:
        """Get the image ID.

        Returns:
            str: The Earth Engine image ID.
        """
        if not self._id:
            self._id = self.image.id().getInfo()
        return self._id

    @id.setter
    def id(self, value: str):
        """Set the image ID.

        Args:
            value (str): The image ID to set.

        Raises:
            AssertionError: If value is empty.
        """
        assert value, "Image ID cannot be empty"
        self._id = value

    @property
    def bands(self) -> List[str]:
        """Get the bands to fetch.

        Returns:
            List[str]: List of band names to fetch from the image.
        """
        if not self._bands:
            self._bands = self._all_bands
        return self._bands

    @bands.setter
    def bands(self, value: List[str]):
        """Set the bands to fetch.

        Args:
            value (List[str]): List of band names to fetch.

        Raises:
            ValueError: If any band in value is not available in the image.
        """
        invalid_bands = set(value) - set(self._all_bands)
        if invalid_bands:
            raise ValueError(f"Invalid bands: {invalid_bands}")
        self._bands: List[str] = value

    @property
    def region(self) -> ee.Geometry:
        """Get the region geometry for data fetching.

        Returns:
            ee.Geometry: The region geometry, either from bounds or image geometry.
        """
        if self.bounds:
            return ee.Geometry.Rectangle(
                self.bounds, proj=self.crs, evenOdd=True, geodesic=False
            )
        else:
            return self.image.geometry()

    @property
    def stats(self) -> pd.DataFrame:
        """Get statistical summary of the image bands.

        Returns:
            pd.DataFrame: DataFrame containing min, max, mean, and std for each band.
        """
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
    def params(self) -> Dict[str, Any]:
        """Get the parameters for GEE image fetching.

        Returns:
            Dict[str, Any]: Dictionary of parameters for ee.Image.getDownloadURL.
        """
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
        """Set additional parameters for GEE image fetching.

        Args:
            kwargs (Dict[str, Any]): Additional parameters to update.

        Raises:
            ValueError: If both dimensions and scale are provided.
        """
        if kwargs.get("dimensions") and self._params.get("scale"):
            raise ValueError("Cannot set both dimensions and scale.")
        elif kwargs.get("scale") and self._params.get("dimensions"):
            raise ValueError("Cannot set both dimensions and scale.")
        self._params.update(**kwargs)

    @property
    def vis_params(self) -> Dict[str, Any]:
        """Get visualization parameters for image preview.

        Returns:
            Dict[str, Any]: Dictionary of visualization parameters.
        """
        if not self._vis_params:
            self._vis_params = {
                "min": self.stats["min"].min().astype(float),
                "max": self.stats["max"].max().astype(float),
                "gamma": [0.95, 1.1, 1],
            }
        return self._vis_params

    @vis_params.setter
    def vis_params(self, kwargs: Dict[str, Any]):
        """Set visualization parameters for image preview.

        Args:
            kwargs (Dict[str, Any]): Visualization parameters to update.
        """
        self._vis_params.update(**kwargs)

    # @property
    # def properties(self):
    #     return geemap.image_props(self.image).getInfo()

    def _reduce_image(self, reducer) -> List[float]:
        """Reduce image region using the specified reducer.

        Args:
            reducer: Earth Engine reducer function.

        Returns:
            List[float]: Reduced values for each band.
        """
        return (
            self.image.select(self.bands)
            .reduceRegion(reducer, self.region, self.scale)
            .values()
            .getInfo()
        )

    def _get_url(
        self,
        params: Dict[str, Any] = None,
        preview: bool = False,
    ) -> str:
        """Get URL to download Earth Engine image.

        Args:
            params (Dict[str, Any], optional): Additional parameters to pass to
                ee.Image.getDownloadURL. If None, defaults to instance parameters.
                Possible arguments include: name, scale, crs, crs_transform, region,
                format, dimensions, filePerBand, etc.
            preview (bool, optional): If True, returns a preview PNG URL using
                ee.Image.visualize. Defaults to False.

        Returns:
            str: A string containing the download or preview URL.
        """
        params = copy(params)
        if preview:
            params["format"] = "png"
            return self.image.visualize(**self.vis_params).getThumbURL(params)
        else:
            return self.image.getDownloadURL(params)

    @retry((HTTPError), tries=5, delay=10)
    def fetch(self) -> Tuple[numpy.ndarray, Dict[str, Any]]:
        """Fetch image data from Google Earth Engine.

        Returns:
            Tuple[numpy.ndarray, Dict[str, Any]]: Image data array and profile metadata.

        Raises:
            HTTPError: If the request fails with a non-200 status code.
        """
        url = self.image.getDownloadURL(self.params)

        with requests.get(url, stream=True) as response:
            if response.status_code != 200:
                raise HTTPError(
                    f"Request failed with status code: {response.status_code}"
                )
            self._zip = ZipFile(BytesIO(response.content))
            imgfile = self._zip.infolist()[0]
            with MemoryFile(self._zip.read(imgfile.filename)) as memfile:
                with memfile.open() as src:
                    data = src.read()
                    profile = src.profile

            return data, profile

    def save(self, dest_path: str, filename: str = None, overwrite: bool = False):
        """Unzip and save image to disk.

        Args:
            dest_path (str): Directory to save the downloaded image.
            filename (str, optional): Filename to use. If None, uses the image ID.
            overwrite (bool, optional): Overwrites the file if True. Defaults to False.
        """
        if self._zip is None:
            print("Run fetch() method first.")
            return

        _filename = self._zip.infolist()[0]
        if filename:
            _filename.filename = filename

        target = os.path.join(dest_path, _filename.filename)
        if os.path.exists(target) and not overwrite:
            print(f"File {target} already exists. Set overwrite=True to replace.")
            return

        self._zip.extract(_filename, path=dest_path)

    def preview(self) -> Image.Image:
        """Fetch a preview of the image in PNG format.

        Returns:
            Image.Image: PIL Image object containing the preview.

        Raises:
            HTTPError: If the request fails with a non-200 status code.
        """
        params = copy(self.params)
        params["format"] = "png"
        url = self.image.visualize(**self.vis_params).getThumbURL(params)
        with requests.get(url, stream=True) as response:
            if response.status_code != 200:
                raise HTTPError(
                    f"Request failed with status code: {response.status_code}"
                )
            return Image.open(BytesIO(response.content))


class GEERasterDataset(CloudRasterDataset):
    """Abstract class to fetch imagery from Earth Engine.

    This class provides an abstract interface for fetching geospatial imagery
    from Google Earth Engine and integrating it with TorchGeo datasets.

    Attributes:
        nodata (int): NoData value for the dataset.
        gee_asset_id (Union[Dict, str]): Earth Engine asset ID or dictionary of asset IDs.
        instrument (str): Name of the sensor/instrument.
        date_start (str): Start date for data collection.
        date_end (str): End date for data collection.
        _cmap (dict): Color map for visualization.
        filename_suffix (str): Suffix to append to filenames.
        errors (list): List of errors encountered during data fetching.
    """

    nodata: int = None

    gee_asset_id: Union[Dict, str] = None

    instrument: str

    date_start: str = None

    date_end: str = None

    _cmap: dict = None

    filename_suffix: str = ""

    errors: list = []

    def __init__(
        self,
        roi: Optional[BoundingBox] = None,
        path: Optional[str] = None,
        res: Union[int, None] = None,
        crs: Optional[CRS] = CRS.from_epsg(5070),
        transforms: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        download: bool = False,
        bypass_errors: bool = True,
        overwrite: bool = False,
        cache: bool = True,
    ):
        """Initialize a new GEERasterDataset instance.

        Args:
            roi (BoundingBox, optional): Region of interest to fetch data from.
            path (str, optional): Directory where data are stored or will be stored
                if download option is set to True. If path is provided and a matching
                file exists, the image will be loaded from that file unless overwrite = True.
            res (int, optional): Pixel resolution of the image to fetch.
            crs (CRS, optional): Coordinate Reference System for fetching images.
                Defaults to EPSG:5070.
            transforms (Callable, optional): Function/transform that takes in a sample
                and returns a transformed version.
            download (bool, optional): If True, download the dataset to the path directory.
                Defaults to False.
            bypass_errors (bool, optional): If True, errors during data fetching will be
                logged but not raised. Defaults to True.
            overwrite (bool, optional): If True, overwrite the dataset if it already exists.
                Defaults to False.
            cache (bool, optional): If True, cache the dataset in memory. Defaults to True.
        """
        super().__init__(
            path=path,
            # tiles=tiles,
            roi=roi,
            res=res,
            transforms=transforms,
            crs=crs,
            download=download,
            cache=cache,
        )
        self.overwrite = overwrite
        self.bypass_errors = bypass_errors

    def _get_cmap(self) -> Tuple[ListedColormap, colors.BoundaryNorm]:
        """Get color map and normalization for visualization.

        Returns:
            Tuple[ListedColormap, colors.BoundaryNorm]: Color map and normalization object.

        Raises:
            ValueError: If color map is not set.
        """
        if not self._cmap:
            raise ValueError("Property `self._cmap` not set")

        k, v = list(self._cmap.keys()), list(self._cmap.values())
        cmap = ListedColormap(v)
        norm = colors.BoundaryNorm(k, cmap.N)

        return cmap, norm

    @property
    @abc.abstractmethod
    def collection(self) -> ee.ImageCollection:
        """Initialize GEE image collection and apply filters.

        Returns:
            ee.ImageCollection: Filtered Earth Engine image collection.

        Note:
            This is an abstract method that must be implemented by subclasses.
        """
        pass

    def _preprocess(self, image: ee.Image) -> ee.Image:
        """Preprocess image before fetching.

        Args:
            image (ee.Image): Earth Engine image to preprocess.

        Returns:
            ee.Image: Preprocessed Earth Engine image.

        Note:
            This method should be overridden by subclasses to implement
            specific preprocessing logic.
        """
        pass

    @abc.abstractmethod
    def _reducer(self, collection: ee.ImageCollection) -> ee.Image:
        """Reduce image collection to a single image.

        Args:
            collection (ee.ImageCollection): Earth Engine image collection to reduce.

        Returns:
            ee.Image: Reduced Earth Engine image.

        Note:
            This is an abstract method that must be implemented by subclasses.
        """
        pass

    def _get_pixels(self, query: BoundingBox) -> numpy.ndarray:
        """Fetch data from Earth Engine or local file if exists.

        Args:
            query (BoundingBox): Bounding box defining the region to fetch.

        Returns:
            numpy.ndarray: Image data array.

        Raises:
            ValueError: If number of hits in index is not exactly 1.
            IndexError: If query is outside of ROI.
        """
        minx, maxx, miny, maxy, _, _ = query
        dimensions = (
            math.ceil((maxx - minx) // self.res),
            math.ceil((maxy - miny) // self.res),
        )
        tile_id = hashlib.md5(f"({minx}, {miny}, {maxx}, {maxy})".encode()).hexdigest()

        hits = [
            hit.object for hit in self.index.intersection(tuple(query), objects=True)
        ]
        # Check if the query matches a unique tile
        if len([hit for hit in hits if hit == tile_id]) != 1 and not self.roi:
            raise ValueError(
                f"Number of hits in {self.__class__.__name__} index should be exactly 1"
            )
        # Check if the query intersects the ROI
        if len(hits) != 1 and self.roi:
            raise IndexError(f"query: {query} outside of ROI {self.roi}")

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
        else:
            try:
                geeimage = GEEMSImage(
                    image=self._reducer(self.collection),
                    scale=self.res,
                    bounds=[minx, miny, maxx, maxy],
                    epsg=self.crs.to_epsg(),
                    bands=self.bands,
                    dimensions=dimensions,
                    nodata=self.nodata,
                )
                data, _ = geeimage.fetch()

                if self._download:
                    geeimage.save(
                        self.paths, Path(filepath).name, overwrite=self.overwrite
                    )

            except ValueError as e:
                if self.bypass_errors:
                    self.errors.append((query, e))
                    return numpy.zeros((len(self.bands), *dimensions))
                else:
                    raise e

        return data

    def _minmax_scaling(self, data: torch.Tensor) -> torch.Tensor:
        """Apply min-max scaling to multi-dimensional tensor.

        Args:
            data (torch.Tensor): Input tensor with shape CxHxW.

        Returns:
            torch.Tensor: Scaled tensor with same shape.

        Raises:
            ValueError: If input tensor has more than 3 dimensions.

        Note:
            Assumes bands are stacked along the first dimension (channel dimension).
        """
        dim = (1, 2)
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        if len(data.shape) > 3:
            raise ValueError("Input tensor must have shape CxHxW")
        mask = data == self.nodata
        data[data == self.nodata] = float("inf")
        min_val = data.amin(dim=dim)
        data[data == float("inf")] = float("-inf")
        max_val = data.amax(dim=dim)
        data[mask] = self.nodata

        scaled = (data - min_val.reshape(-1, 1, 1)) / (max_val - min_val).reshape(
            -1, 1, 1
        )
        scaled[mask] = self.nodata
        return scaled

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
            sample (dict[str, Any]): Sample returned by RasterDataset.__getitem__
            show_titles (bool): Whether to show titles above each panel
            suptitle (str | None): Optional text to use as a suptitle
            contrast (float): Contrast adjustment
            brightness (float): Brightness adjustment
            denormalizer (Callable[[torch.Tensor], torch.Tensor] | None): Optional function to denormalize the image

        Returns:
            Figure: Matplotlib Figure with the rendered sample
        """
        cmap = None
        norm = None
        if self._cmap:
            cmap, norm = self._get_cmap()

        k = "image" if self.is_image else "mask"
        image = sample[k].squeeze()
        # mask = image == self.nodata
        if self.rgb_bands and self.bands:
            if denormalizer:
                image = denormalizer(image)

            image = minmax_scaling(image, self.nodata)
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
        title = (
            f"{self.instrument}\nRGB: {', '.join([b[-1] for b in self.rgb_bands])}"
            if k == "image"
            else self.instrument
        )

        if showing_predictions:
            axs[0].imshow(image, cmap=cmap, norm=norm)
            axs[0].axis("off")
            axs[1].imshow(pred, cmap=cmap, norm=norm)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title(title)
                axs[1].set_title("Prediction")
        else:
            axs.imshow(image, cmap=cmap, norm=norm)
            axs.axis("off")
            if show_titles:
                axs.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
