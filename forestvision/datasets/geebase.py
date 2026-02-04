import os
import abc
import hashlib
import queue
import threading
import time
import sys
import traceback
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from typing import Any, Callable, Dict, Optional, Union, Tuple, List, TYPE_CHECKING
from enum import Enum, auto
import requests
from requests.exceptions import HTTPError, SSLError, ConnectionError
import math
from copy import copy
from retry import retry
from PIL import Image

from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import logging


import ee
import torch
import numpy
from rasterio import merge
from rasterio import MemoryFile
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox
import torchvision.transforms.functional as tvF

from .cloudgeo import CloudRasterDataset
from .utils import minmax_scaling


# Hack to suppress rasterio GDAL warnings
rasterio_logger = logging.getLogger("rasterio._env")
rasterio_logger.setLevel(logging.ERROR)

# GEE Queue Management
class RequestStatus(Enum):
    """Request lifecycle states."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

class GEEQueueRequest:
    """Thread-safe wrapper for GEE API requests.
    
    This class provides thread-safe execution of GEE API requests with
    proper error handling, status tracking, and timeout support.
    
    Attributes:
        func: The function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        callback: Optional callback function called on completion
        request_id: Unique identifier for this request
        result: Execution result (set after completion)
        error: Exception that occurred (set if execution failed)
        status: Current request status
        timestamp: Creation timestamp
        _lock: Thread lock for synchronizing access
        _complete_event: Event that signals completion
        _cancelled: Flag for cancellation status
        timeout: Optional timeout in seconds
        max_age: Optional maximum age in seconds
    """
    
    def __init__(
        self,
        func: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        callback: Optional[Callable[['GEEQueueRequest'], None]] = None,
        timeout: Optional[float] = None,
        max_age: Optional[float] = None
    ):
        """Initialize a new GEEQueueRequest.
        
        Args:
            func: Callable function to execute
            args: Positional arguments for func
            kwargs: Keyword arguments for func
            callback: Optional callback to call on completion
            timeout: Optional timeout in seconds
            max_age: Optional maximum age in seconds
            
        Raises:
            TypeError: If inputs are invalid
        """
        self._validate_inputs(func, args, kwargs)
        
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.callback = callback
        self.request_id = hashlib.md5(
            f"{time.time()}_{id(func)}".encode()
        ).hexdigest()[:8]
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.traceback: Optional[str] = None
        self.exc_info: Optional[Tuple] = None
        self.status = RequestStatus.PENDING
        self.timestamp = time.time()
        self.timeout = timeout
        self.max_age = max_age
        
        self._lock = threading.Lock()
        self._complete_event = threading.Event()
        self._cancelled = False
    
    def _validate_inputs(self, func, args, kwargs):
        """Validate input parameters.
        
        Raises:
            TypeError: If inputs are invalid
        """
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)}")
        if not isinstance(args, tuple):
            raise TypeError(f"args must be tuple, got {type(args)}")
        if not isinstance(kwargs, dict):
            raise TypeError(f"kwargs must be dict, got {type(kwargs)}")
    
    def execute(self) -> bool:
        """Execute the GEE request.
        
        This method is called by the worker thread. It executes the
        function with optional timeout protection.
        
        Returns:
            bool: True if successful, False if failed
            
        Raises:
            KeyboardInterrupt: If interrupted
            SystemExit: If system exit requested
            TimeoutError: If request times out
        """
        # Check for cancellation before starting
        if self._cancelled:
            return False
            
        # Check for expiration
        if self.max_age and (time.time() - self.timestamp) > self.max_age:
            self.error = TimeoutError(
                f"Request expired after {self.max_age}s"
            )
            return False
        
        with self._lock:
            self.status = RequestStatus.RUNNING
        
        try:
            logging.debug(
                f"Request {self.request_id} [{self.func.__name__}] started"
            )
            
            # Execute with optional timeout
            if self.timeout:
                self._execute_with_timeout()
            else:
                self.result = self.func(*self.args, **self.kwargs)
            
            with self._lock:
                self.status = RequestStatus.COMPLETED
                self.error = None
                
            logging.debug(
                f"Request {self.request_id} [{self.func.__name__}] completed"
            )
            
            # Call callback if provided
            if self.callback:
                try:
                    self.callback(self)
                except Exception as e:
                    logging.error(
                        f"Callback error for request {self.request_id}: {e}"
                    )
            
            return True
            
        except KeyboardInterrupt:
            logging.warning(
                f"Request {self.request_id} [{self.func.__name__}] interrupted"
            )
            raise
        except SystemExit:
            logging.warning(
                f"Request {self.request_id} [{self.func.__name__}] system exit"
            )
            raise
        except Exception as e:
            with self._lock:
                self.status = RequestStatus.FAILED
                self.error = e
                self.traceback = traceback.format_exc()
                self.exc_info = sys.exc_info()
            
            logging.error(
                f"Request {self.request_id} [{self.func.__name__}] failed: {e}\n"
                f"Traceback: {self.traceback}"
            )
            return False
        finally:
            self._complete_event.set()
    
    def _execute_with_timeout(self):
        """Execute function with timeout protection using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.func, *self.args, **self.kwargs)
            try:
                self.result = future.result(timeout=self.timeout)
            except FuturesTimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"Request {self.request_id} timed out after {self.timeout}s"
                )
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the request to complete.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            bool: True if completed, False if timeout
        """
        return self._complete_event.wait(timeout)
    
    def cancel(self) -> bool:
        """Attempt to cancel the request.
        
        Returns:
            bool: True if cancellation was attempted
        """
        if self.status == RequestStatus.PENDING:
            self._cancelled = True
            self.status = RequestStatus.FAILED
            self.error = TimeoutError("Request was cancelled")
            return True
        return False
    
    def get_result(self) -> Tuple[Any, Optional[Exception]]:
        """Thread-safe getter for result and error.
        
        Returns:
            Tuple[Any, Optional[Exception]]: (result, error)
        """
        with self._lock:
            return self.result, self.error
    
    @property
    def successful(self) -> bool:
        """Check if request completed successfully."""
        with self._lock:
            return self.status == RequestStatus.COMPLETED
    
    @property
    def completed(self) -> bool:
        """Check if request has completed (success or failure)."""
        with self._lock:
            return self.status in (RequestStatus.COMPLETED, RequestStatus.FAILED)
    
    @property
    def age(self) -> float:
        """Get the age of the request in seconds."""
        return time.time() - self.timestamp


class GEEQueueManager:
    """Thread-safe manager for GEE request queue.
    
    This manager handles GEE API requests with URL expiration awareness.
    GEE download URLs expire after approximately 5-10 minutes, so requests
    should not stay in the queue for too long.
    """
    
    def __init__(self, max_concurrent: int = 10, rate_limit_delay: float = 0.05, max_queue_age: float = 600.0):
        """Initialize a new GEEQueueManager.
        
        Args:
            max_concurrent: Maximum number of concurrent GEE requests
            rate_limit_delay: Delay between requests to avoid rate limiting
            max_queue_age: Maximum age in seconds for requests in queue (default 10 minutes)
        """
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._active = False
        self._thread = None
        self._max_concurrent = max_concurrent
        self._rate_limit_delay = rate_limit_delay
        self._max_queue_age = max_queue_age  # Maximum time requests can stay in queue
        self._stats = {
            'processed': 0,
            'failed': 0,
            'queued': 0,
            'concurrent': 0,
            'expired': 0,
            'queue_expired': 0,  # Requests that expired while in queue
            'avg_queue_time': 0.0,  # Average time requests spend in queue
        }
        self._worker_thread = None
        self._queue_times = []  # Track queue times for statistics
    
    def start(self):
        """Start the queue worker thread."""
        with self._lock:
            if self._active:
                return
                
            self._active = True
            self._worker_thread = threading.Thread(
                target=self._worker,
                daemon=True
            )
            self._worker_thread.start()
    
    def stop(self):
        """Stop the queue worker thread."""
        with self._lock:
            if not self._active:
                return
                
            self._active = False
            self._queue.put(None)  # Poison pill
            
            if self._worker_thread:
                self._worker_thread.join(timeout=5.0)
    
    def submit(
        self,
        func: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        callback: Optional[Callable] = None,
        timeout: Optional[float] = None,
        max_age: Optional[float] = None
    ) -> GEEQueueRequest:
        """Submit a request to the queue.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            callback: Optional callback
            timeout: Optional timeout
            max_age: Optional maximum age
            
        Returns:
            GEEQueueRequest: The request object
        """
        request = GEEQueueRequest(
            func, args, kwargs, callback, timeout, max_age
        )
        self._queue.put(request)
        
        with self._lock:
            self._stats['queued'] += 1
            
        return request
    
    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        with self._lock:
            return self._stats.copy()
    
    def _worker(self):
        """Worker thread function."""
        while self._active:
            try:
                request = self._queue.get(timeout=1.0)
                
                if request is None:  # Poison pill
                    break
                
                # Check for queue expiration (prevent URLs from expiring in queue)
                queue_age = request.age
                if queue_age > self._max_queue_age:
                    request.error = TimeoutError(
                        f"Request expired in queue after {queue_age:.1f}s (max: {self._max_queue_age}s)"
                    )
                    with self._lock:
                        self._stats['queue_expired'] += 1
                        self._stats['expired'] += 1
                    self._queue.task_done()
                    logging.warning(
                        f"Request {request.request_id} expired in queue after {queue_age:.1f}s. "
                        f"Consider increasing max_concurrent or reducing queue size."
                    )
                    continue
                
                # Check for request-specific expiration
                if request.max_age and request.age > request.max_age:
                    request.error = TimeoutError(
                        f"Request expired after {request.max_age}s"
                    )
                    with self._lock:
                        self._stats['expired'] += 1
                    self._queue.task_done()
                    continue
                
                # Track queue time for statistics
                queue_time = request.age
                with self._lock:
                    self._queue_times.append(queue_time)
                    # Keep only last 1000 samples for statistics
                    if len(self._queue_times) > 1000:
                        self._queue_times.pop(0)
                    self._stats['avg_queue_time'] = sum(self._queue_times) / len(self._queue_times)
                
                # Process request
                with self._lock:
                    self._stats['concurrent'] += 1
                
                success = request.execute()
                
                with self._lock:
                    self._stats['concurrent'] -= 1
                
                if success:
                    self._stats['processed'] += 1
                else:
                    self._stats['failed'] += 1
                
                self._queue.task_done()
                
                # Rate limiting
                time.sleep(self._rate_limit_delay)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Queue worker error: {e}")
                with self._lock:
                    self._stats['failed'] += 1


# Global queue manager instance with thread-safe initialization
_queue_manager = None
_queue_manager_lock = threading.Lock()

def _get_queue_manager() -> GEEQueueManager:
    """Get or create global queue manager (thread-safe).
    
    Returns:
        GEEQueueManager: The global queue manager instance.
    """
    global _queue_manager
    if _queue_manager is None:
        with _queue_manager_lock:
            if _queue_manager is None:  # Double-check locking
                # Increase concurrency for multi-worker DataLoaders
                # 10 concurrent requests is a safe default for most GEE accounts
                _queue_manager = GEEQueueManager(max_concurrent=10, rate_limit_delay=0.05)
                _queue_manager.start()  # Auto-start the worker thread
    return _queue_manager

def start_gee_queue(max_concurrent: int = 10, rate_limit_delay: float = 0.05):
    """Start the GEE request queue processor.
    
    Args:
        max_concurrent: Maximum number of concurrent GEE requests
        rate_limit_delay: Delay between requests to avoid rate limiting
    """
    manager = _get_queue_manager()
    # If manager was already started with different params, we might want to update them
    with manager._lock:
        manager._max_concurrent = max_concurrent
        manager._rate_limit_delay = rate_limit_delay
    manager.start()
    logging.info(f"GEE queue started with {max_concurrent} max concurrent requests")

def stop_gee_queue():
    """Stop the GEE request queue processor."""
    manager = _get_queue_manager()
    manager.stop()
    logging.info("GEE queue stopped")

def submit_gee_request(func, args=None, kwargs=None, callback=None):
    """Submit a GEE request to the queue.
    
    Args:
        func: Function to execute
        args: Arguments for the function
        kwargs: Keyword arguments for the function
        callback: Callback function to call on completion
        
    Returns:
        GEEQueueRequest: The request object that was queued
    """
    if args is None:
        args = ()  # Empty tuple, not list
    if kwargs is None:
        kwargs = {}
        
    manager = _get_queue_manager()
    request = manager.submit(func, args, kwargs, callback)
    return request

def get_gee_queue_stats():
    """Get current statistics about the GEE queue."""
    manager = _get_queue_manager()
    return manager.get_stats()

def reset_gee_queue_stats():
    """Reset the GEE queue statistics."""
    manager = _get_queue_manager()
    manager._stats = {
        'processed': 0,
        'failed': 0,
        'queued': 0,
        'concurrent': 0,
        'expired': 0
    }

# TODO: Add option to overwrite existing files

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
    
    # Class-level shared session for connection pooling
    _session: Optional[requests.Session] = None
    _session_lock = threading.Lock()

    def __init__(
        self,
        image: ee.Image,
        scale: int,
        bounds: Tuple[float, float, float, float] = None,
        epsg: Union[str, int] = None,
        bands: List[str] = None,
        dimensions: Tuple[int, int] = None,
        nodata: int = 0,
        timeout: Optional[float] = None,
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
            timeout (float, optional): Timeout in seconds for fetch operations.
                Defaults to 300 seconds.

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
        self.timeout = timeout or 300.0  # Default timeout: 300 seconds
    
    def __del__(self):
        """Cleanup resources when object is deleted."""
        self.close()
    
    def close(self):
        """Explicitly close the zip file and release resources."""
        if self._zip:
            try:
                self._zip.close()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._zip = None
    
    @classmethod
    def get_session(cls) -> requests.Session:
        """Get or create shared requests session for connection pooling.
        
        Returns:
            requests.Session: Shared session with connection pooling enabled.
        """
        if cls._session is None:
            with cls._session_lock:
                if cls._session is None:  # Double-check locking
                    session = requests.Session()
                    session.headers.update({
                        'User-Agent': 'forestvision-gEE-client/1.0',
                        'Accept': 'application/octet-stream',
                    })
                    # Configure connection pooling
                    adapter = requests.adapters.HTTPAdapter(
                        max_retries=3,
                        pool_connections=10,
                        pool_maxsize=20,
                        pool_block=False
                    )
                    session.mount('https://', adapter)
                    session.mount('http://', adapter)
                    cls._session = session
        return cls._session
    
    @classmethod
    def close_session(cls):
        """Close the shared session and release all connections."""
        if cls._session:
            with cls._session_lock:
                if cls._session:
                    cls._session.close()
                    cls._session = None

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
            # Set one based on user input
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
        return self._vis_params

    @vis_params.setter
    def vis_params(self, kwargs: Dict[str, Any]):
        """Set visualization parameters for image preview.

        Args:
            kwargs (Dict[str, Any]): Visualization parameters to update.
        """
        self._vis_params.update(**kwargs)


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

        Maximum request size is 32 MB, maximum grid dimension is 10000.

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

    def _fetch_internal(self, max_url_regenerations: int = 3) -> Tuple[numpy.ndarray, Dict[str, Any]]:
        """Internal fetch method that performs actual GEE request.
        
        This method generates a fresh download URL just before making the request
        to avoid URL expiration issues. GEE download URLs are temporary and
        expire after a short period (typically 5-10 minutes).
        
        Args:
            max_url_regenerations: Maximum number of times to regenerate URL on expiration
            
        Returns:
            Tuple[numpy.ndarray, Dict[str, Any]]: Image data and profile
            
        Raises:
            HTTPError: If request fails after all retries
            SSLError: If SSL connection fails
            ConnectionError: If connection fails
            TimeoutError: If request times out
        """
        # Use shared session for connection pooling
        session = self.get_session()
        
        url_regeneration_attempts = 0
        last_exception = None
        
        while url_regeneration_attempts <= max_url_regenerations:
            try:
                # Generate fresh URL just before request to avoid expiration
                url = self.image.getDownloadURL(self.params)
                url_generated_at = time.time()
                logging.debug(f"Generated GEE download URL (attempt {url_regeneration_attempts + 1}/{max_url_regenerations + 1}): {url[:100]}...")
                
                with session.get(url, stream=True, timeout=self.timeout) as response:
                    response.raise_for_status()
                    if response.status_code != 200:
                        # Check for URL expiration (503 Service Unavailable)
                        if response.status_code == 503:
                            url_age = time.time() - url_generated_at
                            logging.warning(
                                f"GEE URL expired (503) after {url_age:.1f}s. "
                                f"Regenerating URL (attempt {url_regeneration_attempts + 1}/{max_url_regenerations})"
                            )
                            url_regeneration_attempts += 1
                            if url_regeneration_attempts <= max_url_regenerations:
                                # Exponential backoff before retrying
                                backoff_time = min(2 ** url_regeneration_attempts, 30)  # Cap at 30 seconds
                                logging.debug(f"Backing off for {backoff_time}s before retry")
                                time.sleep(backoff_time)
                                continue  # Try again with new URL
                            else:
                                raise HTTPError(
                                    f"GEE URL expired after {max_url_regenerations} regeneration attempts. "
                                    f"Last URL age: {url_age:.1f}s"
                                )
                        raise HTTPError(
                            f"Request failed with status code: {response.status_code}"
                        )
                    
                    # Success! Process the response
                    self._zip = ZipFile(BytesIO(response.content))
                    imgfile = self._zip.infolist()[0]
                    with MemoryFile(self._zip.read(imgfile.filename)) as memfile:
                        with memfile.open() as src:
                            data = src.read()
                            profile = src.profile

                    logging.debug(f"Successfully fetched image data (shape: {data.shape})")
                    return data, profile
                    
            except (SSLError, ConnectionError) as e:
                # Log the specific SSL/connection error
                logging.error(f"SSL/Connection error in fetch (attempt {url_regeneration_attempts + 1}): {str(e)}")
                last_exception = e
                url_regeneration_attempts += 1
                if url_regeneration_attempts <= max_url_regenerations:
                    # Exponential backoff for network errors
                    backoff_time = min(2 ** url_regeneration_attempts, 60)  # Cap at 60 seconds
                    logging.debug(f"Network error, backing off for {backoff_time}s before retry")
                    time.sleep(backoff_time)
                    continue
                else:
                    raise
            except HTTPError as e:
                # Re-raise HTTP errors that aren't 503 (URL expiration)
                if "503" not in str(e):
                    raise
                last_exception = e
                url_regeneration_attempts += 1
                if url_regeneration_attempts <= max_url_regenerations:
                    backoff_time = min(2 ** url_regeneration_attempts, 30)
                    logging.debug(f"HTTP error, backing off for {backoff_time}s before retry")
                    time.sleep(backoff_time)
                    continue
                else:
                    raise
            except Exception as e:
                logging.error(f"Unexpected error in fetch (attempt {url_regeneration_attempts + 1}): {str(e)}")
                raise
        
        # If we exit the loop without returning, raise the last exception
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error: fetch loop exited without result or exception")

    def fetch(self, use_queue: bool = False) -> Tuple[numpy.ndarray, Dict[str, Any]]:
        """Fetch image data from Google Earth Engine.

        Args:
            use_queue: If True, use the queue system for async execution

        Returns:
            Tuple[numpy.ndarray, Dict[str, Any]]: Image data array and profile metadata.

        Raises:
            HTTPError: If the request fails with a non-200 status code.
            SSLError: If SSL connection fails.
            ConnectionError: If connection fails.
            TimeoutError: If request times out
        """
        # Check if queue is active and use_queue is True
        if use_queue:
            # Submit to queue system (queue handles its own retries/timeouts)
            def queue_fetch():
                return self._fetch_internal()
            
            request = submit_gee_request(queue_fetch)
            
            # Wait for completion with instance timeout
            if not request.wait(timeout=self.timeout):
                stats = get_gee_queue_stats()
                msg = (
                    f"Queue request timed out after {self.timeout} seconds. "
                    f"Queue Stats: {stats}. "
                    f"Request Age: {request.age:.1f}s. "
                    f"Status: {request.status.name}. "
                )
                if request.status == RequestStatus.PENDING:
                    msg += "Request is still pending in queue. Consider increasing max_concurrent."
                elif request.status == RequestStatus.RUNNING:
                    msg += "Request is currently running but taking too long."
                
                raise TimeoutError(msg)
            
            # Get result and error in thread-safe manner
            result, error = request.get_result()
            
            if error:
                raise error
                
            return result
        else:
            # Use direct requests with retry decorator for backward compatibility
            @retry((HTTPError, SSLError, ConnectionError), tries=10, delay=10, backoff=2)
            def fetch_with_retry():
                return self._fetch_internal()
            
            return fetch_with_retry()

    def save(self, dest_path: str, filename: str = None, overwrite: bool = False):
        """Unzip and save image to disk.

        Args:
            dest_path (str): Directory to save the downloaded image.
            filename (str, optional): Filename to use. If None, uses the image ID.
            overwrite (bool, optional): Overwrites the file if True. Defaults to False.

        Raises:
            RuntimeError: If fetch() hasn't been called first.
            FileExistsError: If file already exists and overwrite is False.
        """
        if self._zip is None:
            raise RuntimeError("Run fetch() method first.")

        _filename = self._zip.infolist()[0]
        if filename:
            _filename.filename = filename

        target = os.path.join(dest_path, _filename.filename)
        if os.path.exists(target) and not overwrite:
            raise FileExistsError(
                f"File {target} already exists. Set overwrite=True to replace."
            )

        self._zip.extract(_filename, path=dest_path)
        # logging.info(f"Image saved to {target}")

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

            # Safely check if file exists in files collection
            try:
                # Try direct membership test first
                if filepath in self.files and not self.overwrite:
                    load_from_file = True
            except (TypeError, AttributeError):
                # Fallback: convert to list if possible
                try:
                    files_list = list(self.files) if hasattr(self.files, '__iter__') else []
                    if filepath in files_list and not self.overwrite:
                        load_from_file = True
                except (TypeError, AttributeError):
                    # Last resort: check if file exists on disk
                    if os.path.exists(filepath) and not self.overwrite:
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
                data, _ = geeimage.fetch(use_queue=True)

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
