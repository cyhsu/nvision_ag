"""Main module."""

import warnings

warnings.filterwarnings("ignore")

import os
import pyproj
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rx
import geopandas as gpd
import rasterio as rio
from datetime import datetime
from shapely.geometry import box, Polygon

from typing import Dict, Union


def subset_geom(
    path: str, crs: Union[str, pyproj.crs.crs.CRS] = "epsg:32615", grid_resolution: int = 20
) -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame of rectangular polygons covering the area of a given GeoDataFrame,
    with each rectangle having dimensions of grid_resolution x grid_resolution.

    The generated GeoDataFrame will be returned in the same coordinate reference system as the input GeoDataFrame.

    Parameters
    ----------
    path : str
        The file path to the GeoDataFrame to be processed.

    crs : str or pyproj.crs.crs.CRS, optional
        The coordinate reference system to be used for the grid. 
        This could either be a string in the form of an EPSG code or a pyproj CRS object.
        Default is "epsg:32615", which is the UTM zone 15N.

    grid_resolution : int, optional
        The resolution of the grid to be generated in the unit of the provided CRS. 
        Default is 20, indicating a 20x20 grid.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame of polygons covering the area of the input GeoDataFrame.
    """

    # Load the two GeoDataFrames
    gpd_df = gpd.read_file(path)
    original_crs = gpd_df.crs

    # convert degrees to meter
    gpd_df = gpd_df.to_crs(crs)

    # Get the bounding box of the shapefile
    xmin, ymin, xmax, ymax = gpd_df.total_bounds

    # Calculate the number of rows and columns in the grid based on the resolution
    num_cols = int((xmax - xmin) / grid_resolution)
    num_rows = int((ymax - ymin) / grid_resolution)

    # Generate a grid of polygons within the bounding box
    polygons = []
    for row in range(num_rows + 2):
        for col in range(num_cols + 2):
            x1 = xmin + col * grid_resolution
            y1 = ymin + row * grid_resolution
            x2 = x1 + grid_resolution
            y2 = y1 + grid_resolution
            polygons.append(box(x1, y1, x2, y2))

    # Create a GeoDataFrame from the list of polygons
    return gpd.GeoDataFrame(geometry=polygons, crs=crs).to_crs(original_crs)


def clip_with_harvest(
    geom: gpd.GeoDataFrame, 
    yield_path: str, 
    MinAreaPercentage: float = 0.6, 
    TotalArea: float = 400.0
) -> gpd.GeoDataFrame:
    """
    Clips the yield data with the provided geometry (usually in the form of a grid).
    Eliminates any polygon where the harvest area is less than the minimum area percentage of the total area.

    Parameters
    ----------
    geom : gpd.GeoDataFrame
        A GeoDataFrame of polygons, typically grids or cells, to which the yield data should be clipped.

    yield_path : str
        The file path to the yield data to be processed.

    MinAreaPercentage : float, optional
        The minimum ratio of harvest area to total area for a grid or cell to be included in the output.
        Default is 0.6, indicating that at least 60% of the cell must be harvested for it to be included.

    TotalArea : float, optional
        The total area of a single grid or cell. This is used to calculate the area threshold for clipping.
        Default is 400.0, which is typically used for a 20x20 grid assuming the area unit is square meters.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame of yield data, clipped to the provided geometry and with boxes below the minimum area ratio removed.
    """
    # - Step 01: Load Yield field
    yield_df = gpd.read_file(yield_path)

    # - Step 02: generate a empty list to store the outputs
    valid_geom = []

    # - Step 03: Loop for each box geom
    # -    (a) eliminated if total area of yield data is less than 0.6 of total area
    # -    (b) spatial-weighted average
    for num, (idx, row) in enumerate(geom.iterrows()):
        # - make geometry dataframe by box
        row_gdf = gpd.GeoDataFrame(
            row.to_frame().T, geometry="geometry", crs=yield_df.crs
        )

        # - clipped yield in box
        clipped = gpd.clip(yield_df, row_gdf)
        clipped["clipped_area"] = clipped.to_crs("epsg:32615").area

        # - spatial-weighted average if overlapped area > 0.6 of total area
        if clipped["clipped_area"].sum() / TotalArea > MinAreaPercentage:
            valid_geom.append(spatial_normalized_average(row_gdf, clipped))

    if len(valid_geom) == 0:
        raise ValueError(
            {
                "status": "failed",
                "reason": "no overlap, please check the inputs are correct.",
            }
        )

    valid_geom_df = pd.concat(valid_geom)
    return gpd.GeoDataFrame(
        valid_geom_df.reset_index().drop(columns=["index"], axis=1),
        geometry="geometry",
        crs=yield_df.crs,
    )


def spatial_normalized_average(row, clipped):
    """
    Argument
    --------
       row: 20 m x 20 m box geopandas dataframe
       clipped: clipped yield dataframe

    Return
    ------
       normally-spaced yield data

    """
    selected_clipped = clipped.select_dtypes(include=[int, float]).iloc[:, :-1]
    selected_clipped_average = (
        selected_clipped.multiply(clipped["clipped_area"], axis="index").sum()
        / clipped["clipped_area"].sum()
    )
    return selected_clipped_average.to_frame().T.assign(
        geometry=row["geometry"].iloc[0]
    )


def ndvi(nir: Union[float, np.ndarray], red: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).

    NDVI is a commonly used remote sensing index to measure and monitor plant growth, vegetation cover,
    and biomass production from multispectral remote sensing data.

    Parameters
    ----------
    nir : float or numpy.ndarray
        The near-infrared band data. This can be a single value or an array of values.

    red : float or numpy.ndarray
        The red band data. This can be a single value or an array of values.

    Returns
    -------
    float or numpy.ndarray
        The calculated NDVI value or values.
    """
    return (nir - red) / (nir + red)


def gndvi(nir: Union[float, np.ndarray], green: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the Green Normalized Difference Vegetation Index (GNDVI).

    GNDVI is a remote sensing index used to estimate the vegetation fraction covered by green leaf.

    Parameters
    ----------
    nir : float or numpy.ndarray
        The near-infrared band data. This can be a single value or an array of values.

    green : float or numpy.ndarray
        The green band data. This can be a single value or an array of values.

    Returns
    -------
    float or numpy.ndarray
        The calculated GNDVI value or values.
    """
    return (nir - green) / (nir + green)


def masked_bare_soil(
    red_band: Union[float, np.ndarray], 
    green_band: Union[float, np.ndarray], 
    images: str = "Aeroptic"
) -> Union[float, np.ndarray]:
    """
    Create a mask for bare soil in an image.

    This function uses the ratio of the green band to the red band and a given threshold to create a binary mask. 
    Pixels with a ratio above the threshold are considered bare soil and marked as 1, while the rest are marked as 0.

    Parameters
    ----------
    red_band : float or numpy.ndarray
        The red band data. This can be a single value or an array of values.

    green_band : float or numpy.ndarray
        The green band data. This can be a single value or an array of values.

    images : str, optional
        The source of the images. Different sources have different thresholds for bare soil.
        Options are 'Spot', 'Aeroptic', 'Pleiades'. Default is 'Aeroptic'.

    Returns
    -------
    float or numpy.ndarray
        The binary mask indicating bare soil. Pixels with a ratio above the threshold are marked as 1, the rest as 0.
    """
    masked = green_band / red_band
    threshold = (
        1.35
        if images == "Spot"
        else 1.2
        if images == "Aeroptic"
        else 1.15
        if images == "Pleiades"
        else None
    )
    masked = green_band / red_band
    return xr.where(masked >= threshold, 1, 0)


def image_processed(
    valid_geometry_df: gpd.GeoDataFrame, 
    image: str, 
    image_path: str, 
    bands: Dict[str, int]
) -> gpd.GeoDataFrame:
    """
    Process an image to derive relative values on RGB+NIR bands, NDVI, GNDVI, 
    and average values of them with the reference values set as the 10th percentile on RGB, 
    and 90th percentile on NIR, NDVI, and GNDVI.

    Parameters
    ----------
    valid_geometry_df : gpd.GeoDataFrame
        A GeoDataFrame of polygons, typically grids or cells, within which the image values should be processed.

    image : str
        The name of the image to be processed.

    image_path : str
        The file path to the directory containing the image to be processed.

    bands : Dict[str, int]
        A dictionary mapping band names ('R', 'G', 'B', 'N') to their respective positions in the image data array.

    Returns
    -------
    gpd.GeoDataFrame
        The original GeoDataFrame with additional columns for the processed image values.
    """
    # - Step 01: Load Image
    image_df = rx.open_rasterio(image_path)

    # - Step 02: Generate a Mask using bare soil
    masked = masked_bare_soil(
        image_df.sel(band=bands["R"]), image_df.sel(band=bands["G"]), images=image
    )

    # - Step 03: Generate NDVI and GNDVI
    NDVI = ndvi(
        image_df.sel(band=bands["N"]).where(image_df["masked" == 1]),
        image_df.sel(band=bands["R"]).where(image_df["masked" == 1]),
    )

    GNDVI = gndvi(
        image_df.sel(band=bands["N"]).where(image_df["masked" == 1]),
        image_df.sel(band=bands["G"]).where(image_df["masked" == 1]),
    )

    # - Step 04: 10th percential on RGB bands
    REDp10 = np.nanpercentile(
        xr.where(masked == 1, image_df.sel(band=bands["R"]), np.nan), 10
    )
    GRNp10 = np.nanpercentile(
        xr.where(masked == 1, image_df.sel(band=bands["G"]), np.nan), 10
    )
    BLUp10 = np.nanpercentile(
        xr.where(masked == 1, image_df.sel(band=bands["B"]), np.nan), 10
    )

    # - Step 05: 90th percential on NIR, NDVI, GNDVI
    NIRp90 = np.nanpercentile(
        xr.where(masked == 1, image_df.sel(band=bands["N"]), np.nan), 90
    )
    NDVIp90 = np.nanpercentile(xr.where(masked == 1, NDVI, np.nan), 90)
    GNDVIp90 = np.nanpercentile(xr.where(masked == 1, GNDVI, np.nan), 90)

    print(
        f"{image}  RGBN NDVI GNDVI percentile: ",
        REDp10,
        GRNp10,
        BLUp10,
        NIRp90,
        NDVIp90,
        GNDVIp90,
    )
    print(f"{image}  GNDVI: ", GNDVI.min().values, GNDVI.max().values)
    print(f"{image}  NDVI: ", NDVI.min().values, NDVI.max().values, "\n\n")

    # - Step 06: Clip and Generate the final outputs
    image_trimmed_df = []
    for idx, row in valid_geometry_df.iterrows():
        row_gdf = gpd.GeoDataFrame(
            row.to_frame().T, geometry="geometry", crs=valid_geometry_df.crs
        ).to_crs(image_df.rio.crs)

        clipped = image_df.rio.clip([row_gdf.geometry.item()])
        clipped_mask = masked.rio.clip([row_gdf.geometry.item()])
        clipped_NDVI = NDVI.rio.clip([row_gdf.geometry.item()])
        clipped_GNDVI = GNDVI.rio.clip([row_gdf.geometry.item()])

        NIR, RED = clipped.sel(band=bands["N"]), clipped.sel(band=bands["R"])
        GRN, BLU = clipped.sel(band=bands["G"]), clipped.sel(band=bands["B"])

        # - Relative value in box
        RlvRED = xr.where(clipped_mask == 1, RED / REDp10, np.nan).mean().values
        RlvGRN = xr.where(clipped_mask == 1, GRN / GRNp10, np.nan).mean().values
        RlvBLU = xr.where(clipped_mask == 1, BLU / BLUp10, np.nan).mean().values
        RlvNIR = xr.where(clipped_mask == 1, NIR / NIRp90, np.nan).mean().values
        RlvNDVI = (
            xr.where(clipped_mask == 1, clipped_NDVI / NDVIp90, np.nan).mean().values
        )
        RlvGNDVI = (
            xr.where(clipped_mask == 1, clipped_GNDVI / GNDVIp90, np.nan).mean().values
        )

        # - Average in box
        AveRED = xr.where(clipped_mask == 1, RED, np.nan).mean().values
        AveGRN = xr.where(clipped_mask == 1, GRN, np.nan).mean().values
        AveBLU = xr.where(clipped_mask == 1, BLU, np.nan).mean().values
        AveNIR = xr.where(clipped_mask == 1, NIR, np.nan).mean().values
        AveNDVI = xr.where(clipped_mask == 1, clipped_NDVI, np.nan).mean().values
        AveGNDVI = xr.where(clipped_mask == 1, clipped_GNDVI, np.nan).mean().values

        row_gdf[
            [
                f"{image} Average NIR",
                f"{image} Average RED",
                f"{image} Average Blue",
                f"{image} Average Green",
                f"{image} Average NDVI",
                f"{image} Average GNDVI",
            ]
        ] = [AveNIR, AveRED, AveBLU, AveGRN, AveNDVI, AveGNDVI]

        row_gdf[
            [
                f"{image} Relative NIR",
                f"{image} Relative RED",
                f"{image} Relative Blue",
                f"{image} Relative Green",
                f"{image} Relative NDVI",
                f"{image} Relative GNDVI",
            ]
        ] = [RlvNIR, RlvRED, RlvBLU, RlvGRN, RlvNDVI, RlvGNDVI]

        image_trimmed_df.append(row_gdf)

    image_trimmed_df = pd.concat(image_trimmed_df)
    return gpd.GeoDataFrame(image_trimmed_df, geometry="geometry", crs=image_df.rio.crs)
