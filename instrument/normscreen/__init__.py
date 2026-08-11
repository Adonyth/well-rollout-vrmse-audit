"""normscreen -- conditioning screen for variance-normalized error metrics."""
from .screen import (
    NORMALIZERS,
    split_components,
    spatial_rms2,CONDITION_BANDS, FieldReport, ScreenReport, band_for,
                     eps_sweep, floor_share, screen_array, screen_fields,
                     spatial_variance)

__version__ = "0.1.0"
__all__ = [
    "NORMALIZERS", "split_components", "spatial_rms2","CONDITION_BANDS", "FieldReport", "ScreenReport", "band_for", "eps_sweep",
           "floor_share", "screen_array", "screen_fields", "spatial_variance"]
