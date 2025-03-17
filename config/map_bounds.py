"""
Configuration file for map bounds and grid resolution.
These settings are used for data loading and visualization.
"""

# Longitude and latitude bounds for the Antarctic Peninsula region
LON_MIN = -65  # Western boundary
LON_MAX = -53  # Eastern boundary
LAT_MIN = -66  # Southern boundary
LAT_MAX = -58  # Northern boundary

# Grid resolution in degrees
GRID_STEP = 0.5  # Grid step size in degrees

# Map visualization extent (slightly larger than the data bounds)
MAP_LON_MIN = LON_MIN - 2
MAP_LON_MAX = LON_MAX + 2
MAP_LAT_MIN = LAT_MIN - 2
MAP_LAT_MAX = LAT_MAX + 2
