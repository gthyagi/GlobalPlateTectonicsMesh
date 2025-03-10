import coordinate_transform as ct
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# define the region
lon_min, lon_max = -50, -20
lat_min, lat_max = -20, 20

# +
# get lon lat array
lon_arr = np.linspace(lon_min, lon_max, num=31, )
lat_arr = np.linspace(lat_min, lat_max, num=41, )

# Generate 2D grids for lon and lat
lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

# Optionally, if you need a combined 2D array where each row is a (lon, lat) pair:
lonlat_arr = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

# Create a radius column with a constant value of 6300 for each coordinate
radius = 6300
radius_arr = np.full(lonlat_arr.shape[0], radius)

# Combine lon, lat, and radius into one array
lonlatrad_arr = np.column_stack((lonlat_arr, radius_arr))

lonlatrad_arr

# +
# Set up a map with Cartopy using PlateCarree projection
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()  # use a global extent for the map
ax.coastlines()

# Plot the grid points on the map
# Using transform=ccrs.PlateCarree() ensures the points are interpreted as lon/lat.
sc = ax.scatter(lon_grid, lat_grid, color='red', s=20, transform=ccrs.PlateCarree())
# -

coord_trans = ct.CoordinateTransformCubedsphere(g_lon_min=lon_min, 
                                                g_lon_max=lon_max,
                                                g_lat_min=lat_min, 
                                                g_lat_max=lat_max,)

c_xyz = coord_trans.geo_llr_to_cubedsphere_xyz(lonlatrad_arr)
c_xyz

g_llr = coord_trans.cubedsphere_xyz_to_geo_llr(c_xyz)
g_llr

lonlatrad_arr[:,0] = np.mod(lonlatrad_arr[:,0], 360)
lonlatrad_arr

np.isclose(g_llr, lonlatrad_arr)

coord_trans_sph = ct.CoordinateTransformSphere()

sph_xyz = coord_trans_sph.llr_to_xyz(lonlatrad_arr)
sph_xyz

llr = coord_trans_sph.xyz_to_llr(sph_xyz)
np.round(llr, 1)

np.isclose(g_llr, np.round(llr, 1))


