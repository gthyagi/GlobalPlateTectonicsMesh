"""
This file contains classes needed for coordinate transform.

@author: Thyagarajulu Gollapalli
"""

import numpy as np
import math

class CoordinateTransformSphere(object):
	"""
	Transform coordinates in geographical longitude, latitude, radius to spherical xyz and vice versa. 

	Notes:
	llr - longitude, latitude, radius (km)
	xyz - x, y, z
	"""
	def __init__(self, radius: float = 6371.0):
		"""
		Initialize the CoordinateTransformSphere with a default radius.
		"""
		self.radius = radius

	def llr_to_xyz(self, llr: np.ndarray) -> np.ndarray:
		"""
		Converts geographical coordinates (longitude, latitude, radius) to spherical (x, y, z).
		Assumes input llr is an array of shape (N, 3) with:
			- longitude in degrees (expected range: [-180, 180] or [0, 360]),
			- latitude in degrees (expected range: [-90, 90]),
			- and radius in consistent units.
		Returns:
			An array of shape (N, 3) with spherical coordinates (x, y, z).
		"""
		llr = np.asarray(llr)
		lon_deg = np.mod(llr[:, 0], 360) # Normalize longitude to [0, 360]
		lon_rad = np.deg2rad(lon_deg)
		lat_rad = np.deg2rad(llr[:, 1])
		r = llr[:, 2]/self.radius
		x = r * np.cos(lat_rad) * np.cos(lon_rad)
		y = r * np.cos(lat_rad) * np.sin(lon_rad)
		z = r * np.sin(lat_rad)
		return np.column_stack((x, y, z))

	def xyz_to_llr(self, xyz: np.ndarray) -> np.ndarray:
		"""
		Converts spherical (x, y, z) coordinates to geographical (longitude, latitude, radius).
		Assumes input xyz is an array of shape (N, 3).
		Returns:
			An array of shape (N, 3) where:
			- longitude is in degrees in the range [0, 360),
			- latitude is in degrees in the range [-90, 90],
			- and radius is computed based on the input x, y, z.
		"""
		xyz = np.asarray(xyz)
		x = xyz[:, 0]
		y = xyz[:, 1]
		z = xyz[:, 2]
		r = np.sqrt(x**2 + y**2 + z**2)
		lat = np.degrees(np.arcsin(z / r))
		lon = np.degrees(np.arctan2(y, x))
		# Ensure longitude is in [0, 360)
		lon = np.mod(lon, 360)
		return np.column_stack((lon, lat, r*self.radius))


class CoordinateTransformCubedsphere(object):
	"""
	Transform coordinates in geographical longitude, latitude, radius to cubedsphere xyz and vice versa.

	Notes:
	g-geographical, t-transformed, c-cubedsphere
	llr - longitude, latitude, radius (km)
	lld - longitude, latitude, depth (km)
	xyz - x, y, z
	"""
	def __init__(self, g_lon_min: float = -45., g_lon_max: float = 45., g_lat_min: float = -45., g_lat_max: float = 45., radius: float = 6371.0):
		
		if abs(g_lon_max - g_lon_min) > 180 or abs(g_lat_max - g_lat_min) > 180:
			raise ValueError("Longitude and Latitude extent should be less than 180 degrees")
		
		self.radius = radius
		self.g_lon_min, self.g_lon_max = g_lon_min, g_lon_max
		self.g_lat_min, self.g_lat_max = g_lat_min, g_lat_max
		self.t_lon_min, self.t_lon_max = np.mod(g_lon_min, 360), np.mod(g_lon_max, 360)
		self.t_lat_min, self.t_lat_max = 90 - g_lat_max, 90 - g_lat_min
		self.mid_t_lon = (self.t_lon_max + self.t_lon_min) / 2
		self.mid_t_lat = (self.t_lat_max + self.t_lat_min) / 2

		print(f"Transformed longitude ranges from 0 to 360 degrees, and transformed latitude ranges from 0 to 180 degrees. \n"
			  f"The North Pole is at 0 degrees, and the South Pole is at 180 degrees. \n"
			  f"Model Extent. In longitude: {abs(g_lon_max - g_lon_min)}, latitude: {abs(g_lat_max - g_lat_min)}. \n"
			  f"Geographical longitudes: [{self.g_lon_min}, {self.g_lon_max}] -> Transformed longitudes: [{self.t_lon_min}, {self.t_lon_max}]. \n"
			  f"Geographical latitudes: [{self.g_lat_min}, {self.g_lat_max}] -> Transformed latitudes: [{self.t_lat_max}, {self.t_lat_min}]. \n"
			  f"Midpoint of transformed longitude and latitude: {self.mid_t_lon}, {self.mid_t_lat}")
	
	
	def geo_llr_to_cubedsphere_xyz(self, geo_lonlatrad: np.ndarray) -> np.ndarray:
		"""
		Converts geographical coordinates (longitude, latitude, radius) to cubedsphere coordinates (x, y, z).

		Steps:
		1. Converts geographical longitude range to [0, 360] and latitude range to [0, 180].
		2. Converts transformed (longitude, latitude, radius) into cubedsphere domain (longitude, latitude, depth).
		3. Converts cubedsphere (longitude, latitude, depth) to cubedsphere (x, y, z).
		"""

		if geo_lonlatrad.shape[1] != 3:
			raise ValueError("Input data must be in the format (longitude, latitude, radius) in geographical coordinates.")

		# Step 1
		g_llr = self._convert_geo_to_transformed_llr(geo_lonlatrad)

		# Step 2
		c_lld = self._convert_transformed_llr_to_cubedsphere_lld(g_llr)

		# Step 3
		c_xyz = self._convert_cubedsphere_lld_to_xyz(c_lld)
		return c_xyz

	def _convert_geo_to_transformed_llr(self, geo_lonlatrad: np.ndarray) -> np.ndarray:
		"""
		Converts geographical coordinates (longitude, latitude, radius) to transformed coordinates.
		"""
		t_llr = geo_lonlatrad.copy()
		t_llr[:, 0] = np.mod(t_llr[:, 0], 360) # Normalize longitude to [0, 360]
		t_llr[:, 1] = 90 - t_llr[:, 1] # Transform latitude from [-90, 90] to [0, 180]
		return t_llr

	def _convert_transformed_llr_to_cubedsphere_lld(self, t_llr: np.ndarray) -> np.ndarray:
		"""
		Converts transformed coordinates (longitude, latitude, radius) into cubedsphere domain (longitude, latitude, depth).
		"""
		c_lon = t_llr[:, 0] - self.mid_t_lon
		c_lat = self.mid_t_lat - t_llr[:, 1]
		c_depth = (self.radius - t_llr[:, 2]) / self.radius
		return np.column_stack((c_lon, c_lat, c_depth))

	def _convert_cubedsphere_lld_to_xyz(self, c_lld: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere coordinates from (longitude, latitude, depth) to (x, y, z).

		Calculations:
		  - Compute the tangent of the longitude and latitude (in radians).
		  - Compute d = depth / sqrt(tan(lon)^2 + tan(lat)^2 + 1).
		  - Compute:
			  x = d * tan(lon)
			  y = d * tan(lat)
			  z = d
		"""
		# Compute tangent values for longitude and latitude (converted from degrees to radians)
		tan_lon = np.tan(np.deg2rad(c_lld[:, 0]))
		tan_lat = np.tan(np.deg2rad(c_lld[:, 1]))
		denom = np.sqrt(tan_lon**2 + tan_lat**2 + 1)
		d = c_lld[:, 2] / denom
		return np.column_stack((d * tan_lon, d * tan_lat, d))
	

	def cubedsphere_xyz_to_geo_llr(self, c_xyz: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere (x, y, z) coordinates to geographical (longitude, latitude, radius).
		Steps:
		  1. Convert cubedsphere (x, y, z) to cubedsphere (lon, lat, depth).
		  2. Convert cubedsphere (lon, lat, depth) to transformed (lon, lat, rad).
		  3. Convert transformed (lon, lat, rad) to geographical (lon, lat, rad).
		"""
		# Step 1
		c_lld = self._convert_cubedsphere_xyz_to_lld(c_xyz)
		# Step 2
		t_llr = self._convert_cubedsphere_lld_to_transformed_llr(c_lld)
		# Step 3
		g_llr = self._convert_transformed_llr_to_geo_llr(t_llr)
		return g_llr

	def _convert_cubedsphere_xyz_to_lld(self, c_xyz: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere (x, y, z) coordinates to cubedsphere (longitude, latitude, depth).
		"""
		if np.any(np.isclose(c_xyz[:, 2], 0)):
			raise ValueError("z coordinate is zero for one or more points; cannot perform conversion.")

		tan_lon = c_xyz[:, 0] / c_xyz[:, 2]
		tan_lat = c_xyz[:, 1] / c_xyz[:, 2]
		factor = np.sqrt(tan_lon**2 + tan_lat**2 + 1)

		lon = np.degrees(np.arctan(tan_lon))
		lat = np.degrees(np.arctan(tan_lat))
		depth = c_xyz[:, 2] * factor

		return np.column_stack((lon, lat, depth))

	def _convert_cubedsphere_lld_to_transformed_llr(self, c_lld: np.ndarray) -> np.ndarray:
		"""
		Convert cubedsphere coordinates (longitude, latitude, depth) to
		transformed coordinates (longitude, latitude, radius).
		"""
		t_llr = np.empty_like(c_lld)
		t_llr[:, 0] = c_lld[:, 0] + self.mid_t_lon
		t_llr[:, 1] = self.mid_t_lat - c_lld[:, 1]
		t_llr[:, 2] = (1.0 - c_lld[:, 2]) * self.radius
		return t_llr

	def _convert_transformed_llr_to_geo_llr(self, t_llr: np.ndarray) -> np.ndarray:
		"""
		Converts transformed (lon, lat, rad) to geographical (lon, lat, rad) coordinates.
		The transformation converts the latitude by subtracting from 90 degrees.
		"""
		g_llr = t_llr.copy()
		g_llr[:, 1] = 90 - t_llr[:, 1]
		return g_llr