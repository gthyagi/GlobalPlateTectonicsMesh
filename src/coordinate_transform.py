"""
This file contains classes needed for coordinate transform.

@author: Thyagarajulu Gollapalli
"""

import numpy as np
import math

class CoordinateTransformSphere(object):
	"""
	Transform coordinates in geographical longitude, latitude, depth to spherical xyz and vice versa. 

	Notes:
	lld - longitude, latitude, depth (km)
	llr - longitude, latitude, radius (km)
	xyz - x, y, z
	"""
	def __init__(self, radius: float = 6371.0):
		"""
		Initialize the CoordinateTransformSphere with a default radius.
		"""
		self.radius = radius

	def lld_to_xyz(self, lld: np.ndarray) -> np.ndarray:
		"""
		Converts geographical coordinates (longitude, latitude, depth) to spherical (x, y, z).
		Assumes input lld is an array of shape (N, 3) with:
			- longitude in degrees (expected range: [-180, 180] or [0, 360]),
			- latitude in degrees (expected range: [-90, 90]),
			- and depth in km.
		Returns:
			An array of shape (N, 3) with spherical coordinates (x, y, z).
		"""
		lon_deg = np.mod(lld[:, 0], 360) # Normalize longitude to [0, 360]
		lon_rad = np.deg2rad(lon_deg)
		lat_rad = np.deg2rad(lld[:, 1])
		r = (self.radius - lld[:, 2])/self.radius
		x = r * np.cos(lat_rad) * np.cos(lon_rad)
		y = r * np.cos(lat_rad) * np.sin(lon_rad)
		z = r * np.sin(lat_rad)
		return np.column_stack((x, y, z))

	def xyz_to_lld(self, xyz: np.ndarray) -> np.ndarray:
		"""
		Converts spherical (x, y, z) coordinates to geographical (longitude, latitude, depth).
		Assumes input xyz is an array of shape (N, 3).
		Returns:
			An array of shape (N, 3) where:
			- longitude is in degrees in the range [0, 360),
			- latitude is in degrees in the range [-90, 90],
			- and depth in km.
		"""
		x = xyz[:, 0]
		y = xyz[:, 1]
		z = xyz[:, 2]
		r = np.sqrt(x**2 + y**2 + z**2)
		lat = np.degrees(np.arcsin(z / r))
		lon = np.degrees(np.arctan2(y, x))
		lon = np.mod(lon, 360) # Ensure longitude is in [0, 360)
		return np.column_stack((lon, lat, (1 - r)*self.radius))


class CoordinateTransformCubedsphere(object):
	"""
	Transform coordinates in geographical longitude, latitude, depth to cubedsphere xyz and vice versa.

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
	
	
	def geo_lld_to_cubedsphere_xyz(self, geo_lonlatdep: np.ndarray) -> np.ndarray:
		"""
		Converts geographical coordinates (longitude, latitude, depth) to cubedsphere coordinates (x, y, z).

		Steps:
		1. Converts geographical longitude range to [0, 360] and latitude range to [0, 180].
		2. Converts transformed (longitude, latitude, depth) into cubedsphere domain (longitude, latitude, radius).
		3. Converts cubedsphere (longitude, latitude, radius) to cubedsphere (x, y, z).
		"""

		if geo_lonlatdep.shape[1] != 3:
			raise ValueError("Input data must be in the format (longitude, latitude, depth) in geographical coordinates.")

		# Step 1
		g_lld = self._convert_geo_to_transformed_lld(geo_lonlatdep)

		# Step 2
		c_llr = self._convert_transformed_lld_to_cubedsphere_llr(g_lld)

		# Step 3
		c_xyz = self._convert_cubedsphere_llr_to_xyz(c_llr)
		return c_xyz

	def _convert_geo_to_transformed_lld(self, geo_lonlatdep: np.ndarray) -> np.ndarray:
		"""
		Converts geographical coordinates (longitude, latitude, depth) to transformed coordinates.
		"""
		t_lld = geo_lonlatdep.copy()
		t_lld[:, 0] = np.mod(t_lld[:, 0], 360) # Normalize longitude to [0, 360]
		t_lld[:, 1] = 90 - t_lld[:, 1] # Transform latitude from [-90, 90] to [0, 180]
		return t_lld

	def _convert_transformed_lld_to_cubedsphere_llr(self, t_lld: np.ndarray) -> np.ndarray:
		"""
		Converts transformed coordinates (longitude, latitude, depth) into cubedsphere domain (longitude, latitude, radius).
		"""
		c_lon = t_lld[:, 0] - self.mid_t_lon
		c_lat = self.mid_t_lat - t_lld[:, 1]
		c_radius = (self.radius - t_lld[:, 2]) / self.radius
		return np.column_stack((c_lon, c_lat, c_radius))

	def _convert_cubedsphere_llr_to_xyz(self, c_llr: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere coordinates from (longitude, latitude, radius) to (x, y, z).

		Calculations:
		  - Compute the tangent of the longitude and latitude (in radians).
		  - Compute d = radius / sqrt(tan(lon)^2 + tan(lat)^2 + 1).
		  - Compute:
			  x = d * tan(lon)
			  y = d * tan(lat)
			  z = d
		"""
		# Compute tangent values for longitude and latitude (converted from degrees to radians)
		tan_lon = np.tan(np.deg2rad(c_llr[:, 0]))
		tan_lat = np.tan(np.deg2rad(c_llr[:, 1]))
		denom = np.sqrt(tan_lon**2 + tan_lat**2 + 1)
		d = c_llr[:, 2] / denom
		return np.column_stack((d * tan_lon, d * tan_lat, d))
	

	def cubedsphere_xyz_to_geo_lld(self, c_xyz: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere (x, y, z) coordinates to geographical (longitude, latitude, depth).
		Steps:
		  1. Convert cubedsphere (x, y, z) to cubedsphere (lon, lat, radius).
		  2. Convert cubedsphere (lon, lat, radius) to transformed (lon, lat, depth).
		  3. Convert transformed (lon, lat, depth) to geographical (lon, lat, depth).
		"""
		# Step 1
		c_llr = self._convert_cubedsphere_xyz_to_llr(c_xyz)
		# Step 2
		t_lld = self._convert_cubedsphere_llr_to_transformed_lld(c_llr)
		# Step 3
		g_lld = self._convert_transformed_lld_to_geo_lld(t_lld)
		return g_lld

	def _convert_cubedsphere_xyz_to_llr(self, c_xyz: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere (x, y, z) coordinates to cubedsphere (longitude, latitude, radius).
		"""
		if np.any(np.isclose(c_xyz[:, 2], 0)):
			raise ValueError("z coordinate is zero for one or more points; cannot perform conversion.")

		tan_lon = c_xyz[:, 0] / c_xyz[:, 2]
		tan_lat = c_xyz[:, 1] / c_xyz[:, 2]
		factor = np.sqrt(tan_lon**2 + tan_lat**2 + 1)

		lon = np.degrees(np.arctan(tan_lon))
		lat = np.degrees(np.arctan(tan_lat))
		radius = c_xyz[:, 2] * factor
		return np.column_stack((lon, lat, radius))

	def _convert_cubedsphere_llr_to_transformed_lld(self, c_llr: np.ndarray) -> np.ndarray:
		"""
		Convert cubedsphere coordinates (longitude, latitude, radius) to
		transformed coordinates (longitude, latitude, depth).
		"""
		t_lld = np.empty_like(c_llr)
		t_lld[:, 0] = c_llr[:, 0] + self.mid_t_lon
		t_lld[:, 1] = self.mid_t_lat - c_llr[:, 1]
		t_lld[:, 2] = (1.0 - c_llr[:, 2]) * self.radius
		return t_lld

	def _convert_transformed_lld_to_geo_lld(self, t_lld: np.ndarray) -> np.ndarray:
		"""
		Converts transformed (lon, lat, depth) to geographical (lon, lat, depth) coordinates.
		The transformation converts the latitude by subtracting from 90 degrees.
		"""
		g_lld = t_lld.copy()
		g_lld[:, 1] = 90 - t_lld[:, 1]
		return g_lld