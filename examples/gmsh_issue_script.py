import gmsh
import numpy as np
import os
from enum import Enum
import math

# output dir
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

# +
# mesh details
r_o = 1.0
r_i = (6371. - 800.)/6371.
lon_ext = 52.0
lat_ext = 47.0
cellsize = 1/32

meshname = f'uw_sos_ro{r_o}_ri{np.round(r_i, 2)}_lon{lon_ext}_lat{lat_ext}_csize{np.round(cellsize, 3)}'


# -

def SegmentofSphere(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    longitudeExtent: float = 90.0,
    latitudeExtent: float = 90.0,
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
    centroid = (0.0, 0.0, 0.0),
):
    """
    Generate a 3D mesh of a spherical segment using Gmsh, defined by inner/outer radii and angular extents in longitude and latitude. 
    Parameters control mesh size, element degree, file output, refinement, and verbosity. The mesh is centered at a specified centroid 
    and saved to a file if a name is provided.
    """

    class boundaries(Enum):
        Lower = 11
        Upper = 12
        East = 13
        West = 14
        South = 15
        North = 16

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_segmentofsphere_ro{radiusOuter}_ri{radiusInner}_longext{longitudeExtent}_latext{latitudeExtent}_csize{cellSize}.msh"
    else:
        uw_filename = filename

    if (
        radiusInner <= 0
        or not (0 < longitudeExtent < 180)
        or not (0 < latitudeExtent < 180)
    ):
        raise ValueError(
            "Invalid input parameters: "
            "radiusInner must be greater than 0, "
            "and longitudeExtent and latitudeExtent must be within the range (0, 180)."
        )


    def getSphericalXYZ(point):
        """
        Perform Cubed-sphere projection on coordinates.
        Converts (radius, lon, lat) in spherical region to (x, y, z) in spherical region.

        Parameters
        ----------
        Input:
            Coordinates in rthetaphi format (radius, lon, lat)
        Output
            Coordinates in XYZ format (x, y, z)
        """

        (x, y) = (
            math.tan(point[1] * math.pi / 180.0),
            math.tan(point[2] * math.pi / 180.0),
        )
        d = point[0] / math.sqrt(x**2 + y**2 + 1)
        coordX, coordY, coordZ = (
            centroid[0] + d * x,
            centroid[1] + d * y,
            centroid[2] + d,
        )

        return (coordX, coordY, coordZ)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)
    
    gmsh.model.add("SegmentOfSphere")

    p0 = gmsh.model.geo.addPoint(
        centroid[0], centroid[1], centroid[2], meshSize=cellSize
    )

    # Create segment of sphere
    dim = 3

    long_half = longitudeExtent / 2
    lat_half = latitudeExtent / 2

    pt1 = getSphericalXYZ((radiusInner, -long_half, -lat_half))
    pt2 = getSphericalXYZ((radiusInner, long_half, -lat_half))
    pt3 = getSphericalXYZ((radiusInner, long_half, lat_half))
    pt4 = getSphericalXYZ((radiusInner, -long_half, lat_half))
    pt5 = getSphericalXYZ((radiusOuter, -long_half, -lat_half))
    pt6 = getSphericalXYZ((radiusOuter, long_half, -lat_half))
    pt7 = getSphericalXYZ((radiusOuter, long_half, lat_half))
    pt8 = getSphericalXYZ((radiusOuter, -long_half, lat_half))

    p1 = gmsh.model.geo.addPoint(pt1[0], pt1[1], pt1[2], meshSize=cellSize)
    p2 = gmsh.model.geo.addPoint(pt2[0], pt2[1], pt2[2], meshSize=cellSize)
    p3 = gmsh.model.geo.addPoint(pt3[0], pt3[1], pt3[2], meshSize=cellSize)
    p4 = gmsh.model.geo.addPoint(pt4[0], pt4[1], pt4[2], meshSize=cellSize)
    p5 = gmsh.model.geo.addPoint(pt5[0], pt5[1], pt5[2], meshSize=cellSize)
    p6 = gmsh.model.geo.addPoint(pt6[0], pt6[1], pt6[2], meshSize=cellSize)
    p7 = gmsh.model.geo.addPoint(pt7[0], pt7[1], pt7[2], meshSize=cellSize)
    p8 = gmsh.model.geo.addPoint(pt8[0], pt8[1], pt8[2], meshSize=cellSize)

    l1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
    l2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
    l3 = gmsh.model.geo.addCircleArc(p3, p0, p4)
    l4 = gmsh.model.geo.addCircleArc(p4, p0, p1)
    l5 = gmsh.model.geo.addCircleArc(p5, p0, p6)
    l6 = gmsh.model.geo.addCircleArc(p6, p0, p7)
    l7 = gmsh.model.geo.addCircleArc(p7, p0, p8)
    l8 = gmsh.model.geo.addCircleArc(p8, p0, p5)
    l9 = gmsh.model.geo.addLine(p5, p1)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p7, p3)
    l12 = gmsh.model.geo.addLine(p4, p8)

    cl = gmsh.model.geo.addCurveLoop((l1, l2, l3, l4))
    lower = gmsh.model.geo.addSurfaceFilling([cl], tag=boundaries.Lower.value)

    cl = gmsh.model.geo.addCurveLoop((l5, l6, l7, l8))
    upper = gmsh.model.geo.addSurfaceFilling([cl], tag=boundaries.Upper.value)

    cl = gmsh.model.geo.addCurveLoop((l10, l6, l11, -l2))
    east = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.East.value)

    cl = gmsh.model.geo.addCurveLoop((l9, -l4, l12, l8))
    west = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.West.value)

    cl = gmsh.model.geo.addCurveLoop((l1, l10, -l5, l9))
    south = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.South.value)

    cl = gmsh.model.geo.addCurveLoop((-l3, -l11, l7, -l12))
    north = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.North.value)

    sloop = gmsh.model.geo.addSurfaceLoop([south, east, north, upper, west, lower])
    volume = gmsh.model.geo.addVolume([sloop])

    gmsh.model.geo.synchronize()

    # Add Physical groups
    for b in boundaries:
        tag = b.value
        name = b.name
        gmsh.model.addPhysicalGroup(2, [tag], tag)
        gmsh.model.setPhysicalName(2, tag, name)

    # Add the volume entity to a physical group with a high tag number (99999) and name it "Elements"
    gmsh.model.addPhysicalGroup(3, [volume], 99999)
    gmsh.model.setPhysicalName(3, 99999, "Elements")

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.write(uw_filename)

    # visual check
    gmsh.fltk.run()

    gmsh.finalize()

    return

# create refined mesh
SegmentofSphere(radiusOuter=r_o, radiusInner=r_i,
                longitudeExtent=lon_ext, latitudeExtent=lat_ext,
                cellSize=cellsize, filename=f'{output_dir}{meshname}_refined.msh')


