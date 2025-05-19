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


# +
# using occ inbuilt cad

def SegmentofSphere_occ(
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

    p0 = gmsh.model.occ.addPoint(
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

    p1 = gmsh.model.occ.addPoint(pt1[0], pt1[1], pt1[2], meshSize=cellSize)
    p2 = gmsh.model.occ.addPoint(pt2[0], pt2[1], pt2[2], meshSize=cellSize)
    p3 = gmsh.model.occ.addPoint(pt3[0], pt3[1], pt3[2], meshSize=cellSize)
    p4 = gmsh.model.occ.addPoint(pt4[0], pt4[1], pt4[2], meshSize=cellSize)
    p5 = gmsh.model.occ.addPoint(pt5[0], pt5[1], pt5[2], meshSize=cellSize)
    p6 = gmsh.model.occ.addPoint(pt6[0], pt6[1], pt6[2], meshSize=cellSize)
    p7 = gmsh.model.occ.addPoint(pt7[0], pt7[1], pt7[2], meshSize=cellSize)
    p8 = gmsh.model.occ.addPoint(pt8[0], pt8[1], pt8[2], meshSize=cellSize)

    l1 = gmsh.model.occ.addCircleArc(p1, p0, p2)
    l2 = gmsh.model.occ.addCircleArc(p2, p0, p3)
    l3 = gmsh.model.occ.addCircleArc(p3, p0, p4)
    l4 = gmsh.model.occ.addCircleArc(p4, p0, p1)
    l5 = gmsh.model.occ.addCircleArc(p5, p0, p6)
    l6 = gmsh.model.occ.addCircleArc(p6, p0, p7)
    l7 = gmsh.model.occ.addCircleArc(p7, p0, p8)
    l8 = gmsh.model.occ.addCircleArc(p8, p0, p5)
    l9 = gmsh.model.occ.addLine(p5, p1)
    l10 = gmsh.model.occ.addLine(p2, p6)
    l11 = gmsh.model.occ.addLine(p7, p3)
    l12 = gmsh.model.occ.addLine(p4, p8)

    cl = gmsh.model.occ.addCurveLoop((l1, l2, l3, l4))
    lower = gmsh.model.occ.addSurfaceFilling(cl, tag=boundaries.Lower.value)

    cl = gmsh.model.occ.addCurveLoop((l5, l6, l7, l8))
    upper = gmsh.model.occ.addSurfaceFilling(cl, tag=boundaries.Upper.value)

    cl = gmsh.model.occ.addCurveLoop((l10, l6, l11, -l2))
    east = gmsh.model.occ.addPlaneSurface([cl], tag=boundaries.East.value)

    cl = gmsh.model.occ.addCurveLoop((l9, -l4, l12, l8))
    west = gmsh.model.occ.addPlaneSurface([cl], tag=boundaries.West.value)

    cl = gmsh.model.occ.addCurveLoop((l1, l10, -l5, l9))
    south = gmsh.model.occ.addPlaneSurface([cl], tag=boundaries.South.value)

    cl = gmsh.model.occ.addCurveLoop((-l3, -l11, l7, -l12))
    north = gmsh.model.occ.addPlaneSurface([cl], tag=boundaries.North.value)

    sloop = gmsh.model.occ.addSurfaceLoop([south, east, north, upper, west, lower])
    volume = gmsh.model.occ.addVolume([sloop])

    gmsh.model.occ.synchronize()

    # Add Physical groups
    for b in boundaries:
        tag = b.value
        name = b.name
        gmsh.model.addPhysicalGroup(2, [tag], tag)
        gmsh.model.setPhysicalName(2, tag, name)

    #---------------------------------------------------------------------------------------------
    # Read in the surface step 
    gmsh.merge("./output/sum_top_surf_mesh_mmg.step")

    

    #---------------------------------------------------------------------------------------------
    ''' 
    how to include this surface as entity of volume mesh
    Note: need help from gmsh community
    '''

    

    #---------------------------------------------------------------------------------------------
    # compute a distance field based on the surface
    # gmsh examples are available to do this


    #---------------------------------------------------------------------------------------------
    # generate the mesh using distance field
    

    # Add the volume entity to a physical group with a high tag number (99999) and name it "Elements"
    gmsh.model.addPhysicalGroup(3, [volume], 99999)
    gmsh.model.setPhysicalName(3, 99999, "Elements")

    gmsh.model.occ.synchronize()
   
    gmsh.model.mesh.generate(3)
    gmsh.write(filename)

    # visual check
    gmsh.fltk.run()

    gmsh.finalize()

    return
# -

# create refined mesh
SegmentofSphere_occ(radiusOuter=r_o, radiusInner=r_i,
                    longitudeExtent=lon_ext, latitudeExtent=lat_ext,
                    cellSize=cellsize, filename=f'{output_dir}{meshname}_refined_occ.msh')

# +
'''
for simplicity I have included the surface in the box.
where each face of the slab surface is insert as surface in gmsh
'''

def mesh_box_with_surface(
    stepFile: str,
    meshSize: float = 0.1,
    boxMargin: float = 0.1,
    outMsh: str = "box_with_surface.msh"
):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("BoxWithSurface")

    # 1) Import the surface (dim=2 entities) from STEP
    ents = gmsh.model.occ.importShapes(stepFile)
    gmsh.model.occ.synchronize()

    # collect only the surface tags
    surfTags = [tag for (dim, tag) in ents if dim == 2]
    if not surfTags:
        raise RuntimeError("No surfaces found in STEP file")

    # 2) Compute bounding box of all surfaces
    xmin = ymin = zmin =  1e9
    xmax = ymax = zmax = -1e9
    for tag in surfTags:
        bb = gmsh.model.getBoundingBox(2, tag)
        xmin = min(xmin, bb[0]); ymin = min(ymin, bb[1]); zmin = min(zmin, bb[2])
        xmax = max(xmax, bb[3]); ymax = max(ymax, bb[4]); zmax = max(zmax, bb[5])

    # add a margin
    dx = xmax - xmin; dy = ymax - ymin; dz = zmax - zmin
    xmin -= boxMargin*dx; xmax += boxMargin*dx
    ymin -= boxMargin*dy; ymax += boxMargin*dy
    zmin -= boxMargin*dz; zmax += boxMargin*dz

    # 3) Create the box volume
    boxTag = gmsh.model.occ.addBox(xmin, ymin, zmin,
                                   xmax-xmin, ymax-ymin, zmax-zmin)
    gmsh.model.occ.synchronize()

    # 4) Fragment the box by the imported surface(s)
    #    – treat the surface as a cutting tool
    fragments, _ = gmsh.model.occ.fragment(
        [(3, boxTag)],
        [(2, tag) for tag in surfTags]
    )
    gmsh.model.occ.synchronize()

    # Optionally: tag each resulting volume as a PhysicalGroup
    volTags = [tag for (dim, tag) in fragments if dim == 3]
    gmsh.model.addPhysicalGroup(3, volTags, 1)
    gmsh.model.setPhysicalName(3, 1, "BoxVolumes")

    # 5) Set a uniform mesh size on all nodes
    allPoints = [p[1] for p in gmsh.model.getEntities(0)]
    gmsh.model.mesh.setSize([(0, pt) for pt in allPoints], meshSize)

    # 6) Generate and save
    gmsh.model.mesh.generate(3)
    gmsh.write(outMsh)
    gmsh.write("./output/box_with_surface.vtk")
    # print(f"Mesh written to {outMsh}")

    # visual check
    gmsh.fltk.run()

    gmsh.finalize()


# -

# call the def
step_file = "./output/sum_top_surf_mesh_mmg.step"
mesh_box_with_surface(step_file,
                      meshSize=0.05,
                      boxMargin=0.2,
                      outMsh="./output/box_with_surface.msh")


# +
# using occ inbuilt cad

def SegmentofSphere_occ(
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
    stepFile=''
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
    

    p0 = gmsh.model.occ.addPoint(
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

    p1 = gmsh.model.occ.addPoint(pt1[0], pt1[1], pt1[2], meshSize=cellSize)
    p2 = gmsh.model.occ.addPoint(pt2[0], pt2[1], pt2[2], meshSize=cellSize)
    p3 = gmsh.model.occ.addPoint(pt3[0], pt3[1], pt3[2], meshSize=cellSize)
    p4 = gmsh.model.occ.addPoint(pt4[0], pt4[1], pt4[2], meshSize=cellSize)
    p5 = gmsh.model.occ.addPoint(pt5[0], pt5[1], pt5[2], meshSize=cellSize)
    p6 = gmsh.model.occ.addPoint(pt6[0], pt6[1], pt6[2], meshSize=cellSize)
    p7 = gmsh.model.occ.addPoint(pt7[0], pt7[1], pt7[2], meshSize=cellSize)
    p8 = gmsh.model.occ.addPoint(pt8[0], pt8[1], pt8[2], meshSize=cellSize)

    l1 = gmsh.model.occ.addCircleArc(p1, p0, p2)
    l2 = gmsh.model.occ.addCircleArc(p2, p0, p3)
    l3 = gmsh.model.occ.addCircleArc(p3, p0, p4)
    l4 = gmsh.model.occ.addCircleArc(p4, p0, p1)
    l5 = gmsh.model.occ.addCircleArc(p5, p0, p6)
    l6 = gmsh.model.occ.addCircleArc(p6, p0, p7)
    l7 = gmsh.model.occ.addCircleArc(p7, p0, p8)
    l8 = gmsh.model.occ.addCircleArc(p8, p0, p5)
    l9 = gmsh.model.occ.addLine(p5, p1)
    l10 = gmsh.model.occ.addLine(p2, p6)
    l11 = gmsh.model.occ.addLine(p7, p3)
    l12 = gmsh.model.occ.addLine(p4, p8)

    cl = gmsh.model.occ.addCurveLoop((l1, l2, l3, l4))
    lower = gmsh.model.occ.addSurfaceFilling(cl, tag=boundaries.Lower.value)

    cl = gmsh.model.occ.addCurveLoop((l5, l6, l7, l8))
    upper = gmsh.model.occ.addSurfaceFilling(cl, tag=boundaries.Upper.value)

    cl = gmsh.model.occ.addCurveLoop((l10, l6, l11, -l2))
    east = gmsh.model.occ.addPlaneSurface([cl], tag=boundaries.East.value)

    cl = gmsh.model.occ.addCurveLoop((l9, -l4, l12, l8))
    west = gmsh.model.occ.addPlaneSurface([cl], tag=boundaries.West.value)

    cl = gmsh.model.occ.addCurveLoop((l1, l10, -l5, l9))
    south = gmsh.model.occ.addPlaneSurface([cl], tag=boundaries.South.value)

    cl = gmsh.model.occ.addCurveLoop((-l3, -l11, l7, -l12))
    north = gmsh.model.occ.addPlaneSurface([cl], tag=boundaries.North.value)

    sloop = gmsh.model.occ.addSurfaceLoop([south, east, north, upper, west, lower])
    volume = gmsh.model.occ.addVolume([sloop])

    gmsh.model.occ.synchronize()

    # Add Physical groups
    for b in boundaries:
        tag = b.value
        name = b.name
        gmsh.model.addPhysicalGroup(2, [tag], tag)
        gmsh.model.setPhysicalName(2, tag, name)
    
    
    # 1) Import the surface (dim=2 entities) from STEP
    ents = gmsh.model.occ.importShapes(stepFile)
    gmsh.model.occ.synchronize()

    # collect only the surface tags
    surfTags = [tag for (dim, tag) in ents if dim == 2]
    if not surfTags:
        raise RuntimeError("No surfaces found in STEP file")
        
    # 4) Fragment the box by the imported surface(s)
    #    – treat the surface as a cutting tool
    fragments, _ = gmsh.model.occ.fragment(
        [(3, volume)],
        [(2, tag) for tag in surfTags]
    )
    gmsh.model.occ.synchronize()

    # Optionally: tag each resulting volume as a PhysicalGroup
    volTags = [tag for (dim, tag) in fragments if dim == 3]
    
    # Add the volume entity to a physical group with a high tag number (99999) and name it "Elements"
    gmsh.model.addPhysicalGroup(3, volTags, 99999)
    gmsh.model.setPhysicalName(3, 99999, "Elements")

    gmsh.model.occ.synchronize()

    # after your fragment + synchronize, but before meshing…

    # 1. Identify your volume tag(s) from fragments:
    volumes = [tag for (dim, tag) in fragments if dim == 3]
    
    # 2. Ask Gmsh for their boundary surfaces:
    #    - oriented=False: don’t worry about normal directions
    #    - combined=True: return each face only once
    boundary = gmsh.model.getBoundary(
        [(3, v) for v in volumes],
        oriented=False,
        combined=True
    )

    # 3. Extract just the surface tags
    bound_surf_tags = [tag for (dim, tag) in boundary if dim == 2]
    
    print("Surfaces bounding the volume:", bound_surf_tags)


    gmsh.model.mesh.generate(3)
    gmsh.write(filename)

    # visual check
    gmsh.fltk.run()

    gmsh.finalize()

    return
# -

# create refined mesh
step_file = "./output/sum_top_surf_mesh_mmg.step"
SegmentofSphere_occ(radiusOuter=r_o, radiusInner=r_i,
                    longitudeExtent=lon_ext, latitudeExtent=lat_ext,
                    cellSize=cellsize, filename=f'{output_dir}{meshname}_refined_occ_with_slab.msh', 
                    stepFile=step_file)

Lower = 11
Upper = 12
East = 13
West = 14
South = 15
North = 16
