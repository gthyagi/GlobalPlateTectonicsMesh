"""
Author: G Thyagarajulu
Original file is downloaded from slab2.0.

pointShift is essentially epCalc, but for a single point.  
It is used to calculate the endpoint of a vector within the earth given a local lat/lon/dep, strike/dip, and distance.
"""

import math
import numpy as np

def pointShift(lon, lat, dep, dip, strike, mag):
    
    # Rotate from strike to direction of motion
    if strike > 270:
        az = (strike + 90) - 360
    else:
        az = strike + 90
    az = 360 - az    # Accounts for the fact that azimuth counts goes opposite of the positive rotation of the x-axis (points north)
    
    # Convert input angles to radians
    
    latrad = math.radians(90 - lat)
    lonrad = math.radians(lon)
    azrad = math.radians(az)
    diprad = math.radians(dip)
    
    # Define initial location in spherical coordinates
    
    crad = 6371 - dep
    ctheta = latrad
    cphi = lonrad
    
    # Convert initial location to cartesian coordinates
    
    cx = crad * math.sin(ctheta) * math.cos(cphi)
    cy = crad * math.sin(ctheta) * math.sin(cphi)
    cz = crad * math.cos(ctheta)
    
    # Define lon/lat of new coordinate system
    
    if latrad < (math.pi/2):
        x1lat = abs(latrad-(math.pi/2))
        if lonrad > 0:
            x1lon = lonrad - math.pi
        else:
            x1lon = lonrad + math.pi
    else:
        x1lon = lonrad
        x1lat = latrad - (math.pi/2)
    if lonrad < (-1 * (math.pi/2)):
        x2lon = lonrad + 3 * (math.pi/2)
    else:
        x2lon = lonrad - (math.pi/2)
    x2lat = (math.pi/2)
    x3lon = lonrad
    x3lat = latrad

    # Calculate transformation matrix

    a11 = math.sin(x1lat) * math.cos(-1 * x1lon)
    a12 = math.sin(x2lat) * math.cos(-1 * x2lon)
    a13 = math.sin(x3lat) * math.cos(-1 * x3lon)
    a21 = math.sin(x1lat) * math.cos((math.pi/2) - x1lon)
    a22 = math.sin(x2lat) * math.cos((math.pi/2) - x2lon)
    a23 = math.sin(x3lat) * math.cos((math.pi/2) - x3lon)
    a31 = math.cos(x1lat)
    a32 = math.cos(x2lat)
    a33 = math.cos(x3lat)
    
    # Define translation vector in spherical coordinates
    
    trad = mag
    ttheta = diprad
    tphi = azrad
    
    # Convert translation vector to cartesian coordinates
    
    tx = trad * math.sin(ttheta) * math.cos(tphi)
    ty = trad * math.sin(ttheta) * math.sin(tphi)
    tz = trad * math.cos(ttheta)
    
    # Transform translation vector into base coordinate system
    
    txnew = a11 * tx + a12 * ty + a13 * tz
    tynew = a21 * tx + a22 * ty + a23 * tz
    tznew = a31 * tx + a32 * ty + a33 * tz
    
    # Add new vector to original position vector
    
    eptx = cx + txnew
    epty = cy + tynew
    eptz = cz + tznew
    
    # Convert new sum to spherical coordinates
    
    eptrad = math.sqrt(math.pow(eptx, 2) + math.pow(epty, 2) + math.pow(eptz, 2))
    eptphirad = math.atan2(epty, eptx)
    eptthetarad = math.acos(eptz / (math.sqrt(math.pow(eptx, 2) + math.pow(epty, 2) + math.pow(eptz, 2))))
    
    # Convert into lat, lon, depth
    
    eptdep = 6371 - eptrad
    eptlat = 90 - (math.degrees(eptthetarad))
    eptlon = math.degrees(eptphirad)
     
    return [eptlon, eptlat, eptdep] 
