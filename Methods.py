import matplotlib.pyplot as plt
import tensorflow

import numpy
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import keras
import math

from PIL import Image

origin = [0, 0] # point that serves as center of clockwise rotation
refvec = [-1, 0] # vector direction
# function returns angle and length because if we have two points with same angle
# first point in sorting will be point with shortest length
# points are sorted counterclockwise
def clockwiseangle_and_distance(point):
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

# function takes points and direction parameters
# and returns furthest points in both directions
def Extracting_outer_points_of_contour(x_coord, y_coord, center_point, k,l):

    #all points that are inside distance
    distance = 1
    points = []
    for i in range(len(x_coord)):
        calc_distance = abs(k*x_coord[i] + l - y_coord[i])/math.sqrt(1+math.pow(k,2))
        if calc_distance <= distance:
            point = [x_coord[i],y_coord[i]]
            points.append(point)
    upper_points = []
    lower_points = []
    #dividing points by horizontal axis on upper and lower
    if len(points) != 0:
        for point in points:
            if point[1] > center_point[1]:
                upper_points.append(point)
            else:
                lower_points.append(point)

    #selecting furthest points
    furthest_point_upper = []
    if len(upper_points) != 0:
        max_distance = math.sqrt(math.pow(center_point[0] - upper_points[0][0],2) + math.pow(center_point[1] - upper_points[0][1],2))
        furthest_point_upper = upper_points[0]
        for point in upper_points:
            calc_distance = math.sqrt(math.pow(center_point[0] - point[0],2) + math.pow(center_point[1] - point[1],2))
            if calc_distance > max_distance:
                max_distance=calc_distance
                furthest_point_upper = point
    furthest_point_lower = []
    if len(lower_points) != 0:
        max_distance = math.sqrt(math.pow(center_point[0] - lower_points[0][0], 2) + math.pow(center_point[1] - lower_points[0][1], 2))
        furthest_point_lower = lower_points[0]
        for point in lower_points:
            calc_distance = math.sqrt(math.pow(center_point[0] - point[0], 2) + math.pow(center_point[1] - point[1], 2))
            if calc_distance > max_distance:
                max_distance = calc_distance
                furthest_point_lower = point

    return furthest_point_upper,furthest_point_lower

# getting rid of tire curves in third and fourth quadrant
def Getting_rid_of_tires(points, center_point):
    third_quadrant = []
    for point in points:
        if point[0] > center_point[0] and point[1] > center_point[1]: # third quadrant ()mora bit vece u drugom uvjetu jer je obrdnut coordinatni sustav
            third_quadrant.append(point)
    third_quadrant.sort()
    car_underbody_point = third_quadrant[0]  # taken from third quadrant, it is same for fourth

    for point in points:
        if point[0] > center_point[0] and point[1] > car_underbody_point[1]:  # third quadrant
            point[1] = car_underbody_point[1]
        elif point[0] < center_point[0] and point[1] > car_underbody_point[1]:  # fourth quadrant
            point[1] = car_underbody_point[1]
    return points

# getting rid of tire curves in third and fourth quadrant
def Getting_rid_of_side_mirrors_top_view(points, center_point):
    second_quadrant = []
    third_quadrant = []
    for point in points:
        if point[0] > center_point[0] and point[1] > center_point[1]: # third quadrant ()mora bit vece u drugom uvjetu jer je obrdnut coordinatni sustav
            third_quadrant.append(point)
    third_quadrant.sort()
    third_quadrant_point = third_quadrant[0]
    for point in points:
        if point[0] > center_point[0] and point[1] < center_point[1]: # second quadrant
            second_quadrant.append(point)
    second_quadrant.sort()
    second_quadrant_point = second_quadrant[0]

    for point in points:
        if point[0] > center_point[0] and point[1] > third_quadrant_point[1]:  # third quadrant
            point[1] = third_quadrant_point[1]
        elif point[0] > center_point[0] and point[1] < second_quadrant_point[1]:  # second quadrant
            point[1] = second_quadrant_point[1]
    return points

# getting rid of side mirror in first and second quadrant
# use only after points are scaled
# points in quadrants go in counter - clockwise order
def Getting_rid_of_side_mirrors(points, center_point):
    second_quadrant = []
    first_quadrant = []
    # region defining height tolerance in percentages of vehicle height
    xcoords = []
    ycoords = []
    for point in points:
        xcoords.append(point[0])
        ycoords.append(point[1])

    step_y_dir = (max(ycoords) - min(ycoords))*0.05
    #endregion

    for point in points:
        if point[0] < center_point[0] and point[1] < center_point[1]:  # first quadrant
            first_quadrant.append(point)

    for point in points:
        if point[0] > center_point[0] and point[1] < center_point[1]: # second quadrant ()mora bit manje u drugom uvjetu jer je obrdnut coordinatni sustav
            second_quadrant.append(point)

    #region ------------- upper and lower points
    first_quadrant_lower_point = []
    first_quadrant_upper_point = []

    for point in first_quadrant:
        if point[0] < first_quadrant[0][0] and abs(first_quadrant[0][1] - point[1]) > step_y_dir:
            first_quadrant_upper_point = point
            break

    for i in range(len(first_quadrant)-1,0,-1):
        if first_quadrant[i][0] > first_quadrant[len(first_quadrant)-1][0] and abs(first_quadrant[len(first_quadrant)-1][1] - first_quadrant[i][0]) > step_y_dir:
            first_quadrant_lower_point = first_quadrant[i]
            break

    second_quadrant_upper_point = []
    second_quadrant_lower_point = []

    for point in second_quadrant:
        if point[0] < second_quadrant[0][0] and abs(second_quadrant[0][1] - point[1]) > step_y_dir:
            second_quadrant_lower_point = point
            break

    for i in range(len(second_quadrant)-1,0,-1):
        if second_quadrant[i][0] > second_quadrant[len(second_quadrant)-1][0] and abs(second_quadrant[len(second_quadrant)-1][1] - second_quadrant[i][1]) > step_y_dir:
            second_quadrant_upper_point = second_quadrant[i]
            break

    #endregion

    # region equation of direction => y = kx + l
    k_fq=(first_quadrant_upper_point[1] - first_quadrant_lower_point[1])/(first_quadrant_upper_point[0] - first_quadrant_lower_point[0])
    l_fq = first_quadrant_lower_point[1] - first_quadrant_lower_point[0]*k_fq

    k_sq=(second_quadrant_upper_point[1] - second_quadrant_lower_point[1])/(second_quadrant_upper_point[0] - second_quadrant_lower_point[0])
    l_sq = second_quadrant_lower_point[1] - second_quadrant_lower_point[0]*k_sq
    #endregion

    # region getting rid of side mirrors
    for point in points:
        if point[0] > first_quadrant_lower_point[0] and point[0] < first_quadrant_upper_point[0]\
                and point[1] < first_quadrant_lower_point[1] and point[1] > first_quadrant_upper_point[1]:  # first quadrant
            point[1] = k_fq*point[0] + l_fq
        elif point[0] > second_quadrant_upper_point[0] and point[0] < second_quadrant_lower_point[0] \
                and point[1] > second_quadrant_upper_point[1] and point[1] < second_quadrant_lower_point[1]:  # second quadrant
            point[1] = k_sq * point[0] + l_sq
    #endregion

    #removing duplicates
    for i in range(len(points)-1,0,-1):
        if points[i] == points[i-1]:
            points.remove(points[i])

    points.remove(first_quadrant_lower_point)
    points.remove(first_quadrant_upper_point)
    points.remove(second_quadrant_upper_point)
    points.remove(second_quadrant_lower_point)
    return points

# mirroring points around x axis
def Mirror_point(y_coords):
    for i in range(len(y_coords)):
        y_coords[i] = y_coords[i]*(-1)
    return y_coords

# translating points to origin point
def Translate_points(x_coords,y_coords,center_point):
    for i in range(len(x_coords)):
        x_coords[i] = x_coords[i]-center_point[0]
    for i in range(len(y_coords)):
        y_coords[i] = y_coords[i]-center_point[1]
    return x_coords,y_coords

# scaling of points to height 1000, points must be around origin
def Scaling_points(x_coords,y_coords):
    max_height = max(y_coords)
    scale_factor = 1000/max_height
    for i in range(len(x_coords)):
        x_coords[i] = x_coords[i]*scale_factor
    for i in range(len(y_coords)):
        y_coords[i] = y_coords[i]*scale_factor
    return x_coords, y_coords

# scaling for top view so front and top have same width
# half_width means from origin to widest point because contour is symmetric around center point
def Scaling_top_view(x_coords,y_coords, half_width_of_front_view):
    max_height = max(y_coords)
    scale_factor = half_width_of_front_view / max_height
    for i in range(len(x_coords)):
        x_coords[i] = x_coords[i] * scale_factor
    for i in range(len(y_coords)):
        y_coords[i] = y_coords[i] * scale_factor
    return x_coords, y_coords






