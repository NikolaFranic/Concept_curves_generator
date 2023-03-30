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
import Methods

#region ----------------- mrcnn
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)


model = mrcnn.model.MaskRCNN(mode="inference",
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

model.load_weights(filepath="mask_rcnn_coco.h5",
                   by_name=True)
#endregion

#region -------------------- SIDE VIEW
image = cv2.imread("tesla_s_side_view.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)
r = r[0]

"""mrcnn.visualize.display_instances(image=image,
                                  boxes=r['rois'],
                                  masks=r['masks'],
                                  class_ids=r['class_ids'],
                                  class_names=CLASS_NAMES,
                                  scores=r['scores'])"""

###################### retrieving a mask
masks=r['masks']
im = Image.fromarray(masks[:,:,0])
im.convert("1") #converting to true black and white
im.save("mask.jpg")

################################## generating contours

img = cv2.imread('mask.jpg',0)


contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

biggest_contour=max(contours, key=cv2.contourArea)
#epsilon = 0.001*cv2.arcLength(biggest_contour,True)
#approx = cv2.approxPolyDP(biggest_contour,epsilon,True) ######## creating approximation points

img_copy = cv2.imread('mask.jpg')

"""cv2.drawContours(img_copy, biggest_contour, -1, (0,255,0), 2)
plt.figure()
plt.imshow(img_copy)
plt.show()"""

#################### extracting contour points
x_coords_b, y_coords_b = [], []

for first_bracket in biggest_contour: #biggest_contour [[[112 376
    for second_bracket in first_bracket:
        x_coords_b.append(second_bracket[0])
        y_coords_b.append(second_bracket[1])

#################### generating outer points of contour
center_point = [(max(x_coords_b) - min(x_coords_b)) / 2 + min(x_coords_b),
                    (max(y_coords_b) - min(y_coords_b)) / 2 + min(y_coords_b)]

furthest_points = []

angle_step = 2
for i in numpy.arange(2,180, angle_step):
    # direction equation y = math.tan(abs(angle))*X + l
    k1 = math.tan(abs(math.radians(i)))
    l = -math.tan(abs(math.radians(i))) * center_point[0] + center_point[1]

    value1,value2 = Methods.Extracting_outer_points_of_contour(x_coords_b,y_coords_b,center_point,k1,l)
    if value1 != []:
        furthest_points.append(value1)
    if value2 != []:
        furthest_points.append(value2)

############################# sorting of points clockwise
Methods.origin = center_point
furthest_points = sorted(furthest_points,key=Methods.clockwiseangle_and_distance)

############################# getting rid of tires

furthest_points = Methods.Getting_rid_of_tires(furthest_points,center_point)

############################# interpolation
x_coords = []
y_coords = []
for point in furthest_points:
    x_coords.append(point[0])
    y_coords.append(point[1])

#new center point after cutting tires and mirrors
center_point = [(max(x_coords) - min(x_coords)) / 2 + min(x_coords),
                    (max(y_coords) - min(y_coords)) / 2 + min(y_coords)]

from scipy import interpolate

tck,u = interpolate.splprep([x_coords,y_coords],s=400,k=2)
unew = numpy.arange(0, 1.01, 0.01) #ovo nam da 100 novih tocaka kad bi koristili obicni u onda bi dobili stare tocke
out_side_view = interpolate.splev(unew, tck)
plt.figure()
plt.imshow(img_copy)
plt.plot(out_side_view[0], out_side_view[1], 'b')#x_coords,y_coords,"x",
plt.show()

out_side_view[0], out_side_view[1] = Methods.Translate_points(out_side_view[0],out_side_view[1],center_point)

out_side_view[1] = Methods.Mirror_point(out_side_view[1])
print("side_view_done")
#endregion

#region -------------------- FRONT VIEW
image = cv2.imread("tesla_s_front_view.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)
r = r[0]

"""mrcnn.visualize.display_instances(image=image,
                                  boxes=r['rois'],
                                  masks=r['masks'],
                                  class_ids=r['class_ids'],
                                  class_names=CLASS_NAMES,
                                  scores=r['scores'])"""

###################### retrieving a mask
masks=r['masks']
im = Image.fromarray(masks[:,:,0])
im.convert("1") #converting to true black and white
im.save("mask.jpg")

################################## generating contours

img = cv2.imread('mask.jpg',0)


contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

biggest_contour=max(contours, key=cv2.contourArea)
#epsilon = 0.001*cv2.arcLength(biggest_contour,True)
#approx = cv2.approxPolyDP(biggest_contour,epsilon,True) ######## creating approximation points

img_copy = cv2.imread('mask.jpg')

"""cv2.drawContours(img_copy, biggest_contour, -1, (0,255,0), 2)
plt.figure()
plt.imshow(img_copy)
plt.show()"""

#################### extracting contour points
x_coords_b, y_coords_b = [], []

for first_bracket in biggest_contour: #biggest_contour [[[112 376
    for second_bracket in first_bracket:
        x_coords_b.append(second_bracket[0])
        y_coords_b.append(second_bracket[1])

#################### generating outer points of contour
center_point = [(max(x_coords_b) - min(x_coords_b)) / 2 + min(x_coords_b),
                    (max(y_coords_b) - min(y_coords_b)) / 2 + min(y_coords_b)]

furthest_points = []

angle_step = 2
for i in numpy.arange(2,180, angle_step):
    # direction equation y = math.tan(abs(angle))*X + l
    k1 = math.tan(abs(math.radians(i)))
    l = -math.tan(abs(math.radians(i))) * center_point[0] + center_point[1]

    value1,value2 = Methods.Extracting_outer_points_of_contour(x_coords_b,y_coords_b,center_point,k1,l)
    if value1 != []:
        furthest_points.append(value1)
    if value2 != []:
        furthest_points.append(value2)

############################# sorting of points clockwise
Methods.origin = center_point
furthest_points = sorted(furthest_points,key=Methods.clockwiseangle_and_distance)

############################# getting rid of tires

furthest_points = Methods.Getting_rid_of_tires(furthest_points,center_point)

############################# getting rid of side mirrors (only for front view)

furthest_points = Methods.Getting_rid_of_side_mirrors(furthest_points,center_point)

############################# interpolation
x_coords = []
y_coords = []
for point in furthest_points:
    x_coords.append(point[0])
    y_coords.append(point[1])

#new center point after cutting tires and mirrors
center_point = [(max(x_coords) - min(x_coords)) / 2 + min(x_coords),
                    (max(y_coords) - min(y_coords)) / 2 + min(y_coords)]

from scipy import interpolate

tck,u = interpolate.splprep([x_coords,y_coords],s=400,k=2)
unew = numpy.arange(0, 1.01, 0.01) #ovo nam da 100 novih tocaka kad bi koristili obicni u onda bi dobili stare tocke
out_front_view = interpolate.splev(unew, tck)
plt.figure()
plt.imshow(img_copy)
plt.plot(out_front_view[0], out_front_view[1], 'b')#x_coords,y_coords,"x",
plt.show()

out_front_view[0], out_front_view[1] = Methods.Translate_points(out_front_view[0],out_front_view[1],center_point)

out_front_view[1] = Methods.Mirror_point(out_front_view[1])
print("front_view_done")
#endregion

#region -------------------- TOP VIEW
image = cv2.imread("tesla_s_top_view.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)
r = r[0]

"""mrcnn.visualize.display_instances(image=image,
                                  boxes=r['rois'],
                                  masks=r['masks'],
                                  class_ids=r['class_ids'],
                                  class_names=CLASS_NAMES,
                                  scores=r['scores'])"""

###################### retrieving a mask
masks=r['masks']
im = Image.fromarray(masks[:,:,0])
im.convert("1") #converting to true black and white
im.save("mask.jpg")

################################## generating contours

img = cv2.imread('mask.jpg',0)


contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

biggest_contour=max(contours, key=cv2.contourArea)
#epsilon = 0.001*cv2.arcLength(biggest_contour,True)
#approx = cv2.approxPolyDP(biggest_contour,epsilon,True) ######## creating approximation points

img_copy = cv2.imread('mask.jpg')

"""cv2.drawContours(img_copy, biggest_contour, -1, (0,255,0), 2)
plt.figure()
plt.imshow(img_copy)
plt.show()"""

#################### extracting contour points
x_coords_b, y_coords_b = [], []

for first_bracket in biggest_contour: #biggest_contour [[[112 376
    for second_bracket in first_bracket:
        x_coords_b.append(second_bracket[0])
        y_coords_b.append(second_bracket[1])

#################### generating outer points of contour
center_point = [(max(x_coords_b) - min(x_coords_b)) / 2 + min(x_coords_b),
                    (max(y_coords_b) - min(y_coords_b)) / 2 + min(y_coords_b)]

furthest_points = []

angle_step = 2
for i in numpy.arange(2,180, angle_step):
    # direction equation y = math.tan(abs(angle))*X + l
    k1 = math.tan(abs(math.radians(i)))
    l = -math.tan(abs(math.radians(i))) * center_point[0] + center_point[1]

    value1,value2 = Methods.Extracting_outer_points_of_contour(x_coords_b,y_coords_b,center_point,k1,l)
    if value1 != []:
        furthest_points.append(value1)
    if value2 != []:
        furthest_points.append(value2)

############################# sorting of points clockwise
Methods.origin = center_point
furthest_points = sorted(furthest_points,key=Methods.clockwiseangle_and_distance)

############################# getting rid of side mirrors (only for top view)

furthest_points = Methods.Getting_rid_of_side_mirrors_top_view(furthest_points,center_point)

############################# interpolation
x_coords = []
y_coords = []
for point in furthest_points:
    x_coords.append(point[0])
    y_coords.append(point[1])

#new center point after cutting tires and mirrors
center_point = [(max(x_coords) - min(x_coords)) / 2 + min(x_coords),
                    (max(y_coords) - min(y_coords)) / 2 + min(y_coords)]

from scipy import interpolate

tck,u = interpolate.splprep([x_coords,y_coords],s=400,k=2)
unew = numpy.arange(0, 1.01, 0.01) #ovo nam da 100 novih tocaka kad bi koristili obicni u onda bi dobili stare tocke
out_top_view = interpolate.splev(unew, tck)
plt.figure()
plt.imshow(img_copy)
plt.plot(out_top_view[0], out_top_view[1], 'b')#x_coords,y_coords,"x",
plt.show()

out_top_view[0], out_top_view[1] = Methods.Translate_points(out_top_view[0],out_top_view[1],center_point)

out_top_view[1] = Methods.Mirror_point(out_top_view[1])
print("top_view_done")
#endregion

#region scaling
out_side_view[0], out_side_view[1] = Methods.Scaling_points(out_side_view[0],out_side_view[1])
out_front_view[0], out_front_view[1] = Methods.Scaling_points(out_front_view[0],out_front_view[1])

from scipy.integrate import simps

povrsina = (max(out_front_view[0])- min(out_front_view[0]))*(max(out_front_view[1])- min(out_front_view[1]))

print((numpy.trapz(out_front_view[1], out_front_view[0]))/povrsina)

# scaling out_top_view

out_top_view[0], out_top_view[1] = Methods.Scaling_top_view(out_top_view[0],out_top_view[1],max(out_front_view[0]))
#endregion

#exporting points
import pandas as pd

col1 = "X"
col2 = "Y"
col3 = "Z"
col0 = "Num"

zeros = [0]*len(out_side_view[0])
numeration = list(range(len(out_side_view[0])))

data_side = pd.DataFrame({col0:numeration,col1:out_side_view[0],col2:zeros,col3:out_side_view[1]})
data_front = pd.DataFrame({col0:numeration,col1:zeros,col2:out_front_view[0],col3:out_front_view[1]})
data_top = pd.DataFrame({col0:numeration,col1:out_top_view[0],col2:out_top_view[1],col3:zeros})

data_side.to_excel('side_view_points.xlsx', sheet_name='sheet1', index=False)
data_front.to_excel('front_view_points.xlsx', sheet_name='sheet1', index=False)
data_top.to_excel('top_view_points.xlsx', sheet_name='sheet1', index=False)












