import os
import json
import numpy as np

from argparse import ArgumentParser
from pprint import pprint
from skimage.io import imread
from skimage.io import imsave
from shapely.geometry import Point, Polygon

def process_image(image_attributes, src_path):
    # identify original image
    filename = image_attributes['filename']
    image = imread(os.path.join(src_path, filename))
    image_shape = image.shape
    # grab polygon data
    regions_data = image_attributes['regions']
    # process and paste each polygon
    gt_image = np.zeros((image_shape[0], image_shape[1]))
    for region_key in regions_data:
        # construct polygon
        region = regions_data[region_key]
        x_points = region['shape_attributes']['all_points_x']
        y_points = region['shape_attributes']['all_points_y']
        polygon_coordinates = list(zip(x_points, y_points))
        polygon = Polygon(polygon_coordinates)

        # define bounding box of polygon
        x_min, y_min, x_max, y_max = np.min(x_points), np.min(y_points), np.max(x_points), np.max(y_points)

        # fill in ground-truth image
        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                if Point(x, y).within(polygon):
                    gt_value = 1+int(region['region_attributes']['foreground_level'])
                    gt_image[y, x] = gt_value

    # save result
    gt_path = os.path.join(src_path, 'groundtruth')
    if not os.path.exists(gt_path):
        os.mkdir(gt_path)
    gt_image = (gt_image*255/2).astype(np.uint8)
    imsave(os.path.join(gt_path, filename), gt_image)
    
if __name__ == '__main__':
    # parse input argument(s)
    parser = ArgumentParser()
    parser.add_argument('--src', type=str, help='path to folder with images and annotation file', required=True)
    args = parser.parse_args()
    src_path = args.src

    # load annotations
    with open(os.path.join(src_path, 'via_region_data.json'), 'r') as f:
        annotations = json.load(f)

    # process each image
    for image_key in annotations:
        attributes = annotations[image_key]
        process_image(attributes, src_path)
