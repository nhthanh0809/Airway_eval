import os
import cv2
import numpy as np

import xml.etree.ElementTree as ET

def get_points_from_CVAT_xml(imageName, points_list, xml_imageEles):
    labels = []
    x_pos = []
    y_pos = []

    for imageEle in xml_imageEles:
        if imageName == imageEle.attrib['name']:
            for i, pointName in enumerate(points_list):
                for pointEle in imageEle.iter('points'):
                    if pointName == pointEle.attrib['label']:
                        labels.append(pointEle.attrib['label'])
                        x_pos.append(float(pointEle.attrib['points'].split(',')[0]))
                        y_pos.append(float(pointEle.attrib['points'].split(',')[1]))

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    # print(imageName, labels, x_pos, y_pos)

    return labels, x_pos, y_pos

def get_mask_from_CVAT_xml(imageName, xml_imageEles):

    x_pos = []
    y_pos = []
    points = []

    for imageEle in xml_imageEles:
        if imageName == imageEle.attrib['name']:
                for pointEle in imageEle.iter('polygon'):
                    if pointEle.attrib['label'] == 'airway_mask':
                        pointList = pointEle.attrib['points'].split(';')
                        # print(pointList)
                        for point in pointList:
                            points.append([int(float(point.split(',')[0])), int(float(point.split(',')[1]))])

    points = np.array(points)

    return points




if __name__ == '__main__':

    DATA_DIR = '/data/Airway/private/v1.0/original/'

    annotationFile = '/data/Airway/private/v1.0/annotations.xml'

    imageDir = DATA_DIR + dataset + '/images/'
    maskDir = DATA_DIR + dataset + '/masks/'

    os.makedirs(imageDir, exist_ok=True)
    os.makedirs(maskDir, exist_ok=True)

    for imageName in os.listdir(imageDir):

        ########## Working with xml files ##########
        image = cv2.imread(imageDir + imageName, 1)
        mask = np.zeros(image.shape[:2], dtype='uint8')

        doc = ET.parse(annotationFile)
        root = doc.getroot()
        xml_imageEles = root.findall('image')

        # points_name, gt_x, gt_y = get_mask_from_CVAT_xml(imageName, xml_imageEles)
        points = get_mask_from_CVAT_xml(imageName, xml_imageEles)
        # print(len(points))

        # cv2.polylines(image, [points], True, color=(0, 255, 0))

        if len(points) > 2:
            mask = cv2.fillPoly(mask, pts=[points], color=(255, 255, 255))
            cv2.imwrite(maskDir + imageName, mask)
        else:
            print(imageName)

        # cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Resized_Window", 1300, 1900)
        # cv2.imshow("Resized_Window", image)
        # cv2.waitKey(0)






