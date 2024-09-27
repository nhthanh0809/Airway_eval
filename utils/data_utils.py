import numpy as np
import json
import cv2
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


def resize_mask_point(maskPoint, scal_ratio_w, scal_ratio_h):
    newMaskPoint = []
    for point in maskPoint:
        x = int(point[0]/scal_ratio_w)
        y = int(point[1]/scal_ratio_h)
        newMaskPoint.append([x,y])
    return newMaskPoint
def get_points_from_txt(point_num, path):
    flag = 0
    x_pos = []
    y_pos = []
    points_name = []
    with open(path) as note:
        for line in note:
            if flag >= point_num:
                break
            else:
                flag += 1
                x, y = [float(i) for i in line.split(',')]
                x_pos.append(x)
                y_pos.append(y)
                points_name.append(str(flag))
        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)
    return points_name, x_pos, y_pos


def get_points_from_json(point_list, json_path):
    f = open(json_path, 'r')
    data = json.loads(f.read())

    labels = []
    x_pos = []
    y_pos = []

    for i in range(0,len(point_list)):
        points_value = data["markerPoints"][point_list[i]]
        labels.append(point_list[i])
        x_pos.append(points_value.get("x"))
        y_pos.append(points_value.get("y"))

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    return labels, x_pos, y_pos

def get_predict_point_from_heatmap(heatmaps, scal_ratio_w, scal_ratio_h, points_num, visualization=False):

    pred = np.zeros((points_num, 2))

    for i in range(points_num):
        heatmap = heatmaps[i]

        if visualization:
            heatmap_output = heatmap.astype(np.uint8)
            cv2.imshow("Predicted heatmap[" + str(i) + "]" , heatmap_output)
            cv2.imwrite('./Predicted_heatmap_' + str(i) + '.jpeg', heatmap_output)
            cv2.waitKey(0)

        pre_y, pre_x = np.where(heatmap == np.max(heatmap))
        pred[i][1] = pre_y[0] * scal_ratio_h
        pred[i][0] = pre_x[0] * scal_ratio_w
    return pred


def visualization(image_path, points_x, points_y, points_num, points_name):
    image = cv2.imread(image_path, 1)

    for j in range(points_num):
        x, y = int(points_x[j]), int(points_y[j])
        point_name = points_name[j]
        image = cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=2)
        image = cv2.putText(image, str(point_name), (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (0, 0, 255), 1, cv2.LINE_AA)

    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Resized_Window", image)
    cv2.waitKey(0)





# test_json = 'D:/Self-projects/2D_Cephalometry/data/steinerAnno/train/points/1589963708196-573300089.json'
# get_points_from_json(17, test_json)