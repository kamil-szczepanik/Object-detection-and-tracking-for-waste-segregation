import numpy as np


def x1y1wh_to_x1y1x2y2(x1,y1,w,h):
    x2, y2 = x1+w, y1+h
    return x1, y1, x2, y2

def x1y1x2y2_to_x1y1wh(x1, y1, x2, y2):
    w =  x2- x1
    h = y2 - y1
    return x1, y1, w, h

def get_iou(ground_truth, pred):
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
    area_of_intersection = i_height * i_width
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    iou = area_of_intersection / area_of_union
    return iou

