import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import os, copy
import time

import OMRobjects as OMR

line_opt = ['red', 'darkorange', 'gold', 'greenyellow', 'aquamarine', 'skyblue']
color_list = ['red', 'tomato', 'darkorange', 'navajowhite', 'gold', 'khaki', 'greenyellow', 'limegreen', 'aquamarine', 'aqua', 'skyblue', 'slateblue', 'blueviolet']


def get_image_path(path, type_list=['.jpg', '.jpeg', '.png']):
    path_list = []
    type_list = [i.lower() for i in type_list]
    for file_full_name in os.listdir(path):
        file_name, file_ext = os.path.splitext(file_full_name)
        if file_ext.lower() in type_list:
            path_list.append(os.path.join(path, file_full_name))
    return path_list

def get_sheet_list(sheet_path):
    return [OMR.Sheet(os.path.join(sheet_path, page_path), order=i) for i, page_path in enumerate(sorted(os.listdir(sheet_path)))]

def non_maximum_suppression(bboxes, threshold, higher_best=True):
    if bboxes.size <= 0:
        return bboxes
    
    pick = []
    score = bboxes[:,0]
    left, top, right, bottom = (*bboxes[:,1:].astype(np.int32).T, )
    area = (right-left+1)*(bottom-top+1)
    if higher_best:
        idxs = np.argsort(score)
    else:
        idxs = np.argsort(-score)
    
    while len(idxs) > 0:        
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)

        iLeft = np.maximum(left[i], left[idxs[:last]])
        iTop = np.maximum(top[i], top[idxs[:last]])
        iRight = np.minimum(right[i], right[idxs[:last]])
        iBottom = np.minimum(bottom[i], bottom[idxs[:last]])

        iWidth = np.maximum(0, iRight-iLeft+1)
        iHeight = np.maximum(0, iBottom-iTop+1)
        overlap = (iWidth*iHeight)/area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap>threshold)[0])))
    unique_bboxes = bboxes[pick]
    return unique_bboxes