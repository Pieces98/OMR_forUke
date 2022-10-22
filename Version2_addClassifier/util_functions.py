import numpy as np
import matplotlib.pyplot as plt

import cv2
import os, copy

def image_threshold(image, threshold=None):
    if threshold==None:
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        ret, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        
    assert ret, '[Image does not exist]'
    return image
    
def get_images(path, resize_factor=1.0, denoise=False, threshold=None, log=True):
    if os.path.isdir(path):
        if log: print(f'>>> Folder : {path} : {len(os.listdir(path))} files')
        image_path = sorted([os.path.join(path, i) for i in os.listdir(path)])
    else:
        if log: print(f'>>> File : {path}')
        image_path = [path]
    
    images = [cv2.imread(pth, cv2.IMREAD_GRAYSCALE) for pth in image_path]
    if denoise: images = [cv2.fastNlMeansDenoising(image) for image in images]
    if log : print(*[f'{i+1:02d} : {pth} : {images[i].shape}' for i, pth in enumerate(image_path)], sep='\n')
    
    if resize_factor != 1.0:
        if log: print(f'>>> Resize factor : x{resize_factor} : {images[0].shape}->', end='')
        images = [cv2.resize(img, dsize=(0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR) for img in images]
        print(images[0].shape)
        
    if log:
        if threshold == None: print(">Inverse binary thresholding with Otsu's method")
        else: print(f'>>> Inverse binary thresholding with threshold value {threshold}') 
    images = [image_threshold(img, threshold=threshold) for img in images]
    
    if os.path.isdir(path): return images
    else: return images[0]

def plot_single_image(img, figsize=(10,20), off_axis=True, show=True):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    
    ax.imshow(img, cmap='gray')
    if off_axis: ax.axis('off')
    if show: plt.show()
    
def overlap_object(img_shape, obj_info):
    mask = np.zeros(img_shape)
    
    for x, y, w, h, _ in obj_info:
        mask[y:y+h, x:x+w] = 255
    mask = np.uint8(mask)
    
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    stats = stats[1:,:]
    
    return stats
        
def get_main_components(image, width_threshold=0.6, height_std_coef=1.0):
    _, _, stats, _ = cv2.connectedComponentsWithStats(image)
    image_height, image_width = image.shape
    
    stats_ = overlap_object(image.shape, stats[1:,:])
    ind = np.lexsort((-stats_[:,2], -stats_[:,3]))
    stats = stats_[ind,:]
    
    w_filter = stats[:,2]>width_threshold*image_width
    h_avg, h_std = np.mean(stats[w_filter,3]), np.std(stats[w_filter,3])
    h_filter = stats[:,3]>h_avg-height_std_coef*h_std
    ind = np.logical_or(w_filter, h_filter)
    #stats = stats[ind,:]
    
    #ind = np.lexsort((stats[:,1], stats[:,0]))
    main_comp = stats[ind,:]
    component_image = [image[y:y+h, x:x+w] for x, y, w, h, _ in main_comp]
    
    return component_image, main_comp, stats_

def arrange_components_idx(main_components, position):
    center = np.array([[i, x+w/2, y+h/2] for i, (x, y, w, h, _) in enumerate(position)], dtype=int)
    y_list = sorted(center[:,2])
    tmp = np.diff(y_list)
    interval = np.mean(tmp[np.where(tmp>np.mean(tmp)-0.5*np.std(tmp))])
    
    idx_y = np.round((center[:,2]-y_list[0])/interval)
    idx = np.lexsort((center[:,1], idx_y))
    
    rearrange_components = list(np.array(main_components, dtype=object)[idx])
    rearrange_position = position[idx]
    return rearrange_components, rearrange_position
    
def find_best_match(image, templates, scales, search_area_ratio=None):
    if not hasattr(templates, '__iter__'):
        templates = [templates]
    
    best_score, best_score_idx, best_scale, best_template, shape_info = -1, None, -1, None, None
    score_log = [[] for i in range(len(templates))]

    image_height, image_width = image.shape
    for i, template_ in enumerate(templates):
        template_height, template_width = template_.shape
        for scale in scales:
            if (scale*template_width > image_width) or (scale*template_height > image_height):
                break
            template = cv2.resize(template_, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            template_height, template_width = template.shape
            search_to = image_width if search_area_ratio == None else min(image_width, search_area_ratio*template_width)
            result = cv2.matchTemplate(image[:,:search_to], template, cv2.TM_CCOEFF_NORMED)

            best_idx = np.unravel_index(np.argmax(result), result.shape)
            score_log[i].append(result[best_idx])

            if best_score < result[best_idx]:
                best_score = result[best_idx]
                best_scale = scale
                best_score_idx = best_idx
                best_template = template
                shape_info = template.shape

    return best_score, best_scale, best_score_idx, shape_info, best_template

def find_high_matches(image, templates, scales, threshold):
    if not hasattr(templates, '__iter__'):
        templates = [templates]
        
    match_infos = [{'num':0, 'scale':-1, 'shape':None, 'location':None, 'score':None, 'key_score':-1} for i in range(len(templates))]

    image_height, image_width = image.shape
    for i, template_ in enumerate(templates):
        template_height, template_width = template_.shape
        for scale in scales:
            if (scale*template_width > image_width) or (scale*template_height > image_height):
                break
            template = cv2.resize(template_, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            template_height, template_width = template.shape
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            
            locations = np.where(result>threshold)
            count = len(locations[0])
            
            new_score = np.mean(result[locations]) if count != 0 else 0
            
            if match_infos[i]['key_score'] < new_score:
                match_infos[i]['num'] = count
                match_infos[i]['scale'] = scale
                match_infos[i]['location'] = locations
                match_infos[i]['shape'] = template.shape
                match_infos[i]['score'] = result[locations]
                match_infos[i]['key_score'] = new_score

    return match_infos

def merge_match_infos(match_infos):
    merge_info = {'num':0, 'shape':np.zeros((0, 2)), 'location':None, 'score':np.zeros((0, ))}
    loc_row, loc_col = [], []
    for info in match_infos:
        if info['num'] == 0: 
            continue
        merge_info['num'] += info['num']
        
        merge_info['score'] = np.concatenate((merge_info['score'], info['score']))
        
        shape_array = np.array([list(shape) for _ in range(info['num'])])
        mege_info['shape'] = np.concatenate((merge_info['score'], shape_array), axis=0)
        loc_row.append(info['location'][0])
        loc_col.append(info['location'][1])
        
        
        
    
    return merge_info

def match_info_to_bboxes(match_info):
    shape = match_info['shape']
    
    left = match_info['location'][0]
    top = match_info['location'][1]
    right = left+shape[1]
    bottom = top+shape[0]
    score = match_info['score']
        
    bboxes = np.array([left, top, right, bottom, score]).T.astype(np.float32)
    return bboxes

def non_maximum_suppression(match_info, threshold):
    if match_info['num'] == 0:
        return match_info
    bboxes = match_info_to_bboxes(match_info)
    match_info2 = copy.deepcopy(match_info)
    
    pick = []
    score = bboxes[:,4]
    left, top, right, bottom = (*bboxes[:,:4].astype(np.int32).T, )
    area = (right-left+1)*(bottom-top+1)
    idxs = np.argsort(score)
    
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
    
    bboxes2 = bboxes[pick]
    match_info2['num'] = len(bboxes2)
    match_info2['location'] = (bboxes2[:,0], bboxes2[:,1])
    match_info2['score'] = bboxes2[:,4]
    return match_info2

def get_pixel_histogram(image, grid=[0.3, 0.5, 0.7, 0.9]):
    image = np.uint8(np.ceil(image/255))
    pix_hist1 = np.sum(image, axis=1)
    pix_hist_image = np.zeros(image.shape)
    
    pix_total = 0
    for row, num in enumerate(pix_hist1):
        pix_hist_image[row,:num] = 255
        pix_total += num
        
    grid_idx = []
    pix_hist2 = np.sum(pix_hist_image/255, axis=0)
    pix_sum = 0
    i, j = 0, 0
    already_sum = False
    while i < pix_hist2.size:
        if not already_sum: pix_sum += float(pix_hist2[i]/pix_total)          
        if pix_sum > grid[j]:
            grid_idx.append(i)
            already_sum = True
            j += 1
            if j >= len(grid):
                break
        else:
            already_sum = False
            i += 1
            
    return pix_hist_image, pix_hist2, grid, grid_idx
        
        
        
        
        
        
        
        
        
        