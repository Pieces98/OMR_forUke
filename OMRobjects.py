import numpy as np
import matplotlib.pyplot as plt

import cv2
import os, copy
import time

#Class : Sheet ---------------------------------------------------------------------------------------------------------#
class Sheet():
    def __init__(self, sheet_path, order='Unknown'):
        self.path = sheet_path
        self.raw_image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.order = order
        print(f'* Sheet Object Created :: Sheet #{order} :: {self.path} :: {self.raw_image.shape}')
        
        self.image = None
        
        self.systems = None
        self.num_systems = None
        
    def update_image(self, new_image):
        self.image = new_image
        self.shape = self.image.shape
        return
    

    def preprocess_image(self, deskewing=False, resize_factor=1.0, denoise=False, threshold=None):
        print(f'>>> Preprocessing Page #{self.order:02d} :: {self.path}')
        self.image = self.raw_image
        
        if deskewing:  #TODO : Deskewing process (If necessary)
            pass 
        else:
            print('* Deskewing :: PASS')
        
        if resize_factor != 1.0:
            print(f'* Resizing :: x{resize_factor} :: {self.image.shape}->', end='')
            self.image = cv2.resize(self.image, dsize=(0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
            print(self.image.shape)
        else:
            print('* Resizing :: PASS')
        self.shape = self.image.shape

            
        if denoise:
            print(f'* Denoising with <cv2.fastNlMeansDenoising>...', end='')
            self.image = cv2.fastNlMeansDenoising(sefl.image)
            print('Done')
        else:
            print('* Denoising :: PASS')
            
        if threshold is None:
            print("* Inverse binary thresholding :: Otsu's method")
            ret, self.image = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        else:
            print(f'* Inverse binary thresholding :: threshold={threshold}') 
            ret, self.image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY_INV)
        return
    
    def __sort_objects(self, position):
        center = np.array([[i, x+w/2, y+h/2] for i, (x, y, w, h) in enumerate(position)], dtype=int)
        y_list = sorted(center[:,2])
        tmp = np.diff(y_list)
        interval = np.mean(tmp[np.where(tmp>np.mean(tmp)-0.5*np.std(tmp))])
        
        idx_y = np.round((center[:,2]-y_list[0])/interval)
        idx = np.lexsort((center[:,1], idx_y))
        
        position = position[idx]
        return position
    
    def __get_system_position(self, width_threshold=0.6, sort=True):
        _, _, stats1, _ = cv2.connectedComponentsWithStats(self.image)
        
        #Overlap all objects
        mask = np.zeros(self.shape, dtype=np.uint8)
        for x, y, w, h, _ in stats1[1:]:
            mask[y:y+h, x:x+w] = 255
        _, _, stats2, _ = cv2.connectedComponentsWithStats(mask)
        stats2 = stats2[1:]
        
        #Filtering depends on width and height of objects
        filter_width = stats2[:,2]>width_threshold*self.shape[1]
        filter_height = stats2[:,3]>np.mean(stats2[filter_width,3])-np.std(stats2[filter_width,3])
        idx = np.logical_or(filter_width, filter_height)
        
        system_position = stats2[idx, :4]
        if sort:
            system_position = self.__sort_objects(system_position)
        return system_position
    
    def create_system(self, sheet, width_threshold=0.6):
        sys_pos = self.__get_system_position(width_threshold=width_threshold, sort=True)
        print(f'>>> Extracting systems : {len(sys_pos)} systems')
        systems = []
        for i, (x, y, w, h) in enumerate(sys_pos):
            system = System(sheet, (x, y, w, h), order=i)
            systems.append(system)
            
        self.systems = systems
        self.num_systems = len(systems)
        
        return systems
        
#Class : System ---------------------------------------------------------------------------------------------------------#
class System():
    def __init__(self, sheet, position, order='Unknown'):
        assert len(position) == 4, f'System position is not acceptable value it must have 4 value (x, y, w, h) :: position'
        self.sheet = sheet
        self.order = order
        self.x, self.y, self.w, self.h = tuple(position)
        self.basis = position
        self.image = self.sheet.image[self.y:self.y+self.h, self.x:self.x+self.w]
        self.shape = self.image.shape
        print(f'* System Object Created :: System #{order} of [...{self.sheet.path[-10:]}] :: (x, y, w, h) = {self.basis}')
        
        self.grid = {'row':{'val':None, 'idx':None}, 'col':{'val':None, 'idx':None}}   
        self.histogram = {'row': None, 'col':None}
        self.vline_idx, self.vline_thickness = None, None
        self.hline_idx, self.hline_thickness = None, None
        self.image_line_removed = self.image.copy()
        self.measures = []
        self.staves = []
    
    def get_pixel_histogram(self, depend_on='row', grid=[0.3, 0.5, 0.7, 0.8, 0.9]): 
        assert depend_on in ['row', 'col'], 'Histograms should be created depending on rows or columns'
        axis = 1 if depend_on == 'row' else 0
        zerone_image = np.uint8(np.ceil(self.image/255))        
        hist_value = np.sum(zerone_image, axis=axis)
        hist_image = np.zeros(self.shape, dtype=np.uint8)
                
        total_pixel = 0
        for i, num in enumerate(hist_value):
            if axis == 1: hist_image[i, :num] = 255
            else: hist_image[:num, i] = 255
            total_pixel += num
        
        grid_idx = []
        hist_value2 = np.sum(np.uint8(np.ceil(hist_image/255)), axis=int(not axis))
        pixel_sum, i, j = 0, 0, 0
        already_sum = False
        while i < hist_value2.size:
            if not already_sum: pixel_sum += float(hist_value2[i]/total_pixel)
            if pixel_sum > grid[j]:
                grid_idx.append(i)
                already_sum = True
                j += 1
                if j >= len(grid):
                    break
            else:
                already_sum = False
                i += 1
                
        self.grid[depend_on]['val'] = grid
        self.grid[depend_on]['idx'] = grid_idx
        self.histogram[depend_on] = hist_image
        
        return hist_image, grid, grid_idx
        
    def get_line_info(self, hist_by='row'):
        if hist_by == 'row':
            line_pos = np.squeeze(self.histogram[hist_by][:, self.grid[hist_by]['idx'][-1]]).nonzero()[0]
        elif hist_by == 'col':
            line_pos = np.squeeze(self.histogram[hist_by][self.grid[hist_by]['idx'][-1], :]).nonzero()[0]
        else:
            line_pos = hist_by
              
        line_idx, line_thickness = [], []
        cur_thickness = 1
        for i in range(len(line_pos)):
            if cur_thickness == 1:
                line_idx.append(line_pos[i])
            
            if i == int(len(line_pos)-1):
                line_thickness.append(cur_thickness)
            elif line_pos[i]+1 == line_pos[i+1]:
                cur_thickness += 1
            else:
                line_thickness.append(cur_thickness)
                cur_thickness = 1
        
        line_idx, line_thickness = np.array(line_idx), np.array(line_thickness)
        
        if hist_by == 'row':
            self.hline_idx, self.hline_thickness = line_idx, line_thickness
        elif hist_by == 'col':
            self.vline_idx, self.vline_thickness = line_idx, line_thickness
        return line_idx, line_thickness
    
    def check_bar(self, row_fromto=None):
        candidates = copy.deepcopy(self.vline_idx)
        if row_fromto is None:
            row_fromto = (self.hline_idx[0], self.hline_idx[-1]) 
        
        pick_idx = np.min(self.image[row_fromto[0]:row_fromto[1]+1, candidates], axis=0).nonzero()[0]
        
        self.vline_idx = list(np.array(self.vline_idx)[pick_idx])
        self.vline_thickness = list(np.array(self.vline_thickness)[pick_idx])
        return self.vline_idx, self.vline_thickness
    
    def clustering_staff(self, hline_pos=None, thickness=None):
        if hline_pos is None:
            hline_pos = self.hline_idx
        if thickness is None:
            thickness = self.hline_thickness
        
        interval = np.diff(hline_pos)
        val, edges = np.histogram(interval, bins=10, range=(0, 50))
        
        idx_sort = np.argsort(val)
        i = idx_sort[-1]
        threshold = edges[min(i+2, len(edges-1))]
        
        '''
        if val[i]>0.8*np.sum(val):
            threshold = edges[i+1]
        else:
            j = idx_sort[-2]
            threshold = max(edges[i+1], edges[j+1])
        '''
        
        cluster_list = []
        idx = [i for i in range(len(hline_pos))]
        cluster = []
                
        while idx != []:
            i = idx.pop(0)
            if i == 0:
                cluster.append(i)
                continue
                
            if interval[i-1] > threshold:
                cluster_list.append(copy.deepcopy(cluster))
                cluster = [i, ]
            else:
                cluster.append(i)
        if cluster != []:
            cluster_list.append(cluster)
        
        staves = []
        for i, idx_list in enumerate(cluster_list):
            idx = np.array(idx_list)
            staves.append(StaffLines(hline_pos[idx], thickness[idx], order=i))
        self.staves = staves
            
       
        return staves
        
    '''
    def create_staff_info(self, hline_pos=None, thickness=None):
        if hline_pos is None:
            hline_pos = self.hline_idx
        if thickness is None:
            thickness = self.hline_thickness
            
        interval = np.diff(hline_pos)
        
        threshold = np.median(interval)+np.std(interval)
        split_idx = np.where(interval>threshold)[0]
        split_idx = np.append(np.append(0, split_idx+1), len(hline_pos))
        
        for i in range(len(split_idx)-1):
            j = i+1
            sep_pos = hline_pos[split_idx[i]:split_idx[j]]
            if sep_pos == []:
                continue
            self.staves.append(Staff(line_pos=sep_pos, thickness=thickness[split_idx[i]:split_idx[j]]))
        return self.staves
    '''
    
    def __remove_single_line(self, image_, line_from, thickness, b=255, w=0):
        image = image_.copy()
        image = np.pad(image, ((1,1),(1,1)), 'constant', constant_values=0)
        height, width = image.shape
        line_from = line_from+1
        line_to = line_from+thickness-1
        
        for col in range(width):
            if image[line_from, col] == b or image[line_to,col] == b:
                if image[line_from-1, col] == w and image[line_to+1, col] == w:
                    for i in range(thickness):
                        image[line_from+i, col] = w

                elif image[line_from-1, col] == w and image[line_to+1, col] == b:
                    if(col>0 and image[line_to+1, col-1] != b) and (col<width-1 and image[line_to+1, col+1] != b):
                        thick = thickness+1
                        if thick<1:
                            thick = 1
                        for i in range(int(thick)):
                            image[line_from+i, col] = w

                elif image[line_from-1, col] == b and image[line_to+1, col] == w:
                    if (col>0 and image[line_from-1, col-1] != b) and (col<width-1 and image[line_from-1, col+1] != b):
                        thick = thickness+1
                        if thick<1:
                            for i in range(int(thick)):
                                image[line_to-i, col] = w
        
        return image[1:-1,1:-1]

    def remove_lines(self, line_idxs, thicknesses, direction='hor', b=255, w=0):
        if direction == 'hor':
            image = self.image_line_removed.copy()
        elif direction == 'ver':
            image = self.image_line_removed.copy().T
        else:
            return
        
        for line_from , thickness in zip(line_idxs, thicknesses):
            image = self.__remove_single_line(image, line_from, thickness, b=b, w=w)
        if direction == 'hor':
            self.image_line_removed = image
            return image
        else:
            self.image_line_removed = image.T
            return image.T

    def reset_line_removed_image(self):
        self.image_line_removed = self.image.copy()
        return self.image_line_removed
    
    def split_into_measures(self, image=None, vline_pos=None, threshold_px=100):
        if image is None:
            image = self.image
        elif image == 'removed':
            image = self.image_line_removed
            
        if vline_pos is None:
            vline_pos = self.vline_idx
        vline_pos = list(vline_pos)
        
        measures = []
        vline_pos.insert(0, 0)
        vline_pos.append(self.shape[1])
        
        for i in range(len(vline_pos)-1):
            j = i+1
            img = image[:, vline_pos[i]:vline_pos[j]]
            _, width = img.shape
            if width >= threshold_px:
                measures.append(img)
                
        self.measures = measures
        return measures
    
    def create_measure(self, system, threshold_px=100):
        vline_pos = list(self.vline_idx)
        
        measures = []
        vline_pos.insert(0, 0)
        vline_pos.append(system.shape[1])
        
        measure_id = 0
        for i in range(len(vline_pos)-1):
            j = i+1
            if vline_pos[j]-vline_pos[i] >= threshold_px:
                measure = Measure(system, np.array([vline_pos[i], 0, vline_pos[j]-vline_pos[i], system.shape[0]]), order=measure_id)
                measures.append(measure)
                measure_id += 1
                
        self.measures = measures
        return measures
            

#Class : Staff lines ---------------------------------------------------------------------------------------------------------#
class StaffLines():
    def __init__(self, line_pos, thickness, staff_type=None, order='Unknown'):
        self.pos = line_pos
        self.thickness = thickness
        
        self.interval = int(np.round(np.mean(np.diff(line_pos))))
        self.num_line = self.pos.size
        self.staff_type = staff_type
        self.order = order
        
        print(f'* StaffLines Object Created :: Staff Line Cluster #{order} with {self.num_line} lines :: y = {self.pos}, average_interval = {self.interval}')
    
#Class : Measures ---------------------------------------------------------------------------------------------------------#
class Measure():
    def __init__(self, system, position, order='Unknown'):
        assert len(position) == 4, f'Measure position is not acceptable value it must have 4 value (x, y, w, h) :: position'
        self.system = system
        self.order = order
        self.x, self.y, self.w, self.h = tuple(position)
        self.basis = position
        self.image = self.system.image[self.y:self.y+self.h, self.x:self.x+self.w]
        self.image_line_removed = self.system.image_line_removed[self.y:self.y+self.h, self.x:self.x+self.w]

        self.shape = self.image.shape
        print(f'* Measure Object Created:: Measure #{order} of System #{self.system.order} of [...{self.system.sheet.path[-10:]}] :: (x, y, w, h) = {self.basis}')
        self.staves = self.system.staves

    def extract_objects(self, blur=):
        _, _, stats1, _ = cv2.connectedComponentsWithStats(self.image)
        
        #Overlap all objects
        mask = np.zeros(self.shape, dtype=np.uint8)
        for x, y, w, h, _ in stats1[1:]:
            mask[y:y+h, x:x+w] = 255
        _, _, stats2, _ = cv2.connectedComponentsWithStats(mask)
        stats2 = stats2[1:]
        
        #Filtering depends on width and height of objects
        filter_width = stats2[:,2]>width_threshold*self.shape[1]
        filter_height = stats2[:,3]>np.mean(stats2[filter_width,3])-np.std(stats2[filter_width,3])
        idx = np.logical_or(filter_width, filter_height)
        
        system_position = stats2[idx, :4]
        if sort:
            system_position = self.__sort_objects(system_position)
        return system_position


#Class : Template ---------------------------------------------------------------------------------------------------------#

class Template():
    def __init__(self, template_path):
        assert os.path.exists(template_path) and template_path.endswith('.jpg'), 'File Not Exists or Not suitable extension(.jpg)'
        self.path = template_path
        self.info = self.get_template_info(self.path)
        self.image = self.binarize(cv2.imread(self.path, cv2.IMREAD_GRAYSCALE), threshold=50)
        self.resize_image = {'factor':1.0, 'image':self.image}
        self.shape = self.image.shape
        
        self.significant_scale_factor = self.__get_significant_scale_factor(self.shape)
        self.ssf = self.significant_scale_factor
        
    def get_template_info(self, path=None):
        if path is None:
            path = self.path
        else:
            assert os.path.exists(path) and path.endswith('.jpg'), 'File Not Exists or Not suitable extension(.jpg)'
            
        _, file_full_name = os.path.split(path)
        file_name, file_ext = os.path.splitext(file_full_name)
        comp = file_name.split('_')
        
        info = {'type':None, 'src':None, 'staff_interval':None, 'obj_id':None, 'idx':None}
        
        info['type'] = comp[0]
        if comp[0] == 'Time':
            info['type'] = '_'.join(comp[0:3])
        info['src'] = int(comp[-4])
        info['staff_interval'] = int(comp[-3])
        info['obj_id'] = int(comp[-2])
        info['idx'] = int(comp[-1])
        
        self.info = info
        return info
    
    def resize(self, factor=1.0):
        image = cv2.resize(self.image, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        image = self.binarize(image)
        
        self.resize_image['factor'] = factor
        self.resize_image['image'] = image
        return image

    def binarize(self, image=None, threshold=125):
        if image is None:
            image = self.image
        
        mask = image>threshold
        image[mask] = 255
        image[np.logical_not(mask)] = 0
        return image

    def __get_significant_scale_factor(self, image_size):
        ssf = lambda x: (x+1)/x-1
        h, w = image_size
        ssf_h = np.round(ssf(h), 2)
        ssf_w = np.round(ssf(w), 2)
        return (ssf_h, ssf_w)
    
    def find_best_match(self, image, resize_factor=1.0, method=cv2.TM_SQDIFF_NORMED, return_result=True):
        if resize_factor != 1.0:
            template = self.resize(resize_factor)
        else:
            template = self.image
        
        template_height, template_width = template.shape
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            result = cv2.matchTemplate(image, template, method, mask=template)
        else:
            result = cv2.matchTemplate(image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            value = min_val
        else:
            top_left = max_loc 
            value = max_val
            
        bbox = np.array([value, top_left[0], top_left[1], top_left[0]+template_width, top_left[1]+template_height])
        if return_result:
            return bbox, result
        else:
            return bbox
            
    def find_high_matches(self, image, resize_factor=1.0, method=cv2.TM_SQDIFF_NORMED, threshold=0.7, return_result=True):
        if resize_factor != 1.0:
            template = self.resize(resize_factor)
        else:
            template = self.image
            
        template_height, template_width = template.shape
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            result = cv2.matchTemplate(image, template, method, mask=template)
        else:
            result = cv2.matchTemplate(image, template, method)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            locations = np.where(result<threshold)
        else:
            locations = np.where(result>threshold)

        score = result[locations]
        locations = np.array([[y, x] for x, y in zip(*locations)])
        if locations.size != 0:
            bboxes = np.concatenate([score.reshape((-1,1)), locations, locations+np.array([template_width, template_height])], axis=1)
        else:
            bboxes = np.array([]).reshape((0, 5))
        if return_result:
            return bboxes, result
        else:
            return bboxes


