import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import os, copy
import time

import OMRobjects as OMR

template_path = {
    'Notes' : {
        'Full' : {},
        'Half' : {'TM_SQDIFF_NORMED':0.03, 'TM_CCOEFF_NORMED':0.7}, 
        'Quarter' : {'TM_SQDIFF_NORMED':0.02, 'TM_CCOEFF_NORMED':0.7}
    }, 
    'Rests' : {
        'Full' : {},
        'Half' : {}, 
        'Quarter' : {},
        'Eighth' : {},
        'Sixteenth' : {},
    },
    'KeySignatures' : {
        'Flat' : {}, 
        'Sharp' : {},
        'Natural' : {}
    },
    'Times' : {
        'Time_4_4' : {}, 
        'Time_3_4' : {},
        'Time_2_4' : {},
        'Time_6_8' : {},
        'Time_8_12' : {},
    }, 
    'Cleves' : {
        'Treble' : {'TM_SQDIFF_NORMED':0.2, 'TM_CCOEFF_NORMED':0.7}, 
        'Bass' : {}, 
        'Alto' : {}
    },
    'Etc':{}
}