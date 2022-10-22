import numpy as np
import cv2
import util_functions as fs

path = {
    "treble":[
        "../data/symbols_old/clef/treble_1.jpg",
        "../data/symbols_old/clef/treble_2.jpg"
        ],
    "bass":[
        "../data/symbols_old/clef/bass_1.jpg"
        ],
    "sharp": [
        "../data/symbols_old/sharp-line.png",
        "../data/symbols_old/sharp-space.png"
    ],
    "flat": [
        "../data/symbols_old/flat-line.png",
        "../data/symbols_old/flat-space.png"
    ],
    "quarter_note": [
        "../data/symbols_old/note/quarter.png",
        "../data/symbols_old/note/solid-note.png"
    ],
    "half_note": [
        "../data/symbols_old/note/half-space.png",
        "../data/symbols_old/note/half-note-line.png",
        "../data/symbols_old/note/half-line.png",
        #"data/symbols_old/note/half-note-space.png"
    ],
    "whole_note": [
        "../data/symbols_old/note/whole-space.png",
        #"data/symbols_old/note/whole-note-line.png",
        #"data/symbols_old/note/whole-line.png",
        "../data/symbols_old/note/whole-note-space.png"
    ],
    "eighth_rest": [
        "../data/symbols_old/rest/eighth_rest.jpg"
    ],
    "quarter_rest": [
        "../data/symbols_old/rest/quarter_rest.jpg"
    ],
    "half_rest": [
        "../data/symbols_old/rest/half_rest_1.jpg",
        "../data/symbols_old/rest/half_rest_2.jpg"
    ],
    "whole_rest": [
        "../data/symbols_old/rest/whole_rest.jpg"
    ], 
    "barline":[
        "../data/symbols_old/barline/barline_1.jpg",
         "../data/symbols_old/barline/barline_2.jpg",
         "../data/symbols_old/barline/barline_3.jpg",
         "../data/symbols_old/barline/barline_4.jpg"
    ], 
    "time":{
        "common":[
            "../data/symbols_old/time/common.jpg"
            ],
        "44":[
            "../data/symbols_old/time/44.jpg"
            ], 
        "34":[
            "../data/symbols_old/time/34.jpg"
            ],
        "24":[
            "../data/symbols_old/time/24.jpg"
            ],
        "68":[
            "../data/symbols_old/time/68.jpg"
            ]
        }
}

get_image_by_dict = lambda x: {
    k:[fs.get_images(pth, resize_factor=1.0, threshold=200, log=False) for pth in v] for k, v in x.items() if k != "time"
}


template = get_image_by_dict(path)
template["time"] = get_image_by_dict(path["time"])