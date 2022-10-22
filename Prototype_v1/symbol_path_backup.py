import util_functions as fs
import numpy as np
import cv2

clef_paths = {
    "treble": [
        "../data/symbols_old/clef/treble_1.jpg",
        "../data/symbols_old/clef/treble_2.jpg"
    ],
    "bass": [
        "../data/symbols_old/clef/bass_1.jpg"
    ]
}

accidental_paths = {
    "sharp": [
        "../data/symbols_old/sharp-line.png",
        "../data/symbols_old/sharp-space.png"
    ],
    "flat": [
        "../data/symbols_old/flat-line.png",
        "../data/symbols_old/flat-space.png"
    ]
}

note_paths = {
    "quarter": [
        #"../data/symbols_old/note/quarter.png",
        "../data/symbols_old/note/solid-note.png"
    ],
    "half": [
        "../data/symbols_old/note/half-space.png",
        "../data/symbols_old/note/half-note-line.png",
        "../data/symbols_old/note/half-line.png",
        #"../data/symbols_old/note/half-note-space.png"
    ],
    "whole": [
        "../data/symbols_old/note/whole-space.png",
        #"../data/symbols_old/note/whole-note-line.png",
        #"../data/symbols_old/note/whole-line.png",
        "../data/symbols_old/note/whole-note-space.png"
    ]
}
rest_paths = {
    "eighth": ["../data/symbols_old/rest/eighth_rest.jpg"],
    "quarter": ["../data/symbols_old/rest/quarter_rest.jpg"],
    "half": ["../data/symbols_old/rest/half_rest_1.jpg",
            "../data/symbols_old/rest/half_rest_2.jpg"],
    "whole": ["../data/symbols_old/rest/whole_rest.jpg"]
}

flag_paths = ["../data/symbols_old/flag/eighth_flag_1.jpg",
                "../data/symbols_old/flag/eighth_flag_2.jpg",
                "../data/symbols_old/flag/eighth_flag_3.jpg",
                "../data/symbols_old/flag/eighth_flag_4.jpg",
                "../data/symbols_old/flag/eighth_flag_5.jpg",
                "../data/symbols_old/flag/eighth_flag_6.jpg"]

barline_paths = ["../data/symbols_old/barline/barline_1.jpg",
                 "../data/symbols_old/barline/barline_2.jpg",
                 "../data/symbols_old/barline/barline_3.jpg",
                 "../data/symbols_old/barline/barline_4.jpg"]


#---------------------------------------------------------
# Clefs
clef_imgs = {
    "treble": [fs.get_images(clef_file, resize_factor=1.0, threshold=200, log=False) for clef_file in clef_paths["treble"]],
    "bass": [fs.get_images(clef_file, resize_factor=1.0, threshold=200, log=False) for clef_file in clef_paths["bass"]]
}

# Time Signatures


time_imgs = {
    "common": [fs.get_images(time, resize_factor=1.0, threshold=200, log=False) for time in ["../data/symbols_old/time/common.jpg"]],
    "44": [fs.get_images(time, resize_factor=1.0, threshold=200, log=False) for time in ["../data/symbols_old/time/44.jpg"]],
    "34": [fs.get_images(time, resize_factor=1.0, threshold=200, log=False) for time in ["../data/symbols_old/time/34.jpg"]],
    "24": [fs.get_images(time, resize_factor=1.0, threshold=200, log=False) for time in ["../data/symbols_old/time/24.jpg"]],
    "68": [fs.get_images(time, resize_factor=1.0, threshold=200, log=False) for time in ["../data/symbols_old/time/68.jpg"]]
}

# Accidentals
sharp_imgs = [fs.get_images(sharp_file, resize_factor=1.0, threshold=200, log=False) for sharp_file in accidental_paths["sharp"]]
flat_imgs = [fs.get_images(flat_file, resize_factor=1.0, threshold=200, log=False) for flat_file in accidental_paths["flat"]]

# Notes
quarter_note_imgs = [fs.get_images(quarter, resize_factor=1.0, threshold=200, log=False) for quarter in note_paths["quarter"]]
half_note_imgs = [fs.get_images(half, resize_factor=1.0, threshold=200, log=False) for half in note_paths["half"]]
whole_note_imgs = [fs.get_images(whole, resize_factor=1.0, threshold=200, log=False) for whole in note_paths['whole']]

# Rests
eighth_rest_imgs = [fs.get_images(eighth, resize_factor=1.0, threshold=200, log=False) for eighth in rest_paths["eighth"]]
quarter_rest_imgs = [fs.get_images(quarter, resize_factor=1.0, threshold=200, log=False) for quarter in rest_paths["quarter"]]
half_rest_imgs = [fs.get_images(half, resize_factor=1.0, threshold=200, log=False) for half in rest_paths["half"]]
whole_rest_imgs = [fs.get_images(whole, resize_factor=1.0, threshold=200, log=False) for whole in rest_paths['whole']]

# Eighth Flag
eighth_flag_imgs = [fs.get_images(flag, resize_factor=1.0, threshold=200, log=False) for flag in flag_paths]

# Bar line
bar_imgs = [fs.get_images(barline, resize_factor=1.0, threshold=200, log=False) for barline in barline_paths]