import numpy as np
import cv2

def get_limits(color):
    c = np.uint8([[color]]) # Convert to hsv
    # The input "color" is provided as RGB tuple constants in detector.py
    # Convert from RGB to HSV to derive an appropriate hue window.
    hsv_color = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)[0][0]

    # Build a reasonable window around the hue with relaxed S/V minima
    lower_limit = np.array([max(int(hsv_color[0]) - 20, 0), 50, 50])
    upper_limit = np.array([min(int(hsv_color[0]) + 20, 179), 255, 255])
    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)
    return lower_limit, upper_limit