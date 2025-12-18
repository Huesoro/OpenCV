import cv2
from util import  get_limits
from PIL import Image

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Handle red color wrapping (red exists at both ends of HSV hue range)
    # Tighten S/V to avoid skin tones
    lower_limit_red1 = (0, 100, 70)
    upper_limit_red1 = (10, 255, 255)
    lower_limit_red2 = (170, 100, 70)
    upper_limit_red2 = (180, 255, 255)
    red_mask1 = cv2.inRange(hsvImage, lower_limit_red1, upper_limit_red1)
    red_mask2 = cv2.inRange(hsvImage, lower_limit_red2, upper_limit_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Build a skin mask in YCrCb space and subtract from red to avoid body/face
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_lower = (0, 133, 77)
    skin_upper = (255, 173, 127)
    skin_mask = cv2.inRange(ycrcb, skin_lower, skin_upper)
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(skin_mask))
    
    # Light morphological clean-up on red mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    lower_limit_green, upper_limit_green = get_limits(green)
    lower_limit_blue, upper_limit_blue = get_limits(blue)
    # Apply the masks to all colors
    green_mask = cv2.inRange(hsvImage, lower_limit_green, upper_limit_green)
    blue_mask = cv2.inRange(hsvImage, lower_limit_blue, upper_limit_blue)
    # Suppress skin for all colors
    inv_skin = cv2.bitwise_not(skin_mask)
    green_mask = cv2.bitwise_and(green_mask, inv_skin)
    blue_mask = cv2.bitwise_and(blue_mask, inv_skin)
    # Morphological cleanup (open then close)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Suppress skin regions from green/blue as well
    inv_skin = cv2.bitwise_not(skin_mask)
    green_mask = cv2.bitwise_and(green_mask, inv_skin)
    blue_mask = cv2.bitwise_and(blue_mask, inv_skin)
    mask_red = Image.fromarray(red_mask)
    bbox_red = mask_red.getbbox()
    mask_red = Image.fromarray(red_mask)
    bbox_red = mask_red.getbbox()
    mask_green = Image.fromarray(green_mask)
    bbox_green = mask_green.getbbox()
    mask_blue = Image.fromarray(blue_mask)
    bbox_blue = mask_blue.getbbox()
    bboxes = {'red': bbox_red, 'green': bbox_green, 'blue': bbox_blue}
    # SHOW THE DIFFERENT MASKS
    cv2.imshow('Skin Mask', skin_mask)
    cv2.imshow('Red Mask (no skin)', red_mask)
    cv2.imshow('Green Mask (no skin)', green_mask)
    cv2.imshow('Blue Mask (no skin)', blue_mask)

    # Area-based filtering to ignore tiny noise and huge regions
    h, w = frame.shape[:2]
    frame_area = w * h
    min_area = int(0.003 * frame_area)   # ~1% of frame
    max_area = int(0.50 * frame_area)    # ignore very large blobs

    for color, box in bboxes.items():
        if box is None:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area < min_area or area > max_area:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f'Detected: {color}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()