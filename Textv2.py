import cv2
import numpy as np

document = cv2.imread('document1.png')
gray_doc = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray_doc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=4)

contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

counter = 0

for contour in contours:
    color = (255, 0, 0)
    x,y,w,h = cv2.boundingRect(contour)
    contour_image = thresh[y:y+h, x:x+w]
    
    image_copy = contour_image.copy()
    nrow, ncol = image_copy.shape
    ncol = ncol // 30
    
    if ncol % 2 == 0:
        ncol += 1
        
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (ncol, 1))
    hori_lines = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, SE)
    
    lines = cv2.HoughLinesP(hori_lines, 1, np.pi / 180, 250, 100, 0, 10)

    if lines is not None and len(lines) > 4:
        counter += 1
        color = (20, 20, 100)
        cv2.putText(document, f"Table {counter}" , (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    cv2.rectangle(document, (x, y), (x + w, y + h), color, 2)  

cv2.namedWindow('Text & Table Localization', cv2.WINDOW_NORMAL)                            
document = cv2.resize(document, (1000, 900))  

cv2.imshow('Text & Table Localization', document)
cv2.waitKey(0)
cv2.destroyAllWindows()