import cv2
import numpy as np

doc = cv2.imread('document1.png')
gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)

#blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=4)

contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# 2 - Image
# 5 - Table (2 row)
# 7 - Table (More row)

counter = 0

for contour in contours:
    color = (255, 0, 0)
    x,y,w,h = cv2.boundingRect(contour)
    image = thresh[y:y+h, x:x+w]
    
    horizontal = image.copy()
    
    nrow, ncol = horizontal.shape
    horizontal_size = ncol // 30
    
    if horizontal_size % 2 == 0:
        horizontal_size += 1
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
   
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    
#    edges = cv2.Canny(image, 150, 200, apertureSize = 7)
#    cv2.imshow('hori', horizontal)
#    cv2.waitKey()
    minLineLength = 100
    maxLineGap = 0
    lines = cv2.HoughLinesP(horizontal, 1, np.pi / 180, 250, minLineLength, maxLineGap, 10)

    if lines is not None and len(lines) > 4:
        counter += 1
        color = (20, 20, 100)
        cv2.putText(doc, f"Table {counter}" , (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    cv2.rectangle(doc, (x, y), (x + w, y + h), color, 2)  

cv2.namedWindow('image', cv2.WINDOW_NORMAL)                            
doc = cv2.resize(doc, (1000, 900))  
cv2.imshow('image', doc)
cv2.waitKey(0)
cv2.destroyAllWindows()