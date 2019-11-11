import cv2
import numpy as np

doc = cv2.imread('document1.png')
gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=4)

contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#cnts = cnts[0] if len(cnts) == 2 else cnts[1]

## 2 - Image
## 5 - Table (2 row)
## 7 - Table (More row)

#x,y,w,h = cv2.boundingRect(cnts[5])
#image = thresh[y:y+h, x:x+w]
#edges = cv2.Canny(image,100,200,apertureSize = 3)
#    
#minLineLength = 100
#maxLineGap = 0
#lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 350, minLineLength, maxLineGap, 10)
#
#if lines is not None:
#    for x in range(0, len(lines)):
#        for x1,y1,x2,y2 in lines[x]:
#            cv2.line(doc,(x1,y1),(x2,y2),(0,255,0),2)
#            
#cv2.imshow('Hough', doc)
#cv2.waitKey()
counter = 0

for contour in contours:
    color = (255, 0, 0)
    x,y,w,h = cv2.boundingRect(contour)
    image = thresh[y:y+h, x:x+w]
    
    edges = cv2.Canny(image, 100, 200, apertureSize = 3)
    
    minLineLength = 100
    maxLineGap = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 350, minLineLength, maxLineGap, 10)
    
    if lines is not None and len(lines) > 4:
        counter += 1
        color = (20, 20, 100)
        cv2.putText(doc, f"Table {counter}" , (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    cv2.rectangle(doc, (x, y), (x + w, y + h), color, 2)  
    
       
#    cv2.imshow('ROI', image)
#    cv2.waitKey()
##    cv2.rectangle(doc, (x, y), (x + w, y + h), (36,255,12), 2)  
#    edges = cv2.Canny(image,100,200,apertureSize = 3)
  
##    cv2.imshow('ROI', edges)
#    cv2.waitKey()
#    
#    minLineLength = 30
#    maxLineGap = 10
#    lines = cv2.HoughLinesP(edges,1, np.pi/180,15,minLineLength,maxLineGap)
#    
#    if lines is not None:
#        for x in range(0, len(lines)):
#            for x1,y1,x2,y2 in lines[x]:
#                cv2.line(doc,(x1,y1),(x2,y2),(0,255,0),2)

#new_image_matrix, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#proposals = list(filter(
#    lambda x: x[2] > 40 and x[3] > 40,
#    map(cv2.boundingRect, contours)
#))

#cv2.imshow('edges',edges)
#cv2.waitKey(0)

#res = []
#
#for p in proposals:
#    x, y, w, h = p
#    cv2.rectangle(doc, (x, y), (x + w, y + h), (36,255,12), 2)
#    res.append((x, y, x+w, y+h))

cv2.namedWindow('image', cv2.WINDOW_NORMAL)                            
doc = cv2.resize(doc, (1000, 900))  
cv2.imshow('image', doc)
cv2.waitKey(0)
cv2.destroyAllWindows()