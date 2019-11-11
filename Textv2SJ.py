import cv2
import numpy as np

doc = cv2.imread('twoTables.png')
gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)

gray = cv2.bitwise_not(gray)
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

horizontal = bw.copy()
vertical = bw.copy()

cols = horizontal.shape[1]
horizontal_size = cols // 30
# Create structure element for extracting horizontal lines through morphology operations
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
rHorizontalStructure = np.rot90(horizontalStructure,2)
# Apply morphology operations
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, rHorizontalStructure)

rows = vertical.shape[0]
verticalsize = rows // 30
# Create structure element for extracting vertical lines through morphology operations
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
rVerticalStructure = np.rot90(verticalStructure,2)
# Apply morphology operations
vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, rVerticalStructure)

mask = horizontal + vertical

sE = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
erosion = cv2.erode(mask,sE)
opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,sE)

joints = cv2.bitwise_and(horizontal, vertical, 100)
print(len(erosion[erosion > 0]))
#cv2.imshow('joints', joints)
#cv2.waitKey()

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(doc, (x, y), (x + w, y + h), (0,255,0), 3)

#contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#proposals = list(filter(
#    lambda x: x[2] > 40 and x[3] > 40,
#    map(cv2.boundingRect, contours)
#))
#
#res = []
#
#for p in proposals:
#    x, y, w, h = p
#    cv2.rectangle(doc, (x, y), (x + w, y + h), (36,255,12), 2)
#    res.append((x, y, x+w, y+h))

#cv2.namedWindow('image', cv2.WINDOW_NORMAL)                            
#doc = cv2.resize(doc, (1000, 900))  
#cv2.imshow('image', doc)
#cv2.waitKey()
#cv2.destroyAllWindows()

# Show extracted lines
#cv2.imshow("horizontal", horizontal)
#cv2.imshow("vertical", vertical)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)                            
mask = cv2.resize(mask, (1000, 900))  
cv2.imshow("mask", mask)
cv2.imshow("opening",erosion)
cv2.waitKey()
cv2.destroyAllWindows()
