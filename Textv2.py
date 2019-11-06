import cv2

doc = cv2.imread('twoTables.png')
gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=4)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(doc, (x, y), (x + w, y + h), (36,255,12), 2)

new_image_matrix, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
proposals = list(filter(
    lambda x: x[2] > 40 and x[3] > 40,
    map(cv2.boundingRect, contours)
))

res = []

for p in proposals:
    x, y, w, h = p
    cv2.rectangle(doc, (x, y), (x + w, y + h), (36,255,12), 2)
#    res.append((x, y, x+w, y+h))

cv2.namedWindow('image', cv2.WINDOW_NORMAL)                            
doc = cv2.resize(doc, (1000, 900))  
cv2.imshow('image', doc)
cv2.waitKey(0)
cv2.destroyAllWindows()