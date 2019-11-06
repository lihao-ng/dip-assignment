import cv2

#image = cv2.imread('Page1.png')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray, (9,9), 0)
#thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)
#
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
#dilate = cv2.dilate(thresh, kernel, iterations=4)
#
#cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
#ROI_number = 0
#for c in cnts:
#    area = cv2.contourArea(c)
#    if area > 10000:
#        x,y,w,h = cv2.boundingRect(c)
#        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
#        # ROI = image[y:y+h, x:x+w]
#        # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
#        # ROI_number += 1
##
##cv2.imshow('thresh', thresh)
##cv2.imshow('dilate', dilate)
##cv2.imshow('image', image)
#cv2.waitKey()
#cv2.destroyAllWindows()

#image = cv2.imread('Page1.png')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)
#
#image, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE )
#
#for c in contours:
#    rect = cv2.boundingRect(c)
#    if rect[2] < 50 or rect[3] < 50 : continue
#
#    print (cv2.contourArea(c))
#    x,y,w,h = rect
#    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#
##kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
##dilate = cv2.dilate(thresh, kernel, iterations=4)
#
#cv2.imshow('image', )
#cv2.waitKey()
#cv2.destroyAllWindows()

#import cv2
#import numpy as np
#import matplotlib.pyplot as plt

#image = cv2.imread('page1.png', 0)
#
## find lines by horizontally blurring the image and thresholding
#blur = cv2.blur(image, (91,9))
#b_mean = np.mean(blur, axis=1)/256
#
## hist, bin_edges = np.histogram(b_mean, bins=100)
## threshold = bin_edges[66]
#threshold = np.percentile(b_mean, 66)
#t = b_mean > threshold
#'''
#get the image row numbers that has text (non zero)
#a text line is a consecutive group of image rows that 
#are above the threshold and are defined by the first and 
#last row numbers
#'''
#tix = np.where(1-t)
#tix = tix[0]
#lines = []
#start_ix = tix[0]
#for ix in range(1, tix.shape[0]-1):
#    if tix[ix] == tix[ix-1] + 1:
#        continue
#    # identified gap between lines, close previous line and start a new one
#    end_ix = tix[ix-1]
#    lines.append([start_ix, end_ix])
#    start_ix = tix[ix]
#end_ix = tix[-1]
#lines.append([start_ix, end_ix])
#
#l_starts = []
#for line in lines:
#    center_y = int((line[0] + line[1]) / 2)
#    xx = 500
#    for x in range(0,500):
#        col = image[line[0]:line[1], x]
#        if np.min(col) < 64:
#            xx = x
#            break
#    l_starts.append(xx)
#
#median_ls = np.median(l_starts)
#
#paragraphs = []
#p_start = lines[0][0]
#
#for ix in range(1, len(lines)):
#    if l_starts[ix] > median_ls * 2:
#        p_end = lines[ix][0] - 10
#        paragraphs.append([p_start, p_end])
#        p_start = lines[ix][0]
#
#p_img = np.array(image)
#n_cols = p_img.shape[1]
#for paragraph in paragraphs:
#    cv2.rectangle(p_img, (5, paragraph[0]), (n_cols - 5, paragraph[1]), (128, 128, 0), 5)
#
#cv2.imshow('Show', p_img)



# Load the image
img = cv2.imread('Page1.png')

# convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# smooth the image to avoid noises
gray = cv2.medianBlur(gray,5)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
thresh = cv2.dilate(thresh,None,iterations = 12)
thresh = cv2.erode(thresh,None,iterations = 12)

# Find the contours
image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)

# Finally show the image
cv2.namedWindow('image', cv2.WINDOW_NORMAL)                            
imS = cv2.resize(img, (1000, 900))  
cv2.imshow('image', imS)
#cv2.imshow('res',thresh_color)
cv2.waitKey(0)
cv2.destroyAllWindows()