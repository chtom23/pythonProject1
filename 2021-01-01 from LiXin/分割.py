# -*- coding: utf-8 -*- 
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
 
# 使用2g-r-b分离土壤与背景

src = cv2.imread('C:/Users/86130/Pictures/IMG_1491.JPG')
"""
#缩放为原来的十分之一，针对高清图
src = cv2.resize(src,None,fx=0.1, fy=0.1, 
                 interpolation = cv2.INTER_CUBIC)
"""
cv2.imshow('src', src)


# 转换为浮点数进行计算
fsrc = np.array(src, dtype=np.float32) / 255.0
(b,g,r) = cv2.split(fsrc)
gray = 2 * g - b - r
 
# 求取最大值和最小值
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
 
# 计算直方图
hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
plt.plot(hist)
plt.show()
 
cv2.waitKey()

# 转换为u8类型，进行otsu二值化
gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
(thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)

cv2.imshow('bin_img', bin_img)


# 得到彩色的图像
(b8, g8, r8) = cv2.split(src)
color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
cv2.imshow('color_img', color_img)

import cv2  
import os  
"""
if __name__ == '__main__':  
    img = cv2.imread(color_img, -1)  
    if img == None: 
         
        os._exit(0)  
      
    height, width = img.shape[:2]  
"""
 
img = color_img
 
img = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,118) #这里对最后一个参数使用了经验型的值
result = img.copy()
for line in lines:
	rho = line[0][0]  #第一个元素是距离rho
	theta= line[0][1] #第二个元素是角度theta
	print (rho)
	print (theta)
	if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
		pt1 = (int(rho/np.cos(theta)),0)               #该直线与第一行的交点
		#该直线与最后一行的焦点
		pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
		cv2.line( result, pt1, pt2, (255))             # 绘制一条白线
	else:                                                  #水平直线
		pt1 = (0,int(rho/np.sin(theta)))               # 该直线与第一列的交点
		#该直线与最后一列的交点
		pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
		cv2.line(result, pt1, pt2, (255), 1)           # 绘制一条直线

cv2.imshow('Canny', edges )
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
#去除较小的团块
img = edges
print(np.shape(img))
kernel = np.ones((3,3),np.uint8)
dilate = cv2.dilate(img,kernel,iterations = 1)
# 显示图片
# ## 效果展示
# cv2.imshow('origin', img)
 
 
cv2.imwrite('lishuwang_dilate.jpg',dilate)
# erosion = cv2.erode(img,kernel,iterations = 1)
# cv2.imwrite('lishuwang_erosion.jpg',erosion)
 
canny1=cv2.Canny(dilate,100,200)
cv2.imwrite('lishuwang_canny.jpg',canny1)
 
# kernel2 = np.ones((2,1),np.uint8)
# erosion = cv2.erode(canny,kernel2,iterations = 1)
# cv2.imwrite('lishuwang_erosion.jpg',erosion)
 
_, labels, stats, centroids = cv2.connectedComponentsWithStats(canny1)
print(centroids)
print("stats",stats)
i=0
for istat in stats:
    if istat[4]<120:
        #print(i)
        print(istat[0:2])
        if istat[3]>istat[4]:
            r=istat[3]
        else:r=istat[4]
        cv2.rectangle(canny1,tuple(istat[0:2]),tuple(istat[0:2]+istat[2:4]) , 0,thickness=-1)  # 26
    i=i+1
 
cv2.imshow('Canny1', canny1 )
"""
