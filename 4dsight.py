import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np
from tqdm import tqdm
import sys

starmap = sys.argv[1]
area = sys.argv[2]
#starmap = 'StarMap.png'
#area = 'Small_area_rotated.png'

img_map = cv.imread(starmap)

gray= cv.cvtColor(img_map,cv.COLOR_BGR2GRAY)
img_height, img_width, = gray.shape

patch = cv.imread(area)
gray_patch= cv.cvtColor(patch,cv.COLOR_BGR2GRAY)
patch_height, patch_width = gray_patch.shape
x = patch_height//2 
y = patch_width//2 

sift = cv.SIFT_create()
bf = cv.BFMatcher()

kp_patch, des_patch = sift.detectAndCompute(gray_patch,None)
tm=cv.drawKeypoints(gray_patch,kp_patch,patch)

window = np.zeros(gray_patch.shape)
list_of_good=[]
max_m_pts=0
for i in tqdm(range(x,img_height - x,5)):
	for j in range(y, img_width - y,5):
		window = gray[i-x:i+x,j-y:j+y]
		img = img_map[i-x:i+x,j-y:j+y,:]
		kp, des = sift.detectAndCompute(window,None)
		wn=cv.drawKeypoints(window,kp,img)	
			

		


		matches = bf.knnMatch(des,des_patch,k=2)
		good = []
		for m,n in matches :
			if m.distance < 0.8*n.distance:
				good.append([m])
		matching_points = len(good)
		if matching_points >= max_m_pts:
			max_m_pts = matching_points
			list_of_good.append((good,wn,kp,(i,j)))
			

best_candidates = [l for l in list_of_good if len(l[0])==max_m_pts]


b = best_candidates[0]
print('Centre:')
print(b[3])
img3 = cv.drawMatchesKnn(b[1],b[2],patch,kp_patch,b[0],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
	


i=best_candidates[0][3][0]
j=best_candidates[0][3][1]
found_map = cv.rectangle(img_map,(i-x,j-y),(i+x,j+y),(0,255,0),1)

print('Corners:')
print('\t('+str(i-x)+';'+str(j-y)+')')
print('\t('+str(i-x)+';'+str(j+y)+')')
print('\t('+str(i+x)+';'+str(j-y)+')')
print('\t('+str(i+x)+';'+str(j+y)+')')

cv.imwrite('sift_keypoints'+area+'.jpg',img3)




#cv.imwrite('sift_keypoints_small_area.png',patch)



