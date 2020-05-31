import math
import cv2
import os
def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    tempImg = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4,borderValue=(0,255,0))
    seedx = np.linspace(0.4,0.8,20)
    seedy = np.linspace(0.4,0.8,20)
    indexx = np.random.randint(0,len(seedx))
    indexy = np.random.randint(0,len(seedy))
    res = cv2.resize(tempImg,(int(tempImg.shape[1]*seedx[indexx]),int(tempImg.shape[0]*seedy[indexy])),interpolation=cv2.INTER_CUBIC)   #dsize=（2*width,2*height）
    return res


import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

def roate_zoom_img(img):
    angle = np.linspace(-30,30,20)
    scale = np.linspace(0.6,1.0,20)
    index = np.random.randint(0,len(angle))
    new_image = rotate_about_center(img,angle[index],scale[index])
    return new_image


def img_random_brightness(img):
    brightness = iaa.Multiply((0.6,1.5))
    #print(brightness.augment)
    image = brightness.augment_image(img)
    return image


def random_augment(img):
   
    image = img_random_brightness(img)
    image = roate_zoom_img(image)

    return image

def checkOverlap(rectangles,b):
    for a in rectangles:
        widthmin = min([a[0],a[2],b[0],b[2]])
        widthmax = max([a[0],a[2],b[0],b[2]])
        heightmin = min([a[1],a[3],b[1],b[3]])
        heightmax = max([a[1],a[3],b[1],b[3]])
        if (a[2]-a[0] + b[2]-b[0]) < (widthmax -widthmin) and (a[3]-a[1] + b[3]-b[1]) < (heightmax -heightmin):
            continue
        else :
            return True
    return False
#随机位置生成图片
def transparentOverlay(path):
    bgImg = cv2.imread(path, -1)
    src = cv2.resize(bgImg, (416,416))
    rows, cols, _ = src.shape  # 背景图
    rectangles = []
    label = ' '
    for i in range(0,10):
        index = np.random.randint(1,53)
        pukeimg = cv2.imread('./puke/'+str(index)+'.png', -1)
        overlay = random_augment(pukeimg)
        h, w, _ = overlay.shape  # 扑克牌

        x = np.random.randint(0,416-h)
        y = np.random.randint(0,416-w)
        
        if checkOverlap(rectangles,(y, x,y+w,x+h)):
            continue
        
        hsv = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV)

        # 图像合并
        for i in range(h):
            for j in range(w):
                if x + i >= rows or y + j >= cols:
                    continue
                if  (hsv[i][j][0:1]>36 and hsv[i][j][1:2]>25  and hsv[i][j][2:3]>25 and hsv[i][j][0:1]<70):
                    continue
                src[x + i][y + j] = overlay[i][j][:3]
#        cv2.rectangle(src, (y, x), (y+w,x+h), (255, 0, 0), 1)
#         print("{},{},{},{},{} ".format(y, x, y+w,x+h,index) )
        rectangles.append((y, x, y+w,x+h))
        label += "{},{},{},{},{} ".format(y, x, y+w,x+h,index-1) 
#     print(label)
    return src,label


def generateImage():
   
         
    rootdir = './texture'
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    with open("./data/dataset/test.txt", "w") as wf: 
        for i in range(0,2000):
            index = np.random.randint(0,len(list))
            path = os.path.join(rootdir,list[index])
            overlay,label = transparentOverlay(path)
    #         print(label)
    #         cv2.imshow('demo', overlay)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
            cv2.imwrite("./data/dataset/test/"+str(i)+".jpg",overlay)
            annotation = ("./data/dataset/test/"+str(i)+".jpg"+label)
            wf.write(annotation + "\n")
            wf.flush()
    
generateImage()

