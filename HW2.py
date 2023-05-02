import cv2
import numpy as np
import random
import math
import sys

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# SIFT
def SIFT(img):
    SIFT_Detector = cv2.SIFT_create()
    kp, descript = SIFT_Detector.detectAndCompute(img, None)
    return kp, descript

def plot_SIFT(img, kp):
    tmp = img.copy()
    tmp = cv2.drawKeypoints(tmp, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return tmp

# feature matching
def matcher(kp1, des1, kp2, des2, threshold):
    # Using BFMatcher:
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # matches is a tuple, containing Dmatches.
    # Apply Lowe's ratio test
    good = []
    for m,n in matches:
        # m, n are DMatch
        if m.distance < threshold*n.distance:
            good.append([m])
    
    # Brutal Force
    matches1 = []
    print(des1.shape)
    for i in range(len(kp1)):
        dmatch = {"distance":1e7, "queryIdx":0, "trainIdx":0} # smallest
        dmatch1 = {"distance":1e7, "queryIdx":0, "trainIdx":0} # 2nd smallest
        v1 = des1[i,:]
        for j in range(len(kp2)):
            v2 = des2[j,:]
            distance = 0
            # print(v1.shape[0], v2.shape[0])
            for k in range(v1.shape[0]):
                distance+=(v1[k]-v2[k])**2
            distance = math.sqrt(distance)
            if distance < dmatch["distance"]:
                dmatch1["distance"] = dmatch["distance"]
                dmatch1["queryIdx"] = dmatch["queryIdx"]
                dmatch1["trainIdx"] = dmatch["queryIdx"]
                dmatch["distance"] = distance
                dmatch["queryIdx"] = i
                dmatch["trainIdx"] = j
            elif distance<dmatch1["distance"]:
                dmatch1["distance"] = distance
                dmatch1["queryIdx"] = i
                dmatch1["trainIdx"] = j
        print(i, "done.")
        matches1.append((dmatch, dmatch1))

    good1 = []
    for m,n in matches1:
        # m, n are DMatch
        if m["distance"] < threshold*n["distance"]:
            good1.append([m])
    
    return good, good1

    # matches = []
    # for pair in good:
    #     print(pair[0])
    #     matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    # matches = np.array(matches)
    # return matches

def plot_matches(good, kp1, kp2, total_img):
    import matplotlib.pyplot as plt
    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0]["queryIdx"]].pt + kp2[pair[0]["trainIdx"]].pt))
    matches = np.array(matches)

    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)
    plt.show()

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # the example of image window
    base = "./baseline"
    imgs = []
    img_grays = []
    for i in range(1,7):
        path = base + "/m"+str(i)+".jpg"
        img, img_gray = read_img(path)
        imgs.append(img)
        img_grays.append(img_gray)
        # creat_im_window("img"+str(i),img)

    img_left_g = img_grays[0]
    img_right_g = img_grays[1]
    img_left = imgs[0]
    img_right = imgs[1]

    # Step 1 SIFT
    kp_left, des_left = SIFT(img_left_g)
    kp_right, des_right = SIFT(img_right_g)
    # plot the SIFT image
    kp_left_img = plot_SIFT(img_left, kp_left)
    kp_right_img = plot_SIFT(img_right, kp_right)
    total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
    # creat_im_window("total_kp", total_kp)

    # Step 2 Feature matching
    good, good1 = matcher(kp_left, des_left, kp_right, des_right, 0.7)
    # plot the good matches
    img_matches = cv2.drawMatchesKnn(img_left_g,kp_left,img_right_g,kp_right,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # img_matches1 = cv2.drawMatchesKnn(img_left_g,kp_left,img_right_g,kp_right,good1,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    total_img = np.concatenate((img_left, img_right), axis=1)
    plot_matches(good1,kp_left,kp_right,total_img)
    creat_im_window("img_matches", img_matches)
    # creat_im_window("img_matches1", img_matches1)

    # Show the final image
    im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)