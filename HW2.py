import cv2
import numpy as np
import random
import math
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    # Using BFMatcher (For debugging):
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
    for i in tqdm(range(len(kp1))):
        dmatch = {"distance":1e7, "queryIdx":0, "trainIdx":0} # smallest
        dmatch1 = {"distance":1e7, "queryIdx":0, "trainIdx":0} # 2nd smallest
        v1 = des1[i,:]
        for j in range(len(kp2)):
            v2 = des2[j,:]
            distance = 0
            distance = np.linalg.norm(v1-v2)
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
        matches1.append((dmatch, dmatch1))

    good1 = []
    # Apply Lowe's ratio test
    for m,n in matches1:
        # m, n are DMatch
        if m["distance"] < threshold*n["distance"]:
            good1.append([m])
    
    return good, good1


def get_matches(good, kp1, kp2):
    '''
    out: matches [[x1, y1, x1', y1'], ....., [xn, yn, xn', yn']]
    '''
    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0]["queryIdx"]].pt + kp2[pair[0]["trainIdx"]].pt))  # Use for self define pair
        # matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))      # Use for dmatch class (when using BFMatcher())
    matches = np.array(matches)
    return matches

def plot_matches(matches, total_img):

    match_img = total_img.copy()
    match_img = cv2.cvtColor(match_img,cv2.COLOR_BGR2RGB)
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)
    plt.show()

# Step 3 Homography
def homography(pairs):
    # pairs: [[(x1, y1), (x1', y1')],[(x2, y2), (x2', y2')] [(x3, y3), (x3', y3')], [(x4, y4), (x4', y4')]]
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)     # A in spec, rows.shape is (8, 9)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)   # use the last vector since np.linalg.svd has vector sorted in descending order
    H = H/H[2, 2] # normalize h33 to 1  
    return H

# Helper functions of RANSEC
def random_point(matches, k=4):
    # Get k pairs from the matches (Used in RANSEC)
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(matches, H):
    # matches: [[x1, y1, x1', y1'], ..., [xn, yn, xn', yn']]
    num_points = len(matches)
    all_p1 = np.concatenate((matches[:, 0:2], np.ones((num_points, 1))), axis=1) # change (x1, y1) to (x1, y1, 1)
    all_p2 = matches[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

# Step 3 RANSAC
def ransac(matches, threshold, iters):
    num_best_inliers = 0
    best_H = np.zeros((3, 3))
    for i in tqdm(range(iters)):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

# use for blendering the images
class Blender:
    def linearBlending(self, imgs):
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        
        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
        
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # Plot the overlap mask
        plt.figure(21)
        plt.title("overlap_mask")
        plt.imshow(overlap_mask.astype(int), cmap="gray")
        plt.show()
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        for i in range(hr): 
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
        
        
        linearBlending_img = np.copy(img_right)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
                elif (np.count_nonzero(img_left_mask[i, j])>0):
                    linearBlending_img[i, j] = img_left[i,j]
                elif (np.count_nonzero(img_right_mask[i, j])>0):
                    linearBlending_img[i, j] = img_right[i,j]
        
        return linearBlending_img
    
    def linearBlendingWithConstantWidth(self, imgs):
        '''
        linear Blending with Constat Width, avoiding ghost region
        # you need to determine the size of constant with
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        constant_width = 10 # constant width
        
        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
                    
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        for i in range(hr):
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            
            # Find the middle line of overlapping regions, and only do linear blending to those regions very close to the middle line.
            middleIdx = int((maxIdx + minIdx) / 2)
            
            # left 
            for j in range(minIdx, middleIdx + 1):
                if (j >= middleIdx - constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (i - minIdx))
                else:
                    alpha_mask[i, j] = 1
            # right
            for j in range(middleIdx + 1, maxIdx + 1):
                if (j <= middleIdx + constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (i - minIdx))
                else:
                    alpha_mask[i, j] = 0

        # plt.figure(21)
        # plt.title("overlap_mask")
        # plt.imshow(overlap_mask.astype(int), cmap="gray")
        # plt.show()

        linearBlendingWithConstantWidth_img = np.copy(img_right)
        # linear blending with constant width
        for i in range(hr):
            for j in range(wr):
                if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
                elif (np.count_nonzero(img_left_mask[i, j])>0):
                    linearBlendingWithConstantWidth_img[i, j] = img_left[i,j]
                elif (np.count_nonzero(img_right_mask[i, j])>0):
                    linearBlendingWithConstantWidth_img[i, j] = img_right[i,j]
        
        return linearBlendingWithConstantWidth_img

def removeBlackBorder(img):
    '''
    Remove img's the black border 
    '''
    h, w = img.shape[:2]
    reduced_h, reduced_w = h, w
    # right to left
    for col in range(w - 1, -1, -1):
        all_black = True
        for i in range(h):
            if (np.count_nonzero(img[i, col]) > 0):
                all_black = False
                break
        if (all_black == True):
            reduced_w = reduced_w - 1
            
    # bottom to top 
    for row in range(h - 1, -1, -1):
        all_black = True
        for i in range(reduced_w):
            if (np.count_nonzero(img[row, i]) > 0):
                all_black = False
                break
        if (all_black == True):
            reduced_h = reduced_h - 1
    
    return img[:reduced_h, :reduced_w]

# Step 4 Stitich image
def stitch_img(left, right, H):
    
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)
    y_min = min(y_min, 0)
    x_min = min(x_min, 0)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    translation_reverse = np.array([[1, 0, 0], [0, 1, y_min], [0, 0, 1]])
    H1 = np.copy(H)
    H = np.dot(translation_mat, H)
    
    # Get height, width
    h2, w2 ,c2= right.shape
    # print(round(y_min), round(x_min))
    y_min = round(y_min)
    x_min = round(x_min)
    height_new = int(round(abs(y_min)) + h2)
    width_new = int(round(abs(x_min)) + w2)
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    height_new_last = height_r
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    size_last = (width_new, height_new_last)
    
    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # cv2.imshow("left",warped_l)
    # cv2.imshow("right",warped_r)

    # Stitching procedure, store results in warped_r.
    # warped_img = np.zeros((size[0], size[1], 3), dtype="int")

    blender = Blender()
    # warped_img= blender.linearBlendingWithConstantWidth([warped_l, warped_r])
    warped_img= blender.linearBlending([warped_l, warped_r])
    # cv2.imshow("blender", warped_img)
    warped_img = removeBlackBorder(warped_img)
    # Used in baseline
    # warped_img = cv2.warpPerspective(src=warped_img, M=translation_reverse, dsize=size_last)
    warped_l*=255.0
    warped_l = warped_l.astype(np.uint8)
    warped_r*=255.0
    warped_r = warped_r.astype(np.uint8)
    cv2.imwrite("left.jpg",warped_l)
    cv2.imwrite("right.jpg",warped_r)
    return warped_img

def stitch(img_left_g, img_right_g, img_left, img_right):

    # Step 1 SIFT
    print("Step 1 SIFT ...")
    kp_left, des_left = SIFT(img_left_g)
    kp_right, des_right = SIFT(img_right_g)

    # plot the SIFT image

    # kp_left_img = plot_SIFT(img_left, kp_left)
    # kp_right_img = plot_SIFT(img_right, kp_right)
    # total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
    # creat_im_window("total_kp", total_kp)

    # Step 2 Feature matching
    print("Step 2 Feature Matching ...")
    good, good1 = matcher(kp_left, des_left, kp_right, des_right, 0.7)
    matches = get_matches(good1, kp_left, kp_right)

    # plot the good matches
    # img_matches = cv2.drawMatchesKnn(img_left_g,kp_left,img_right_g,kp_right,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # creat_im_window("img_matches", img_matches)
    # total_img = np.concatenate((img_left, img_right), axis=1)
    # plot_matches(matches,total_img)
    # print(matches.shape)

    # Step 3 Homography
    print("Step 3 RANSC ...")
    inliers, H = ransac(matches, 0.4, 8000)
    # plot_matches(inliers, total_img)

    # Step 4 Stitch Image
    print("Step 4 stiching image ...")
    img_stitched = stitch_img(img_left, img_right, H)

    # creat_im_window("img_stitched", img_stitched)

    # Change the img back to [0, 255] inorder to store using cv2.imwrite
    img_stitched*=255.0
    img_stitched = img_stitched.astype(np.uint8)
    return img_stitched
    

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
    img_number = 6

    # For bouns
    # base = "./bonus"
    # img_number = 4
    for i in range(1,img_number+1):
        path = base +f"/m{str(i)}.jpg"
        img, img_gray = read_img(path)
        imgs.append(img)
        img_grays.append(img_gray)
        # creat_im_window("img"+str(i),img)

    # Start from right hand side.
    img_left_g = img_grays[4]
    img_right_g = img_grays[5]
    img_left = imgs[4]
    img_right = imgs[5]

    # img_left, img_left_g = read_img("./bonus_1+2p.jpg")
    # img_right, img_right_g = read_img("./bonus_3+4p.jpg")
    img = stitch(img_left_g, img_right_g, img_left, img_right)
    # img = img_right
    # for i in range(4,-1,-1):
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img = stitch(img_grays[i], img_gray, imgs[i], img)
    

    creat_im_window("img_stitched", img)
    # cv2.imwrite("testing1.jpg",img)
    
    # Show the final image
    im_show()