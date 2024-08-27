"""
---

# Problem 2

Given four overlapping images of a far-way building taken from the same camera (images may be taken with both
rotation and slight translation). Design an image processing pipeline to stitch these images to create a panoramic
image. (Note: You can consider far away objects/features to be approximately on the same plane.)



---

Tasks done in the below cell:
1. Extact/Read the images
2. Convert the images to grayscale\
3. Initialize the SIFT feature detector
4. Initialize FLANN (Fast Library for Approximate Nearest Neighbors) feature matcher

The main reason for selecting SIFT feature detector is its scale and rotation invarience feature. Even when the image is rotated or scaled up/down, the feature will get detected. In this case, as there are multiple images where some regions are overlapping, with possible sclae change and rotation, SIFT will detect the features regardless which will lead to strong matches. '

The reason for selecting FLANN is because it is computaionally inexpensive when compared to brute force matcher
"""

## Extract features from each frame (You can use any feature extractor and justify).
## Match the features between each consecutive image and visualize them. (hint: Use RANSAC)
## Compute the homographies between the pairs of images.
## Combine these frames together using the computed homographies.

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

## Reading the images
image1 = cv.imread('PA120275.JPG')
image2 = cv.imread('PA120274.JPG')
image3 = cv.imread('PA120273.JPG')
image4 = cv.imread('PA120272.JPG')

## Converting the images to grayscale
gray_image_1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray_image_2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
gray_image_3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)
gray_image_4 = cv.cvtColor(image4, cv.COLOR_BGR2GRAY)

## Initializing SIFT feature detector for detecting features
sift_detector = cv.SIFT_create()

## Initializing FLANN for feature matching
flann_index_kdtree = 1
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

"""The below cell contains the functions used in the process.

1. The homography function finds the matches, extracts good matches and finds the homography between 2 images.
2. The draw_matches function draws/links the matches for visualization
3. The stitch_image function performs translation and rotation of 2 images and overlaps them (stitch) according to the homography.
"""

## Defining a function to match the features and find the homography between 2 images
def homography(key1, desc1, key2, desc2):
    # matching the features using FLANN
    matches = flann.knnMatch(desc1, desc2, k=2)

    # selecting good matches and discarding bad matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    if len(good_matches) > 5:
        ## Using the good matches to find the homography
        src_pts = np.float32([key1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([key2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        h, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
        return h,mask,good_matches
    else:
        return None,None,None

## Defining a function and using cv.drawMatches to visualize the matches
def draw_matches(img1,key1,img2,key2,good_matches):
    match_img = cv.drawMatches(
        img1, key1,
        img2, key2,
        good_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('matches.jpg', match_img)

## Defining a function to stitch the images
def stitch_image(image1, image2, homography_rotation):
    # Extracting the height and width of the two images
    height2, width2 = image2.shape[0],image2.shape[1]
    height1, width1 = image1.shape[0],image1.shape[1]

    # computing the corners of the 2nd image
    corners2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    # Transforming the corners of the first image with respect to the calculated homography
    corners1_transformed = cv.perspectiveTransform(np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2), homography_rotation)
    corners = np.concatenate((corners1_transformed, corners2), axis=0)

    # Extracting the minimum and maximum corners
    min_x = np.min(corners[:, 0, 0])
    min_y = np.min(corners[:, 0, 1])
    max_x = np.max(corners[:, 0, 0])
    max_y = np.max(corners[:, 0, 1])

    # Computing the width and height of the stitched image
    final_width = int(max_x - min_x)
    final_height = int(max_y - min_y)

    # Formulating a translation matrix to position the image
    homography_translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    # Formulating a translation+rotation matrix to rotate and position the image
    homography_translation_rotation = homography_translation.dot(homography_rotation)

    # translating, rotating and positioning the image 1 for stitching
    warped_image1 = cv.warpPerspective(image1, homography_translation_rotation, (final_width, final_height))
    # translating and positioning the image 2 for stitching
    warped_image2 = cv.warpPerspective(image2, homography_translation, (final_width, final_height))
    # stitching the images togeather
    stitched_image = np.where(warped_image1 != 0, warped_image1, warped_image2)

    # Note: The image 1 will be rotated to match with image 2
    return stitched_image

"""In the below cell, the feature keypoints are extracted between image 1 and 2, homography is computed and the matches are drawn for visualization"""

## Extracting keypoints and descriptions of images 1 and 2 (wrt features) using sift detector
keypts1, desc1 = sift_detector.detectAndCompute(gray_image_1, None)
keypts2, desc2 = sift_detector.detectAndCompute(gray_image_2, None)
# Finding homography and matching the features between image 1 and 2
h_1_2,mask_1_2,good_matches12 = homography(keypts1, desc1, keypts2, desc2)
draw_matches(image1,keypts1,image2,keypts2,good_matches12)

"""In the below code, the images 1 and 2 are stitched togeather"""

# Stitching image 1 and 2
pano12 = stitch_image(image1,image2,h_1_2)

"""In the below cell, the feature keypoints are extracted between image 3 and 4, homography is computed and the matches are drawn for visualization"""

## Extracting keypoints and descriptions of images 3 and 4 (wrt features) using sift detector
keypts3, desc3 = sift_detector.detectAndCompute(gray_image_3, None)
keypts4, desc4 = sift_detector.detectAndCompute(gray_image_4, None)
# Finding homography and matching the features between image 3 and 4
h_3_4,mask_3_4,good_matches34 = homography(keypts3, desc3, keypts4, desc4)
draw_matches(image3,keypts3,image4,keypts4,good_matches34)

"""In the below code, the images 3 and 4 are stitched togeather"""

# stitching image 3 and 4
pano34 = stitch_image(image3,image4,h_3_4)

"""In the below cell, the 2 previously stitched images are considered. These images are converted to grayscale, their feature keypoints are extracted, homography is computed and matched are drawn for visualization"""

# Finding keypoints, matching features and finding homography between the 2 stitched images

gray_image_12 = cv.cvtColor(pano12, cv.COLOR_BGR2GRAY)
gray_image_34 = cv.cvtColor(pano34, cv.COLOR_BGR2GRAY)

keypts12, desc12 = sift_detector.detectAndCompute(gray_image_12, None)
keypts34, desc34 = sift_detector.detectAndCompute(gray_image_34, None)

h_12_34,mask_12_34,good_matches1234 = homography(keypts12, desc12, keypts34, desc34)

draw_matches(pano12,keypts12,pano34,keypts34,good_matches1234)

"""The two stitched images are further stitched togeather to get a singe panoramic image"""

# stitching image pano12 and pano34
pano1234 = stitch_image(pano12,pano34,h_12_34)
cv.imwrite('Pano.jpg', pano1234)

"""In the above panoramic image, we can observe that the 4th image is considered as the base image (it is not rotated), whereas all the other images are rotated to match the features and are stitched togeather.

Question: In general, why does panoramic mosaicing work better when the camera is only allowed to rotate at its camera
center?

Answer: Panoramic mosaicing works better when the camera is only allowed to rotate at its camera center because if the position of the camera is shifted, it leads to parallax error. Parallax is the displacement of the object when viewed from 2 different line of sight. This makes the final panoramic image and its features distorted. Also, the homography is simple and works better when the camera is rotated around its camera centre. If translation is involved, the process of homography and stitching the images becomes more complex.
"""