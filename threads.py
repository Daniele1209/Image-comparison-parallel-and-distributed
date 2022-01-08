import math
import threading
import cv2

import main
import properties
import copy
from threading import Lock
import time
import numpy as np
from skimage.metrics import structural_similarity

diff_patches_list = {}
# define a mutex for writing in list
mutex = Lock()
needed_image_size = 0


def thread_function2(idx, patch1, patch2):
    print(f"Thread {idx} started ...")
    # Convert images to grayscale
    patch1_gray = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
    patch2_gray = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(patch1_gray, patch2_gray, full=True)
    print(f"Thread {idx} - patch similarity: {score}")
    # optimisation
    if (needed_image_size <= 300000 and score > 0.9) or \
            (needed_image_size > 300000 and score > 0.99):
        mutex.acquire()
        diff_patches_list[idx] = patch1
        mutex.release()
        return

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(patch1.shape, dtype='uint8')
    filled_after = patch2.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    # add the difference to the list
    mutex.acquire()
    diff_patches_list[idx] = filled_after
    mutex.release()


def thread_function(idx, patch1, patch2):
    # old implementation
    print(f"Thread {idx} started ...")
    difference = cv2.subtract(patch1, patch2)

    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    # add the red mask to the images to make the differences obvious
    patch1[mask != 255] = [0, 0, 255]

    # add the difference to the list
    mutex.acquire()
    diff_patches_list[idx] = mask
    mutex.release()


def main_threads():
    print('Main thread')
    isFirst = True
    threads = []
    global needed_image_size
    needed_image_size = main.getNeededImageSize()
    start_time = time.time()

    for index in range(properties.NB_TASKS):
        patch_image1 = cv2.imread("patches/1_patch_" + str(index) + ".jpg")
        patch_image2 = cv2.imread("patches/2_patch_" + str(index) + ".jpg")
        # difference = cv2.subtract(patch_image1, patch_image2)

        x = threading.Thread(target=thread_function2, args=(index, patch_image1, patch_image2,))
        threads.append(x)
        x.start()

    for x in threads:
        x.join()

    # put patches together to form the initial image
    final_image = None
    nr_rows_cols = int(math.sqrt(properties.NB_TASKS))

    for idx_v in range(0, nr_rows_cols):
        image_index = int(idx_v * (math.sqrt(properties.NB_TASKS)-1) + idx_v)

        # patch_image = cv2.imread("patches/1_patch_" + str(image_index) + ".jpg")
        mask = diff_patches_list[image_index]
        # h_img = cv2.bitwise_and(patch_image, patch_image, mask=mask)
        # h_img[mask != 255] = [0, 0, 255]

        for idx_h in range(1, nr_rows_cols):

            image_index = int(idx_v * (math.sqrt(properties.NB_TASKS)-1) + idx_h + idx_v)
            # patch_image_1 = cv2.imread("patches/1_patch_" + str(image_index) + ".jpg")
            mask_1 = diff_patches_list[image_index]
            # h_img_1 = cv2.bitwise_and(patch_image_1, patch_image_1, mask=mask_1)
            # h_img_1[mask_1 != 255] = [0, 0, 255]

            mask = cv2.hconcat([mask, mask_1])

        if isFirst:
            final_image = copy.deepcopy(mask)
            isFirst = False
        else:
            final_image = cv2.vconcat([final_image, mask])

    print(f"\nThreads done: {'{:.2f}'.format(time.time() - start_time)} s")
    cv2.imshow('final_image', final_image)
    cv2.imwrite('results/final_image.jpg', final_image)
    cv2.waitKey(0)
