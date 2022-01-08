import math
import threading
import cv2
import properties
import copy
from threading import Lock
import time

diff_patches_list = {}
# define a mutex for writing in list
mutex = Lock()


def thread_function(idx, patch1, patch2):
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
    start_time = time.time()

    for index in range(properties.NB_TASKS):
        patch_image1 = cv2.imread("patches/1_patch_" + str(index) + ".jpg")
        patch_image2 = cv2.imread("patches/2_patch_" + str(index) + ".jpg")
        # difference = cv2.subtract(patch_image1, patch_image2)

        x = threading.Thread(target=thread_function, args=(index, patch_image1, patch_image2,))
        threads.append(x)
        x.start()

    for x in threads:
        x.join()

    # patch images together to form the initial image
    final_image = None

    for idx_v in range(0, int(math.sqrt(properties.NB_TASKS))):
        image_index = int(idx_v * (math.sqrt(properties.NB_TASKS)-1) + idx_v)

        patch_image = cv2.imread("patches/1_patch_" + str(image_index) + ".jpg")
        mask = diff_patches_list[image_index]
        h_img = cv2.bitwise_and(patch_image, patch_image, mask=mask)
        h_img[mask != 255] = [0, 0, 255]

        for idx_h in range(1, int(math.sqrt(properties.NB_TASKS))):

            image_index = int(idx_v * (math.sqrt(properties.NB_TASKS)-1) + idx_h + idx_v)
            patch_image_1 = cv2.imread("patches/1_patch_" + str(image_index) + ".jpg")
            mask_1 = diff_patches_list[image_index]
            h_img_1 = cv2.bitwise_and(patch_image_1, patch_image_1, mask=mask_1)
            h_img_1[mask_1 != 255] = [0, 0, 255]

            h_img = cv2.hconcat([h_img, h_img_1])

        if isFirst:
            final_image = copy.deepcopy(h_img)
            isFirst = False
        else:
            final_image = cv2.vconcat([final_image, h_img])

    print(f"\nThreads done: {'{:.2f}'.format(time.time() - start_time)} s")
    cv2.imshow('final_image', final_image)
    cv2.imwrite('results/final_image.jpg', final_image)
    cv2.waitKey(0)
