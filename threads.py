import threading
import cv2
import properties


def thread_function(idx, patch1, patch2):
    print(idx)
    print(patch1.shape)
    print(patch2.shape)
    difference = cv2.subtract(patch1, patch2)
    #print(difference)


def main_threads():
    print('Main thread')
    threads = []
    for index in range(properties.NB_TASKS):
        patch_image1 = cv2.imread("patches/1_patch_" + str(index) + ".jpg")
        patch_image2 = cv2.imread("patches/2_patch_" + str(index) + ".jpg")
        difference = cv2.subtract(patch_image1, patch_image2)
        cv2.imshow("lmao", difference)
        cv2.waitKey(0)

        x = threading.Thread(target=thread_function, args=(index, patch_image1, patch_image2,))
        threads.append(x)
        x.start()

    for x in threads:
        x.join()
