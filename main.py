import glob
import math
import cv2
import properties
import os
import threads


# delete the previous patches from dir
def clean_directory():
    patch_files = glob.glob('patches/*.jpg')
    for file in patch_files:
        os.remove(file)


def getNeededImageSize():
    img_1 = cv2.imread(properties.IMAGES[0])
    img_2 = cv2.imread(properties.IMAGES[1])
    if img_1.shape >= img_2.shape:
        return img_2.shape[0] * img_2.shape[1]
    elif img_1.shape < img_2.shape:
        return img_1.shape[0] * img_1.shape[1]


# down scales the bigger image to the size of the smaller one
# returns both of the images, now having the same size
def down_scale(img_1, img_2):
    im1_height, im1_width, channels1 = img_1.shape
    im2_height, im2_width, channels2 = img_2.shape
    if img_1.shape > img_2.shape:
        img_1 = cv2.resize(img_1, (im2_width, im2_height))
    elif img_1.shape < img_2.shape:
        img_2 = cv2.resize(img_2, (im1_width, im1_height))
    return img_1, img_2


# divides both of the images into patches and saves them to another dir
# each roi is of size: total size of width or height // number of columns or rows
def split_into_cells(img_1, img_2):
    nRows = int(math.sqrt(properties.NB_TASKS))
    mCols = int(math.sqrt(properties.NB_TASKS))
    sizeY, sizeX, ch = img_1.shape

    clean_directory()
    idx = 0
    for i in range(0, nRows):
        for j in range(0, mCols):
            roi_image_1 = img_1[i * sizeY // mCols: (i+1) * sizeY // mCols,
                                j * sizeX // nRows: (j+1) * sizeX // nRows]
            roi_image_2 = img_2[i * sizeY // mCols: (i+1) * sizeY // mCols,
                                j * sizeX // nRows: (j+1) * sizeX // nRows]
            cv2.imwrite('patches/1_patch_' + str(idx) + ".jpg", roi_image_1)
            cv2.imwrite('patches/2_patch_' + str(idx) + ".jpg", roi_image_2)
            idx += 1


if __name__ == '__main__':

    first_image_path = properties.IMAGES[0]
    second_image_path = properties.IMAGES[1]

    image_1 = cv2.imread(first_image_path)
    image_2 = cv2.imread(second_image_path)

    image_1, image_2 = down_scale(image_1, image_2)
    split_into_cells(image_1, image_2)

    # cv2.imshow('image_1', image_1)
    # cv2.imshow('image_2', image_2)
    # cv2.waitKey(0)

    if properties.IMPLEMENTATION == "threaded":
        threads.main_threads()
    else:
        os.system(f"mpiexec -n {properties.NB_TASKS+1} python distributed.py")
