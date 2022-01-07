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


# down scales the bigger image to the size of the smaller one
# returns both of the images, now having the same size
def down_scale(image_1, image_2):
    im1_height, im1_width, channels1 = image_1.shape
    im2_height, im2_width, channels2 = image_2.shape

    if image_1.shape > image_2.shape:
        image_1 = cv2.resize(image_1, (im2_width, im2_height))
    elif image_1.shape < image_2.shape:
        image_2 = cv2.resize(image_2, (im1_width, im1_height))

    return image_1, image_2


# divides both of the images into patches and saves them to another dir
# each roi is of size: total size of width or height // number of columns or rows
def split_into_cells(image_1, image_2):
    nRows = int(math.sqrt(properties.NB_TASKS))
    mCols = int(math.sqrt(properties.NB_TASKS))
    sizeY, sizeX, ch = image_1.shape

    clean_directory()
    idx = 0
    for i in range(0, nRows):
        for j in range(0, mCols):
            roi_image_1 = image_1[i * sizeY // mCols : (i+1) * sizeY // mCols,
                          j * sizeX // nRows : (j+1) * sizeX // nRows]
            roi_image_2 = image_2[i * sizeY // mCols : (i+1) * sizeY // mCols,
                          j * sizeX // nRows : (j+1) * sizeX // nRows]
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

    threads.main_threads()




