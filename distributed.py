import copy
import math
import time
from mpi4py import MPI
import cv2
from skimage.metrics import structural_similarity
import numpy as np
import properties

# mpi run command
# mpiexec -n 10 python distributed.py
diff_patches_list = {}


def node_function(idx, patch1, patch2):
    print(f"Node {idx} started ...")
    # Convert images to grayscale
    patch1_gray = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
    patch2_gray = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(patch1_gray, patch2_gray, full=True)
    print(f"Image similarity: {score}")
    # optimisation
    if (properties.NEEDED_IMAGE_SIZE <= 300000 and score > 0.9) or \
            (properties.NEEDED_IMAGE_SIZE > 300000 and score > 0.99):
        return patch1

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
            x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            # cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    # add the difference to the list
    return filled_after


def reconstruct_final_image(start_time):
    # put patches together to form the initial image
    isFirst = True
    final_image = None
    nr_rows_cols = int(math.sqrt(properties.NB_TASKS))

    for idx_v in range(0, nr_rows_cols):
        image_index = int(idx_v * (math.sqrt(properties.NB_TASKS) - 1) + idx_v)

        # patch_image = cv2.imread("patches/1_patch_" + str(image_index) + ".jpg")
        mask = diff_patches_list[image_index]
        # h_img = cv2.bitwise_and(patch_image, patch_image, mask=mask)
        # h_img[mask != 255] = [0, 0, 255]

        for idx_h in range(1, nr_rows_cols):
            image_index = int(idx_v * (math.sqrt(properties.NB_TASKS) - 1) + idx_h + idx_v)
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

    print(f"\nMPI done: {'{:.2f}'.format(time.time() - start_time)} s")
    cv2.imshow('final_image', final_image)
    cv2.imwrite('C:/Users/S/Desktop/Prog-par-distrib/Image-comparison-parallel-and-distributed/results/final_image.jpg',
                final_image)
    cv2.waitKey(0)


def main_distributed():
    print('Main distributed')

    start_time = time.time()

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    print(rank, size)

    # master process
    if rank == 0:
        for index in range(1, properties.NB_TASKS + 1):
            patch_image1 = cv2.imread("C:/Users/S/Desktop/Prog-par-distrib/Image-comparison-parallel-and-distributed/patches/1_patch_" + str(index-1) + ".jpg")
            patch_image2 = cv2.imread("C:/Users/S/Desktop/Prog-par-distrib/Image-comparison-parallel-and-distributed/patches/2_patch_" + str(index-1) + ".jpg")
            MPI.COMM_WORLD.send((patch_image1, patch_image2), dest=index, tag=0)

        for rank_idx in range(1, properties.NB_TASKS + 1):
            data = MPI.COMM_WORLD.recv(source=rank_idx, tag=0)
            diff_patches_list[rank_idx-1] = data

        reconstruct_final_image(start_time)

    else:
        data = MPI.COMM_WORLD.recv(source=0, tag=0)
        returned_patch = node_function(rank, data[0], data[1])
        MPI.COMM_WORLD.send(returned_patch, dest=0, tag=0)

    MPI.Finalize()


if __name__ == '__main__':
    main_distributed()
