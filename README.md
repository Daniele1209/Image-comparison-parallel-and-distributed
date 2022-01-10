# Image-comparison-parallel-and-distributed
Final project for the Parallel and Distributed programming course from CS year 3

This application helps in visualising the exact differences between 2 similar images.

The project has 2 implementations: one with "regular" threads, and one distributed (using MPI).

The algorithms

#Genreal idea of the algorithm
- we use 2 images, that are quite the same with some differences (some drawings on top)
- in the pre-processing phase we make sure that our 2 input images have the same size (which in any case is size of the smaller image)
- after that, we split into n cells the 2 images that we have (n is the number of processes that we run, this number should be a square number)
- each image patch is saved separately in another directory
- the following 2 implementations for parallelism and distribution

#Threaded version
- use a list for the threads
- read 2 images at a time (representing patches from the first and second image) and start the thread function
- in the thread function we grayscale the 2 patches and we apply structural_similarity using skimage library
- check if the score that we got passes the threshold, if not we just write the patches as they are, without highlights
- aquire the mutex, add the output patch in the dictionary and release the mutex
- and in case we pass the threshold, use cv2 to draw contours for the highlights and after that write the image to the dictionary
- after all threads finish execution, we join all the threads and use the master thread to build the image back as it was before, but this time highlighted
- write the final image to disk

#Distributed version
- the distributed part is done using MPI
- get the rank and size
- check for the master process
- the master process sends 2 patches at a time to the worker nodes
- the workers grayscale the image, find and draw the differences in the final patch image
- workers send the final patch to the master process
- the main node then receives each of the patches and rebuilds the image
- at the end finalize

The synchronization used in the parallelized variants
- using mutex to keep the dictionary of patches consistent
- mutex aquire when we want to assign a certain patch to a key in the global dictionary
- release Lock afterwards

The performance measurements
-
- hardware specifications: 
   - Processor	Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz, 2208 Mhz, 6 Core(s)
   - Installed Physical Memory (RAM) 16.0 GB
- OS: Microsoft Windows 11 Pro
- number of parallel workers set to: 4  -> photo split in 4 patches
- time measurement unit: s

| Implementation | Photo size (no. of pixels) | Time elapsed |
|----------------|----------------------------|--------------|
| threaded       | 614400                     | 0.12         |
| threaded       | 128390                     | 0.07         |
| threaded       | 261671                     | 0.08         |
| distributed    | 614400                     | 0.18         |
| distributed    | 128390                     | 0.08         |
| distributed    | 261671                     | 0.14         |