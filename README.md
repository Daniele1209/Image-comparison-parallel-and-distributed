# Image-comparison-parallel-and-distributed
Final project for the Parallel and Distributed programming course from CS year 3

This application helps in visualising the exact differences between 2 similar images.

The project has 2 implementations: one with "regular" threads, and one distributed (using MPI).

The algorithms
- 

The synchronization used in the parallelized variants
-

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