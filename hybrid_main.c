/**
 *
 * CENG342 Project-2
 *
 * Downscaling OpenMP
 *
 * Usage:  mpirun -n 4 ./hybrid <input.jpg> <output.jpg> <num_threads>
 *
 *  mpicc -o hybrid_main hybrid_main.c -fopenmp -lm -cc=gcc-13
 *  mpirun -n 4 ./hybrid_main input.jpg output.jpg 4
 *
 * @Grup10
 * @author
 *  Ertuğrul ÇINAR
 *  Fadime Eda Nur BAYSAL
 *  Saidakhmad USMANALIEV
 *
 * @version 1.0, 28 May 2023
 */


#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1

double cubicInterpolate (double p[4], double x) {
   return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])));
}

double bicubicInterpolate (double p[4][4], double x, double y) {
   double arr[4];
   arr[0] = cubicInterpolate(p[0], y);
   arr[1] = cubicInterpolate(p[1], y);
   arr[2] = cubicInterpolate(p[2], y);
   arr[3] = cubicInterpolate(p[3], y);
   return cubicInterpolate(arr, x);
}

void seq_downscaling(uint8_t* input_image, uint8_t* output_image, int width, int start_height, int end_height) {
   for(int i=start_height; i<end_height ; i++){
       for(int j=0; j<width; j++){
           int newI = i / 2;
           int newJ = j / 2;
           output_image[newI * width / 2 + newJ] = input_image[i * width + j];
       }
   }
}

int main(int argc,char* argv[])
{
    MPI_Init(&argc,&argv);

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_threads = atoi(argv[3]);

    int width, height, bpp;
    uint8_t* input_image;
    uint8_t* output_image;
    if(rank == 0) {
        // Reading the image in grey colors
        input_image = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);
        output_image = malloc(sizeof(uint8_t) * (width / 2) * (height / 2));
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0) {
        input_image = (uint8_t*)malloc(sizeof(uint8_t) * width * height);
        output_image = malloc(sizeof(uint8_t) * (width / 2) * (height / 2));
    }

    MPI_Bcast(input_image, width*height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    printf("Rank %d: Width: %d  Height: %d \n",rank,width,height);
    printf("Rank %d: Input: %s , Output: %s  \n",rank,argv[1],argv[2]);

    // start the timer
    double time1= MPI_Wtime();
     
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        int start_height = thread_id * height / num_threads;
        int end_height = (thread_id + 1) * height / num_threads;

        if (thread_id == num_threads - 1) {
            end_height = height; // Make sure the last thread goes up to the end
        }

        seq_downscaling(input_image, output_image, width, start_height, end_height);
    }
        
    double time2= MPI_Wtime();
    printf("Elapsed time: %lf \n",time2-time1);
    
    if(rank == 0) {
        // Storing the image
        stbi_write_jpg(argv[2], width / 2, height / 2, CHANNEL_NUM, output_image, 100);
    }
    stbi_image_free(input_image);
    free(output_image);

    MPI_Finalize();
    return 0;
}


/**
 mpicc -o hybrid_main hybrid_main.c -fopenmp -lm -cc=gcc-13
 mpirun -n 4 ./hybrid_main input.jpg output.jpg 4

 */
