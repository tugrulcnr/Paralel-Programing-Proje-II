/**
 *
 * CENG342 Project-2
 *
 * Downscaling OpenMP
 *
 * Usage:  openmp_main <input.jpg> <output.jpg><num_thread>
 *
 *  gcc-13 -fopenmp openmp_main.c -o openmp_main -lm
 *  ./openmp_main aybu.jpg output.jpg 4
 *
 * @Grup10
 * @author
 *  Ertuğrul ÇINAR
 *  Fadime Eda Nur BAYSAL
 *  Saidakhmad USMANALIEV
 *
 * @version 1.0, 28 May 2023
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

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

double clamp(double x, double lower, double upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

void downscaleBicubic(uint8_t* input_image, uint8_t* output_image, int width, int height, int num_threads, omp_sched_t schedule, int chunk) {
    int new_width = width / 2;
    int new_height = height / 2;
    double grid_x = (double)width / (double)new_width;
    double grid_y = (double)height / (double)new_height;

    omp_set_schedule(schedule, chunk);

    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < new_height; i++){
        for(int j = 0; j < new_width; j++){
            double grid_i = i * grid_y;
            double grid_j = j * grid_x;

            double arr[4][4];
            for (int m = -1; m <= 2; ++m) {
                for (int n = -1; n <= 2; ++n) {
                    int x = (int)floor(grid_i) + m;
                    int y = (int)floor(grid_j) + n;
                    x = x < 0 ? 0 : x >= height ? height - 1 : x;
                    y = y < 0 ? 0 : y >= width ? width - 1 : y;
                    arr[m+1][n+1] = input_image[x*width + y];
                }
            }

            double val = bicubicInterpolate(arr, grid_i - floor(grid_i), grid_j - floor(grid_j));
            output_image[i*new_width + j] = (uint8_t) clamp(val, 0.0, 255.0);
        }
    }
}

int main(int argc,char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.jpg> <output.jpg> <number_of_threads>\n", argv[0]);
        return 1;
    }

    int width, height, bpp;
    int total_threads = atoi(argv[3]);
    int perf_threads = total_threads / 2; // use half of the threads for performance cores
    int effi_threads = total_threads - perf_threads; // rest of the threads for efficiency cores

    uint8_t* rgb_image = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);
    uint8_t* output_image = (uint8_t*) malloc(width * height / 4);

    if(!rgb_image) {
        fprintf(stderr, "Error in loading the image\n");
        exit(1);
    }

    printf("Width: %d  Height: %d\n", width, height);

    // Default schedule
    printf("Using default schedule\n");
    double start = omp_get_wtime();
    downscaleBicubic(rgb_image, output_image, width, height, total_threads, omp_sched_auto, 0);
    double end = omp_get_wtime();
    printf("Elapsed time with schedule 1 and chunk size 0: %f\n", end-start);

    // Static schedule
    printf("Using static schedule with chunk size 1\n");
    start = omp_get_wtime();
    downscaleBicubic(rgb_image, output_image, width, height, total_threads, omp_sched_static, 1);
    end = omp_get_wtime();
    printf("Elapsed time with schedule 1 and chunk size 1: %f\n", end-start);

    printf("Using static schedule with chunk size 100\n");
    start = omp_get_wtime();
    downscaleBicubic(rgb_image, output_image, width, height, total_threads, omp_sched_static, 100);
    end = omp_get_wtime();
    printf("Elapsed time with schedule 1 and chunk size 100: %f\n", end-start);

    // Dynamic schedule
    printf("Using dynamic schedule with chunk size 1\n");
    start = omp_get_wtime();
    downscaleBicubic(rgb_image, output_image, width, height, total_threads, omp_sched_dynamic, 1);
    end = omp_get_wtime();
    printf("Elapsed time with schedule 2 and chunk size 1: %f\n", end-start);

    printf("Using dynamic schedule with chunk size 100\n");
    start = omp_get_wtime();
    downscaleBicubic(rgb_image, output_image, width, height, total_threads, omp_sched_dynamic, 100);
    end = omp_get_wtime();
    printf("Elapsed time with schedule 2 and chunk size 100: %f\n", end-start);

    // Guided schedule
    printf("Using guided schedule with chunk size 100\n");
    start = omp_get_wtime();
    downscaleBicubic(rgb_image, output_image, width, height, total_threads, omp_sched_guided, 100);
    end = omp_get_wtime();
    printf("Elapsed time with schedule 3 and chunk size 100: %f\n", end-start);

    printf("Using guided schedule with chunk size 1000\n");
    start = omp_get_wtime();
    downscaleBicubic(rgb_image, output_image, width, height, total_threads, omp_sched_guided, 1000);
    end = omp_get_wtime();
    printf("Elapsed time with schedule 3 and chunk size 1000: %f\n", end-start);

    // Performance cores
    printf("Using performance cores...\n");
    start = omp_get_wtime();
    downscaleBicubic(rgb_image, output_image, width, height, perf_threads, omp_sched_static, 100);
    end = omp_get_wtime();
    printf("Elapsed time with performance cores: %f\n", end-start);

    // Efficiency cores
    printf("Using efficiency cores...\n");
    start = omp_get_wtime();
    downscaleBicubic(rgb_image, output_image, width, height, effi_threads, omp_sched_static, 100);
    end = omp_get_wtime();
    printf("Elapsed time with efficiency cores: %f\n", end-start);

    stbi_write_jpg(argv[2], width/2, height/2, CHANNEL_NUM, output_image, 100);

    stbi_image_free(rgb_image);
    free(output_image);

    return 0;
}

/**
 gcc-13 -fopenmp openmp_main.c -o openmp_main -lm
 ./openmp_main aybu.jpg output.jpg 4
 */

