/**
 *
 * CENG342 Project-2
 *
 * Downscaling OpenMP
 *
 * Usage:  seq_main <input.jpg> <output.jpg>
 *
 *  gcc-13 seq_main.c -o seq_main -lm
 *  ./seq_main aybu.jpg output.jpg
 *
 * @Grup10
 *
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
#include <time.h>

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

void downscaleBicubic(uint8_t* input_image, uint8_t* output_image, int width, int height) {
    int new_width = width / 2;
    int new_height = height / 2;
    double grid_x = (double)width / (double)new_width;
    double grid_y = (double)height / (double)new_height;

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
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.jpg> <output.jpg>\n", argv[0]);
        return 1;
    }

    int width, height, bpp;

    // Reading the image in grey colors
    uint8_t* input_image = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);
    uint8_t* output_image = (uint8_t *)malloc(sizeof(uint8_t) * (width/2) * (height/2));

    printf("Width: %d  Height: %d \n",width,height);
    printf("Input: %s , Output: %s\n",argv[1],argv[2]);

    // start the timer
    double time1 = clock();

    //downscaling
    downscaleBicubic(input_image, output_image, width, height);

    double time2 = clock();
    printf("Elapsed time: %lf \n",(time2-time1)/CLOCKS_PER_SEC);

    // Storing the image
    stbi_write_jpg(argv[2], width/2, height/2, CHANNEL_NUM, output_image, 100);
    stbi_image_free(input_image);
    free(output_image);

    return 0;
}

/**
 gcc-13 seq_main.c -o seq_main -lm
 ./seq_main aybu.jpg output.jpg
 */
