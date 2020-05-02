#pragma once

#include "cuda.h"
#include "cuda_runtime_api.h"


/**
* Remaps image from x and y previously built maps
* Author: Fabio Bagni
*
* @param d_input source device image
* @param s_w source image width
* @param s_h source image height
* @param s_c source image channel number
* @param d_map1 x axis pixel map
* @param d_map2 y axis pixel map
* @param d_output destination device image
* @param d_w destination image width
* @param d_h destination image height
* @param d_c destination image channel number
*/
void remap(uint8_t *d_input, int s_w, int s_h, int s_c, float *d_map1, float *d_map2, uint8_t *d_output, int d_w, int d_h, int d_c);
