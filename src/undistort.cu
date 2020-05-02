#include "undistort.h"

__global__ 
void remapKernel(uint8_t *d_input, int s_w, int s_h, int s_c, float *d_map1, float *d_map2, uint8_t *d_output, int d_w, int d_h, int d_c){
	uint u = min( (int)(blockIdx.x * blockDim.x + threadIdx.x), d_w-1);
	uint v = min( (int)(blockIdx.y * blockDim.y + threadIdx.y), d_h-1);

	int x = min( max(0, (int)(d_map1[(v * d_w + u)])), s_w-1);
	int y = min( max(0, (int)(d_map2[(v * d_w + u)])), s_h-2);

	for(int i = 0; i < d_c; i++){
		d_output[(v * d_w + u) * d_c + i] = d_input[(y * s_w + x) * s_c + i];
	}
}

void remap(uint8_t *d_input, int s_w, int s_h, int s_c, float *d_map1, float *d_map2, uint8_t *d_output, int d_w, int d_h, int d_c){
	dim3 dg( ceil( (float)d_w/32 ), ceil( (float)d_h/8 ) );
	dim3 db( 32, 8);

	remapKernel<<< dg, db >>>(d_input, s_w, s_h, s_c, d_map1, d_map2, d_output, d_w, d_h, d_c);
	cudaDeviceSynchronize();
}
