/*
 * kernels.h
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void CalculateForcesGlobal(float* d_positions, float* d_velocities, uint64_t* d_particleNum, float* d_timeDelta, float* d_mass);
__global__ void UpdatePositionsGlobal(float* d_positions, float* d_velocities, uint64_t* d_particleNum, float* d_timeDelta);

#endif /* KERNELS_H_ */
