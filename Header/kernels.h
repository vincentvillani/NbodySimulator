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

__global__ void CalculateForcesGlobal(float* d_positions, float* d_velocities, uint64_t* particleNum, float* timeDelta);
__global__ void UpdatePositionsGlobal(float* d_positions, float* d_velocities, uint64_t* particleNum, float* timeDelta);

#endif /* KERNELS_H_ */
