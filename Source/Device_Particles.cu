/*
 * Device_Particles.cu
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */


#include "../Header/Device_Particles.h"

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

Device_Particles::Device_Particles(uint64_t particleNum)
{

	d_particleNumber = NULL;
	d_arrayLength = NULL;
	d_positions = NULL;
	d_velocities = NULL;


	size_t arrayByteSize = sizeof(float) * particleNum * 3;

	cudaError_t cudaError;

	//Allocate memory
	cudaError = cudaMalloc(&d_particleNumber, sizeof(uint64_t));
	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Device_Particles::Device_Particles: Allocating memory for d_particleNumber failed\n");
		exit(1);
	}

	cudaError = cudaMalloc(&d_arrayLength, sizeof(uint64_t));
	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Device_Particles::Device_Particles: Allocating memory for d_arrayLength failed\n");
		exit(1);
	}

	cudaError = cudaMalloc(&d_positions, arrayByteSize);
	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Device_Particles::Device_Particles: Allocating memory for d_positions failed\n");
		exit(1);
	}

	cudaError = cudaMalloc(&d_velocities, arrayByteSize);
	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "Device_Particles::Device_Particles: Allocating memory for d_positions failed\n");
		exit(1);
	}

	//Copy the particle number and length values over
	uint64_t h_arrayLength = particleNum * 3;
	cudaMemcpy(d_particleNumber, &particleNum, sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arrayLength, &h_arrayLength, sizeof(uint64_t), cudaMemcpyHostToDevice);

}

Device_Particles::~Device_Particles()
{
	if(d_particleNumber != NULL)
		cudaFree(d_particleNumber);

	if(d_arrayLength != NULL)
		cudaFree(d_arrayLength);

	if(d_positions != NULL)
		cudaFree(d_positions);

	if(d_velocities != NULL)
		cudaFree(d_velocities);
}


