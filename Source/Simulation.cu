/*
 * Simulation.cu
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */

#include <cuda.h>
#include <stdio.h>

#include <stdlib.h>
#include <cuda.h>
#include <stdint.h>

#include "../Header/Simulation.h"

#include "../Header/Device_Particles.h"
#include "../Header/Host_Particles.h"
#include "../Header/ParticlesUtility.h"
#include "../Header/kernels.h"


#define BLOCK_DIM 1024

void DeviceSimulation(Host_Particles* hostParticles, Device_Particles* deviceParticles, float* d_timeDelta)
{
	dim3 block(BLOCK_DIM);
	dim3 grid( ceilf( hostParticles->h_particleNumber / (float)BLOCK_DIM) );

	CalculateForcesGlobal<<<grid, block>>> (deviceParticles->d_positions, deviceParticles->d_velocities, deviceParticles->d_particleNumber, d_timeDelta);
	UpdatePositionsGlobal<<<grid, block>>>(deviceParticles->d_positions, deviceParticles->d_velocities, deviceParticles->d_particleNumber, d_timeDelta);
}


void WriteHostDataToFile(Host_Particles* hostParticles)
{

}


void Simulate(Host_Particles* hostParticles, Device_Particles* deviceParticles, uint64_t h_totalNumberOfSteps, float h_timeDelta)
{

	//Allocate memory for the time delta and copy it to the device
	float* d_timeDelta = NULL;
	cudaMalloc(&d_timeDelta, sizeof(float));
	cudaMemcpy(d_timeDelta, &h_timeDelta, sizeof(float), cudaMemcpyHostToDevice);


	for(uint64_t i = 0; i < h_totalNumberOfSteps; ++i)
	{
		//Sync with the device
		if( cudaDeviceSynchronize() != cudaSuccess)
		{
			fprintf(stderr, "Simulate: cudaError when trying to sync\n");
			exit(1);
		}

		//Copy the new position data over to the host
		CopyDeviceParticlesPositionsToHost(deviceParticles, hostParticles);

		//Call the two simulation kernels, one by one (async)
		DeviceSimulation(hostParticles, deviceParticles, d_timeDelta);

		//While the simulation is occuring, write the host data to a file

	}
}


