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

void DeviceSimulationGlobal(Host_Particles* hostParticles, Device_Particles* deviceParticles, float* d_timeDelta, float* d_mass, uint32_t blockDim)
{
	dim3 block(blockDim);
	dim3 grid( ceilf( hostParticles->h_particleNumber / (float)blockDim) );
	//printf("block: %u\ngrid: %u\n", block.x, grid.x);

	CalculateForcesGlobal<<<grid, block>>> (deviceParticles->d_positions, deviceParticles->d_velocities, deviceParticles->d_particleNumber, d_timeDelta, d_mass);
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );

	UpdatePositionsGlobal<<<grid, block>>>(deviceParticles->d_positions, deviceParticles->d_velocities, deviceParticles->d_particleNumber, d_timeDelta);
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );
}


void DeviceSimulationShared(Host_Particles* hostParticles, Device_Particles* deviceParticles, float* d_timeDelta, float* d_mass, uint32_t blockDim)
{
	dim3 block(blockDim);
	dim3 grid( ceilf( hostParticles->h_particleNumber / (float)blockDim) );
	//printf("block: %u\ngrid: %u\n", block.x, grid.x);

	CalculateForcesShared<<<grid, block, sizeof(float) * blockDim * 3>>> (deviceParticles->d_positions, deviceParticles->d_velocities, deviceParticles->d_particleNumber, d_timeDelta, d_mass);
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );

	UpdatePositionsGlobal<<<grid, block>>>(deviceParticles->d_positions, deviceParticles->d_velocities, deviceParticles->d_particleNumber, d_timeDelta);
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );
}


void WriteHostDataToFile(Host_Particles* hostParticles, uint64_t frameNumber, FILE* outputFile)
{

	fwrite(&frameNumber, sizeof(uint64_t), 1, outputFile);
	fwrite(hostParticles->h_positions, sizeof(float) * hostParticles->h_arrayLength, 1, outputFile);

}


void WriteHeaderInformationToOutputFile(Host_Particles* hostParticles, uint64_t h_totalNumberOfSteps, float h_timeDelta, float mass, FILE* outputFile)
{
	uint64_t headerByteSize = sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(float) + sizeof(float);
	uint64_t expectedFileByteSize = headerByteSize + (h_totalNumberOfSteps * hostParticles->h_particleNumber * 3 * sizeof(float)) +
			h_totalNumberOfSteps * sizeof(uint64_t);

	fwrite(&expectedFileByteSize, sizeof(uint64_t),1, outputFile);
	fwrite(&h_totalNumberOfSteps, sizeof(uint64_t), 1, outputFile);
	fwrite(&(hostParticles->h_particleNumber), sizeof(uint64_t), 1, outputFile);
	fwrite(&h_timeDelta, sizeof(float), 1, outputFile);
	fwrite(&mass, sizeof(float), 1, outputFile);

}


void Simulate(Host_Particles* hostParticles, Device_Particles* deviceParticles, uint64_t h_totalNumberOfSteps, float h_timeDelta, float h_mass, uint32_t blockDim)
{

	//Open up the output file
	FILE* outputFile = fopen("OutputFile.sim", "wb");

	if(outputFile == NULL)
	{
		fprintf(stderr, "Simulate: Unable to open output file\n");
		exit(1);
	}

	//Write the file header
	WriteHeaderInformationToOutputFile(hostParticles, h_totalNumberOfSteps, h_timeDelta, h_mass, outputFile);


	//Allocate memory for the time delta and mass and copy it to the device
	float* d_timeDelta = NULL;
	cudaMalloc(&d_timeDelta, sizeof(float));
	cudaMemcpy(d_timeDelta, &h_timeDelta, sizeof(float), cudaMemcpyHostToDevice);

	float* d_mass = NULL;
	cudaMalloc(&d_mass, sizeof(float));
	cudaMemcpy(d_mass, &h_mass, sizeof(float), cudaMemcpyHostToDevice);



	uint32_t frameCounter = 0;

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
		//DeviceSimulationGlobal(hostParticles, deviceParticles, d_timeDelta, d_mass, blockDim);
		DeviceSimulationShared(hostParticles, deviceParticles, d_timeDelta, d_mass, blockDim);

		//While the simulation is occuring, write the host data to a file
		WriteHostDataToFile(hostParticles, i, outputFile);

		frameCounter += 1;

		if(frameCounter == 9)
		{
			frameCounter = 0;
			printf("Frame: %lu/%lu\n", i, h_totalNumberOfSteps);
		}


	}


	fclose(outputFile);

}



