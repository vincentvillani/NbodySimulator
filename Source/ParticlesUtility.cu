/*
 * ParticlesUtility.cu
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */


#include "../Header/ParticlesUtility.h"

#include <random>


#define POSITION_MIN -10.0f
#define POSITION_MAX 10.0f

#define VELOCITY_MIN -1.0f
#define VELOCITY_MAX 1.0f

void SetInitialParticleStateHost(Host_Particles* hostParticles)
{
	std::default_random_engine randomNumberGenerator;
	std::uniform_real_distribution<float> positionDistribution(POSITION_MIN, POSITION_MAX);
	std::uniform_real_distribution<float> veclocityDistribution(VELOCITY_MIN, VELOCITY_MAX);

	//Set initial positions and velocities
	for(uint64_t i = 0; i < hostParticles->h_arrayLength; ++i)
	{
		hostParticles->h_positions[i] = positionDistribution(randomNumberGenerator);
		hostParticles->h_velocities[i] = veclocityDistribution(randomNumberGenerator);
	}


	printf("Setting initial state complete\n");

}


void CopyHostParticlesToDevice(Host_Particles* hostParticles, Device_Particles* deviceParticles)
{
	//Don't copy the array length and particle number, those are set on instantiation
	//Copy only the position and velocities

	size_t arrayByteLength = sizeof(float) * hostParticles->h_particleNumber * 3;

	cudaMemcpy( &(deviceParticles->d_positions), &(hostParticles->h_positions), arrayByteLength, cudaMemcpyHostToDevice);
	cudaMemcpy( &(deviceParticles->d_velocities), &(hostParticles->h_velocities), arrayByteLength, cudaMemcpyHostToDevice);
}


void CopyDeviceParticlesPositionsToHost(Device_Particles* deviceParticles, Host_Particles* hostParticles)
{
	//Don't copy the array length and particle number, those are set on instantiation
	//Copy only the position

	size_t arrayByteLength = sizeof(float) * hostParticles->h_particleNumber * 3;

	cudaMemcpy( &(hostParticles->h_positions), &(deviceParticles->d_positions), arrayByteLength, cudaMemcpyDeviceToHost);
}

