/*
 * ParticlesUtility.cu
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */


#include "../Header/ParticlesUtility.h"

#include <random>


#define POSITION_MIN -100.0f
#define POSITION_MAX 100.0f

#define MEAN 0.0f
#define STD_DEV_POS 100.0f
#define STD_DEV_VELOCITY 10.0f

#define VELOCITY_MIN -20.0f
#define VELOCITY_MAX 20.0f

void SetInitialParticleStateHost(Host_Particles* hostParticles)
{
	std::default_random_engine randomNumberGenerator;
	//std::normal_distribution<float> positionDistribution(MEAN, STD_DEV_POS);
	//std::normal_distribution<float> veclocityDistribution(MEAN, STD_DEV_VELOCITY);
	std::uniform_real_distribution<float> positionDistribution(POSITION_MIN, POSITION_MAX);
	std::uniform_real_distribution<float> veclocityDistribution(VELOCITY_MIN, VELOCITY_MAX);

	//Set initial positions and velocities
	for(uint64_t i = 0; i < hostParticles->h_arrayLength; ++i)
	{
		hostParticles->h_positions[i] = positionDistribution(randomNumberGenerator);
		hostParticles->h_velocities[i] = 0; //veclocityDistribution(randomNumberGenerator);
	}


	printf("Setting initial state complete\n");

}


void SetInitialParticleStateUniformClustersHost(Host_Particles* hostParticles)
{
	std::default_random_engine randomNumberGenerator;

	std::uniform_real_distribution<float> positionMeanDistribution(POSITION_MIN, POSITION_MAX);
	std::uniform_real_distribution<float> stdDeviationDistribution(10, STD_DEV_POS);
	std::uniform_real_distribution<float> veclocityDistribution(VELOCITY_MIN, VELOCITY_MAX);

	//Number of clusters to generate (MUST BE DIVISIABLE BY BLOCK SIZE) (power of two)
	const uint32_t numberOfClusters = 8;
	const uint32_t clusterParticleCount = hostParticles->h_particleNumber / numberOfClusters;

	//For each cluster,
	for (uint32_t i = 0; i < numberOfClusters; ++i)
	{
		float currentMin = positionMeanDistribution(randomNumberGenerator);
		float currentMax = stdDeviationDistribution(randomNumberGenerator);

		//If min is actually smaller than max
		if (currentMin <= currentMax)
		{
			std::uniform_real_distribution<float> positionDistribution(currentMin, currentMax);

			//Generate inital values for this cluster
			for (uint32_t j = 0; j < clusterParticleCount * 3; ++j)
			{
				uint32_t index = (i * clusterParticleCount * 3) + j;
				hostParticles->h_positions[index] = positionDistribution(randomNumberGenerator);
				hostParticles->h_velocities[index] = 0; //veclocityDistribution(randomNumberGenerator);
			}
		}
		else //Min is bigger than max
		{
			std::uniform_real_distribution<float> positionDistribution(currentMax, currentMin);

			//Generate inital values for this cluster
			for (uint32_t j = 0; j < clusterParticleCount * 3; ++j)
			{
				uint32_t index = (i * clusterParticleCount * 3) + j;
				hostParticles->h_positions[index] = positionDistribution(randomNumberGenerator);
				hostParticles->h_velocities[index] = 0; //veclocityDistribution(randomNumberGenerator);
			}
		}



		//printf("i: %u\n", i);
	}

	printf("Setting initial state complete\n");
}



void SetInitalParticlesStateHostNormalDistributionClusters(Host_Particles* hostParticles)
{

	std::default_random_engine randomNumberGenerator;

	std::normal_distribution<float> positionMeanDistribution(0, 50);
	std::uniform_real_distribution<float> stdDeviationDistribution(10, 50);
	//std::normal_distribution<float> veclocityDistribution(VELOCITY_MIN, VELOCITY_MAX);

	//Number of clusters to generate (MUST BE DIVISIABLE BY BLOCK SIZE) (power of two)
	const uint32_t numberOfClusters = 32;
	const uint32_t clusterParticleCount = hostParticles->h_particleNumber / numberOfClusters;

	//For each cluster,
	for (uint32_t i = 0; i < numberOfClusters; ++i)
	{
		float currentMean = positionMeanDistribution(randomNumberGenerator);
		float currentStdDev = stdDeviationDistribution(randomNumberGenerator);

		std::normal_distribution<float> positionDistribution(currentMean, currentStdDev);

		//Generate inital values for this cluster
		for (uint32_t j = 0; j < clusterParticleCount * 3; ++j)
		{
			uint32_t index = (i * clusterParticleCount * 3) + j;
			hostParticles->h_positions[index] = positionDistribution(randomNumberGenerator);
			hostParticles->h_velocities[index] = 0; //veclocityDistribution(randomNumberGenerator);
		}


		//printf("i: %u\n", i);
	}

	printf("Setting initial state complete\n");
}


void CopyHostParticlesToDevice(Host_Particles* hostParticles, Device_Particles* deviceParticles)
{
	//Don't copy the array length and particle number, those are set on instantiation
	//Copy only the position and velocities

	size_t arrayByteLength = sizeof(float) * hostParticles->h_particleNumber * 3;

	cudaMemcpy( deviceParticles->d_positions, hostParticles->h_positions, arrayByteLength, cudaMemcpyHostToDevice);
	cudaMemcpy( deviceParticles->d_velocities, hostParticles->h_velocities, arrayByteLength, cudaMemcpyHostToDevice);
}


void CopyDeviceParticlesPositionsToHost(Device_Particles* deviceParticles, Host_Particles* hostParticles)
{
	//Don't copy the array length and particle number, those are set on instantiation
	//Copy only the position

	size_t arrayByteLength = sizeof(float) * hostParticles->h_particleNumber * 3;

	cudaMemcpy( hostParticles->h_positions, deviceParticles->d_positions, arrayByteLength, cudaMemcpyDeviceToHost);
}

