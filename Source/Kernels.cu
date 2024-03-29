/*
 * Kernels.cu
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */


#include "../Header/kernels.h"


__global__ void CalculateForcesGlobal(float* d_positions, float* d_velocities, uint64_t* d_particleNum, float* d_timeDelta, float* d_mass)
{
	uint64_t globalIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if(globalIndex >= *d_particleNum)
		return;


	float mass = *d_mass;
	const float softeningFactorSquared = 0.5f * 0.5f;


	//Get this threads masses's position
	float3 ourMassPosition;
	ourMassPosition.x = d_positions[ globalIndex * 3 ];
	ourMassPosition.y = d_positions[ globalIndex * 3 + 1 ];
	ourMassPosition.z = d_positions[ globalIndex * 3 + 2 ];


	float3 forceVector;
	forceVector.x = 0;
	forceVector.y = 0;
	forceVector.z = 0;

	for(uint64_t i = 0; i < *d_particleNum; ++i)
	{

		//Get a vector from our mass to the current mass
		float3 vectorToCurrentMass;
		vectorToCurrentMass.x = d_positions[i * 3 ] - ourMassPosition.x;
		vectorToCurrentMass.y = d_positions[i * 3 + 1 ] - ourMassPosition.y;
		vectorToCurrentMass.z = d_positions[i * 3 + 2 ] - ourMassPosition.z;


		//Calculate distances
		float distanceSquared = vectorToCurrentMass.x * vectorToCurrentMass.x + vectorToCurrentMass.y * vectorToCurrentMass.y +
				vectorToCurrentMass.z * vectorToCurrentMass.z;


		//Normalise the vectorToCurrentMass
		vectorToCurrentMass.x *= mass;
		vectorToCurrentMass.y *= mass;
		vectorToCurrentMass.z *= mass;

		//Calculate the force between them
		float denominator = powf( (distanceSquared + softeningFactorSquared), 3.0f / 2.0f);

		//Add the force this object applies to the force vector
		forceVector.x += (vectorToCurrentMass.x / denominator) * *d_timeDelta;
		forceVector.y += (vectorToCurrentMass.y / denominator) * *d_timeDelta;
		forceVector.z += (vectorToCurrentMass.z / denominator) * *d_timeDelta;
	}


	//Add the total force contribution from all objects this timestep to this objects velocity vector
	d_velocities[globalIndex * 3] += forceVector.x;
	d_velocities[globalIndex * 3 + 1] += forceVector.y;
	d_velocities[globalIndex * 3 + 2] += forceVector.z;

}



__global__ void CalculateForcesShared(float* d_positions, float* d_velocities, uint64_t* d_particleNum, float* d_timeDelta, float* d_mass)
{
	uint64_t globalIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if(globalIndex >= *d_particleNum)
		return;

	extern __shared__ int sharedMemory[];

	float mass = *d_mass;
	const float softeningFactorSquared = 0.5f * 0.5f;


	//Get this threads masses's position
	float3 ourMassPosition;
	ourMassPosition.x = d_positions[ globalIndex * 3 ];
	ourMassPosition.y = d_positions[ globalIndex * 3 + 1 ];
	ourMassPosition.z = d_positions[ globalIndex * 3 + 2 ];

	//if(globalIndex == 0)
	//	printf("%f, %f, %f\n", ourMassPosition.x, ourMassPosition.y, ourMassPosition.z);

	float3 forceVector;
	forceVector.x = 0;
	forceVector.y = 0;
	forceVector.z = 0;


	//For each shared memory loop
	for(uint64_t i = 0; i < *d_particleNum; i+= blockDim.x)
	{

		//Each thread in the block should load one x,y,z position
		sharedMemory[threadIdx.x * 3] = d_positions [(i + threadIdx.x) * 3];
		sharedMemory[threadIdx.x * 3 + 1] = d_positions [(i + threadIdx.x) * 3 + 1];
		sharedMemory[threadIdx.x * 3  + 2] = d_positions [(i + threadIdx.x) * 3 + 2];

		//Wait till all threads in this block have written to shared memory
		__syncthreads();


		//Go through each position in the shared memory cache and do our calculations
		for(uint64_t j = 0; j < blockDim.x; ++j)
		{
			//Get a vector from our mass to the current mass
			float3 vectorToCurrentMass;
			vectorToCurrentMass.x = sharedMemory[j * 3 ] - ourMassPosition.x;
			vectorToCurrentMass.y = sharedMemory[j * 3 + 1 ] - ourMassPosition.y;
			vectorToCurrentMass.z = sharedMemory[j * 3 + 2 ] - ourMassPosition.z;


			//Calculate distances
			float distanceSquared = vectorToCurrentMass.x * vectorToCurrentMass.x + vectorToCurrentMass.y * vectorToCurrentMass.y +
					vectorToCurrentMass.z * vectorToCurrentMass.z;


			//Scale the vector by the mass
			vectorToCurrentMass.x *= mass;
			vectorToCurrentMass.y *= mass;
			vectorToCurrentMass.z *= mass;

			//Calculate the force between them
			float denominator = powf( (distanceSquared + softeningFactorSquared), 3.0f / 2.0f);

			//Add the force this object applies to the force vector
			forceVector.x += (vectorToCurrentMass.x / denominator) * *d_timeDelta;
			forceVector.y += (vectorToCurrentMass.y / denominator) * *d_timeDelta;
			forceVector.z += (vectorToCurrentMass.z / denominator) * *d_timeDelta;
		}

		//Make sure all threads in this thread block have finished writing before continuing
		__syncthreads();

	}


	//Add the total force contribution from all objects this timestep to this objects velocity vector
	d_velocities[globalIndex * 3] += forceVector.x;
	d_velocities[globalIndex * 3 + 1] += forceVector.y;
	d_velocities[globalIndex * 3 + 2] += forceVector.z;
}



__global__ void UpdatePositionsGlobal(float* d_positions, float* d_velocities, uint64_t* particleNum, float* d_timeDelta)
{
	uint64_t globalIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if(globalIndex >= *particleNum)
		return;

	d_positions[globalIndex * 3] += d_velocities[globalIndex * 3] * *d_timeDelta;
	d_positions[globalIndex * 3 + 1] += d_velocities[globalIndex * 3 + 1] * *d_timeDelta;
	d_positions[globalIndex * 3 + 2] += d_velocities[globalIndex * 3 + 2] * *d_timeDelta;

}

