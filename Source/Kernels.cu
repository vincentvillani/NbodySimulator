/*
 * Kernels.cu
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */


#include "../Header/kernels.h"


__global__ void CalculateForcesGlobal(float* d_positions, float* d_velocities, uint64_t* particleNum, float* timeDelta)
{
	uint64_t globalIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if(globalIndex >= *particleNum)
		return;

	const float mass = 10.0f;

	//Get this threads masses's position
	float3 ourMassPosition;
	ourMassPosition.x = d_positions[ globalIndex * 3 ];
	ourMassPosition.y = d_positions[ globalIndex * 3 + 1 ];
	ourMassPosition.z = d_positions[ globalIndex * 3 + 2 ];

	float3 forceVector;
	forceVector.x = 0;
	forceVector.y = 0;
	forceVector.z = 0;

	for(uint64_t i = 0; i < *particleNum; ++i)
	{

		//Get a vector from our mass to the current mass
		float3 vectorToCurrentMass;
		vectorToCurrentMass.x = d_positions[i * 3 ] - ourMassPosition.x;
		vectorToCurrentMass.y = d_positions[i * 3 + 1 ] - ourMassPosition.y;
		vectorToCurrentMass.z = d_positions[i * 3 + 2 ] - ourMassPosition.z;

		//Calculate distances
		float distanceSquared = vectorToCurrentMass.x * vectorToCurrentMass.x + vectorToCurrentMass.y * vectorToCurrentMass.y +
				vectorToCurrentMass.z * vectorToCurrentMass.z;
		float distance = sqrtf(distanceSquared);

		//Normalise the vectorToCurrentMass
		vectorToCurrentMass.x /= distance;
		vectorToCurrentMass.y /= distance;
		vectorToCurrentMass.z /= distance;

		//Calculate the force between them
		float force = (mass * mass) / distanceSquared;

		//Add the force this object applies to the force vector
		forceVector.x += force * vectorToCurrentMass.x * *timeDelta;
		forceVector.y += force * vectorToCurrentMass.y * *timeDelta;
		forceVector.z += force * vectorToCurrentMass.z * *timeDelta;
	}

	//Add the total force contribution from all objects this timestep to this objects velocity vector
	d_velocities[globalIndex * 3] += forceVector.x;
	d_velocities[globalIndex * 3 + 1] += forceVector.y;
	d_velocities[globalIndex * 3 + 2] += forceVector.z;

}



__global__ void UpdatePositionsGlobal(float* d_positions, float* d_velocities, uint64_t* particleNum, float* timeDelta)
{
	uint64_t globalIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if(globalIndex >= *particleNum)
		return;


	d_positions[globalIndex * 3] += d_velocities[globalIndex * 3] * *timeDelta;
	d_positions[globalIndex * 3 + 1] += d_velocities[globalIndex * 3 + 1] * *timeDelta;
	d_positions[globalIndex * 3 + 2] += d_velocities[globalIndex * 3 + 2] * *timeDelta;

}

