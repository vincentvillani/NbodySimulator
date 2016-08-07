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

	//if(globalIndex == 0)
	//	printf("particleCount: %llu\n", *d_particleNum);


	float mass = *d_mass;
	const float softeningFactorSquared = 0.5f * 0.5f;

	//if(globalIndex == 0)
	//	printf("Time Delta: %f\nMass: %f\n", *d_timeDelta, *d_mass);


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

	for(uint64_t i = 0; i < *d_particleNum; ++i)
	{

		//if(i == globalIndex)
		//	continue;

		//Get a vector from our mass to the current mass
		float3 vectorToCurrentMass;
		vectorToCurrentMass.x = d_positions[i * 3 ] - ourMassPosition.x;
		vectorToCurrentMass.y = d_positions[i * 3 + 1 ] - ourMassPosition.y;
		vectorToCurrentMass.z = d_positions[i * 3 + 2 ] - ourMassPosition.z;

		//if(globalIndex == 0 && i == 1)
			//printf("%f, %f, %f\n", vectorToCurrentMass.x, vectorToCurrentMass.y, vectorToCurrentMass.z);


		//Calculate distances
		float distanceSquared = vectorToCurrentMass.x * vectorToCurrentMass.x + vectorToCurrentMass.y * vectorToCurrentMass.y +
				vectorToCurrentMass.z * vectorToCurrentMass.z;
		//float distance = sqrtf(distanceSquared);

		/*
		if(fabs(distanceSquared) == 0.0f)
		{
			//printf("ZERO!\n");
			continue;
		}
		*/

		/*
		if(globalIndex == 0)
		{
			printf("d2: %f\n", distanceSquared);
			printf("distance: %f\n", distance);
		}
		*/


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

	//if(globalIndex == 0)
	//	printf("%f, %f, %f\n", forceVector.x, forceVector.y, forceVector.z);

	//if(globalIndex == 0)
	//	printf("%f, %f, %f\n", d_velocities[globalIndex * 3], d_velocities[globalIndex * 3 + 1], d_velocities[globalIndex * 3 + 2]);

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

	/*
	if(globalIndex == 0)
	{
		printf("Time Delta: %f\n", *d_timeDelta);
		printf("%f, %f, %f\n\n", d_velocities[globalIndex * 3], d_velocities[globalIndex * 3 + 1], d_velocities[globalIndex * 3 + 2]);
	}
	*/

	d_positions[globalIndex * 3] += d_velocities[globalIndex * 3] * *d_timeDelta;
	d_positions[globalIndex * 3 + 1] += d_velocities[globalIndex * 3 + 1] * *d_timeDelta;
	d_positions[globalIndex * 3 + 2] += d_velocities[globalIndex * 3 + 2] * *d_timeDelta;

}

