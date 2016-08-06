/*
 * Device_Particles.h
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */

#ifndef DEVICE_PARTICLES_H_
#define DEVICE_PARTICLES_H_


#include <stdint.h>

class Device_Particles
{
public:

	uint64_t* d_particleNumber; //Number of particles in total

	float* d_positions; //Array of floats containing the x,y,z positions for each particle
	float* d_velocities; //Arrays of float containing the x,y,z elements of the velocity vectors
	uint64_t* d_arrayLength; //Length of both the positions and velocity arrays


	Device_Particles(uint64_t particleNum);
	~Device_Particles();
};


#endif /* DEVICE_PARTICLES_H_ */
