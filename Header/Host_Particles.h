/*
 * Particles.h
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */

#ifndef PARTICLES_H_
#define PARTICLES_H_

#include <stdint.h>

class Host_Particles
{
public:

	uint64_t h_particleNumber; //Number of particles in total

	float* h_positions; //Array of floats containing the x,y,z positions for each particle
	float* h_velocities; //Arrays of float containing the x,y,z elements of the velocity vectors
	uint64_t h_arrayLength; //Length of both the positions and velocity arrays


	Host_Particles(uint64_t particleNum);
	~Host_Particles();
};

#endif /* PARTICLES_H_ */
