/*
 * Particles.cpp
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */

#include "../Header/Host_Particles.h"

#include <stdlib.h>
#include <stdio.h>
#include <new>

Host_Particles::Host_Particles(uint64_t particleNum)
{
	h_particleNumber = particleNum;
	h_arrayLength = particleNum * 3;
	h_positions = new (std::nothrow) float[h_arrayLength];
	h_velocities = new (std::nothrow) float[h_arrayLength];

	if(h_positions == NULL)
	{
		fprintf(stderr, "Host_Particles::Host_Particles: Allocating memory for positions failed\n");
		exit(1);
	}

	if(h_velocities == NULL)
	{
		fprintf(stderr, "Host_Particles::Host_Particles: Allocating memory for velocities failed\n");
		exit(1);
	}

}

Host_Particles::~Host_Particles()
{
	if(h_positions != NULL)
		delete [] h_positions;

	if(h_velocities != NULL)
		delete [] h_velocities;
}

