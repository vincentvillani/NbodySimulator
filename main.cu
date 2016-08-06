/*
 * main.cu
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "Header/Host_Particles.h"
#include "Header/Device_Particles.h"
#include "Header/ParticlesUtility.h"
#include "Header/Simulation.h"

#define PARTICLE_NUM 1000

int main()
{

	//Allocate memory for particles on the host
	Host_Particles hostParticles(PARTICLE_NUM);

	//Allocate memory for the particles on the device
	Device_Particles deviceParticles(PARTICLE_NUM);

	//Set the intial state on the host
	SetInitialParticleStateHost(&hostParticles);

	//Copy the initial state over to the device
	CopyHostParticlesToDevice(&hostParticles, &deviceParticles);

	//Run the simulation
	Simulate(&hostParticles, &deviceParticles, 180, 1.0f/60.0f);


	printf("Done...\n");

	return 0;
}



