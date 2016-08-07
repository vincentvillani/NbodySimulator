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


#define BLOCK_SIZE (1024)
#define PARTICLE_NUM (BLOCK_SIZE * 10) //MUST BE A MULTIPLE OF BLOCK SIZE
#define FRAME_RATE (144)
#define TIME_STEP (1.0f / FRAME_RATE)
#define SIMULATION_SECONDS (40)
#define MASS (5.0f)



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
	Simulate(&hostParticles, &deviceParticles, FRAME_RATE * SIMULATION_SECONDS, TIME_STEP, MASS, BLOCK_SIZE);


	printf("Done...\n");

	return 0;
}



