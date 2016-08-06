/*
 * Simulation.h
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */

#ifndef SIMULATION_H_
#define SIMULATION_H_

#include "Host_Particles.h"
#include "Device_Particles.h"
#include "ParticlesUtility.h"


void Simulate(Host_Particles* hostParticles, Device_Particles* deviceParticles, uint64_t h_totalNumberOfSteps, float h_timeDelta);



#endif /* SIMULATION_H_ */
