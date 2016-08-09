/*
 * ParticlesUtility.h
 *
 *  Created on: 6 Aug 2016
 *      Author: vincent
 */

#ifndef PARTICLESUTILITY_H_
#define PARTICLESUTILITY_H_

#include "Host_Particles.h"
#include "Device_Particles.h"

void SetInitialParticleStateHost(Host_Particles* hostParticles);
void SetInitialParticleStateUniformClustersHost(Host_Particles* hostParticles);
void SetInitalParticlesStateHostNormalDistributionClusters(Host_Particles* hostParticles);

void CopyHostParticlesToDevice(Host_Particles* hostParticles, Device_Particles* deviceParticles);
void CopyDeviceParticlesPositionsToHost(Device_Particles* deviceParticles, Host_Particles* hostParticles);


#endif /* PARTICLESUTILITY_H_ */
