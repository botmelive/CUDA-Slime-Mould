#ifndef KERNEL_H
#define KERNEL_H

#include <curand_kernel.h>


struct uchar4;
struct float2;
struct int2;

typedef struct{
	float x, y;
	float angle;
 }Agent;

 typedef struct{
	float diffuseRate = 3.0f;
	float decayRate = 0.2f;

	float sensorAngleSpacing = 40.0f;
	int sensorOffsetDistance = 45;
	int sensorSize = 1;

	float turnSpeed = 30.0f;
	float agentSpeed = 0.025f;

	float deltaTime = 0.001;
 }Settings;

void kernelLauncher(uchar4 *d_out, int w, int h, Settings);
void kernelLauncherAgent(curandState*, Agent*, int, uchar4*, int, int, Settings);
void KernelLauncherSetup(curandState*, Agent*, int, int, int);

#endif