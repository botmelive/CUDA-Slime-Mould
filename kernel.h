#ifndef KERNEL_H
#define KERNEL_H

#include <curand_kernel.h>


struct uchar4;
struct float2;
struct int2;

typedef struct{
	int x, y;
	float angle;
 }Agent;

void kernelLauncher(uchar4 *d_out, int w, int h);
void kernelLauncherAgent(curandState*, Agent*, int, uchar4*, int, int);
void KernelLauncherSetup(curandState*, Agent*, int, int, int);

#endif