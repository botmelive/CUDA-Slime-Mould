#include "kernel.h"
#include <stdio.h>

#define TX 32
#define TY 32

 __device__
 unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

 __global__
 void distanceKernel(uchar4 *d_out, int w, int h) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    
	if ((c >= w) || (r >= h)) return; // Check if within image bounds
    
	const int i = c + r * w; // 1D indexing

    int index = 0;
    int OIX, OIY;
    int sum = 0;
    for(int offsetX = -1; offsetX <= 1; offsetX++){
        for(int offsetY = -1; offsetY <= 1; offsetY++){
            OIX = (c + offsetX);// + (r + offsetY) * w;
            OIY = (r + offsetY);
            if ((OIX >= w) || (OIY >= h) || (OIX < 0) || (OIY < 0)) break; 

            index = OIX + OIY * w;
            sum += d_out[index].x;
        }
    }

    sum = sum / 9;
    d_out[i].x = sum;
    d_out[i].y = sum;//clip(d_out[i].y - 1);
    d_out[i].z = sum;//clip(d_out[i].z - 1);
    
	d_out[i].x = clip(d_out[i].x - 1);
    d_out[i].y = clip(d_out[i].y - 1);
    d_out[i].z = clip(d_out[i].z - 1);
    d_out[i].w = 255;
 }

  __global__
 void agentKernel(curandState* state, Agent* agents, uchar4* d_out, int numAgents, int w, int h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > numAgents) return;

    curandState localState = state[i];

    Agent agent = agents[i];
    float directionx = cosf(agent.angle);
    float directiony = sinf(agent.angle);
    int newPosx = agent.x + directionx * 2;// * 0.001f;
    int newPosy = agent.y + directiony * 2;// * 0.001f;

    if (newPosx < 0 || newPosx >= w || newPosy < 0 || newPosy >= h){
        newPosx = min(w - 1, max(0, newPosx));
        newPosy = min(h - 1, max(0, newPosy));
        agent.angle = curand_uniform(&localState) * 2.0f * 3.14159f;
    }

    agent.x = newPosx;
    agent.y = newPosy;

    agents[i] = agent;
    //int agentx = agent.x * w;
    //int agenty = agent.y * h;
    if (agent.x > w || agent.y > h || agent.x < 0 || agent.y < 0) return;

    int id = agent.x + agent.y * w;
    if (id < 0 || id > (w * h)) return;
    d_out[id].x = 255;
    d_out[id].y = 255;
    d_out[id].z = 255;
 }

 __global__
 void setupKernel(curandState* devStates, Agent* agents, int numAgents, int w, int h){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > numAgents) return;
    
    curand_init(8954, i, 0, &devStates[i]);

    curandState localState = devStates[i];

    float x = curand_uniform(&localState); // float from 0.0 to 1.0f
    float y = curand_uniform(&localState);
    float angle = curand_uniform(&localState) * 2.0f * 3.1415f;

    agents[i].x = x * w;
    agents[i].y = y * h;
    agents[i].angle = angle;//curand_uniform(&localState);
 }

 void kernelLauncher(uchar4 *d_out, int w, int h) {
    const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
    const dim3 blockSize(TX, TY);
    distanceKernel<<<gridSize, blockSize>>>(d_out, w, h);
 }

 void kernelLauncherAgent(curandState* states, Agent* agents, int numAgents, uchar4* d_out, int w, int h){
    const dim3 gridSize(2048, 1, 1);
    const dim3 blockSize(512, 1, 1);
    agentKernel<<<gridSize, blockSize>>>(states, agents, d_out, numAgents, w, h);
 }

 void KernelLauncherSetup(curandState* state, Agent* agents, int numAgents, int w, int h){
    const dim3 gridSize(2048, 1, 1);
    const dim3 blockSize(512, 1, 1);
    setupKernel<<<gridSize, blockSize>>>(state, agents, numAgents, w, h);
 }