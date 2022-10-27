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

    float blurredCol = (sum / 9) / 255.0f;
    float color = d_out[i].x / 255.0f;

    float diffuseWeight = 0.05f;
    blurredCol = color * (1 - diffuseWeight) + blurredCol * diffuseWeight;

    float decayRate = 0.02f;
    color = max(0.0, blurredCol - decayRate * 0.1f);


    //d_out[i].x = sum;
    //d_out[i].y = sum;//clip(d_out[i].y - 1);
    //d_out[i].z = sum;//clip(d_out[i].z - 1);
    
	d_out[i].x = clip(color * 255);
    d_out[i].y = clip(color * 255);
    d_out[i].z = clip(color * 255);
    d_out[i].w = 255;
 }

__device__
float sense(Agent agent, float sensorAngelOffset, uchar4* d_out, int w, int h){
    float sensorAngle = agent.angle + sensorAngelOffset;
    float sensorDirx = cosf(sensorAngle);
    float sensorDiry = sinf(sensorAngle);

    int sensorCenterx = (agent.x * w) + sensorDirx * 35;
    int sensorCentery = (agent.y * h) + sensorDiry * 35;

    int sum = 0;
    int sensorSize = 1;

    for (int offsetX = -sensorSize; offsetX <= sensorSize; offsetX++){
        for (int offsetY = -sensorSize; offsetY <= sensorSize; offsetY++){
            int posx = min(w - 1, max(0, sensorCenterx + offsetX));//sensorCenterx + offsetX;
            int posy = min(h - 1, max(0, sensorCentery + offsetY));

            if (posx >= 0 && posx < w && posy > 0 && posy < h){
                int idx = posx + posy * w;
                sum += d_out[idx].x;
            }
        }
    }

    return sum;
}

__global__
void agentKernel(curandState* state, Agent* agents, uchar4* d_out, int numAgents, int w, int h, float TurnSpeed, float SensorSpacing) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > numAgents) return;

    curandState localState = state[i];

    Agent agent = agents[i];

    float sensorAngleSpacing = SensorSpacing * 2 * 3.1415f;

    float weightForward = sense(agent, 0, d_out, w, h);
    float weightLeft = sense(agent, sensorAngleSpacing, d_out, w, h);
    float weightRight = sense(agent, -sensorAngleSpacing, d_out, w, h);

    float randomSteerStrength = curand_uniform(&localState);
    float turnSpeed = TurnSpeed * 2 * 3.1415;

    if ((weightForward > weightLeft) && (weightForward > weightRight)){
        agent.angle += 0;
    }
    else if ((weightForward < weightLeft) && (weightForward < weightRight)){
        agent.angle = agent.angle + (randomSteerStrength - 0.5) * 2 * turnSpeed;// * 0.1f;
    }
    else if (weightRight > weightLeft){
        agent.angle = agent.angle - randomSteerStrength * turnSpeed;// * 0.1f;
    }
    else if (weightLeft > weightRight){
        agent.angle = agent.angle + randomSteerStrength * turnSpeed;// * 0.1f;
    }

    float directionx = cosf(agent.angle);
    float directiony = sinf(agent.angle);
    float newPosx = agent.x + directionx * 0.00025f;// * 0.001f;
    float newPosy = agent.y + directiony * 0.00025f;// * 0.001f;

    if (newPosx < 0 || newPosx >= 1 || newPosy < 0 || newPosy >= 1){
        newPosx = min(1.0 - 0.005, max(0.0, newPosx));
        newPosy = min(1.0 - 0.005, max(0.0, newPosy));
        agent.angle = curand_uniform(&localState) * 2.0f * 3.14159f;
    }

    agent.x = newPosx;
    agent.y = newPosy;

    agents[i] = agent;
    if (agent.x > 1.0 || agent.y > 1.0 || agent.x < 0.0 || agent.y < 0.0) return;

    int agentx = agent.x * w;
    int agenty = agent.y * h;

    int id = agentx + agenty * w;
    if (id < 0 || id > (w * h)) return;
    d_out[id].x = 255;
    d_out[id].y = 255;
    d_out[id].z = 255;
 }

 __global__
 void setupKernel(curandState* devStates, Agent* agents, int numAgents, int w, int h){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > numAgents) return;
    
    curand_init(1548, i, 0, &devStates[i]);

    curandState localState = devStates[i];
    float aspect = (float)w / h;

    float r = 0.2f * sqrtf(curand_uniform(&localState)); // float from 0.0 to 1.0f
    float theta = curand_uniform(&localState) * 2.0f * 3.1415f;

    float x = 0.5f + r * cosf(theta);
    float y = 0.5f + r * sinf(theta);

    float dx = 0.5 - x;
    float dy = 0.5 - y;
    float angle = atan2f(dy, dx);

    agents[i].x = x;
    agents[i].y = y * aspect - 0.40;
    agents[i].angle = angle;//curand_uniform(&localState);
 }

 void kernelLauncher(uchar4 *d_out, int w, int h) {
    const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
    const dim3 blockSize(TX, TY);
    distanceKernel<<<gridSize, blockSize>>>(d_out, w, h);
 }
// TS = 0.1 SS = 0.2
 void kernelLauncherAgent(curandState* states, Agent* agents, int numAgents, uchar4* d_out, int w, int h, float turnSpeed, float sensorSpacing){
    const dim3 gridSize(2048, 1, 1);
    const dim3 blockSize(512, 1, 1);
    agentKernel<<<gridSize, blockSize>>>(states, agents, d_out, numAgents, w, h, turnSpeed, sensorSpacing);
 }

 void KernelLauncherSetup(curandState* state, Agent* agents, int numAgents, int w, int h){
    const dim3 gridSize(2048, 1, 1);
    const dim3 blockSize(512, 1, 1);
    setupKernel<<<gridSize, blockSize>>>(state, agents, numAgents, w, h);
 }
