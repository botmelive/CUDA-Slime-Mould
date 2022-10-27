 #include "kernel.h"
 #include <stdio.h>
 #include <stdlib.h>
 #ifdef _WIN32
    #define WINDOWS_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
 #endif
 #ifdef __APPLE__
    #include <GLUT/glut.h>
 #else
 #include <GL/glew.h>
 #include <GL/freeglut.h>
 #endif
 #include <cuda_runtime.h>
 #include <cuda_gl_interop.h>
 #include <curand_kernel.h>
 #include "interactions.h"

 // texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
curandState *devStates;
Agent *agents;
struct cudaGraphicsResource *cuda_pbo_resource;

// int frame = 0;
// int time = 0;
// int timebase = 0;

void render() {
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	size_t bufferSize = 0;
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, &bufferSize, cuda_pbo_resource);
	kernelLauncherAgent(devStates, agents, NUM_AGENTS, d_out, W, H, TurnSpeed, SensorSpacing);
	kernelLauncher(d_out, W, H);
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture() {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
	GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

 void display() {
	render();
	drawTexture();
	glutSwapBuffers();
 }

 void initAgents(){

	cudaMalloc((void **)&devStates, NUM_AGENTS * sizeof(curandState));
	cudaMalloc((void**)&agents, NUM_AGENTS * sizeof(Agent));
	
	KernelLauncherSetup(devStates, agents, NUM_AGENTS, W, H);

 }

 void initGLUT(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(W, H);
	glutCreateWindow(TITLE_STRING);
	#ifndef __APPLE__
	glewInit();
	#endif
 }

 void initPixelBuffer() {
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4*W*H*sizeof(GLubyte), 0,
	GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
	cudaGraphicsMapFlagsWriteDiscard);

	initAgents();
 }

 void exitfunc() {
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
 }

 void idle(){
	// frame++;
	// time = glutGet(GLUT_ELAPSED_TIME);
	// if (time - timebase > 1000) {
	// 	printf("FPS:%4.2f",
	// 		frame*1000.0/(time-timebase));
	// 	timebase = time;
	// 	frame = 0;
	// }

	glutPostRedisplay();
 }

 int main(int argc, char** argv) {
	printInstructions();
	
	initGLUT(&argc, argv);
	gluOrtho2D(0, W, H, 0);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(handleSpecialKeypress);
	glutPassiveMotionFunc(mouseMove);
	glutMotionFunc(mouseDrag);
	glutIdleFunc(idle);
	glutDisplayFunc(display);
	
	initPixelBuffer();
	
	glutMainLoop();
	atexit(exitfunc);
	return 0;
 }
