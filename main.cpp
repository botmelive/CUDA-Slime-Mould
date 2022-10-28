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
Settings settings;
int oldTimeSinceStart = 0;

void render(int deltaTime) {
	settings.deltaTime = deltaTime * 0.001f;
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	size_t bufferSize = 0;
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, &bufferSize, cuda_pbo_resource);
	kernelLauncherAgent(devStates, agents, NUM_AGENTS, d_out, W, H, settings);
	kernelLauncher(d_out, W, H, settings);
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
	//printf(".%d .", deltaTime);
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
	int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
    int deltaTime = timeSinceStart - oldTimeSinceStart;
	oldTimeSinceStart = timeSinceStart;
	render(deltaTime);
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
 }

 void exitfunc() {
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
 }

 void idle(){
	glutPostRedisplay();
 }

 void keyboard(unsigned char key, int x, int y) {
  float incerement = 0.5f;
  if (key == 'a')
    settings.turnSpeed += incerement;

  if (key == 's')
    settings.turnSpeed -= incerement;

  if (key == 'q')
    settings.sensorAngleSpacing += incerement;
 
  if (key == 'w')
    settings.sensorAngleSpacing -= incerement;

  printf("Turn Speed: %f --- Sensor Spacing: %f\n", settings.turnSpeed, settings.sensorAngleSpacing);
  fflush(stdout);

  if (key == 27) exit(0);
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
	initAgents();
	
	glutMainLoop();
	atexit(exitfunc);
	return 0;
 }


