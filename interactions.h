#pragma once

#include <stdio.h>
//#include "kernel.h"

#define W 2560
#define H 1440
#define NUM_AGENTS 1000000

#define DELTA 5 // pixel increment for arrow keys
#define TITLE_STRING "flashlight: distance image display app"
int2 loc = {W/2, H/2};
bool dragMode = false; // mouse tracking mode
//float TurnSpeed = 0.55;//0.35;
//float SensorSpacing = 0.45;

void mouseMove(int x, int y) {
  if (dragMode) return;
  loc.x = x;
  loc.y = y;
  //printf("Mouse pos %d, %d\n", x, y);
  glutPostRedisplay();
}

void mouseDrag(int x, int y) {
  if (!dragMode) return;
  loc.x = x;
  loc.y = y;
  glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y) {
  if (key == GLUT_KEY_LEFT) loc.x -= DELTA;
  if (key == GLUT_KEY_RIGHT) loc.x += DELTA;
  if (key == GLUT_KEY_UP) loc.y -= DELTA;
  if (key == GLUT_KEY_DOWN) loc.y += DELTA;
  glutPostRedisplay();
}

void printInstructions() {
printf("AGENTS : %d\n", NUM_AGENTS);
printf("q: Sensor spacing +0.5\n");
printf("w: Sensor spacing -0.5\n"); 
printf("a: Turn speed +0.5\n"); 
printf("d: Turn speed -0.5\n"); 

printf("esc: close graphics window\n");
}