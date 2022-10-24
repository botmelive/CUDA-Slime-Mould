#ifndef INTERACTIONS_H

 #define INTERACTIONS_H
 #include <stdio.h>
 #define W 1920
 #define H 1080
 #define NUM_AGENTS 5000000

 #define DELTA 5 // pixel increment for arrow keys
 #define TITLE_STRING "flashlight: distance image display app"
 int2 loc = {W/2, H/2};
 bool dragMode = false; // mouse tracking mode
 float TurnSpeed = 0.45;
 float SensorSpacing = 0.15;

 void keyboard(unsigned char key, int x, int y) {
    if (key == 'a'){
      TurnSpeed += 0.05f;
      if (TurnSpeed > 1) TurnSpeed = 1.0;
    }

    if (key == 's'){
      TurnSpeed -= 0.05f;
      if (TurnSpeed < 0) TurnSpeed = 0.0;
    }

    if (key == 'q'){
      SensorSpacing += 0.05;
      if (SensorSpacing > 1.0) SensorSpacing = 1.0;
    }
    
    if (key == 'w'){
      SensorSpacing -= 0.05;
      if (SensorSpacing < 0) SensorSpacing = 0.0;
    }

    printf("Turn Speed: %f --- Sensor Spacing: %f\n", TurnSpeed, SensorSpacing);

    if (key == 27) exit(0);
    glutPostRedisplay();
 }

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
   printf("q: Sensor spacing +0.5\n");
   printf("w: Sensor spacing -0.5\n"); 
   printf("a: Turn speed +0.5\n"); 
   printf("d: Turn speed -0.5\n"); 

   printf("esc: close graphics window\n");
 }

 #endif