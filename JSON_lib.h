#ifndef JSON_lib
#define JSON_lib

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char hasLine = 0;

void startJson() {
	printf("{\n");
}

void printJsonLine(char* key, double value) {
	if (hasLine) {
		printf(",\n");
	}
	hasLine = 1;
	printf("\"%s\": %f", key, value);
}

void finishJson() {
	printf("\n}\n");
}

#endif