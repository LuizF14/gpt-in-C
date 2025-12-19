#ifndef MATH_UTILS
#define MATH_UTILS

#define EPS 1e-5

#include <stdlib.h>
#include <string.h>
#include <math.h>

void matmul(float *out, float *a, float *b, int m, int n, int p);
void copy(float *dst, const float *src, size_t n);
void softmax(float *x, int n);
float gelu(float x);
double get_mean(float *input, int n_embd);
double get_var(float *input, int n_embd, double mean);
void norm(float *input, int n_elements, const float *gamma, const float *beta, float *out);
void add_bias(float *input, float *bias, int m, int n);
void add_inplace(float *a, float *b, int m, int n);
void rescale(float *input, int m, int n, float scale);
void transpose(float* output, const float* input, int m, int n);
int argmax(float *arr, int len);

#endif