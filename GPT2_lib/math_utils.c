#include "math_utils.h"

void matmul(float *out, float *a, float *b, int m, int n, int p) {
    memset(out, 0, m * p * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            float val = a[i*n + k];
            for (int j = 0; j < p; j++) {
                out[i*p + j] += val * b[k*p + j];
            }
        }
    }
}

void copy(float *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

void softmax(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) 
        if (x[i] > maxv) 
            maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { 
        x[i] = expf(x[i] - maxv); sum += x[i];
    }
    for (int i = 0; i < n; i++) 
        x[i] /= sum;
}

float gelu(float x) {
    return x * 0.5 * (1 + erf(x / sqrt(2)));
}

double get_mean(float *input, int n_embd) {
    double mean;
    for (int i = 0; i < n_embd; i++)
        mean += input[i];
    mean /= n_embd;
    return mean;
}

double get_var(float *input, int n_embd, double mean) {
    double var;
    for (int i = 0; i < n_embd; i++) {
        float u = input[i] - mean;
        var += u * u;
    }
    var = sqrtf(var / n_embd + EPS);
    return var;
}

void norm(float *input, int n_elements, const float *gamma, const float *beta, float *out) {
    double mean = get_mean(input, n_elements);
    double var = get_var(input, n_elements, mean);
    
    for (int i = 0; i < n_elements; i++) {
        float norm = (input[i] - mean) / var;
        out[i] = norm * gamma[i] + beta[i];
    }
}

void add_bias(float *input, float *bias, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            input[i * n + j] += bias[j];
}

void add_inplace(float *a, float *b, int m, int n) {
    int size = m * n;
    for (int i = 0; i < size; i++)
        a[i] += b[i];
}

void rescale(float *input, int m, int n, float scale) {
    for (int i = 0; i < m * n; i++)
            input[i] *= scale;
}

void transpose(float* output, const float* input, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // output[j, i] = input[i, j]
            output[j * m + i] = input[i * n + j];
        }
    }
}

int argmax(float *arr, int len) {
    if (len <= 0) return -1;
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}