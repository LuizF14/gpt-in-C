#ifndef GPT2_LIB
#define GPT2_LIB

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

#define MAX_DIMS 4
#define MAX_NAME_LEN 128
#define MAX_TENSORS 4096

#define MAX_VOCAB 50257
#define MAX_TOKEN_LEN 128

#define MAX_SEQ 1024
// #define N_EMBD 768

typedef struct {
    char name[MAX_NAME_LEN];
    int shape[MAX_DIMS];
    int ndims;
    size_t num_elements;
    float *data;
} Tensor;

typedef struct {
    Tensor tensors[MAX_TENSORS];
    int count;
} GPT2Model;

typedef struct {
    char token[MAX_TOKEN_LEN];
    int id;
} VocabEntry;

typedef struct {
    VocabEntry entries[MAX_VOCAB];
    int size;
} Vocab;

int read_shape(const char *filename, int *shape, int max_dims);
float *read_bin(const char *filename, size_t *num_elements);
void load_model(GPT2Model *model, const char *dir_path);
void print_tensor_info(const Tensor *t, int n_vals);
void free_model(GPT2Model *model);

Tensor* get_tensor(GPT2Model *model, const char *name);

float gelu(float x);
void matmul(float *out, float *a, float *b, int m, int n, int p) ;
void softmax(float *x, int n);
void multihead_attention(float *Q, float *K, float *V, float *out, int seq_len, int n_embd, int n_heads);

float* build_input_embeddings(GPT2Model *model, int *seq_tokens, int seq_len, int* n_embd);
void transformer_block(GPT2Model *model, float *x, int seq_len, int n_embd, int n_heads, int layer);
float* final_logits(GPT2Model *model, float *x, int seq_len, int n_embd);

void load_vocab(Vocab *vocab, const char *filename);
int encode_token(Vocab *vocab, const char *word);
const char* decode_token(Vocab *vocab, int id);

int argmax(float *arr, int len);

void top_k_argmax(float *arr, int len, int *top_indices, int k);

#endif