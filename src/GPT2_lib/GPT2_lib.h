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
#define MAX_NAME_LEN 256
#define MAX_TENSORS 4096

#define MAX_VOCAB 50257
#define MAX_TOKEN_LEN 127

#define MAX_SEQ 1024
#define N_EMBD 768
#define N_HEADS 12

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

char* generate_text(int* seq_tokens, int* seq_len, GPT2Model *model, Vocab *vocab);

void load_model(GPT2Model *model, const char *dir_path);
void print_tensor_info(const Tensor *t, int n_vals);
void free_model(GPT2Model *model);

Tensor* get_tensor(GPT2Model *model, const char *name);
void load_vocab(Vocab *vocab, const char *filename);

int encode_token(Vocab *vocab, const char *word);
char* decode_token(Vocab *vocab, int id);
int* encode_seq(Vocab *vocab, char** prompt, int seq_len);

#endif