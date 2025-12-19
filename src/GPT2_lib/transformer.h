#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "GPT2_lib.h"

float* build_input_embeddings(GPT2Model *model, int *seq_tokens, int seq_len);
void transformer_block(GPT2Model *model, float *input, int seq_len, int layer);
void final_block(GPT2Model *model, float *input, int seq_len);
float* final_logits(GPT2Model *model, float *x, int seq_len);

int argmax(float *arr, int len);
void multihead_attention(float *Q, float *K, float *V, float *out, int seq_len);

#endif
