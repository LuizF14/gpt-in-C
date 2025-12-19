#include "transformer.h"
#include "math_utils.h"

void softmax_layer(float *input, int m, int n) {
    for (int i = 0; i < m; i++) {
        softmax(input + i*n, n);
    }
}

float* build_input_embeddings(GPT2Model *model, int *seq_tokens, int seq_len) {
    Tensor *Wte = get_tensor(model, "wte_weight");
    Tensor *Wpe = get_tensor(model, "wpe_weight");
    float *emb_output = malloc(sizeof(float) * seq_len * N_EMBD);

    if (!Wte || !Wpe) {
        fprintf(stderr, "Erro: embeddings não encontrados!\n");
        return NULL;
    }

    for (int i = 0; i < seq_len; i++) {
        int token_id = seq_tokens[i];
        const float *token_vec = Wte->data + token_id * N_EMBD;
        const float *pos_vec   = Wpe->data + i * N_EMBD;

        for (int j = 0; j < N_EMBD; j++) {
            emb_output[i * N_EMBD + j] = token_vec[j] + pos_vec[j];
        }
    }
    
    return emb_output;
}

void apply_causal_mask(float *scores, int seq_len) {
    const float NEG_INF = -1e9f; // bastante negativo, evita NaN com exp
    for (int i = 0; i < seq_len; ++i) {
        int row_off = i * seq_len;
        for (int k = i + 1; k < seq_len; ++k) {
            scores[row_off + k] = NEG_INF;
        }
    }
}

void multihead_attention(float *Q, float *K, float *V, float *out,
                         int seq_len) {
    int head_dim = N_EMBD / N_HEADS;

    float *Qh = malloc(sizeof(float) * seq_len * head_dim);
    float *Kh = malloc(sizeof(float) * seq_len * head_dim);
    float *Vh = malloc(sizeof(float) * seq_len * head_dim);
    float *Kt = malloc(sizeof(float) * seq_len * head_dim);

    float *attn = malloc(sizeof(float) * seq_len * seq_len);
    float *context = malloc(sizeof(float) * seq_len * head_dim);
    memset(out, 0, sizeof(float) * seq_len * N_EMBD);

    for (int h = 0; h < N_HEADS; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                Qh[i * head_dim + d] = Q[i * N_EMBD + h * head_dim + d];
                Kh[i * head_dim + d] = K[i * N_EMBD + h * head_dim + d];
                Vh[i * head_dim + d] = V[i * N_EMBD + h * head_dim + d];
            }
        }

        transpose(Kt, Kh, seq_len, head_dim);
        matmul(attn, Qh, Kt, seq_len, head_dim, seq_len);
        rescale(attn, seq_len, seq_len, 1.0f/sqrtf((float)head_dim));

        apply_causal_mask(attn, seq_len);

        softmax_layer(attn, seq_len, seq_len);
        matmul(context, attn, Vh, seq_len, seq_len, head_dim);
        
        for (int i = 0; i < seq_len; i++) {
            memcpy(out + i * N_EMBD + h * head_dim,
                   context + i * head_dim,
                   head_dim * sizeof(float));
        }
    }

    free(Qh);
    free(Kh);
    free(Vh);
    free(Kt);
    free(attn);
    free(context);
}

void layernorm(float *input, int seq_len, const float *weight, const float *bias) {
    for (int i = 0; i < seq_len; i++)
        norm(&input[i * N_EMBD], N_EMBD, weight, bias, &input[i * N_EMBD]);
}

void transformer_block(GPT2Model *model, float *input, int seq_len, int layer) {
    char name[256];

    snprintf(name, sizeof(name), "h_%d_ln_1_weight", layer);
    Tensor *ln1_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_ln_1_bias", layer);
    Tensor *ln1_b = get_tensor(model, name);

    float* residual = malloc(sizeof(float) * seq_len * N_EMBD);
    copy(residual, input, seq_len * N_EMBD);

    layernorm(input, seq_len, ln1_w->data, ln1_b->data);

    snprintf(name, sizeof(name), "h_%d_attn_c_attn_weight", layer);
    Tensor *c_attn_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_attn_c_attn_bias", layer);
    Tensor *c_attn_b = get_tensor(model, name);

    int n_qkv = 3 * N_EMBD;
    float *QKV = malloc(sizeof(float) * seq_len * n_qkv);
    matmul(QKV, input, c_attn_w->data, seq_len, N_EMBD, n_qkv);
    add_bias(QKV, c_attn_b->data, seq_len, n_qkv);

    // separa Q, K, V
    float *Q = malloc(sizeof(float) * seq_len * N_EMBD);
    float *K = malloc(sizeof(float) * seq_len * N_EMBD);
    float *V = malloc(sizeof(float) * seq_len * N_EMBD);
    for (int i = 0; i < seq_len; i++) {
        memcpy(&Q[i * N_EMBD], &QKV[i * n_qkv], N_EMBD * sizeof(float));
        memcpy(&K[i * N_EMBD], &QKV[i * n_qkv + N_EMBD], N_EMBD * sizeof(float));
        memcpy(&V[i * N_EMBD], &QKV[i * n_qkv + 2 * N_EMBD], N_EMBD * sizeof(float));
    }

    free(QKV);

    float *context = malloc(sizeof(float) * seq_len * N_EMBD);
    multihead_attention(Q, K, V, context, seq_len);

    snprintf(name, sizeof(name), "h_%d_attn_c_proj_weight", layer);
    Tensor *c_proj_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_attn_c_proj_bias", layer);
    Tensor *c_proj_b = get_tensor(model, name);

    float *proj = malloc(sizeof(float) * seq_len * N_EMBD);
    matmul(proj, context, c_proj_w->data, seq_len, N_EMBD, N_EMBD);
    add_bias(proj, c_proj_b->data, seq_len, N_EMBD);

    add_inplace(residual, proj, seq_len, N_EMBD);
    copy(input, residual, seq_len * N_EMBD);

    snprintf(name, sizeof(name), "h_%d_ln_2_weight", layer);
    Tensor *ln2_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_ln_2_bias", layer);
    Tensor *ln2_b = get_tensor(model, name);

    layernorm(input, seq_len, ln2_w->data, ln2_b->data);

    snprintf(name, sizeof(name), "h_%d_mlp_c_fc_weight", layer);
    Tensor *fc_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_mlp_c_fc_bias", layer);
    Tensor *fc_b = get_tensor(model, name);

    float *fc_out = malloc(sizeof(float) * seq_len * 4 * N_EMBD);
    matmul(fc_out, input, fc_w->data, seq_len, N_EMBD, 4 * N_EMBD);
    add_bias(fc_out, fc_b->data, seq_len, 4 * N_EMBD);

    for (int i = 0; i < seq_len * 4 * N_EMBD; i++)
        fc_out[i] = gelu(fc_out[i]);

    snprintf(name, sizeof(name), "h_%d_mlp_c_proj_weight", layer);
    Tensor *proj_w = get_tensor(model, name);

    snprintf(name, sizeof(name), "h_%d_mlp_c_proj_bias", layer);
    Tensor *proj_b = get_tensor(model, name);

    float *mlp_out = malloc(sizeof(float) * seq_len * N_EMBD);
    matmul(mlp_out, fc_out, proj_w->data, seq_len, 4*N_EMBD, N_EMBD);
    add_bias(mlp_out, proj_b->data, seq_len, N_EMBD);

    add_inplace(residual, mlp_out, seq_len, N_EMBD);
    copy(input, residual, seq_len * N_EMBD);

    free(proj);
    free(context);
    free(Q);
    free(K);
    free(V);
    free(fc_out);
    free(mlp_out);
    free(residual);

    // free(ln1_w);
    // free(ln1_b);
    // free(c_attn_w);
    // free(c_attn_b);
    // free(c_proj_w);
    // free(c_proj_b);
    // free(ln2_w);
    // free(ln2_b);
    // free(fc_w);
    // free(fc_b);
    // free(proj_w);
    // free(proj_b);
}

void final_block(GPT2Model *model, float *input, int seq_len) {
    Tensor *ln_f_w = get_tensor(model, "ln_f_weight");
    Tensor *ln_f_b = get_tensor(model, "ln_f_bias");

    layernorm(input, seq_len, ln_f_w->data, ln_f_b->data);
    // free(ln_f_w);
    // free(ln_f_b);
}

float* final_logits(GPT2Model *model, float *x, int seq_len) {
    Tensor *Wte = get_tensor(model, "wte_weight");

    float *logits = malloc(sizeof(float) * seq_len * MAX_VOCAB);

    float *wte_transposed = malloc(sizeof(float) * MAX_VOCAB * N_EMBD);
    transpose(wte_transposed, Wte->data, MAX_VOCAB, N_EMBD);
    matmul(logits, x, wte_transposed, seq_len, N_EMBD, MAX_VOCAB);
    softmax(&logits[(seq_len-1)*MAX_VOCAB], MAX_VOCAB);
    return logits;
}