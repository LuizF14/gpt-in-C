#include "GPT2_lib.h"

// --- Lê shape ---
int read_shape(const char *filename, int *shape, int max_dims) {
    FILE *f = fopen(filename, "r");
    if (!f) return -1;
    int count = 0;
    while (count < max_dims && fscanf(f, "%d", &shape[count]) == 1)
        count++;
    fclose(f);
    return count;
}

// --- Lê binário ---
float *read_bin(const char *filename, size_t *num_elements) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *num_elements = file_size / sizeof(float);
    float *data = (float *)malloc(file_size);
    fread(data, sizeof(float), *num_elements, f);
    fclose(f);
    return data;
}

// --- Função para carregar todos os tensores de uma pasta ---
void load_model(GPT2Model *model, const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        perror("Erro ao abrir diretório");
        return;
    }

    struct dirent *entry;
    model->count = 0;

    while ((entry = readdir(dir))) {
        // Só processa arquivos .shape
        if (!strstr(entry->d_name, ".shape")) continue;

        // Extrai nome base (sem .shape)
        char base[MAX_NAME_LEN];
        strncpy(base, entry->d_name, strlen(entry->d_name) - 6);
        base[strlen(entry->d_name) - 6] = '\0';

        // Monta caminhos
        char shape_file[256], bin_file[256];
        snprintf(shape_file, sizeof(shape_file), "%s/%s.shape", dir_path, base);
        snprintf(bin_file, sizeof(bin_file), "%s/%s.bin", dir_path, base);

        Tensor *t = &model->tensors[model->count];
        strncpy(t->name, base, MAX_NAME_LEN);

        // Lê shape
        t->ndims = read_shape(shape_file, t->shape, MAX_DIMS);

        // Lê dados
        t->data = read_bin(bin_file, &t->num_elements);
        if (!t->data) {
            fprintf(stderr, "Falha ao ler %s\n", bin_file);
            continue;
        }

        model->count++;
        if (model->count >= MAX_TENSORS) break;
    }

    closedir(dir);
    // printf("✅ Carregados %d tensores do diretório %s\n", model->count, dir_path);
}

// --- Função de debug ---
void print_tensor_info(const Tensor *t, int n_vals) {
    printf("Tensor: %s\n", t->name);
    printf("Dimensões: ");
    for (int i = 0; i < t->ndims; i++)
        printf("%d ", t->shape[i]);
    printf("\nPrimeiros valores: ");
    for (int i = 0; i < n_vals && i < t->num_elements; i++)
        printf("%.5f ", t->data[i]);
    printf("\n\n");
}

void free_model(GPT2Model *model) {
    for (int i = 0; i < model->count; i++)
        free(model->tensors[i].data);
}

Tensor* get_tensor(GPT2Model *model, const char *name) {
    for (int i = 0; i < model->count; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

float* build_input_embeddings(GPT2Model *model, int *seq_tokens, int seq_len, int* n_embd) {
    Tensor *Wte = get_tensor(model, "wte_weight");
    Tensor *Wpe = get_tensor(model, "wpe_weight");
    *n_embd = Wte->shape[1];
    float *emb_output = malloc(sizeof(float) * seq_len * *n_embd);

    if (!Wte || !Wpe) {
        fprintf(stderr, "Erro: embeddings não encontrados!\n");
        return NULL;
    }

    for (int i = 0; i < seq_len; i++) {
        int token_id = seq_tokens[i];
        const float *token_vec = Wte->data + token_id * *n_embd;
        const float *pos_vec   = Wpe->data + i * *n_embd;

        for (int j = 0; j < *n_embd; j++) {
            emb_output[i * *n_embd + j] = token_vec[j] + pos_vec[j];
        }
    }

    return emb_output;
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3))));
}

void layernorm(float *x, int n_embd, const float *gamma, const float *beta, float *out) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n_embd; i++) mean += x[i];
    mean /= n_embd;
    for (int i = 0; i < n_embd; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n_embd;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n_embd; i++)
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}

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

void softmax(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

void multihead_attention(float *Q, float *K, float *V, float *out,
                         int seq_len, int n_embd, int n_heads) {
    int head_dim = n_embd / n_heads;

    float attn_scores[n_heads][seq_len * seq_len];

    #pragma omp parallel for num_threads(4)
    for (int h = 0; h < n_heads; h++) {
        float *Qh = Q + h * head_dim;
        float *Kh = K + h * head_dim;
        float *Vh = V + h * head_dim;
        float *Oh = out + h * head_dim;

        // QKᵀ / sqrt(d)
        matmul(attn_scores[h], Qh, Kh, seq_len, head_dim, seq_len);
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int i = 0; i < seq_len * seq_len; i++)
            attn_scores[h][i] *= scale;

        // softmax linha a linha
        for (int i = 0; i < seq_len; i++)
            softmax(&attn_scores[h][i * seq_len], seq_len);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < seq_len; k++) {
                    float a = attn_scores[h][i * seq_len + k];
                    float v = Vh[k * n_embd + h * head_dim + j];
                    sum += a * v;
                }
                out[i * n_embd + h * head_dim + j] = sum;
            }
        }
    }
}


void transformer_block(GPT2Model *model, float *x, int seq_len, int n_embd, int n_heads, int layer) {
    char name[128];

    // --- LayerNorm 1 ---
    snprintf(name, sizeof(name), "h_%d_ln_1_weight", layer);
    Tensor *ln1_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_ln_1_bias", layer);
    Tensor *ln1_b = get_tensor(model, name);

    float *x_norm = malloc(sizeof(float) * seq_len * n_embd);
    for (int i = 0; i < seq_len; i++)
        layernorm(&x[i * n_embd], n_embd, ln1_w->data, ln1_b->data, &x_norm[i * n_embd]);

    // --- Atenção: QKV ---
    snprintf(name, sizeof(name), "h_%d_attn_c_attn_weight", layer);
    Tensor *c_attn_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_attn_c_attn_bias", layer);
    Tensor *c_attn_b = get_tensor(model, name);

    int n_qkv = 3 * n_embd;
    float *QKV = malloc(sizeof(float) * seq_len * n_qkv);
    matmul(QKV, x_norm, c_attn_w->data, seq_len, n_embd, n_qkv);

    // adiciona bias
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < n_qkv; j++)
            QKV[i * n_qkv + j] += c_attn_b->data[j];

    // separa Q, K, V
    float *Q = malloc(sizeof(float) * seq_len * n_embd);
    float *K = malloc(sizeof(float) * seq_len * n_embd);
    float *V = malloc(sizeof(float) * seq_len * n_embd);
    for (int i = 0; i < seq_len; i++) {
        memcpy(&Q[i * n_embd], &QKV[i * n_qkv], n_embd * sizeof(float));
        memcpy(&K[i * n_embd], &QKV[i * n_qkv + n_embd], n_embd * sizeof(float));
        memcpy(&V[i * n_embd], &QKV[i * n_qkv + 2 * n_embd], n_embd * sizeof(float));
    }

    free(QKV);

    // --- Atenção Multi-Head ---
    float *context = malloc(sizeof(float) * seq_len * n_embd);
    multihead_attention(Q, K, V, context, seq_len, n_embd, n_heads);

    // --- Projeção final ---
    snprintf(name, sizeof(name), "h_%d_attn_c_proj_weight", layer);
    Tensor *c_proj_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_attn_c_proj_bias", layer);
    Tensor *c_proj_b = get_tensor(model, name);

    float *attn_out = malloc(sizeof(float) * seq_len * n_embd);
    matmul(attn_out, context, c_proj_w->data, seq_len, n_embd, n_embd);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < n_embd; j++)
            attn_out[i * n_embd + j] += c_proj_b->data[j];

    // adiciona resíduo
    for (int i = 0; i < seq_len * n_embd; i++)
        x[i] += attn_out[i];

    // --- LayerNorm 2 ---
    snprintf(name, sizeof(name), "h_%d_ln_2_weight", layer);
    Tensor *ln2_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_ln_2_bias", layer);
    Tensor *ln2_b = get_tensor(model, name);

    for (int i = 0; i < seq_len; i++)
        layernorm(&x[i * n_embd], n_embd, ln2_w->data, ln2_b->data, &x_norm[i * n_embd]);

    // --- MLP ---
    snprintf(name, sizeof(name), "h_%d_mlp_c_fc_weight", layer);
    Tensor *fc_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_mlp_c_fc_bias", layer);
    Tensor *fc_b = get_tensor(model, name);

    float *hidden = malloc(sizeof(float) * seq_len * (4 * n_embd));
    matmul(hidden, x_norm, fc_w->data, seq_len, n_embd, 4 * n_embd);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < 4 * n_embd; j++) {
            hidden[i * 4 * n_embd + j] += fc_b->data[j];
            hidden[i * 4 * n_embd + j] = gelu(hidden[i * 4 * n_embd + j]);
        }

    snprintf(name, sizeof(name), "h_%d_mlp_c_proj_weight", layer);
    Tensor *proj_w = get_tensor(model, name);
    snprintf(name, sizeof(name), "h_%d_mlp_c_proj_bias", layer);
    Tensor *proj_b = get_tensor(model, name);

    float *mlp_out = malloc(sizeof(float) * seq_len * n_embd);
    matmul(mlp_out, hidden, proj_w->data, seq_len, 4 * n_embd, n_embd);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < n_embd; j++)
            mlp_out[i * n_embd + j] += proj_b->data[j];

    // adiciona resíduo
    for (int i = 0; i < seq_len * n_embd; i++)
        x[i] += mlp_out[i];

    // --- limpa ---
    free(Q); free(K); free(V);
    free(context); free(attn_out);
    free(hidden); free(mlp_out); free(x_norm);
}

float* final_logits(GPT2Model *model, float *x, int seq_len, int n_embd) {
    // Última LayerNorm
    Tensor *lnf_w = get_tensor(model, "ln_f_weight");
    Tensor *lnf_b = get_tensor(model, "ln_f_bias");

    for (int i = 0; i < seq_len; i++)
        layernorm(&x[i * n_embd], n_embd, lnf_w->data, lnf_b->data, &x[i * n_embd]);
    
    // Pega os pesos do embedding (compartilhados)
    Tensor *wte = get_tensor(model, "wte_weight");
    int n_vocab = wte->shape[0];

    float* logits = malloc(sizeof(float) * n_vocab);

    // Multiplica último token pelo embedding transposto
    // (só o último token, para prever o próximo)
    float *last_token = &x[(seq_len - 1) * n_embd];

    for (int v = 0; v < n_vocab; v++) {
        float sum = 0.0f;
        for (int j = 0; j < n_embd; j++)
            sum += wte->data[v * n_embd + j] * last_token[j];
        logits[v] = sum;
    }

    softmax(logits, n_vocab);
    return logits;
}

void load_vocab(Vocab *vocab, const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Erro ao abrir vocab.csv");
        exit(1);
    }

    vocab->size = 0;
    char token[MAX_TOKEN_LEN];
    int id;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = 0;

        char *last_comma = strrchr(line, ',');
        if (!last_comma) continue; // linha inválida

        *last_comma = '\0'; // separa token e id
        char *token_str = line;
        char *id_str = last_comma + 1;
        int id = atoi(id_str);

        strncpy(vocab->entries[vocab->size].token, token_str, MAX_TOKEN_LEN - 1);
        vocab->entries[vocab->size].token[MAX_TOKEN_LEN - 1] = '\0';
        vocab->entries[vocab->size].id = id;

        vocab->size++;
        if (vocab->size >= MAX_VOCAB) break;
    }

    fclose(f);
    // printf("✅ Vocabulário carregado com %d tokens\n", vocab->size);
}

int encode_token(Vocab *vocab, const char *word) {
    for (int i = 0; i < vocab->size; i++)
        if (strcmp(vocab->entries[i].token, word) == 0)
            return vocab->entries[i].id;
    return -1; // não encontrado
}

const char* decode_token(Vocab *vocab, int id) {
    for (int i = 0; i < vocab->size; i++)
        if (vocab->entries[i].id == id)
            return vocab->entries[i].token;
    return "?";
}

int argmax(float *arr, int len) {
    if (len <= 0) return -1; // segurança
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

void top_k_argmax(float *arr, int len, int *top_indices, int k) {
    for (int i = 0; i < k; i++) {
        top_indices[i] = -1;
    }

    for (int i = 0; i < k; i++) {
        float max_val = 0;
        int max_idx = -1;
        for (int j = 0; j < len; j++) {
            int already_picked = 0;
            for (int t = 0; t < i; t++) {
                if (top_indices[t] == j) {
                    already_picked = 1;
                    break;
                }
            }
            if (!already_picked && arr[j] > max_val) {
                max_val = arr[j];
                max_idx = j;
            }
        }
        top_indices[i] = max_idx;
    }
}