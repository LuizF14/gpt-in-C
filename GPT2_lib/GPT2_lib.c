#include "GPT2_lib.h"

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

Tensor* get_tensor(GPT2Model *model, const char *name) {
    for (int i = 0; i < model->count; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
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

char* decode_token(Vocab *vocab, int id) {
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