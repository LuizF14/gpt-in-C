#include "GPT2_lib.h"

int read_shape(const char *filename, int *shape) {
    FILE *file = fopen(filename, "r");
    if (!file) 
        return -1;

    int count = 0;
    while (count < MAX_DIMS && fscanf(file, "%d", &shape[count]) == 1)
        count++;

    fclose(file);

    return count;
}

float *read_bin(const char *filename, size_t *num_elements) {
    FILE *file= fopen(filename, "rb");
    if (!file) 
        return NULL;

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);

    fseek(file, 0, SEEK_SET);
    *num_elements = file_size / sizeof(float);

    float *data = (float *)malloc(file_size);
    size_t n = fread(data, sizeof(float), *num_elements, file);
    if (n != *num_elements) {
        fprintf(stderr, "Erro ao ler arquivo binário\n");
        exit(1);
    }

    fclose(file);
    return data;
}

// --- Função para carregar todos os tensores de uma pasta ---
void load_model(GPT2Model *model, const char *dir_path) {
    DIR *dir_stream = opendir(dir_path);

    if (dir_stream == NULL) {
        fprintf(stderr, "Erro ao abrir o diretório");
    }

    struct dirent *file;
    model->count = 0;

    while((file = readdir(dir_stream)) != NULL) {
        if (!strstr(file->d_name, ".shape")) continue;

        // Extrai nome base (sem .shape)
        char base[MAX_NAME_LEN];
        int filename_len = strlen(file->d_name);
        snprintf(base, sizeof(base), "%.*s", filename_len - 6, file->d_name);
        //strncpy(base, file->d_name, filename_len - 6);
        base[filename_len - 6] = '\0';

        char shape_file[MAX_NAME_LEN + 32], bin_file[MAX_NAME_LEN + 32];
        snprintf(shape_file, sizeof(shape_file), "%s/%s.shape", dir_path, base);
        snprintf(bin_file, sizeof(bin_file), "%s/%s.bin", dir_path, base);

        Tensor *t = &model->tensors[model->count];
        strncpy(t->name, base, MAX_NAME_LEN);

        t->ndims = read_shape(shape_file, t->shape);
        t->data = read_bin(bin_file, &t->num_elements);

        if (!t->data) {
            fprintf(stderr, "Falha ao ler %s\n", bin_file);
            continue;
        }

        model->count++;
        if (model->count >= MAX_TENSORS) break;
    }
}

void free_model(GPT2Model *model) {
    for (int i = 0; i < model->count; i++)
        free(model->tensors[i].data);
}

void print_tensor_info(const Tensor *t, int n_vals) {
    printf("Tensor: %s\n", t->name);
    printf("Dimensões: ");
    for (int i = 0; i < t->ndims; i++)
        printf("%d ", t->shape[i]);
    printf("\nPrimeiros valores: ");
    for (size_t i = 0; i < (size_t) n_vals && i < t->num_elements; i++)
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
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = 0;

        char *last_comma = strrchr(line, ',');
        if (!last_comma) continue; // linha inválida

        *last_comma = '\0';
        char *id_str = last_comma + 1;
        int id = atoi(id_str);

        strncpy(vocab->entries[vocab->size].token, line, MAX_TOKEN_LEN - 1);
        vocab->entries[vocab->size].token[MAX_TOKEN_LEN - 1] = '\0';
        vocab->entries[vocab->size].id = id;

        vocab->size++;
        if (vocab->size >= MAX_VOCAB) break;
    }

    fclose(f);
}