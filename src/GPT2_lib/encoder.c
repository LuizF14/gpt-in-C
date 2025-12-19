#include "GPT2_lib.h"

int encode_token(Vocab *vocab, const char *word) {
    for (int i = 0; i < vocab->size; i++)
        if (strcmp(vocab->entries[i].token, word) == 0)
            return vocab->entries[i].id;
    return -1; // não encontrado
}

char* decode_token(Vocab *vocab, int id) {
    if (id == 11) 
        return ",";
    for (int i = 0; i < vocab->size; i++)
        if (vocab->entries[i].id == id)
            return vocab->entries[i].token;
    return "?";
}

int* encode_seq(Vocab *vocab, char** prompt, int seq_len) {
    int* seq_tokens = malloc(sizeof(int) * MAX_SEQ);
    for (int i = 0; i < seq_len; i++){
        seq_tokens[i] = encode_token(vocab, prompt[i]);
    }
    return seq_tokens;
}

void encode_prompt(char **prompt, const char *input, int *seq_len) {
    int i = 0, j = 0, k = 0;

    prompt[j] = malloc(MAX_TOKEN_LEN);
    if (!prompt[j]) return;

    while (input[i] != '\0') {
        if (input[i] == ' ') {
            prompt[j][k] = '\0';  
            j++;
            if (j >= 20) break;

            k = 0;
            prompt[j] = malloc(MAX_TOKEN_LEN);
            if (!prompt[j]) break;

            prompt[j][k++] = 0xC4;
            prompt[j][k++] = 0xA0;

            i++; 
            continue;
        }

        if (k < MAX_TOKEN_LEN - 1) {
            prompt[j][k++] = input[i];
        }
        i++;
    }

    prompt[j][k] = '\0';
    *seq_len = j + 1;
}

void decode_word(char* encoded, char* decoded) {
    int i = 0, j = 0;

    while (encoded[i] != '\0') {
        if ((unsigned char)encoded[i] == 0xC4 &&
            (unsigned char)encoded[i + 1] == 0xA0) {
            decoded[j++] = ' ';
            i += 2;
        } else if ((unsigned char)encoded[i] == 0xC4 &&
            (unsigned char)encoded[i + 1] == 0x8A) {
            decoded[j++] = '\n';
            i += 2;
        } else if ((unsigned char)encoded[i] == 0xC4 &&
            (unsigned char)encoded[i + 1] == 0x89) {
            decoded[j++] = '\t';
            i += 2;
        } else {
            decoded[j++] = encoded[i++];
        }
    }
    decoded[j] = '\0';
}