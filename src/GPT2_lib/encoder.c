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