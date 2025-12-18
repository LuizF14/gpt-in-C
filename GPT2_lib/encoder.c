#include "GPT2_lib.h"

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