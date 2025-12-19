#include "GPT2_lib.h"
#include "transformer.h"

char* generate_text(int* seq_tokens, int* seq_len, GPT2Model *model, Vocab *vocab) {
    float* emb_output = build_input_embeddings(model, seq_tokens, *seq_len);
    for (int i = 0; i < 12; i++) {
        transformer_block(model, emb_output, *seq_len, i);
    }
    final_block(model, emb_output, *seq_len);

    float* logits = final_logits(model, emb_output, *seq_len);

    int next_token = argmax(&logits[(*seq_len-1) * MAX_VOCAB], MAX_VOCAB);
    seq_tokens[*seq_len] = next_token;
    (*seq_len)++;

    char* next_word = decode_token(vocab, next_token);

    free(logits);
    free(emb_output);

    return next_word;
}