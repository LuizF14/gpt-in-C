#include "GPT2_lib/GPT2_lib.h"
#ifdef TEST_TIME
    #include <sys/time.h>
    #include "JSON_lib.h"
#endif

#include <stdio.h>

#define MAX_PROMPT 4096

int main() {
    GPT2Model model;
    load_model(&model, "bin/");
    
    Vocab vocab;
    load_vocab(&vocab, "bin/vocab.csv");

    char* prompt[] = {"May", "Ġthe", "Ġforce", "Ġbe", "Ġwith"};
    int seq_len = 5;
    int seq_tokens[MAX_SEQ];
    for (int i = 0; i < seq_len; i++){
        seq_tokens[i] = encode_token(&vocab, prompt[i]);
        printf("%s ", prompt[i]);
    }

    while(seq_len < 10) {
        float* emb_output = build_input_embeddings(&model, seq_tokens, seq_len);
        for (int i = 0; i < 12; i++) {
            transformer_block(&model, emb_output, seq_len, i);
        }
        final_block(&model, emb_output, seq_len);

        float* logits = final_logits(&model, emb_output, seq_len);

        int next_token = argmax(&logits[(seq_len-1) * MAX_VOCAB], MAX_VOCAB);

        printf("%s ", decode_token(&vocab, next_token));
        fflush(stdout);
        seq_tokens[seq_len] = next_token;
        seq_len++;

        free(logits);
        free(emb_output);
    }
    
    free_model(&model);
    // free(&vocab);

    return 0;
}