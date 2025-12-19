#include "GPT2_lib/GPT2_lib.h"
#include <stdio.h>

#define MAX_PROMPT 4096

int main() {
    GPT2Model model;
    load_model(&model, "bin/");
    
    Vocab vocab;
    load_vocab(&vocab, "bin/vocab.csv");

    // char* prompt[] = {"May", "Ġthe", "Ġforce", "Ġbe", "Ġwith"};
    // int seq_len = 5;
    char* prompt[] = {"May"};
    int seq_len = 1;
    for (int i = 0; i < seq_len; i++){
        printf("%s ", prompt[i]);
        fflush(stdout);
    }

    int* seq_tokens = encode_seq(&vocab, prompt, seq_len);

    while(seq_len < 10) {
        char* next_word = generate_text(seq_tokens, &seq_len, &model, &vocab);
        printf("%s ", next_word);
        fflush(stdout);
    }
    printf("\n");
    
    free_model(&model);
    // free(&vocab);

    return 0;
}