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

    // char* prompt[] = {"May", "Ġthe", "Ġforce", "Ġbe", "Ġwith"};
    // int seq_len = 5;
    char* prompt[] = {"May"};
    int seq_len = 1;

    int seq_tokens[MAX_SEQ];
    for (int i = 0; i < seq_len; i++){
        seq_tokens[i] = encode_token(&vocab, prompt[i]);
        printf("%s ", prompt[i]);
        fflush(stdout);
    }

    while(seq_len < 10) {
        char* next_word = generate_text(seq_tokens, seq_len, &model, &vocab);
        seq_tokens[seq_len] = next_word;
        seq_len++;
        printf("%s", next_word);
        fflush(stdout);
    }
    
    free_model(&model);
    // free(&vocab);

    return 0;
}