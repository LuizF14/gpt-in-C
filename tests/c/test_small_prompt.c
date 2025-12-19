#include "GPT2_lib.h"

int main(void) {
    GPT2Model model;
    load_model(&model, "bin/");
    
    Vocab vocab;
    load_vocab(&vocab, "bin/vocab.csv");

    char* prompt[] = {"May"};
    int seq_len = 1;
    for (int i = 0; i < seq_len; i++){
        printf("%s", prompt[i]);
    }

    int* seq_tokens = encode_seq(&vocab, prompt, seq_len);

    while(seq_len < 10) {
        char* next_word = generate_text(seq_tokens, &seq_len, &model, &vocab);
        printf("%s", next_word);
    }
    
    free_model(&model);
    return 0;
}