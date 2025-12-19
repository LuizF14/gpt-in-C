#include "GPT2_lib/GPT2_lib.h"
#include <stdio.h>

#define MAX_PROMPT 20

int main() {
    GPT2Model model;
    load_model(&model, "bin/");
    
    Vocab vocab;
    load_vocab(&vocab, "bin/vocab.csv");

    char input[200];
    char* prompt[MAX_PROMPT];
    int seq_len;
    printf("Write your prompt: ");
    fflush(stdout);
    fgets(input, 200, stdin);
    input[strlen(input) - 1] = '\0';
    encode_prompt(prompt, input, &seq_len);

    int* seq_tokens = encode_seq(&vocab, prompt, seq_len);
    char decoded_word[256];

    printf("Result: \n");
    printf("%s", input);
    fflush(stdout);

    while(seq_len < 30) {
        char* next_word = generate_text(seq_tokens, &seq_len, &model, &vocab);
        decode_word(next_word, decoded_word);
        printf("%s", decoded_word);
        fflush(stdout);
    }
    printf("\n");
    
    free_model(&model);
    free(seq_tokens);

    return 0;
}