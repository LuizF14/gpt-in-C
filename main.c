#include "GPT2_lib.h"
#ifdef TEST_TIME
    #include <sys/time.h>
    #include "JSON_lib.h"
#endif

int main() {
    
    GPT2Model model;
    load_model(&model, "GPT2/bin");
    
    Vocab vocab;
    load_vocab(&vocab, "GPT2/bin/vocab.csv");
    
    char* prompt[] = {"May", "Ġthe", "Ġforce", "Ġbe", "Ġwith"};
    int seq_len = 5;
    int seq_tokens[MAX_SEQ];
    for (int i = 0; i < seq_len; i++){
        seq_tokens[i] = encode_token(&vocab, prompt[i]);
    }
    
    #ifdef TEST_TIME
        struct timeval t1, t2;
        double Elapsed_Time;
        gettimeofday(&t1, NULL); // Start timer
    #endif

    int n_embd;
    float* emb_output = build_input_embeddings(&model, seq_tokens, seq_len, &n_embd);
    for (int i = 0; i < 12; i++) {
        transformer_block(&model, emb_output, seq_len, n_embd, 12, i);
    }

    float* logits = final_logits(&model, emb_output, seq_len, n_embd);

    int next_token = argmax(logits, vocab.size);

    #ifdef TEST_TIME
        gettimeofday(&t2, NULL);
        
        Elapsed_Time = (t2.tv_sec - t1.tv_sec)*1000.0; 
        Elapsed_Time += (t2.tv_usec - t1.tv_usec)/1000.0;
        
        startJson();
        printJsonLine("ElapsedTime", Elapsed_Time);
        finishJson();
    #else 
        const char* decoded = decode_token(&vocab, next_token);
        printf("Texto: %s\n", decoded);
    #endif

    free_model(&model);
    free(emb_output);

    return 0;
}