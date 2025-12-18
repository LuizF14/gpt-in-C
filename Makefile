CFLAGS := -g

GPT2_LIB_DIR := GPT2_lib
GPT2_LIB_C   := $(wildcard $(GPT2_LIB_DIR)/*.c)

gpt2_files := main.c $(GPT2_LIB_C)

gpt2_serial: $(gpt2_files)
	gcc $(CFLAGS) -o gpt2_serial.exe $^ -lm -DTEST_TIME

gpt2_parallel: $(gpt2_files)
	gcc $(CFLAGS) -o gpt2_parallel.exe $^ -lm -DTEST_TIME -fopenmp

gpt2: $(gpt2_files)
	gcc $(CFLAGS) -o gpt2.exe $^ -lm
