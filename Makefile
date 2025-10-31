gpt2_files := main.c GPT2_lib.c GPT2_lib.h JSON_lib.h

gpt2_serial: $(gpt2_files)
	gcc $(CFLAGS) -o gpt2_serial.exe $^ -lm -DTEST_TIME

gpt2_parallel: $(gpt2_files)
	gcc $(CFLAGS) -o gpt2_parallel.exe $^ -lm -DTEST_TIME -fopenmp

gpt2: $(gpt2_files)
	gcc $(CFLAGS) -o gpt2.exe $^ -lm