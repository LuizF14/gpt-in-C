CC = gcc
CFLAGS = -Wall -Wextra -O2 -Isrc -Isrc/GPT2_lib -Itests/unity/src
LDFLAGS = -lm

BUILD = build

MAIN = src/main.c 

GPT2_lib = src/GPT2_lib/encoder.c \
      src/GPT2_lib/transformer.c \
      src/GPT2_lib/math_utils.c \
      src/GPT2_lib/load.c \
      src/GPT2_lib/GPT2_lib.c

TEST = tests/c/test_small_prompt.c

all: gpt2

gpt2:
	$(CC) $(CFLAGS) $(GPT2_lib) $(MAIN) -o $(BUILD)/gpt2 $(LDFLAGS)

test_small_prompt:
	$(CC) $(CFLAGS) $(GPT2_lib) $(TEST) -o $(BUILD)/test_small_prompt $(LDFLAGS)

clean:
	rm -f gpt2.exe test_math
