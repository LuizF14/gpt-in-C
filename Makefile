CC = gcc
CFLAGS = -Wall -Wextra -O0 -g -Isrc -Isrc/GPT2_lib -Itests/unity/src
LDFLAGS = -lm

DEBUGFLAGS = -fno-omit-frame-pointer -fno-inline -fno-inline-functions

BUILD = build

MAIN = src/main.c 

GPT2_lib = src/GPT2_lib/encoder.c \
      src/GPT2_lib/transformer.c \
      src/GPT2_lib/math_utils.c \
      src/GPT2_lib/load.c \
      src/GPT2_lib/GPT2_lib.c

TEST = tests/c/test_small_prompt.c

all: gpt2

$(BUILD):
	mkdir -p $(BUILD)

gpt2:
	$(CC) $(CFLAGS) $(GPT2_lib) $(MAIN) -o $(BUILD)/gpt2 $(LDFLAGS)

gpt2_debug:
	$(CC) $(CFLAGS) $(DEBUGFLAGS) $(GPT2_lib) $(MAIN) -o $(BUILD)/gpt2 $(LDFLAGS)

test_small_prompt:
	$(CC) $(CFLAGS) $(GPT2_lib) $(TEST) -o $(BUILD)/test_small_prompt $(LDFLAGS)

clean:
	rm -rf $(BUILD)/*
