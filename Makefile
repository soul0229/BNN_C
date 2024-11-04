PREFIX = ${CROSS_COMPILE}
CC = ${PREFIX}gcc
CFLAGS = -Wall -Iinclude
EINCLUDE = -lcjson -lm
SRC_DIR = src
OBJ_DIR = build
SRCS = $(wildcard $(SRC_DIR)/*.c ./*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
EXEC = bnn

ifeq ("$(origin V)", "command line")
  KBUILD_VERBOSE = $(V)
endif

Q = @

ifneq ($(findstring 1, $(KBUILD_VERBOSE)),)
  Q =
endif


all: $(EXEC)

$(EXEC): $(OBJS)
	$(Q)$(CC) $(CFLAGS) $^ -o $@ $(EINCLUDE)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(Q)$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	$(Q)mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR)/* $(EXEC)

.PHONY: clean all
