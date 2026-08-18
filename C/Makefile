CC = gcc
CSPICE_DIR = cspice

CFLAGS = -I$(CSPICE_DIR)/include
LDFLAGS = $(CSPICE_DIR)/lib/cspice.a $(CSPICE_DIR)/lib/csupport.a -lm

SRC_DIR = src
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:.c=.o)

main: $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o main

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) -c $< $(CFLAGS) -o $@

clean:
	rm -f $(SRC_DIR)/*.o main