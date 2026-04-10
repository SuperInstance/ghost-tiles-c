CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
LDFLAGS = -lm

all: test_ghost_tiles

test_ghost_tiles: tests/test_ghost_tiles.c src/ghost-tiles.c
	$(CC) $(CFLAGS) -o $@ tests/test_ghost_tiles.c src/ghost-tiles.c $(LDFLAGS)

test: test_ghost_tiles
	./test_ghost_tiles

clean:
	rm -f test_ghost_tiles
