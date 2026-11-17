#include <stdio.h>
#include <stdlib.h>

#define MAX_ROWS 100

typedef struct Plot {
    int id;
    double area;
} Plot;

static int count_rows(const char *text)
{
    int n = 0;
    for (const char *p = text; *p; p++) {
        if (*p == '\n') {
            n++;
        }
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        printf("%d\n", count_rows(argv[1]));
    }
    return 0;
}
