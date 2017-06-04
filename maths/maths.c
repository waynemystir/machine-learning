#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "maths.h"

#define NSUM 25

// courtesy of http://c-faq.com/lib/gaussian.html
double gaussrand() {
        static int rand_seeded = 0;

        if (!rand_seeded) {
                time_t t;
                srand((unsigned) time(&t));
        }
        rand_seeded = 1;

        double x = 0;
        for (int i = 0; i < NSUM; i++)
                x += (double)rand() / RAND_MAX;
        x -= NSUM / 2.0;
        x /= sqrt(NSUM / 12.0);
        return x;
}

