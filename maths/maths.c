#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "maths.h"

#define NSUM 25

static int rand_seeded = 0;

void seed_rand() {
	if (rand_seeded) return;
	rand_seeded = 1;
	time_t t;
	srand((unsigned) time(&t));
}

// courtesy of http://c-faq.com/lib/gaussian.html
double gaussrand() {
	seed_rand();

	double x = 0;
	for (int i = 0; i < NSUM; i++)
		x += (double)rand() / RAND_MAX;
	x -= NSUM / 2.0;
	x /= sqrt(NSUM / 12.0);
	return x;
}

double gaussrand_0to1() {
	return (double)rand() / RAND_MAX;
}

int mrand(int n) {
	seed_rand();
	int r = rand();
	return r % n;
}
