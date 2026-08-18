#include "SpiceUsr.h"
#include <stdio.h>

int main() {
    /* 1. Load kernels — do this once, at startup */
    furnsh_c("kernels/naif0012.tls");
    furnsh_c("kernels/de440s.bsp");

    /* 2. Convert a calendar date to Ephemeris Time */
    SpiceDouble et;
    str2et_c("2026-08-07T00:00:00", &et);

    /* 3. Query a state vector */
    SpiceDouble state[6];   /* x,y,z,vx,vy,vz */
    SpiceDouble lt;         /* light time, usually ignored for your purposes */

    spkezr_c("MOON", et, "J2000", "NONE", "EARTH", state, &lt);

    printf("Moon position (km): %f %f %f\n", state[0], state[1], state[2]);
    printf("Moon velocity (km/s): %f %f %f\n", state[3], state[4], state[5]);

    return 0;
}