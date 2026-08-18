"""
Then — small-scale JPL cross-check, which is the cheapest and most valuable
thing left before committing to C: pull one real L1 halo state near a comparable Jacobi constant
to something in your swept family, run it through your corrector as the seed, see if it lands close to your Richardson-seeded
orbit at the same C. That's your best available "is this whole pipeline actually correct" signal,
and it's a lot cheaper to get now than after C is built on top of an unverified Python baseline.
"""

import json
import requests
import numpy as np
from math_engine import manifolds, cr3bp
from math_engine.richardson import richardson_halo_seed
from tests import jacobi_constant_test

URL = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
params = {
    "sys": "earth-moon",
    "family": "halo",
    "libr": 1,
    "branch": "N",
    "jacobimin": 3.15,
    "jacobimax": 3.19,
}

response = requests.get(URL, params=params)
data = response.json()
# print(response.status_code)
# print(data)
# print("==================================================================================")
# print(type(data))          # <class 'dict'>
# print(data.keys())         # dict_keys(['signature', 'system', 'family', ...])
# print(data["count"])       # '5731' <- note: string, not int

if response.status_code != 200:
    raise RuntimeError(f"Request failed: {response.status_code}")

if "warning" in data:
    raise ValueError(f"No matching orbits: {data['warning']}")

count = int(data["count"])
if count == 0:
    raise ValueError("Zero orbits matched your filter — widen jacobimin/jacobimax")

fields = data["fields"]  # ['x','y','z','vx','vy','vz','jacobi','period','stability']
row = data["data"][0]  # first matching orbit, list of strings
row_floats = [float(v) for v in row]

orbit = dict(zip(fields, row_floats))
print("Jacobi, Period: ", orbit["jacobi"], orbit["period"])

mu = float(data["system"]["mass_ratio"])  # cross-check this against your hardcoded mu
print(f"MU: {mu}")
state0 = np.array(
    [
        orbit["x"],
        orbit["y"],
        orbit["z"],
        orbit["vx"],
        orbit["vy"],
        orbit["vz"],
    ]
)
period = orbit["period"]

if __name__ == "__main__":

    # ============================================================
    # Integrator and Jacobi Constant Consistency Test with JPL API
    # ============================================================

    # statef = manifolds.integrator((0,period), state6=state0, mu=mu,).y[:6,-1]
    # print("statef: ", statef)
    # print(f"State: {state0}")

    # C_F = jacobi_constant_test.jacobi_constant(statef, mu)
    # C_I = orbit["jacobi"] #jacobi_constant_test(state0, mu)
    # if np.linalg.norm(statef - state0) <= 1e-8:
    #     print("Closure Successful")
    # else:
    #     print("[Error] Closure Unsuccessful")

    # if np.abs(C_F - C_I) <= 1e-12:
    #     print("Jacobi Constant Conserved")
    # else:
    #     print("[Error] Jacobi Constant Not Conserved")

    # ==========================================
    # Richardson IC and JPL API Consistency Test
    # ==========================================

    Az_km_test = 5000.0
    primary_dist_km = 384400.0  # earth moon sep.
    full_Ic_guess = richardson_halo_seed(
        mu, lpoint=1, Az_km=Az_km_test, primary_dist_km=primary_dist_km
    )
    ic_state = full_Ic_guess[:-1]
    ic_period = full_Ic_guess[-1]
    jacobi = jacobi_constant_test.jacobi_constant(ic_state, mu)
    print(f"C: {jacobi}")
    # print(f"State: {ic_state}")

    x_guess = ic_state[0]
    z_guess = ic_state[2]
    vy_guess = ic_state[4]
    integrated_state, test_period = manifolds.differential_corrector(
        T=ic_period,
        x0_guess=x_guess,
        z0_guess=z_guess,
        vy0_guess=vy_guess,
        mu=mu,
    )
    
    corrected_jacobi = jacobi_constant_test.jacobi_constant(integrated_state, mu)
    target_jacobi = corrected_jacobi   # search using the CORRECTED value, not Richardson's

    fields = data["fields"]
    jacobi_idx = fields.index("jacobi")

    chosen_orbit_row = min(
        data["data"], key=lambda row: abs(float(row[jacobi_idx]) - target_jacobi)
    )
    chosen_orbit = dict(zip(fields, [float(v) for v in chosen_orbit_row]))

    jpl_state = np.array([
        chosen_orbit["x"], chosen_orbit["y"], chosen_orbit["z"],
        chosen_orbit["vx"], chosen_orbit["vy"], chosen_orbit["vz"],
    ])

    corrected_error = np.linalg.norm(np.array(integrated_state) - jpl_state)
    print(f"Target Jacobi (corrected): {corrected_jacobi}")
    print(f"Nearest JPL Jacobi: {chosen_orbit['jacobi']}")
    print(f"Jacobi mismatch: {abs(corrected_jacobi - chosen_orbit['jacobi']):.3e}")
    print(f"Corrector output error vs JPL: {corrected_error:.3e}")