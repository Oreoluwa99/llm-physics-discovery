import numpy as np

def projectile(
    v0 = 20.0,                   # initial speed [m/s]
    theta_deg = 45.0,            # launch angle [degrees]
    g = 9.81,                    # gravitational acceleration [m/s^2]
    noise = 0.05,                # relative noise level (e.g., 0.05 = 5%)
    num_points = 200,            # number of time samples
    seed  = 0.01                 # RNG seed for reproducibility 
):
    """
    Generate noisy projectile-motion data (vertical position vs. time).

    Model (no air resistance):
        y(t) = v0 * sin(theta) * t - 0.5 * g * t^2

    Parameters
    ----------
    v0 : Initial launch speed in meters per second.
    theta_deg : Launch angle in degrees (converted to radians internally).
    g : Gravitational acceleration (default 9.81 m/s^2).
    noise : Relative noise level. The noise standard deviation is set to `noise * std(y_clean)`. 
    num_points : Number of uniformly spaced time samples from 0 to total flight time.
    seed : Random seed for reproducibility. Use None for nondeterministic noise.

    Returns
    -------
    t : Time samples (seconds), shape (num_points,).
    y_noisy : Noisy vertical positions (meters), same shape as `t`.
    meta : Metadata with system name, parameters, and the clean trajectory std.
    """

    # Random number generator (reproducible if seed is set)
    rng = np.random.default_rng(seed)

    # Convert angle to radians
    theta_rad = np.deg2rad(theta_deg)

    # Total flight time (time from launch until y returns to 0) - T = 2 * v0 * sin(theta) / g
    total_time = 2.0 * v0 * np.sin(theta_rad) / g
    # Guard against degenerate cases (e.g., v0=0 or sin(theta)=0)
    total_time = float(max(total_time, 1e-6))

    # Uniform time samples from 0 to total_time
    t = np.linspace(0.0, total_time, num_points)

    # Clean (noise-free) vertical position
    y_clean = v0 * np.sin(theta_rad) * t - 0.5 * g * (t ** 2)

    # Noise standard deviation: scale with the signal magnitude so noise is proportional to data scale
    y_std = float(np.std(y_clean))
    sigma = (noise * y_std) if y_std > 0 else max(noise, 1e-6)

    # Add zero-mean Gaussian noise to simulate measurement error
    eps = rng.normal(loc=0.0, scale=sigma, size=t.shape)
    y_noisy = y_clean + eps

    meta = {
        "system": "projectile",
        "params": {
            "v0": v0,
            "theta_deg": theta_deg,
            "g": g,
            "noise": noise,
            "num_points": num_points,
            "seed": seed,
        },
        "y_clean_std": y_std,
        "total_time": total_time,
    }

    return t, y_noisy, meta
