# Matthew Davidson UCID: 30182729
# File Description: Contains functions for analyzing Love Waves
#
#
#
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from solving_roots import root_secant_modified
#
#
#
# Function Purpose: Calculate the Love Wave Dispersion Equation
# Parameters:
# - zeta: (1/c_L)
# - f: frequency in Hz
# - rho1: density of the surface layer
# - rho2: density of the base layer
# - beta1: shear wave velocity of the surface layer
# - beta2: shear wave velocity of the half-space
# - H: thickness of the surface layer in km
#
# Returns:
# - float: value of the dispersion equation
#
def love_wave_dispersion(zeta, f, rho1, rho2, beta1, beta2, H):
    H_m = H * 1000
    k1_squared = 1/beta1**2 - zeta**2
    k2_squared = zeta**2 - 1/beta2**2
    if k1_squared <= 0 or k2_squared <= 0:
        return float('inf')
    arg = 2 * np.pi * f * H_m * np.sqrt(k1_squared)
    ratio = (rho2 * np.sqrt(k2_squared)) / (rho1 * np.sqrt(k1_squared))
    return np.tan(arg) - ratio
#
#
#
# Function Purpose: Find the zero crossings of a function
# Parameters:
# - x_values: array of x values
# - y_values: array of corresponding y values
#
# Returns:
# - list: estimated x values where y crosses zero
#
def find_zero_crossings(x_values, y_values):
    zero_crossings = []
    for i in range(1, len(y_values)):
        if y_values[i-1] * y_values[i] <= 0 and np.isfinite(y_values[i-1]) and np.isfinite(y_values[i]):
            # Linear interpolation to find the zero crossing
            if y_values[i] - y_values[i-1] != 0:  # Avoid division by zero
                x_zero = x_values[i-1] + (x_values[i] - x_values[i-1]) * (-y_values[i-1]) / (y_values[i] - y_values[i-1])
                zero_crossings.append(x_zero)
    return zero_crossings
#
#
#
# Function Purpose: Find the tangent discontinuities in the dispersion curve
# Parameters:
# - zeta_values: array of zeta values
# - f: frequency in Hz
# - rho1: density of the surface layer
# - rho2: density of the half-space
# - beta1: shear wave velocity of the surface layer
# - beta2: shear wave velocity of the half-space
# - H: thickness of the surface layer in km
#
# Returns:
# - list: estimated zeta values where tangent discontinuities occur
#
def find_tangent_discontinuities(zeta_values, f, rho1, rho2, beta1, beta2, H):
    discontinuities = []
    prev_val = None
    for i in range(1, len(zeta_values)):
        z1 = zeta_values[i-1]
        z2 = zeta_values[i]
        try:
            val1 = love_wave_dispersion(z1, f, rho1, rho2, beta1, beta2, H)
            val2 = love_wave_dispersion(z2, f, rho1, rho2, beta1, beta2, H)
            if np.isfinite(val1) and np.isfinite(val2):
                if np.abs(val2 - val1) > 5:
                    discontinuities.append(z2)
        except:
            continue
    return discontinuities
#
#
#
# Function Purpose: Plot Love wave dispersion curves and characteristics
# Parameters:
# - f_values: array of frequencies to analyze
# - rho1: density of the surface layer
# - rho2: density of the half-space
# - beta1: shear wave velocity of the surface layer
# - beta2: shear wave velocity of the half-space
# - H: thickness of the surface layer in km
# - num_modes: number of modes to analyze
#
# Returns:
# - Graph of Love Wave Dispersion Curves
#
def plot_dispersion(f_values, rho1, rho2, beta1, beta2, H, num_modes=3):
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    zeta_crit1 = 1/beta1
    zeta_crit2 = 1/beta2
    zeta_values = np.linspace(zeta_crit2 * 1.001, zeta_crit1 * 0.999, 2000)
    plot_freqs = np.linspace(f_values[0], f_values[-1], 5)

    all_modes = [{"f": [], "c": [], "lambda": []} for _ in range(num_modes)]

    for i, (ax, f) in enumerate(zip(axes, plot_freqs)):
        disp_values = []
        valid_zeta = []
        for zeta in zeta_values:
            try:
                val = love_wave_dispersion(zeta, f, rho1, rho2, beta1, beta2, H)
                if np.abs(val) < 10:
                    disp_values.append(val)
                    valid_zeta.append(zeta)
                else:
                    disp_values.append(np.nan)
                    valid_zeta.append(zeta)
            except:
                disp_values.append(np.nan)
                valid_zeta.append(zeta)

        valid_zeta = np.array(valid_zeta)
        disp_values = np.array(disp_values)

        ax.plot(valid_zeta, disp_values, 'r-')
        ax.axvline(x=zeta_crit2, color='b', linestyle='--', label='ζ = 1/β₂')
        ax.axvline(x=zeta_crit1, color='r', linestyle='--', label='ζ = 1/β₁')

        discontinuities = find_tangent_discontinuities(valid_zeta, f, rho1, rho2, beta1, beta2, H)
        for dc in discontinuities:
            ax.axvline(x=dc, color='b', linestyle='--', alpha=0.6)

        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylim(-5, 5)
        ax.set_title(f'f = {f:.2f} Hz')
        ax.grid(True)
        if i == 2:
            ax.set_ylabel('Dispersion Equation Value')
        if i == 0:
            ax.legend()

        zero_crossings = find_zero_crossings(valid_zeta, disp_values)
        zero_crossings.sort(reverse=True)
        for j, zeta_root in enumerate(zero_crossings[:num_modes]):
            c_L = 1 / zeta_root
            lambda_L = c_L / f
            all_modes[j]["f"].append(f)
            all_modes[j]["c"].append(c_L)
            all_modes[j]["lambda"].append(lambda_L)

    plt.xlabel('Zeta (s/m)')
    plt.suptitle('Love Wave Dispersion Curves')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('figures/love_wave_dispersion.png')
    
#
#
#
# Function Purpose: Analyze Love wave dispersion characteristics
# Parameters:
# - f_values: array of frequencies to analyze
# - rho1: density of the surface layer
# - rho2: density of the half-space
# - beta1: shear wave velocity of the surface layer
# - beta2: shear wave velocity of the half-space
# - H: thickness of the surface layer in km
# - num_modes: number of modes to analyze
#
# Returns:
# - Graph of Love Wave Dispersion Characteristics
#
def analyze_love_waves(f_values, rho1, rho2, beta1, beta2, H, num_modes=3):
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    zeta_crit1 = 1/beta1
    zeta_crit2 = 1/beta2
    
    all_modes = []
    for i in range(num_modes):
        all_modes.append({"f": [], "c": [], "lambda": []})
    
    for f in f_values:
        zeta_values = np.linspace(zeta_crit2 * 1.001, zeta_crit1 * 0.999, 5000)
        disp_values = []
        valid_zeta = []
        for zeta in zeta_values:
            try:
                value = love_wave_dispersion(zeta, f, rho1, rho2, beta1, beta2, H)
                if np.isfinite(value):
                    disp_values.append(value)
                    valid_zeta.append(zeta)
            except:
                continue

        valid_zeta = np.array(valid_zeta)
        disp_values = np.array(disp_values)
        
        zero_crossings = find_zero_crossings(valid_zeta, disp_values)
        zero_crossings.sort(reverse=True)
        
        for i, zeta in enumerate(zero_crossings[:num_modes]):
            if i < num_modes:
                c_L = 1 / zeta
                
                lambda_L = c_L / f
                
                all_modes[i]["f"].append(f)
                all_modes[i]["c"].append(c_L)
                all_modes[i]["lambda"].append(lambda_L)

    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i in range(num_modes):
        if len(all_modes[i]["f"]) > 0:
            plt.plot(all_modes[i]["f"], all_modes[i]["c"], '-o', 
                     color=colors[i % len(colors)], 
                     label=f'Mode {i}',
                     markersize=4)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase Velocity (m/s)')
    plt.title('Love Wave Dispersion Characteristics')
    plt.legend()
    plt.ylim(1800, 3300)
    plt.savefig('figures/love_wave_phase_velocity.png')

    plt.figure(figsize=(10, 8))
    plt.grid(True)
    for i in range(num_modes):
        if len(all_modes[i]["f"]) > 0:
            plt.plot(all_modes[i]["f"], all_modes[i]["lambda"], '-o', 
                     color=colors[i % len(colors)], 
                     label=f'Mode {i}',
                     markersize=4)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Wavelength (m)')
    plt.title('Love Wave Wavelength vs Frequency')
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig('figures/love_wave_wavelength.png')
    
    return all_modes

if __name__ == "__main__":
#
# Provided Parameters
#
    rho1 = 1800  # kg/m^3
    rho2 = 2500  # kg/m^3
    beta1 = 1900  # m/s
    beta2 = 3200  # m/s
    H = 4.0      # km
#   
#Create frequency range
#
    f_values = np.linspace(0.1, 2.0, 50)
    
    plot_dispersion(f_values, rho1, rho2, beta1, beta2, H, num_modes=3)

#   
#Create dispersion characteristics
#
    analyze_love_waves(f_values, rho1, rho2, beta1, beta2, H, num_modes=3)