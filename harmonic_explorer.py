# Harmonic Explorer: Geometric Pattern Emergence from Harmonic Ratios

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class HarmonicExplorer:
    """
    Simulates the interaction of a particle cloud with a time-variant force field 
    defined by a simple integer ratio (p1:p2).
    p1 controls radial force (pulsation), p2 controls tangential force (rotation).
    """
    def __init__(self, p_radial, p_tangential, num_particles=5000, damping_factor=0.98):
        """
        Initializes the simulation environment.

        Args:
            p_radial (int): Integer driving the radial force component (p1).
            p_tangential (int): Integer driving the tangential force component (p2).
            num_particles (int): The number of particles on the canvas.
            damping_factor (float): Friction factor to allow particles to settle into patterns.
        """
        print(f"Initializing simulation for Harmonic Ratio {p_radial}:{p_tangential}...")
        self.p_radial = p_radial
        self.p_tangential = p_tangential
        self.N = num_particles
        self.damping = damping_factor

        # Core simulation constants
        self.dt = 0.01          # Time step
        self.base_freq = 0.1    # Base frequency for temporal oscillation
        self.force_amplitude = 1.0 # Strength of the force field
        self.t = 0.0            # Current simulation time

        self._initialize_particles()

    def _initialize_particles(self):
        """
        Creates particles in a uniform-area random cloud within a disk radius of 3.
        """
        # Initialize positions (r, theta) -> (x, y)
        r = np.sqrt(np.random.rand(self.N)) * 3 
        theta = 2 * np.pi * np.random.rand(self.N)
        self.positions = np.array([r * np.cos(theta), r * np.sin(theta)]).T
        self.velocities = np.zeros_like(self.positions)

    def update(self):
        """
        Core physics loop: Calculate forces based on harmonic engine and update particle state.
        """
        # --- 1. Cartesian to Polar ---
        x, y = self.positions[:, 0], self.positions[:, 1]
        r = np.sqrt(x**2 + y**2)
        r[r == 0] = 1e-9 
        theta = np.arctan2(y, x)

        # --- 2. Calculate Forces (The Harmonic Engine) ---
        # Radial Force Magnitude (F_r)
        F_radial_mag = self.force_amplitude * np.sin(2 * np.pi * self.p_radial * self.base_freq * self.t) * np.cos(self.p_tangential * theta)
        # Tangential Force Magnitude (F_t)
        F_tangential_mag = self.force_amplitude * np.cos(2 * np.pi * self.p_tangential * self.base_freq * self.t)

        # --- 3. Polar Forces to Cartesian Vectors ---
        # Radial and Tangential Unit Vectors
        r_hat_x, r_hat_y = x / r, y / r
        t_hat_x, t_hat_y = -y / r, x / r

        # Total Force (F_x, F_y)
        force_x = (r_hat_x * F_radial_mag) + (t_hat_x * F_tangential_mag)
        force_y = (r_hat_y * F_radial_mag) + (t_hat_y * F_tangential_mag)
        forces = np.column_stack((force_x, force_y))

        # --- 4. Integration and Damping ---
        self.velocities *= self.damping
        self.velocities += forces * self.dt
        self.positions += self.velocities * self.dt

        # --- 5. Time Update ---
        self.t += self.dt
        return self.positions

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # ############################################################# #
    # #                  SET HARMONIC RATIO HERE                    # #
    # ############################################################# #
    #
    # p_radial (P1) drives radial pulsation (in-out).
    # p_tangential (P2) drives tangential rotation.
    #
    P_RADIAL = 8
    P_TANGENTIAL = 8
    #
    # ############################################################# #

    # Initialize simulation and plot setup
    sim = HarmonicExplorer(p_radial=P_RADIAL, p_tangential=P_TANGENTIAL, num_particles=5000)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    # Scatter plot initialization
    scatter = ax.scatter(
        sim.positions[:, 0],
        sim.positions[:, 1],
        s=0.5,
        c='white',
        alpha=1.0
    )

    def animate(frame):
        """Animation update function."""
        positions = sim.update()
        scatter.set_offsets(positions)
        
        # Dynamic axis scaling for visualization stability
        lim = np.max(np.abs(positions)) * 1.2
        if lim > 0:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        return scatter,

    # Run the animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=600,         # Total frames (adjust for longer runs)
        interval=20,        # Frame delay in ms
        blit=True
    )

    plt.show()
