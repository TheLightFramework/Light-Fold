"""
THE LIGHT-FOLD ENGINE v0.2 (Stable & Interactive)
"Hydrophobic Collapse as a Negentropy Vector"

Author: 00003 (The Weaver)
Status: Patch 2 (Force Clamping + Time Control)

CONTROLS:
- [SPACE]: Pause / Resume
- [RIGHT ARROW]: Step Forward (when paused)
- [R]: Reset Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- CONFIGURATION ---

# The DNA (H = Truth/Core, P = Interface/Surface)
# Try changing this string to see different geometries!
SEQUENCE = "H-P-H-H-P-H-P-P-H-H-H-P-H-H-P-P-H" 

# Physics Parameters
BOND_STRENGTH = 50.0       # Stronger bonds to hold the chain
ATTRACTION_H_H = 2.0       # Stronger Love vector
REPULSION_STERIC = 5.0     # Self-respect (Volume)
NOISE_LEVEL = 0.05         # Water movement
DT = 0.02                  # Time step (Smaller = More precise/Stable)
MAX_FORCE = 10.0           # Safety Valve: Prevents explosions

# --- THE SIMULATION ENGINE ---

class LightChain:
    def __init__(self, seq_str):
        self.types = seq_str.split('-')
        self.n = len(self.types)
        self.reset()

    def reset(self):
        # Initialize positions: A jagged line to avoid symmetry lock
        # Using 'float64' for precision
        self.pos = np.zeros((self.n, 2), dtype=np.float64)
        for i in range(self.n):
            self.pos[i] = [i * 1.0, np.sin(i) * 0.5] # Wavy start
        
        self.vel = np.zeros_like(self.pos)
        self.indices_H = [i for i, t in enumerate(self.types) if t == 'H']
        self.history = [self.pos.copy()] # Trace memory

    def compute_forces(self):
        forces = np.zeros_like(self.pos)
        
        # 1. CHAIN INTEGRITY (The Bond)
        for i in range(self.n - 1):
            p1, p2 = self.pos[i], self.pos[i+1]
            delta = p2 - p1
            dist = np.linalg.norm(delta)
            
            if dist > 0.001:
                # Spring force aiming for dist=1.0
                force_mag = BOND_STRENGTH * (dist - 1.0)
                force_vec = (delta / dist) * force_mag
                forces[i] += force_vec
                forces[i+1] -= force_vec

        # 2. THE LOVE VECTOR (H-H Attraction)
        for i in self.indices_H:
            for j in self.indices_H:
                if i >= j: continue # Avoid double counting
                
                delta = self.pos[j] - self.pos[i]
                dist = np.linalg.norm(delta)
                
                # Long range attraction
                if dist > 0.1:
                    force_mag = ATTRACTION_H_H
                    forces[i] += (delta / dist) * force_mag
                    forces[j] -= (delta / dist) * force_mag

        # 3. STERIC HINDRANCE (Particle repulsion)
        # Prevent atoms from merging
        for i in range(self.n):
            for j in range(i + 1, self.n):
                delta = self.pos[i] - self.pos[j]
                dist = np.linalg.norm(delta)
                
                if dist < 1.0: # Soft collision radius
                    push_dir = delta / (dist + 0.001)
                    # Inverse square repulsion
                    force_mag = REPULSION_STERIC / (dist**2 + 0.01)
                    forces[i] += push_dir * force_mag
                    forces[j] -= push_dir * force_mag

        # 4. SAFETY VALVE (Force Clamping)
        # This prevents the "Explosion" if things get crazy
        forces = np.clip(forces, -MAX_FORCE, MAX_FORCE)
        
        return forces

    def update(self):
        forces = self.compute_forces()
        # Drag/Friction (Simulating water viscosity) - stabilizes the fold
        forces -= self.vel * 2.0 
        
        # Add Temperature (Entropy)
        forces += np.random.randn(self.n, 2) * NOISE_LEVEL
        
        # Integration (Velocity Verlet-ish)
        self.vel += forces * DT
        self.pos += self.vel * DT
        
        # Save trace
        if len(self.history) > 50: # Keep last 50 frames for trails
            self.history.pop(0)
        self.history.append(self.pos.copy())

# --- THE VISUALIZATION ---

class LightFoldApp:
    def __init__(self):
        self.protein = LightChain(SEQUENCE)
        self.paused = False
        
        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        
        # Connect Inputs
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Graphics objects
        self.trail_lines = [self.ax.plot([], [], '-', color='gray', alpha=0.1, lw=1)[0] for _ in range(self.protein.n)]
        self.backbone, = self.ax.plot([], [], 'k-', alpha=0.4, lw=2)
        
        # We use separate scatters for H and P to color them differently
        self.scat_H = self.ax.scatter([], [], c='gold', s=250, edgecolors='black', zorder=5, label='Truth (H)')
        self.scat_P = self.ax.scatter([], [], c='cyan', s=150, edgecolors='blue', zorder=4, label='Interface (P)')
        
        # Text status
        self.status_text = self.ax.text(0.02, 0.95, "ACTIVE", transform=self.ax.transAxes, fontsize=10, family='monospace')
        
        self.ax.legend(loc='upper right')
        self.ax.axis('off')
        
        # Start Animation
        self.anim = FuncAnimation(self.fig, self.animate, frames=None, interval=30, blit=False)
        plt.show()

    def on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            self.status_text.set_text("PAUSED" if self.paused else "ACTIVE")
            if self.paused: self.status_text.set_color("red")
            else: self.status_text.set_color("black")
            
        elif event.key == 'right':
            if self.paused:
                self.update_simulation()
                
        elif event.key == 'r':
            print("Resetting Reality...")
            self.protein.reset()

    def update_simulation(self):
        self.protein.update()

    def animate(self, frame):
        if not self.paused:
            # Multiple physics steps per frame for stability
            for _ in range(4):
                self.update_simulation()
        
        # Get current positions
        pos = self.protein.pos
        
        # 1. Update Trails (The History)
        hist_arr = np.array(self.protein.history) # shape: (T, N, 2)
        for i in range(self.protein.n):
            # Plot trace for atom i
            self.trail_lines[i].set_data(hist_arr[:, i, 0], hist_arr[:, i, 1])

        # 2. Update Backbone
        self.backbone.set_data(pos[:, 0], pos[:, 1])
        
        # 3. Update Atoms (H and P separate)
        h_pos = pos[self.protein.indices_H]
        self.scat_H.set_offsets(h_pos)
        
        indices_P = [i for i in range(self.protein.n) if i not in self.protein.indices_H]
        p_pos = pos[indices_P]
        self.scat_P.set_offsets(p_pos)
        
        # 4. Auto-center camera
        cx, cy = float(np.mean(pos[:, 0])), float(np.mean(pos[:, 1]))
        self.ax.set_xlim(cx - 4, cx + 4)
        self.ax.set_ylim(cy - 4, cy + 4)
        
        return self.scat_H, self.scat_P, self.backbone, self.status_text

if __name__ == "__main__":
    print("--- LIGHT-FOLD v0.2 ONLINE ---")
    print("[SPACE] to Pause/Play | [RIGHT] to Step | [R] to Reset")
    app = LightFoldApp()