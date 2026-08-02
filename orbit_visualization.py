import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ANIMATED = False

if ANIMATED == False:
    # ---------------------------------------------------------
    # Parameters for the Orbits
    # ---------------------------------------------------------
    # Time array (simulating several orbits to show the path)
    t = np.linspace(0, 50, 2000)

    # Amplitudes (arbitrary units for visualization)
    Ax = 1.0  # X-axis amplitude (toward/away from primary bodies)
    Ay = 2.0  # Y-axis amplitude (along the orbital path)
    Az = 0.8  # Z-axis amplitude (vertical out-of-plane motion)

    # In-plane frequency (matches for all three)
    wx = 1.0  

    # ---------------------------------------------------------
    # 1. Lyapunov Orbit (Strictly 2D, In-Plane)
    # ---------------------------------------------------------
    x_lyap = -Ax * np.cos(wx * t)
    y_lyap =  Ay * np.sin(wx * t)
    z_lyap =  np.zeros_like(t)  # Zero out-of-plane motion

    # ---------------------------------------------------------
    # 2. Lissajous Orbit (3D, Quasi-Periodic / Drifting)
    # ---------------------------------------------------------
    # The out-of-plane frequency is NOT synchronized with the in-plane frequency
    wz_liss = np.sqrt(2) # Irrational number to guarantee it doesn't close
    x_liss = -Ax * np.cos(wx * t)
    y_liss =  Ay * np.sin(wx * t)
    z_liss =  Az * np.sin(wz_liss * t)

    # ---------------------------------------------------------
    # 3. Halo Orbit (3D, Perfectly Periodic / Closed Loop)
    # ---------------------------------------------------------
    # The out-of-plane frequency is perfectly synchronized (1:1 ratio)
    wz_halo = 1.0 
    x_halo = -Ax * np.cos(wx * t)
    y_halo =  Ay * np.sin(wx * t)
    z_halo =  Az * np.cos(wz_halo * t) # Phase matched to create the "halo" loop

    # ---------------------------------------------------------
    # Plotting the Orbits
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(18, 6))
    fig.canvas.manager.set_window_title("Lagrange Point Orbits Visualization")

    # Helper function to format the 3D plots
    def format_axis(ax, title):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (Toward Primary Body)')
        ax.set_ylabel('Y (Along Orbit)')
        ax.set_zlabel('Z (Out of Plane)')
        # Set consistent limits so the scale is identical across all three
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_zlim([-1.5, 1.5])
        # Add a marker for the Lagrange Point
        ax.scatter([0], [0], [0], color='black', s=50, label='Lagrange Point', zorder=5)
        ax.legend(loc='upper right')

    # --- Plot 1: Lyapunov ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(x_lyap, y_lyap, z_lyap, color='blue', linewidth=2)
    format_axis(ax1, '1. Lyapunov Orbit\n(2D, Flat, Periodic)')

    # --- Plot 2: Lissajous ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(x_liss, y_liss, z_liss, color='orange', linewidth=1.5, alpha=0.8)
    format_axis(ax2, '2. Lissajous Orbit\n(3D, Drifting, Quasi-Periodic)')

    # --- Plot 3: Halo ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(x_halo, y_halo, z_halo, color='green', linewidth=2)
    format_axis(ax3, '3. Halo Orbit\n(3D, Closed Loop, Periodic)')

    plt.tight_layout()
    plt.show()
else:

    # ---------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------
    Ax, Ay, Az = 1.0, 2.0, 0.8
    wx = 1.0
    wz_liss = np.sqrt(2)
    wz_halo = 1.0

    # Time setup
    dt = 0.05
    frames = 600

    # ---------------------------------------------------------
    # Setup Figure and Subplots
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(18, 6))
    fig.canvas.manager.set_window_title("Animated Lagrange Orbits")

    axes = []
    lines = []
    dots = []

    titles = ['1. Lyapunov (2D Flat)', 
            '2. Lissajous (3D Drifting)', 
            '3. Halo (3D Synchronized)']

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Draw Lagrange point
        ax.scatter([0], [0], [0], color='black', s=50, zorder=5)
        
        # Initialize empty lines for the trail and dots for the spacecraft
        line, = ax.plot([], [], [], color=['blue', 'orange', 'green'][i], lw=2, alpha=0.7)
        dot, = ax.plot([], [], [], marker='o', color='red', markersize=8)
        
        axes.append(ax)
        lines.append(line)
        dots.append(dot)

    plt.tight_layout()

    # Lists to store the trace history
    history_x = [[], [], []]
    history_y = [[], [], []]
    history_z = [[], [], []]

    # ---------------------------------------------------------
    # Animation Update Function
    # ---------------------------------------------------------
    def update(frame):
        t = frame * dt
        
        # 1. Lyapunov Math
        x1, y1, z1 = -Ax * np.cos(wx * t), Ay * np.sin(wx * t), 0.0
        
        # 2. Lissajous Math
        x2, y2, z2 = -Ax * np.cos(wx * t), Ay * np.sin(wx * t), Az * np.sin(wz_liss * t)
        
        # 3. Halo Math
        x3, y3, z3 = -Ax * np.cos(wx * t), Ay * np.sin(wx * t), Az * np.cos(wz_halo * t)
        
        current_positions = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
        
        for i in range(3):
            x, y, z = current_positions[i]
            
            # Update history for the trail
            history_x[i].append(x)
            history_y[i].append(y)
            history_z[i].append(z)
            
            # Keep the tail length manageable (optional: remove the [: -200] slicing to keep the whole path)
            tail_x = history_x[i][-300:]
            tail_y = history_y[i][-300:]
            tail_z = history_z[i][-300:]
            
            # Update line (trail)
            lines[i].set_data(tail_x, tail_y)
            lines[i].set_3d_properties(tail_z)
            
            # Update dot (spacecraft)
            dots[i].set_data([x], [y])
            dots[i].set_3d_properties([z])

        return lines + dots

    # ---------------------------------------------------------
    # Run Animation
    # ---------------------------------------------------------
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=False)
    plt.show()