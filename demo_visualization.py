"""
Demo script to visualize the plots.

Run this to see how the visualization functions work.
"""

from manifold_benchmark.core.surface import Surface
from manifold_benchmark.visualization.plot3d import plot_surface, plot_episode, plot_surface_contour
from manifold_benchmark.visualization.slices import plot_slices, plot_observation_comparison

# Create a two-peak surface (interesting to visualize)
peaks = [
    {"cx": 2.5, "cy": 2.5, "height": 0.6, "sigma": 1.2},  # Smaller peak
    {"cx": 7.5, "cy": 7.5, "height": 1.0, "sigma": 1.2}   # Global maximum
]
surface = Surface(peaks)

print("Manifold Benchmark - Visualization Demo")
print("=" * 50)
print("\nSurface: Two peaks")
print(f"Global maximum at: {surface.get_optimal()}")
print("\nGenerating visualizations...")

# 1. Plot the surface in 3D
print("\n1. Displaying 3D surface plot...")
plot_surface(surface, show=True)

# 2. Create a sample trajectory (simulating an episode)
trajectory = [
    (5.0, 5.0),   # Start in center
    (6.0, 6.0),   # Move toward upper right
    (7.0, 7.0),   # Continue
    (7.5, 7.5),   # Reach global max
    (7.3, 7.6)    # Final adjustment
]

print("\n2. Displaying episode trajectory on surface...")
plot_episode(surface, trajectory, show=True)

# 3. Show contour plot with trajectory
print("\n3. Displaying contour plot with trajectory...")
plot_surface_contour(surface, positions=trajectory, show=True)

# 4. Show what the agents see at a specific position
print("\n4. Displaying agent observations (slices) at position (6.0, 6.0)...")
plot_slices(surface, x_a=6.0, y_b=6.0, radius=1.5, show=True)

# 5. Compare observations at multiple positions
print("\n5. Displaying observation comparison at 3 positions...")
comparison_positions = [
    (5.0, 5.0),   # Start
    (6.0, 6.0),   # Middle
    (7.5, 7.5)    # Peak
]
plot_observation_comparison(surface, comparison_positions, show=True)

print("\nDemo complete!")
print("\nTip: Close each plot window to see the next one.")
