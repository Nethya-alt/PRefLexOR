import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from networkx.algorithms.community import louvain_communities
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Parameters
saved_graphs_dir = "saved_graphs"      # Directory containing graphml files
output_file = "graph_evolution_3d.mp4" # Output movie file
max_iterations = 1024                  # Control max iterations (adjust for testing)
fps = 15                               # Slightly higher frames per second for smoother rotation
initial_node_size = 120                # Base starting node size for degree scaling
final_rotation_frames = 240            # Number of frames to show the final graph rotating (4 seconds at 15 fps)
zoom_frames = 240                      # Number of frames for the zoom effect (2 seconds at 15 fps)

# Get list of graph files and sort them
graph_files = [f for f in os.listdir(saved_graphs_dir) if f.endswith('.graphml')]
graph_files.sort()
graph_files = graph_files[:max_iterations]

# Load all graphs
graphs = []
for file in graph_files:
    filepath = os.path.join(saved_graphs_dir, file)
    G = nx.read_graphml(filepath)
    graphs.append(G)

# Set up the 3D figure with a black background
fig = plt.figure(figsize=(20, 16))  # Larger figure for more space and impact
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')  # Black background for universe effect
plt.style.use('default')

# Compute communities for coloring and positioning (using the final graph)
final_communities = louvain_communities(graphs[-1])
community_map = {}
for i, comm in enumerate(final_communities):
    for node in comm:
        community_map[node] = i
n_communities = len(final_communities)
# Use a vibrant, cosmic color map for communities (Spectral for distinct colors)
colors = plt.cm.Spectral(np.linspace(0, 1, n_communities))  # Bright, distinct colors for communities

# Function to compute positions for an expanding 3D graph with community grouping, more distributed
def get_expanding_3d_positions(graphs):
    # Use spring layout for the final graph, grouping by communities, with wider distribution
    G_final = graphs[-1]
    n_nodes = len(G_final.nodes())
    pos_final = {}
    
    # Create subgraphs for each community and position them in 3D space with wider distribution
    for i, community in enumerate(final_communities):
        # Create a subgraph for this community
        G_comm = G_final.subgraph(community)
        if len(G_comm.nodes()) > 0:
            # Use 3D spring layout for each community, with larger k for wider spacing
            comm_pos = nx.spring_layout(G_comm, dim=3, k=2.5/np.sqrt(len(G_comm.nodes())), iterations=50, seed=i)
            # Scale and shift positions to spread communities across x, y, and z
            scale = 1.0 + i * 0.2  # Increase scale by community index for wider spread
            # Random offset for x and y, fixed offset for z to distribute across 3D space
            offset = np.array([
                np.random.uniform(-0.5, 0.5),  # Random x offset
                np.random.uniform(-0.5, 0.5),  # Random y offset
                i * 0.4 - (n_communities * 0.2)  # Spread along z-axis, centered
            ])
            for node in comm_pos:
                # Apply scaling and offset, then add slight randomization for natural distribution
                pos_final[node] = (np.array(comm_pos[node]) * scale + offset) * 0.8 + np.random.rand(3) * 0.1
                # Ensure positions stay within [-2.5, 2.5] before normalization
                pos_final[node] = np.clip(pos_final[node], -2.5, 2.5)
    
    positions = []
    expansion_factor = 0.2  # Increased for more noticeable and distributed expansion
    
    for i, G in enumerate(graphs):
        pos = {}
        t = i / (len(graphs) - 1) if len(graphs) > 1 else 1  # Progress toward final state
        for node in G.nodes():
            if node in pos_final:
                # Start with a random 3D position, interpolate toward the community position, then expand
                start_pos = np.random.rand(3) * 0.2 if i == 0 else pos_final[node]  # Small initial random spread
                base_pos = (1 - t) * start_pos + t * pos_final[node]
                # Expand outward based on iteration (scale by t and add random wobble)
                expansion = t * expansion_factor * np.random.rand(3) * 2 - expansion_factor  # Random outward push
                pos[node] = base_pos + expansion
            else:
                # For nodes not in final graph, assign a random 3D position near the current expansion
                pos[node] = np.random.rand(3) * (0.2 + t * 0.8)  # Start small, expand outward
        # Ensure positions stay within a reasonable 3D space (e.g., [-2.5, 2.5] for each axis, then normalize)
        for node in pos:
            pos[node] = np.clip(pos[node], -2.5, 2.5)
            # Normalize to [0, 1] for visualization
            pos[node] = (pos[node] + 2.5) / 5  # Shift and scale to [0, 1]
        positions.append(pos)
    return positions

positions = get_expanding_3d_positions(graphs)

# Initialization function
def init():
    ax.clear()
    ax.set_facecolor('black')
    # Set 3D axes limits and labels (optional, for clarity)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X', color='white', fontsize=10)
    ax.set_ylabel('Y', color='white', fontsize=10)
    ax.set_zlabel('Z', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    return []

# Animation update function with continuous rotation, final rotations, and zoom
def update(frame):
    ax.clear()
    ax.set_facecolor('black')
    
    # Determine the phase of the animation
    total_frames = len(graphs) + final_rotation_frames + zoom_frames
    if frame < len(graphs):
        # Growing phase: show the graph as it expands and rotates
        G = graphs[frame]
        pos = positions[frame]
    else:
        # Final rotation phase: show only the final graph rotating
        G = graphs[-1]
        pos = positions[-1]
    
    # Calculate node degrees for sizing
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1  # Avoid division by zero
    
    # Dynamic node size: proportional to degree, scaled by growth phase
    if frame < len(graphs):
        base_size = initial_node_size * (1 - frame / len(graphs))  # Shrink as graph grows
    else:
        base_size = 5  # Fixed small size for final rotations and zoom
    
    node_sizes = [base_size * (degree / max_degree * 2 + 0.5) for degree in degrees.values()]  # Scale by degree (min 0.5x, max 2.5x base size)
    node_sizes = [max(size, 5) for size in node_sizes]  # Ensure minimum size of 5
    
    # Get node colors based on communities (vibrant, pulsating alpha)
    node_colors = [colors[community_map.get(node, 0)] for node in G.nodes()]  # Use community colors
    alpha_pulse = 0.6 + 0.2 * np.sin(frame * 0.1)  # Pulsating transparency for dynamic effect
    
    # Draw nodes with transparency, glowing edges, and size proportional to degree, grouped by community
    ax.scatter([pos[n][0] for n in G.nodes()], 
               [pos[n][1] for n in G.nodes()], 
               [pos[n][2] for n in G.nodes()], 
               s=node_sizes, c=node_colors, alpha=alpha_pulse, edgecolors='cyan', linewidths=0.8, zorder=10)
    
    # Draw edges with clearer, thicker lines for cosmic effect in 3D
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], 
                [pos[u][1], pos[v][1]], 
                [pos[u][2], pos[v][2]], 
                c='cyan', alpha=0.1, lw=2.0, zorder=-1)  # Thicker, less transparent cyan edges
    
    # Continuous rotation as the graph grows (slower, smooth rotation)
    angle = frame * 2  # Rotate 2 degrees per frame for smooth rotation
    ax.view_init(elev=30, azim=angle)  # Slightly higher elevation, smooth rotation
    
    # Zoom effect for the final phase
    if frame >= len(graphs) + final_rotation_frames:
        # Calculate zoom progress (0 to 1 over zoom_frames)
        zoom_progress = (frame - (len(graphs) + final_rotation_frames)) / zoom_frames
        if zoom_progress > 1:
            zoom_progress = 1  # Cap at 1
        
        # Define a center point to zoom into (e.g., the mean position of all nodes in the final graph)
        center = np.mean([pos[n] for n in G.nodes()], axis=0)
        zoom_scale = 1 - zoom_progress * 0.8  # Zoom in by reducing the view range (80% zoom at end)
        
        # Set dynamic limits for zoom effect
        ax.set_xlim(center[0] - 0.5 * zoom_scale, center[0] + 0.5 * zoom_scale)
        ax.set_ylim(center[1] - 0.5 * zoom_scale, center[1] + 0.5 * zoom_scale)
        ax.set_zlim(center[2] - 0.5 * zoom_scale, center[2] + 0.5 * zoom_scale)
    else:
        # Full view during growth and final rotations
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
    
    # Add title with iteration number (white for contrast on black background)
    if frame < len(graphs):
        iteration = int(graph_files[frame].split('_')[-1].split('.')[0])
        ax.set_title(f"Graph Evolution - Iteration {iteration:06d}", color='white')
    else:
        ax.set_title("Final Graph - Rotating and Zooming", color='white')
    
    ax.axis('off')
    return []

# Create animation with extended frames for final rotations and zoom
total_frames = len(graphs) + final_rotation_frames + zoom_frames
anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False, interval=1000/fps)

# Save the animation
anim.save(output_file, writer='ffmpeg', fps=fps, dpi=300)
print(f"Movie saved as {output_file}")

# Close plot
plt.close()
