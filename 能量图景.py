import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from collections import defaultdict
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 1. 管道构型系统定义（保持不变）
directions = ['F', 'U', 'B', 'D']  # Forward, Up, Back, Down
n_nodes = 12  # Number of pipeline nodes
fixed_start, fixed_end = 'F', 'D'  # Fixed start and end points

# 2. 优化后的能量计算函数（保持不变）
def calculate_energy(path):
    """Calculate pipeline configuration energy:
    - Sharp turn penalty (180-degree turns)
    - Self-intersection penalty (3D space detection)
    - Path compactness reward
    """
    energy = 0
    pos = np.zeros(3)
    visited = {tuple(pos): 1}
    turn_penalty = 0
    
    prev_dir = path[0]
    for i, d in enumerate(path[1:], 1):
        # Update position
        if d == 'F': pos += [1,0,0]
        elif d == 'B': pos += [-1,0,0]
        elif d == 'U': pos += [0,1,0]
        elif d == 'D': pos += [0,-1,0]
        
        # Sharp turn penalty (180 degrees)
        if (ord(prev_dir) - ord(d)) % 2 == 0 and prev_dir != d:
            energy += 2
            turn_penalty += 1
        
        # Self-intersection detection
        tpos = tuple(pos)
        if tpos in visited:
            energy += 5 * visited[tpos]
        visited[tpos] = visited.get(tpos, 0) + 1
        prev_dir = d
    
    # Path compactness reward (fewer unique positions is better)
    energy += 0.5 * len(visited)
    # Turn diversity penalty (avoid excessive repeated patterns)
    energy += 0.2 * turn_penalty
    return energy

# 3. 路径生成（MCMC采样）（保持不变）
def generate_paths_mcmc(n_samples=1000, temp=0.7):
    """Metropolis-Hastings sampling to generate paths"""
    current_path = fixed_start + ''.join(np.random.choice(directions, n_nodes-2)) + fixed_end
    current_energy = calculate_energy(current_path)
    samples = [current_path]
    energies = [current_energy]
    
    for _ in range(n_samples-1):
        # Generate candidate path (swap two random positions)
        idx = np.random.choice(range(1, n_nodes-1), size=2, replace=False)
        new_path = list(current_path)
        new_path[idx[0]], new_path[idx[1]] = new_path[idx[1]], new_path[idx[0]]
        new_path = ''.join(new_path)
        new_energy = calculate_energy(new_path)
        
        # Metropolis acceptance criterion
        if new_energy < current_energy or np.random.rand() < np.exp((current_energy-new_energy)/temp):
            current_path, current_energy = new_path, new_energy
        
        samples.append(current_path)
        energies.append(current_energy)
    
    return samples, np.array(energies)

# 4. 路径距离计算（保持不变）
def path_distance(path1, path2):
    """Comprehensive distance metric: swap distance + direction pattern distance"""
    # Character swap distance
    swap_dist = sum(c1 != c2 for c1, c2 in zip(path1, path2))
    
    # Direction pattern distance
    changes1 = [path1[i] != path1[i+1] for i in range(len(path1)-1)]
    changes2 = [path2[i] != path2[i+1] for i in range(len(path2)-1)]
    pattern_dist = sum(c1 != c2 for c1, c2 in zip(changes1, changes2))
    
    # Direction distribution distance
    hist1 = np.array([path1.count(d) for d in directions])
    hist2 = np.array([path2.count(d) for d in directions])
    hist_dist = np.linalg.norm(hist1 - hist2)
    
    return swap_dist + 0.4*pattern_dist + 0.2*hist_dist

# 5. 主程序（可视化部分改为英文）
if __name__ == "__main__":
    # Generate path samples
    print("Generating path samples...")
    paths, energies = generate_paths_mcmc(n_samples=800, temp=0.6)
    
    # Sample to calculate distance matrix (control computation)
    sample_size = min(150, len(paths))
    sample_idx = np.random.choice(len(paths), size=sample_size, replace=False)
    sample_paths = [paths[i] for i in sample_idx]
    sample_energies = energies[sample_idx]
    
    print("Calculating distance matrix...")
    dist_matrix = np.zeros((sample_size, sample_size))
    for i, j in combinations(range(sample_size), 2):
        dist = path_distance(sample_paths[i], sample_paths[j])
        dist_matrix[i,j] = dist_matrix[j,i] = dist
    
    # t-SNE dimensionality reduction
    print("Performing dimensionality reduction...")
    tsne = TSNE(n_components=2, metric="precomputed", 
               init="random", random_state=42,
               perplexity=min(30, sample_size//4))
    coords = tsne.fit_transform(dist_matrix)
    
    # Interpolate to generate continuous surface
    print("Generating continuous landscape...")
    xi = np.linspace(coords[:,0].min()-1, coords[:,0].max()+1, 200)
    yi = np.linspace(coords[:,1].min()-1, coords[:,1].max()+1, 200)
    xi, yi = np.meshgrid(xi, yi)
    
    # Cubic spline interpolation
    zi = griddata((coords[:,0], coords[:,1]), sample_energies, 
                 (xi, yi), method='cubic')
    
    # Create figure
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    
    # Plot interpolated energy surface
    contour = ax.contourf(xi, yi, zi, levels=20, cmap='viridis', alpha=0.8)
    
    # Plot original data points
    sc = ax.scatter(coords[:,0], coords[:,1], c=sample_energies,
                   cmap='viridis', edgecolors='k', linewidths=0.5, s=60)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label('Configuration Energy', fontsize=12)  # 改为英文标签
    
    # Mark key points
    min_idx = np.argmin(sample_energies)
    max_idx = np.argmax(sample_energies)
    
    for idx, marker in [(min_idx, 'o'), (max_idx, 'X')]:
        ax.scatter(coords[idx,0], coords[idx,1], 
                  c='red' if idx == max_idx else 'lime',
                  s=200, marker=marker, edgecolors='k', linewidths=1)
        ax.annotate(f"{sample_paths[idx]}\nEnergy={sample_energies[idx]:.1f}",  # 改为英文注解
                   (coords[idx,0], coords[idx,1]),
                   textcoords="offset points", xytext=(0,15),
                   ha='center', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # Add legend and title
    ax.scatter([], [], c='lime', s=100, marker='o', edgecolors='k', label='Minimum Energy')  # 改为英文图例
    ax.scatter([], [], c='red', s=100, marker='X', edgecolors='k', label='Maximum Energy')  # 改为英文图例
    ax.legend(loc='upper left', fontsize=10)
    
    plt.title(f'Pipeline Configuration Energy Landscape\n(Node count={n_nodes}, Sample size={sample_size}, Temperature parameter=0.6)',  # 改为英文标题
             fontsize=14, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)  # 改为英文标签
    plt.ylabel('t-SNE Dimension 2', fontsize=12)  # 改为英文标签
    plt.grid(alpha=0.2)
    
    # Save image
    plt.tight_layout()
    plt.savefig('pipeline_energy_landscape.png', dpi=300, bbox_inches='tight')
    print("Visualization completed, result saved as pipeline_energy_landscape.png")
    plt.show()