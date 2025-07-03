import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from collections import defaultdict
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 管道系统配置
directions = ['F', 'U', 'B', 'D']  # 前进、上升、后退、下降
n_nodes = 12  # 管道节点数
fixed_start, fixed_end = 'F', 'D'  # 固定起点和终点

# 环境区域定义
class EnvironmentZones:
    def __init__(self):
        # 危险区域 (x1, y1, z1, x2, y2, z2)
        self.danger_zones = [
            (2, 1, 0, 4, 3, 0),      # 矩形危险区域1
            (-3, -2, 0, -1, 1, 0),   # 矩形危险区域2
            (3, -3, 0, 5, -1, 0),    # 新增危险区域
        ]
        
        # 障碍物区域
        self.obstacles = [
            (0, 3, 0, 1, 4, 0),      # 障碍物1
            (-2, -1, 0, -1, 1, 0),   # 障碍物2
        ]
        
        # 亲和区域（绿色，鼓励经过）
        self.affinity_zones = [
            (-1, 2, 0, 1, 3, 0),     # 矩形亲和区域1
            (-2, -3, 0, 2, -2, 0),   # 矩形亲和区域2
        ]
        
        # 危险区域安全距离
        self.safety_margin = 0.8
    
    def point_in_danger_zone(self, x, y, z):
        """检查点是否在危险区域内（包含安全边距）"""
        for x1, y1, z1, x2, y2, z2 in self.danger_zones:
            if (x1 - self.safety_margin <= x <= x2 + self.safety_margin and 
                y1 - self.safety_margin <= y <= y2 + self.safety_margin and
                z1 - self.safety_margin <= z <= z2 + self.safety_margin):
                return True
        return False
    
    def point_in_obstacle(self, x, y, z):
        """检查点是否在障碍物内"""
        for x1, y1, z1, x2, y2, z2 in self.obstacles:
            if x1 <= x <= x2 and y1 <= y <= y2 and z1 <= z <= z2:
                return True
        return False
    
    def point_in_affinity_zone(self, x, y, z):
        """检查点是否在亲和区域内"""
        for x1, y1, z1, x2, y2, z2 in self.affinity_zones:
            if x1 <= x <= x2 and y1 <= y <= y2 and z1 <= z <= z2:
                return True
        return False

# 初始化环境
env_zones = EnvironmentZones()

# 2. 改进的能量计算函数（返回总能量和各分项）
def calculate_energy_components(path):
    """计算管道配置能量和各组成部分"""
    components = {
        'total': 0,
        'self_intersect': 0,
        'danger_zone': 0,
        'obstacle': 0,
        'affinity': 0,
        'efficiency': 0,
        'sharp_turn': 0,
        'direction_change': 0,
        'length': 0
    }
    
    pos = np.zeros(3)  # 3D位置跟踪
    visited = defaultdict(int)
    visited[tuple(pos)] = 1
    prev_dir = path[0]
    
    for i, d in enumerate(path[1:], 1):
        # 更新位置
        move = {'F': [1,0,0], 'B': [-1,0,0], 
                'U': [0,1,0], 'D': [0,-1,0]}[d]
        pos += move
        
        # 急转弯惩罚（180度）
        if (ord(prev_dir) - ord(d)) % 2 == 0 and prev_dir != d:
            components['sharp_turn'] += 5
            components['total'] += 5
        
        # 自交检测
        tpos = tuple(pos)
        if tpos in visited:
            penalty = 15 * visited[tpos]
            components['self_intersect'] += penalty
            components['total'] += penalty
        visited[tpos] += 1
        prev_dir = d
        
        # 环境约束检查
        x, y, z = pos
        if env_zones.point_in_danger_zone(x, y, z):
            components['danger_zone'] += 60
            components['total'] += 60
        if env_zones.point_in_obstacle(x, y, z):
            components['obstacle'] += 150
            components['total'] += 150
        if env_zones.point_in_affinity_zone(x, y, z):
            components['affinity'] += 25
            components['total'] -= 25
    
    # 路径效率评估
    unique_points = len(visited)
    total_points = len(path)
    efficiency_ratio = unique_points / total_points
    efficiency_penalty = 80 * (1 - efficiency_ratio)
    components['efficiency'] = efficiency_penalty
    components['total'] += efficiency_penalty
    
    # 转向惩罚（使用简化后路径计算）
    dir_changes = sum(1 for i in range(len(path)-1) if path[i] != path[i+1])
    dir_penalty = 3 * dir_changes
    components['direction_change'] = dir_penalty
    components['total'] += dir_penalty
    
    # 路径长度惩罚
    length_penalty = 2 * len(path)
    components['length'] = length_penalty
    components['total'] += length_penalty
    
    return components

# 包装函数，保持与之前代码兼容
def calculate_energy(path):
    return calculate_energy_components(path)['total']

# 3. 路径生成算法（增强多样性）
def generate_paths_mcmc(n_samples=1500, temp=0.6, diversity=0.4):
    """改进的MCMC采样，增加路径多样性"""
    # 初始路径
    current_path = (fixed_start + 
                   ''.join(np.random.choice(directions, n_nodes-2)) + fixed_end)
    current_energy = calculate_energy(current_path)
    samples = [current_path]
    energies = [current_energy]
    
    for _ in range(n_samples-1):
        # 随机选择变异方式
        r = np.random.rand()
        if r < diversity:
            # 方式1：随机重置部分路径
            idx = np.random.randint(1, n_nodes-1)
            new_path = (current_path[:idx] + 
                       ''.join(np.random.choice(directions, n_nodes-1-idx)) + 
                       fixed_end)
        elif r < 0.7:
            # 方式2：交换两个随机位置
            idx = np.random.choice(range(1, n_nodes-1), size=2, replace=False)
            new_path = list(current_path)
            new_path[idx[0]], new_path[idx[1]] = new_path[idx[1]], new_path[idx[0]]
            new_path = ''.join(new_path)
        else:
            # 方式3：随机翻转一个方向
            idx = np.random.randint(1, n_nodes-1)
            new_dir = np.random.choice([d for d in directions if d != current_path[idx]])
            new_path = current_path[:idx] + new_dir + current_path[idx+1:]
        
        # 计算能量并决定是否接受
        new_energy = calculate_energy(new_path)
        if (new_energy < current_energy or 
            np.random.rand() < np.exp((current_energy-new_energy)/temp)):
            current_path, current_energy = new_path, new_energy
        
        samples.append(current_path)
        energies.append(current_energy)
    
    return samples, np.array(energies)

# 4. 改进的路径距离度量
def path_distance(path1, path2):
    """综合考虑转向点、三维形状和方向分布的复合距离"""
    # 转向点差异
    def get_turns(path):
        return {i for i in range(1, len(path)) if path[i] != path[i-1]}
    turn_dist = len(get_turns(path1).symmetric_difference(get_turns(path2))) * 0.5
    
    # 三维形状差异
    def get_shape(path):
        pos = np.zeros(3)
        shape = {tuple(pos)}
        for d in path:
            pos += {'F': [1,0,0], 'B': [-1,0,0], 
                   'U': [0,1,0], 'D': [0,-1,0]}[d]
            shape.add(tuple(pos))
        return shape
    shape_dist = len(get_shape(path1).symmetric_difference(get_shape(path2))) * 0.8
    
    # 方向分布差异
    hist_dist = sum(abs(path1.count(d) - path2.count(d)) for d in directions) * 0.3
    
    return turn_dist + shape_dist + hist_dist

# 5. 主程序
def main():
    # 生成路径样本
    print("生成多样化路径样本...")
    paths, energies = generate_paths_mcmc(n_samples=1500, temp=0.6, diversity=0.4)
    
    # 选择代表性样本（按能量排序）
    sample_size = min(200, len(paths))
    sample_idx = sorted(np.random.choice(len(paths), size=sample_size, replace=False),
                       key=lambda x: energies[x])
    sample_paths = [paths[i] for i in sample_idx]
    sample_energies = energies[sample_idx]
    
    # 预先计算所有样本的能量组成
    sample_components = [calculate_energy_components(p) for p in sample_paths]
    
    # 计算距离矩阵
    print("计算改进的距离矩阵...")
    dist_matrix = np.zeros((sample_size, sample_size))
    for i, j in combinations(range(sample_size), 2):
        dist = path_distance(sample_paths[i], sample_paths[j])
        dist_matrix[i,j] = dist_matrix[j,i] = dist
    
    # t-SNE降维
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=2, metric="precomputed", 
               init="random", random_state=42,
               perplexity=min(30, sample_size//4),
               n_iter=2000)
    coords = tsne.fit_transform(dist_matrix)
    
    # 创建可视化
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    
    # 绘制能量表面
    xi = np.linspace(coords[:,0].min()-1, coords[:,0].max()+1, 200)
    yi = np.linspace(coords[:,1].min()-1, coords[:,1].max()+1, 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((coords[:,0], coords[:,1]), sample_energies, 
                 (xi, yi), method='cubic')
    contour = ax.contourf(xi, yi, zi, levels=20, cmap='viridis', alpha=0.8)
    
    # 绘制样本点
    sc = ax.scatter(coords[:,0], coords[:,1], c=sample_energies,
                   cmap='viridis', edgecolors='k', linewidths=0.5, s=60)
    
    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label('配置能量', fontsize=12)
    
    # 标记能量最低和最高的点
    min_idx = np.argmin(sample_energies)
    max_idx = np.argmax(sample_energies)
    for idx, marker, color in [(min_idx, 'o', 'lime'), (max_idx, 'X', 'red')]:
        ax.scatter(coords[idx,0], coords[idx,1], c=color,
                  s=200, marker=marker, edgecolors='k', linewidths=1)
        ax.annotate(f"路径: {sample_paths[idx]}\n能量: {sample_energies[idx]:.1f}",
                   (coords[idx,0], coords[idx,1]),
                   xytext=(0,15), textcoords="offset points",
                   ha='center', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # 添加图例
    ax.scatter([], [], c='lime', s=100, marker='o', 
              edgecolors='k', label='最低能量')
    ax.scatter([], [], c='red', s=100, marker='X', 
              edgecolors='k', label='最高能量')
    ax.legend(loc='upper left', fontsize=10)
    
    # 交互功能
    def onclick(event):
        if event.inaxes != ax:
            return
        
        # 找到最近的样本点
        click_pos = np.array([event.xdata, event.ydata])
        distances = np.sqrt(np.sum((coords - click_pos)**2, axis=1))
        closest_idx = np.argmin(distances)
        path = sample_paths[closest_idx]
        components = sample_components[closest_idx]
        
        # 清除旧标注
        for artist in ax.texts + ax.collections:
            if hasattr(artist, '_interactive_annotation'):
                artist.remove()
        
        # 创建详细能量分解文本
        detail_text = (
            f"路径: {path}\n"
            f"总能量: {components['total']:.1f}\n"
            "------------------------\n"
            f"自相交惩罚: {components['self_intersect']:.1f}\n"
            f"危险区域惩罚: {components['danger_zone']:.1f}\n"
            f"障碍物惩罚: {components['obstacle']:.1f}\n"
            f"亲和区域奖励: -{components['affinity']:.1f}\n"
            f"急转弯惩罚: {components['sharp_turn']:.1f}\n"
            f"方向变化惩罚: {components['direction_change']:.1f}\n"
            f"路径效率惩罚: {components['efficiency']:.1f}\n"
            f"路径长度惩罚: {components['length']:.1f}"
        )
        
        # 添加新标注
        annot = ax.annotate(
            detail_text,
            xy=(coords[closest_idx, 0], coords[closest_idx, 1]),
            xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9),
            fontsize=10
        )
        annot._interactive_annotation = True
        
        # 高亮显示选中点
        dot = ax.scatter(
            coords[closest_idx, 0], coords[closest_idx, 1],
            s=100, facecolors='none', edgecolors='red', linewidths=2
        )
        dot._interactive_annotation = True
        
        plt.draw()
    
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    
    # 设置标题和标签
    plt.title('管道配置能量景观（多样性优化）\n'
             f'节点数: {n_nodes}, 样本数: {sample_size}, 温度参数: 0.6', 
             fontsize=14, pad=20)
    plt.xlabel('t-SNE维度1', fontsize=12)
    plt.ylabel('t-SNE维度2', fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    
    print("可视化完成！点击图上任意位置查看路径详情和能量分解")
    plt.show()

if __name__ == "__main__":
    main()