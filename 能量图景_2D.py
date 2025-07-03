import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from collections import defaultdict
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 管道构型系统定义 (使用世界坐标系下的上下左右)
directions = ['U', 'D', 'L', 'R']  # 上、下、左、右
n_nodes = 12  # 管道节点数量
fixed_start, fixed_end = 'R', 'U'  # 固定的起点和终点方向
target_point = (5, 3)  # 明确的目标点位置

# 2. 环境区域定义 (保持不变)
class EnvironmentZones:
    def __init__(self):
        # 危险区域 (x1, y1, x2, y2)
        self.danger_zones = [
            (2, 1, 4, 3),      # 矩形危险区域1
            (-3, -2, -1, 1),   # 矩形危险区域2
            (3, -3, 5, -1),    # 新增危险区域
        ]
        
        # 障碍物区域
        self.obstacles = [
            (0, 3, 1, 4),      # 障碍物1
            (-2, -1, -1, 1),   # 障碍物2
        ]
        
        # 亲和区域（绿色，鼓励经过）
        self.affinity_zones = [
            (-1, 2, 1, 3),     # 矩形亲和区域1
            (-2, -3, 2, -2),   # 矩形亲和区域2
        ]
        
        # 危险区域安全距离
        self.safety_margin = 0.8
    
    def point_in_danger_zone(self, x, y):
        """检查点是否在危险区域内（包含安全边距）"""
        for x1, y1, x2, y2 in self.danger_zones:
            if (x1 - self.safety_margin <= x <= x2 + self.safety_margin and 
                y1 - self.safety_margin <= y <= y2 + self.safety_margin):
                return True
        return False
    
    def point_in_obstacle(self, x, y):
        """检查点是否在障碍物内"""
        for x1, y1, x2, y2 in self.obstacles:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False
    
    def point_in_affinity_zone(self, x, y):
        """检查点是否在亲和区域内"""
        for x1, y1, x2, y2 in self.affinity_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

# 3. 路径转换为2D坐标 (修改为直接处理上下左右)
def path_to_2d_coords(path):
    """将路径字符串转换为2D坐标序列"""
    coords = [(0, 0)]  # 起始点
    x, y = 0, 0
    
    for direction in path:
        if direction == 'U':    # 上
            y += 1
        elif direction == 'D':  # 下
            y -= 1
        elif direction == 'L':  # 左
            x -= 1
        elif direction == 'R':  # 右
            x += 1
        
        coords.append((x, y))
    
    return coords

# 4. 检测路径自相交 (保持不变)
def has_path_intersection(coords):
    """检测路径是否存在自相交"""
    def line_segments_intersect(p1, q1, p2, q2):
        """检测两条线段是否相交"""
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0: return 0
            return 1 if val > 0 else 2
        
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        # 一般情况
        if o1 != o2 and o3 != o4:
            return True
        
        # 特殊情况
        if (o1 == 0 and on_segment(p1, p2, q1)) or \
           (o2 == 0 and on_segment(p1, q2, q1)) or \
           (o3 == 0 and on_segment(p2, p1, q2)) or \
           (o4 == 0 and on_segment(p2, q1, q2)):
            return True
        
        return False
    
    # 检查所有线段对是否相交
    for i in range(len(coords) - 1):
        for j in range(i + 2, len(coords) - 1):
            if line_segments_intersect(coords[i], coords[i+1], 
                                     coords[j], coords[j+1]):
                return True
    return False

# 5. 改进的能量计算函数 (增加危险区域严格惩罚)
def calculate_energy(path, env_zones):
    """计算管道配置能量：
    - 自相交严重惩罚
    - 危险区域严格惩罚
    - 障碍物惩罚
    - 亲和区域奖励
    - 急转弯惩罚
    - 路径长度惩罚
    - 目标点距离惩罚
    """
    coords = path_to_2d_coords(path)
    energy = 0
    
    # 1. 自相交检测 - 严重惩罚
    if has_path_intersection(coords):
        energy += 1000  # 非常高的惩罚，基本淘汰自相交路径
    
    # 2. 环境区域评估
    danger_penalty = 0
    obstacle_penalty = 0
    affinity_reward = 0
    
    for x, y in coords:
        if env_zones.point_in_danger_zone(x, y):
            danger_penalty += 200  # 大幅增加危险区域惩罚
        if env_zones.point_in_obstacle(x, y):
            obstacle_penalty += 300  # 障碍物惩罚（比危险区域更严重）
        if env_zones.point_in_affinity_zone(x, y):
            affinity_reward += 20  # 亲和区域奖励
    
    energy += danger_penalty + obstacle_penalty - affinity_reward
    
    # 3. 急转弯惩罚 (检查连续相反方向)
    for i in range(len(path)-1):
        if (path[i] == 'U' and path[i+1] == 'D') or \
           (path[i] == 'D' and path[i+1] == 'U') or \
           (path[i] == 'L' and path[i+1] == 'R') or \
           (path[i] == 'R' and path[i+1] == 'L'):
            energy += 30  # 增加相反方向移动惩罚
    
    # 4. 路径复杂度惩罚
    direction_changes = sum(1 for i in range(len(path)-1) if path[i] != path[i+1])
    energy += 0.8 * direction_changes
    
    # 5. 路径长度（总步数）
    total_length = len(coords)
    energy += 0.2 * total_length
    
    # 6. 终点约束检查（到目标点的距离）
    final_x, final_y = coords[-1]
    target_distance = np.sqrt((final_x - target_point[0])**2 + (final_y - target_point[1])**2)
    energy += 15 * target_distance  # 增加目标距离的权重
    
    return energy

# 6. 严格保证路径到达目标点的MCMC采样
def generate_paths_mcmc(env_zones, n_samples=1000, initial_temp=1.0, final_temp=0.1):
    """Metropolis-Hastings采样生成合法路径，严格保证到达目标点"""
    def calculate_required_steps():
        """计算从起点到目标点所需的最少步数"""
        dx = abs(target_point[0] - 0)  # 起点x=0
        dy = abs(target_point[1] - 0)  # 起点y=0
        return dx + dy  # 曼哈顿距离
    
    required_steps = calculate_required_steps()
    min_length = required_steps + 2  # 包含起点和终点方向
    
    def is_path_reaching_target(path):
        """检查路径是否到达目标点"""
        coords = path_to_2d_coords(path)
        final_pos = coords[-1]
        return final_pos[0] == target_point[0] and final_pos[1] == target_point[1]
    
    def generate_target_oriented_path():
        """生成指向目标的初始路径"""
        path = [fixed_start]
        current_pos = np.array([0, 0])
        remaining_steps = n_nodes - 2  # 可自由分配的步数
        
        # 首先生成必须的移动步骤
        required_moves = []
        dx = target_point[0] - current_pos[0]
        dy = target_point[1] - current_pos[1]
        
        # 生成水平移动
        x_dir = 'R' if dx > 0 else 'L'
        for _ in range(abs(dx)):
            required_moves.append(x_dir)
        
        # 生成垂直移动
        y_dir = 'U' if dy > 0 else 'D'
        for _ in range(abs(dy)):
            required_moves.append(y_dir)
        
        # 随机打乱必须的移动步骤
        np.random.shuffle(required_moves)
        
        # 插入必须的移动步骤
        for move in required_moves[:min(len(required_moves), remaining_steps)]:
            path.append(move)
            remaining_steps -= 1
        
        # 填充剩余的随机步骤（确保不反向移动）
        for _ in range(remaining_steps):
            last_dir = path[-1] if len(path) > 0 else None
            available_dirs = directions.copy()
            
            # 避免急转弯
            if last_dir == 'U': available_dirs.remove('D')
            elif last_dir == 'D': available_dirs.remove('U')
            elif last_dir == 'L': available_dirs.remove('R')
            elif last_dir == 'R': available_dirs.remove('L')
            
            # 优先选择靠近目标的方向
            current_pos = path_to_2d_coords(path)[-1]
            target_dir = np.array(target_point) - np.array(current_pos)
            if np.linalg.norm(target_dir) > 0:
                target_dir = target_dir / np.linalg.norm(target_dir)
                dir_scores = []
                for d in available_dirs:
                    if d == 'U': move = np.array([0,1])
                    elif d == 'D': move = np.array([0,-1])
                    elif d == 'L': move = np.array([-1,0])
                    elif d == 'R': move = np.array([1,0])
                    score = np.dot(move, target_dir)
                    dir_scores.append(score)
                chosen_dir = available_dirs[np.argmax(dir_scores)]
            else:
                chosen_dir = np.random.choice(available_dirs)
            
            path.append(chosen_dir)
        
        path.append(fixed_end)
        return ''.join(path)
    
    # 初始化
    current_path = generate_target_oriented_path()
    current_energy = calculate_energy(current_path, env_zones)
    samples = []
    energies = []
    valid_samples = 0
    
    # 温度衰减系数
    temp_decay = (final_temp / initial_temp) ** (1.0 / (n_samples * 3))
    current_temp = initial_temp
    
    for i in range(n_samples * 10):  # 大幅增加尝试次数
        # 温度衰减
        current_temp *= temp_decay
        
        # 生成候选路径（保持到达目标点的特性）
        while True:
            if np.random.rand() < 0.7:
                # 单点变异（保持路径终点不变）
                new_path = list(current_path)
                idx = np.random.randint(1, n_nodes-1)
                
                # 保存终点位置
                original_end = path_to_2d_coords(current_path)[-1]
                
                # 尝试变异
                original_dir = new_path[idx]
                for dir in np.random.permutation(directions):  # 随机顺序尝试
                    if dir == original_dir:
                        continue
                    
                    new_path[idx] = dir
                    new_path_str = ''.join(new_path)
                    new_end = path_to_2d_coords(new_path_str)[-1]
                    
                    # 检查是否仍然到达目标点
                    if new_end[0] == target_point[0] and new_end[1] == target_point[1]:
                        new_path = new_path_str
                        break
                else:
                    # 没有有效的变异，保持原路径
                    new_path = current_path
                break
            else:
                # 交换两个位置（保持路径终点不变）
                new_path = list(current_path)
                idx1, idx2 = np.random.choice(range(1, n_nodes-1), size=2, replace=False)
                new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
                new_path_str = ''.join(new_path)
                
                # 检查是否仍然到达目标点
                new_end = path_to_2d_coords(new_path_str)[-1]
                if new_end[0] == target_point[0] and new_end[1] == target_point[1]:
                    new_path = new_path_str
                    break
                # 否则继续循环尝试
        
        new_energy = calculate_energy(new_path, env_zones)
        
        # Metropolis接受准则
        if new_energy < current_energy or np.random.rand() < np.exp((current_energy - new_energy) / current_temp):
            current_path, current_energy = new_path, new_energy
        
        # 确保路径到达目标点且满足能量要求
        if is_path_reaching_target(current_path) and current_energy < 400:
            samples.append(current_path)
            energies.append(current_energy)
            valid_samples += 1
            
            if valid_samples >= n_samples:
                break
    
    # 最终验证所有样本
    valid_samples = []
    valid_energies = []
    for path, energy in zip(samples, energies):
        if is_path_reaching_target(path):
            valid_samples.append(path)
            valid_energies.append(energy)
            if len(valid_samples) >= n_samples:
                break
    
    return valid_samples[:n_samples], np.array(valid_energies[:n_samples])

# 7. 路径距离计算 (保持不变)
def path_distance(path1, path2):
    """综合距离度量：交换距离 + 方向模式距离"""
    swap_dist = sum(c1 != c2 for c1, c2 in zip(path1, path2))
    
    changes1 = [path1[i] != path1[i+1] for i in range(len(path1)-1)]
    changes2 = [path2[i] != path2[i+1] for i in range(len(path2)-1)]
    pattern_dist = sum(c1 != c2 for c1, c2 in zip(changes1, changes2))
    
    hist1 = np.array([path1.count(d) for d in directions])
    hist2 = np.array([path2.count(d) for d in directions])
    hist_dist = np.linalg.norm(hist1 - hist2)
    
    return swap_dist + 0.4*pattern_dist + 0.2*hist_dist

# 8. 改进的路径可视化函数 (增加目标点标记)
def visualize_path(path, ax, color='blue', linewidth=2, label=None, alpha=1.0):
    """可视化管道路径"""
    coords = path_to_2d_coords(path)
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    
    line = ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, 
                   label=label, marker='o', markersize=3, alpha=alpha)[0]
    
    # 标记起点和终点
    ax.scatter(x_coords[0], y_coords[0], c='blue', s=100, marker='o', edgecolors='k')
    ax.scatter(x_coords[-1], y_coords[-1], c=color, s=100, marker='s', edgecolors='k')
    
    return line

# 9. 主程序 (增加目标点可视化)
if __name__ == "__main__":
    # 初始化环境
    env_zones = EnvironmentZones()
    
    # 生成路径样本
    print("生成路径样本...")
    paths, energies = generate_paths_mcmc(env_zones, n_samples=500, initial_temp=1.0, final_temp=0.1)
    
    # 处理路径生成失败的情况（确保至少有起始点和目标点）
    has_valid_paths = len(paths) > 0
    if not has_valid_paths:
        print("警告：没有生成有效路径样本！将仅显示起始点和目标点。")
        # 创建一个默认路径用于获取目标点坐标
        default_path = fixed_start + 'R'*(n_nodes-2) + fixed_end
        final_coords = path_to_2d_coords(default_path)[-1]
        start_coords = (0, 0)  # 起始点固定为(0,0)
    else:
        print(f"生成了 {len(paths)} 个有效路径样本")
    
    # 准备样本数据（处理无有效路径的情况）
    sample_paths = []
    sample_energies = np.array([])
    if has_valid_paths:
        sample_size = min(200, len(paths))
        sample_idx = np.random.choice(len(paths), size=sample_size, replace=False)
        sample_paths = [paths[i] for i in sample_idx]
        sample_energies = energies[sample_idx]
    
    # 创建双子图
    plt.figure(figsize=(22, 10))
    
    # 图1：能量景观（处理无数据的情况）
    ax1 = plt.subplot(1, 2, 1)
    
    if has_valid_paths:
        # 计算距离矩阵
        print("计算距离矩阵...")
        dist_matrix = np.zeros((len(sample_paths), len(sample_paths)))
        for i, j in combinations(range(len(sample_paths)), 2):
            dist = path_distance(sample_paths[i], sample_paths[j])
            dist_matrix[i,j] = dist_matrix[j,i] = dist
        
        # t-SNE降维
        print("执行降维...")
        tsne = TSNE(n_components=2, metric="precomputed", 
                   init="random", random_state=42,
                   perplexity=min(25, len(sample_paths)//5))
        coords = tsne.fit_transform(dist_matrix)
        
        # 插值生成连续表面
        print("生成连续景观...")
        xi = np.linspace(coords[:,0].min()-2, coords[:,0].max()+2, 150)
        yi = np.linspace(coords[:,1].min()-2, coords[:,1].max()+2, 150)
        xi, yi = np.meshgrid(xi, yi)
        
        zi = griddata((coords[:,0], coords[:,1]), sample_energies, 
                     (xi, yi), method='cubic')
        
        # 绘制能量景观
        contour = ax1.contourf(xi, yi, zi, levels=25, cmap='RdYlBu_r', alpha=0.8)
        sc = ax1.scatter(coords[:,0], coords[:,1], c=sample_energies,
                       cmap='RdYlBu_r', edgecolors='k', linewidths=0.5, s=50)
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('配置能量', fontsize=12)
        
        # 标记关键点
        min_idx = np.argmin(sample_energies)
        max_idx = np.argmax(sample_energies)
        
        ax1.scatter(coords[min_idx,0], coords[min_idx,1], 
                  c='lime', s=200, marker='*', edgecolors='k', linewidths=2, zorder=10)
        ax1.scatter(coords[max_idx,0], coords[max_idx,1], 
                  c='red', s=200, marker='X', edgecolors='k', linewidths=2, zorder=10)
        
        ax1.annotate(f"最优路径\n{sample_paths[min_idx]}\n能量={sample_energies[min_idx]:.1f}",
                   (coords[min_idx,0], coords[min_idx,1]),
                   textcoords="offset points", xytext=(0,20),
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8))
        
        ax1.scatter([], [], c='lime', s=100, marker='*', edgecolors='k', label='最优配置')
        ax1.scatter([], [], c='red', s=100, marker='X', edgecolors='k', label='最差配置')
        ax1.legend(loc='upper left', fontsize=10)
        
        ax1.set_title(f'管道配置能量景观\n(节点数={n_nodes}, 样本数={len(sample_paths)})',
                     fontsize=14, pad=20)
    else:
        # 无有效路径时，显示提示信息
        ax1.text(0.5, 0.5, '无有效路径样本，无法生成能量景观', 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, transform=ax1.transAxes)
        ax1.set_title('管道配置能量景观', fontsize=14, pad=20)
    
    ax1.set_xlabel('t-SNE 维度 1', fontsize=12)
    ax1.set_ylabel('t-SNE 维度 2', fontsize=12)
    ax1.grid(alpha=0.3)
    
    # 图2：2D管道可视化
    ax2 = plt.subplot(1, 2, 2)
    
    # 绘制环境元素
    # 危险区域（红色）
    for i, (x1, y1, x2, y2) in enumerate(env_zones.danger_zones):
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           color='red', alpha=0.3, 
                           label='危险区域' if i == 0 else '')
        ax2.add_patch(rect)
        # 安全边距
        safety_rect = plt.Rectangle((x1-env_zones.safety_margin, y1-env_zones.safety_margin), 
                                  (x2-x1)+2*env_zones.safety_margin, (y2-y1)+2*env_zones.safety_margin,
                                  color='red', alpha=0.1, linestyle='--')
        ax2.add_patch(safety_rect)
    
    # 障碍物（灰色）
    for i, (x1, y1, x2, y2) in enumerate(env_zones.obstacles):
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           color='gray', alpha=0.6,
                           label='障碍物' if i == 0 else '')
        ax2.add_patch(rect)
    
    # 亲和区域（绿色）
    for i, (x1, y1, x2, y2) in enumerate(env_zones.affinity_zones):
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           color='green', alpha=0.3,
                           label='亲和区域' if i == 0 else '')
        ax2.add_patch(rect)
    
    # 标记目标点
    ax2.scatter(target_point[0], target_point[1], c='gold', s=200, marker='*', 
               edgecolors='k', linewidth=2, label='目标点', zorder=10)
    
    # 绘制路径（如果有）
    if has_valid_paths:
        # 绘制所有路径（细线，透明）
        for path in sample_paths:
            visualize_path(path, ax2, color='lightblue', linewidth=0.5, alpha=0.3)
        
        # 高亮最优和最差路径
        min_idx = np.argmin(sample_energies)
        max_idx = np.argmax(sample_energies)
        visualize_path(sample_paths[min_idx], ax2, color='green', linewidth=4, 
                       label=f'最优路径 (能量={sample_energies[min_idx]:.1f})')
        visualize_path(sample_paths[max_idx], ax2, color='red', linewidth=3, 
                       label=f'最差路径 (能量={sample_energies[max_idx]:.1f})')
    else:
        # 无路径时，绘制起始点
        ax2.scatter(start_coords[0], start_coords[1], c='blue', s=100, marker='o', 
                   label=f'起始点 {start_coords}', edgecolors='k')
    
    ax2.set_title('2D管道路径可视化\n（世界坐标系：上下左右）', fontsize=14)
    ax2.set_xlabel('X 坐标', fontsize=12)
    ax2.set_ylabel('Y 坐标', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axis('equal')
    
    # 设置合适的显示范围
    all_coords = []
    if has_valid_paths:
        for path in [sample_paths[np.argmin(sample_energies)], sample_paths[np.argmax(sample_energies)]]:
            all_coords.extend(path_to_2d_coords(path))
    else:
        all_coords = [start_coords, target_point]
    
    if all_coords:
        x_coords = [coord[0] for coord in all_coords]
        y_coords = [coord[1] for coord in all_coords]
        margin = 2
        ax2.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        ax2.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    
    plt.tight_layout()
    plt.savefig('pipeline_visualization_with_target.png', dpi=300, bbox_inches='tight')
    print("可视化完成，结果保存为 pipeline_visualization_with_target.png")
    
    # 输出统计信息（如果有有效路径）
    if has_valid_paths:
        print(f"\n=== 路径统计信息 ===")
        min_idx = np.argmin(sample_energies)
        max_idx = np.argmax(sample_energies)
        print(f"最优路径: {sample_paths[min_idx]} (能量: {sample_energies[min_idx]:.2f})")
        print(f"最差路径: {sample_paths[max_idx]} (能量: {sample_energies[max_idx]:.2f})")
        print(f"平均能量: {np.mean(sample_energies):.2f}")
        print(f"能量标准差: {np.std(sample_energies):.2f}")
        
        # 检查最优路径是否满足约束
        best_coords = path_to_2d_coords(sample_paths[min_idx])
        has_intersection = has_path_intersection(best_coords)
        in_danger = any(env_zones.point_in_danger_zone(x, y) for x, y in best_coords)
        in_obstacle = any(env_zones.point_in_obstacle(x, y) for x, y in best_coords)
        in_affinity = any(env_zones.point_in_affinity_zone(x, y) for x, y in best_coords)
        final_x, final_y = best_coords[-1]
        target_dist = np.sqrt((final_x - target_point[0])**2 + (final_y - target_point[1])**2)
        
        print(f"\n=== 最优路径约束检查 ===")
        print(f"是否自相交: {'是' if has_intersection else '否'}")
        print(f"是否经过危险区域: {'是' if in_danger else '否'}")
        print(f"是否经过障碍物: {'是' if in_obstacle else '否'}")
        print(f"是否经过亲和区域: {'是' if in_affinity else '否'}")
        print(f"终点到目标点距离: {target_dist:.2f}")
    else:
        print("\n=== 路径统计信息 ===")
        print("无有效路径可供分析")
        print(f"起始点坐标: (0, 0)")
        print(f"目标点坐标: {target_point}")
    
    plt.show()