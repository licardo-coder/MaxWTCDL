# 导入安装包
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg', 'Agg' 等
import matplotlib.pyplot as plt
import time

# 设置设备 - 如果有GPU则使用GPU,否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class MaxWTCDLGreedySolver:
    def __init__(self, n_targets=300, n_sensors=30, grid_size=1000, 
                 radius=50, distance_move_limit=250, distance_total_B=2000,
                 seed=42):
        """
        初始化MaxWTCDL贪心求解器
        
        参数:
            n_targets: 目标数量
            n_sensors: 传感器数量
            grid_size: 网格大小 (0到grid_size)
            radius: 传感器感知半径
            distance_move_limit: 单个传感器移动距离限制 d
            distance_total_B: 总移动距离限制 B
            seed: 随机种子
        """
        self.n_targets = n_targets
        self.n_sensors = n_sensors
        self.grid_size = grid_size
        self.radius = radius
        self.distance_move_limit = distance_move_limit
        self.distance_total_B = distance_total_B
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 初始化数据
        self._initialize_data()
        
    def _initialize_data(self):
        """初始化目标和传感器数据"""
        # 生成目标位置和权重
        target_x = np.random.randint(0, self.grid_size + 1, size=self.n_targets)
        target_y = np.random.randint(0, self.grid_size + 1, size=self.n_targets)
        target_w = np.random.randint(1, 10, size=self.n_targets)
        
        # 生成传感器初始位置
        sensor_x_init = np.random.randint(0, self.grid_size + 1, size=self.n_sensors)
        sensor_y_init = np.random.randint(0, self.grid_size + 1, size=self.n_sensors)
        
        # 转换为PyTorch张量并移到设备上
        self.target_pos = torch.tensor(
            np.column_stack((target_x, target_y)), 
            dtype=torch.float32, 
            device=device
        )  # shape: (n_targets, 2)
        
        self.target_w = torch.tensor(
            target_w, 
            dtype=torch.float32, 
            device=device
        )  # shape: (n_targets,)
        
        self.sensor_pos_init = torch.tensor(
            np.column_stack((sensor_x_init, sensor_y_init)), 
            dtype=torch.float32, 
            device=device
        )  # shape: (n_sensors, 2)
        
        # 当前传感器位置 (会被修改)
        self.sensor_pos = self.sensor_pos_init.clone()
        
        # 保存初始位置用于绘图
        self.sensor_pos_init_np = np.column_stack((sensor_x_init, sensor_y_init))
        self.target_pos_np = np.column_stack((target_x, target_y))
        self.target_w_np = target_w
        
    def compute_coverage_reward(self, sensor_positions):
        """
        计算给定传感器位置下的覆盖奖励
        使用PyTorch张量操作实现向量化计算
        
        参数:
            sensor_positions: 传感器位置张量 shape: (n_sensors, 2)
        
        返回:
            总奖励值
        """
        # 计算所有传感器与所有目标之间的距离
        # sensor_positions: (n_sensors, 2)
        # target_pos: (n_targets, 2)
        # 扩展维度进行广播: (n_sensors, 1, 2) - (1, n_targets, 2) = (n_sensors, n_targets, 2)
        distances = torch.cdist(sensor_positions, self.target_pos)  # shape: (n_sensors, n_targets)
        
        # 检查哪些目标被至少一个传感器覆盖
        # (n_sensors, n_targets) -> (n_targets,) 每个目标是否被至少一个传感器覆盖
        covered = (distances <= self.radius).any(dim=0)  # shape: (n_targets,)
        
        # 计算总奖励
        reward = (covered.float() * self.target_w).sum()
        
        return reward
    
    def compute_sensor_reach_location(self):
        """
        计算每个传感器可以到达的位置及其移动距离
        使用稀疏表示来节省内存
        
        返回:
            字典,键为(sensor_idx, x, y),值为移动距离
        """
        print("计算传感器可达位置...")
        sensor_reach = {}
        
        # 对每个传感器计算可达区域
        for i in range(self.n_sensors):
            sensor_init_pos = self.sensor_pos_init[i].cpu().numpy()
            
            # 计算可能的移动范围 (在网格边界内)
            x_min = max(0, int(sensor_init_pos[0] - self.distance_move_limit))
            x_max = min(self.grid_size, int(sensor_init_pos[0] + self.distance_move_limit))
            y_min = max(0, int(sensor_init_pos[1] - self.distance_move_limit))
            y_max = min(self.grid_size, int(sensor_init_pos[1] + self.distance_move_limit))
            
            # 遍历可能的位置
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    distance = np.sqrt((x - sensor_init_pos[0])**2 + (y - sensor_init_pos[1])**2)
                    if distance <= self.distance_move_limit:
                        sensor_reach[(i, x, y)] = distance
        
        print(f"传感器可达位置数量: {len(sensor_reach)}")
        return sensor_reach
    
    def compute_action_values_batch(self, sensor_reach):
        """
        批量计算所有可能动作的价值
        利用PyTorch的向量化操作加速
        
        参数:
            sensor_reach: 传感器可达位置字典
            
        返回:
            action_values: 字典,键为(sensor_idx, x, y),值为奖励增益
        """
        # 当前状态的奖励
        current_reward = self.compute_coverage_reward(self.sensor_pos)
        
        action_values = {}
        
        # 按传感器分组处理以减少内存使用
        for sensor_idx in range(self.n_sensors):
            # 获取该传感器的所有可达位置
            positions = [(x, y) for (s, x, y) in sensor_reach.keys() if s == sensor_idx]
            
            if len(positions) == 0:
                continue
            
            # 批量处理该传感器的所有可能位置
            # 创建批量的传感器位置张量
            batch_sensor_pos = self.sensor_pos.repeat(len(positions), 1, 1)  # (batch, n_sensors, 2)
            
            # 更新批量中每个样本的第sensor_idx个传感器位置
            new_positions = torch.tensor(positions, dtype=torch.float32, device=device)  # (batch, 2)
            batch_sensor_pos[:, sensor_idx, :] = new_positions
            
            # 批量计算每个配置的奖励
            batch_rewards = torch.zeros(len(positions), device=device)
            for i in range(len(positions)):
                batch_rewards[i] = self.compute_coverage_reward(batch_sensor_pos[i])
            
            # 计算奖励增益
            reward_gains = batch_rewards - current_reward
            reward_gains = torch.clamp(reward_gains, min=0)  # 只保留正增益
            
            # 存储结果
            for i, (x, y) in enumerate(positions):
                action_values[(sensor_idx, x, y)] = reward_gains[i].item()
        
        return action_values
    
    def compute_total_movement_cost(self):
        """计算当前配置的总移动距离"""
        distances = torch.norm(self.sensor_pos - self.sensor_pos_init, dim=1)
        return distances.sum().item()
    
    def select_best_action(self, action_values, sensor_reach):
        """
        选择最佳动作
        
        参数:
            action_values: 动作价值字典
            sensor_reach: 传感器可达位置字典
            
        返回:
            (sensor_idx, new_x, new_y, reward_gain) 或 None
        """
        best_action = None
        best_value = 0
        
        for (sensor_idx, x, y), value in action_values.items():
            if value <= 0:
                continue
            
            # 检查移动后是否超过总距离限制
            temp_sensor_pos = self.sensor_pos.clone()
            temp_sensor_pos[sensor_idx] = torch.tensor([x, y], dtype=torch.float32, device=device)
            
            # 计算新的总移动距离
            new_total_cost = torch.norm(temp_sensor_pos - self.sensor_pos_init, dim=1).sum().item()
            
            if new_total_cost <= self.distance_total_B and value > best_value:
                best_value = value
                best_action = (sensor_idx, x, y, value)
        
        return best_action
    
    def solve(self):
        """使用贪心算法求解MaxWTCDL问题"""
        print("=" * 50)
        print("开始求解MaxWTCDL问题 (PyTorch版本)")
        print("=" * 50)
        
        start_time = time.time()
        
        # 计算传感器可达位置
        sensor_reach = self.compute_sensor_reach_location()
        
        # 记录求解过程
        action_history = []
        reward_history = []
        cost_history = []
        
        iteration = 0
        initial_reward = self.compute_coverage_reward(self.sensor_pos).item()
        print(f"\n初始覆盖奖励: {initial_reward:.2f}")
        print(f"初始移动成本: 0.00")
        
        while True:
            iteration += 1
            print(f"\n迭代 {iteration}:")
            
            # 计算所有动作的价值
            action_values = self.compute_action_values_batch(sensor_reach)
            
            # 选择最佳动作
            best_action = self.select_best_action(action_values, sensor_reach)
            
            if best_action is None:
                print("  没有可行的改进动作,算法终止")
                break
            
            sensor_idx, new_x, new_y, reward_gain = best_action
            
            # 执行动作
            old_pos = self.sensor_pos[sensor_idx].cpu().numpy()
            self.sensor_pos[sensor_idx] = torch.tensor([new_x, new_y], dtype=torch.float32, device=device)
            
            # 更新记录
            current_reward = self.compute_coverage_reward(self.sensor_pos).item()
            current_cost = self.compute_total_movement_cost()
            
            action_history.append((sensor_idx, new_x, new_y))
            reward_history.append(current_reward)
            cost_history.append(current_cost)
            
            print(f"  传感器 {sensor_idx}: ({old_pos[0]:.0f}, {old_pos[1]:.0f}) -> ({new_x}, {new_y})")
            print(f"  奖励增益: {reward_gain:.2f}")
            print(f"  当前总奖励: {current_reward:.2f}")
            print(f"  当前移动成本: {current_cost:.2f} / {self.distance_total_B}")
        
        end_time = time.time()
        final_reward = self.compute_coverage_reward(self.sensor_pos).item()
        final_cost = self.compute_total_movement_cost()
        
        print("\n" + "=" * 50)
        print("求解完成!")
        print("=" * 50)
        print(f"总迭代次数: {iteration}")
        print(f"初始奖励: {initial_reward:.2f}")
        print(f"最终奖励: {final_reward:.2f}")
        print(f"奖励提升: {final_reward - initial_reward:.2f} ({(final_reward - initial_reward) / initial_reward * 100:.2f}%)")
        print(f"总移动距离: {final_cost:.2f} / {self.distance_total_B}")
        print(f"求解时间: {end_time - start_time:.2f} 秒")
        print("=" * 50)
        
        return {
            'action_history': action_history,
            'reward_history': reward_history,
            'cost_history': cost_history,
            'final_reward': final_reward,
            'final_cost': final_cost,
            'solve_time': end_time - start_time
        }
    
    def plot_initial_state(self, save_path=None):
        """绘制初始状态"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制目标
        ax.scatter(self.target_pos_np[:, 0], self.target_pos_np[:, 1], 
                  color='blue', label='Targets', alpha=0.6, 
                  s=self.target_w_np * 3)
        
        # 绘制传感器
        ax.scatter(self.sensor_pos_init_np[:, 0], self.sensor_pos_init_np[:, 1], 
                  color='red', label='Sensors', marker='*', alpha=1, s=200)
        
        # 绘制传感器覆盖范围
        for i in range(self.n_sensors):
            circle = plt.Circle(self.sensor_pos_init_np[i], self.radius, 
                              edgecolor='red', fill=False, linestyle='dashed', alpha=0.5)
            ax.add_artist(circle)
        
        ax.legend(loc='upper left')
        ax.set_xlim(-100, self.grid_size + 100)
        ax.set_ylim(-100, self.grid_size + 100)
        ax.set_aspect('equal', 'box')
        ax.set_title('Initial State')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"初始状态图已保存到: {save_path}")
        
        plt.show()
    
    def plot_final_state(self, save_path=None):
        """绘制最终状态"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制目标
        ax.scatter(self.target_pos_np[:, 0], self.target_pos_np[:, 1], 
                  color='blue', label='Targets', alpha=0.6, 
                  s=self.target_w_np * 3)
        
        # 获取最终传感器位置
        final_sensor_pos = self.sensor_pos.cpu().numpy()
        
        # 绘制最终传感器位置
        ax.scatter(final_sensor_pos[:, 0], final_sensor_pos[:, 1], 
                  color='red', label='Sensors (Final)', marker='*', alpha=1, s=200)
        
        # 绘制传感器覆盖范围
        for i in range(self.n_sensors):
            circle = plt.Circle(final_sensor_pos[i], self.radius, 
                              edgecolor='red', fill=False, linestyle='dashed', alpha=0.5)
            ax.add_artist(circle)
        
        # 绘制传感器移动限制范围
        for i in range(self.n_sensors):
            circle = plt.Circle(self.sensor_pos_init_np[i], self.distance_move_limit, 
                              edgecolor='green', fill=False, linestyle='dashed', alpha=0.3)
            ax.add_artist(circle)
        
        # 绘制移动路径
        for i in range(self.n_sensors):
            ax.plot([self.sensor_pos_init_np[i, 0], final_sensor_pos[i, 0]],
                   [self.sensor_pos_init_np[i, 1], final_sensor_pos[i, 1]],
                   'g--', alpha=0.5, linewidth=1)
        
        ax.legend(loc='upper left')
        ax.set_xlim(-100, self.grid_size + 100)
        ax.set_ylim(-100, self.grid_size + 100)
        ax.set_aspect('equal', 'box')
        ax.set_title('Final State')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"最终状态图已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    # 创建求解器
    solver = MaxWTCDLGreedySolver(
        n_targets=300,
        n_sensors=30,
        grid_size=1000,
        radius=50,
        distance_move_limit=250,
        distance_total_B=2000,
        seed=42
    )
    
    # 绘制初始状态
    solver.plot_initial_state(save_path='result/initial_state_pytorch.png')
    
    # 求解
    result = solver.solve()
    
    # 绘制最终状态
    solver.plot_final_state(save_path='result/final_state_pytorch.png')
    
    return solver, result


if __name__ == "__main__":
    solver, result = main()
