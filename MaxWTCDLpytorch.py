import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class SensorCoverageOptimizer(nn.Module):
    def __init__(self, n_targets=300, n_sensors=30, grid_size=1001, radius=50,
                 move_limit=250, total_budget=2000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(SensorCoverageOptimizer, self).__init__()

        self.n_targets = n_targets
        self.n_sensors = n_sensors
        self.grid_size = grid_size
        self.radius = radius
        self.move_limit = move_limit
        self.total_budget = total_budget
        self.device = device

        # 初始化数据
        self._initialize_data()

        # 计算传感器可达位置
        self.sensor_reach_mask = self._compute_reach_mask()

    def _initialize_data(self):
        """初始化目标、传感器和位置数据"""
        # 目标位置和权重
        self.target_x = torch.randint(0, self.grid_size, (self.n_targets,), device=self.device)
        self.target_y = torch.randint(0, self.grid_size, (self.n_targets,), device=self.device)
        self.target_w = torch.randint(1, 10, (self.n_targets,), device=self.device)

        # 传感器初始位置
        self.sensor_x = torch.randint(0, self.grid_size, (self.n_sensors,), device=self.device)
        self.sensor_y = torch.randint(0, self.grid_size, (self.n_sensors,), device=self.device)

        # 保存原始传感器位置用于可视化
        self.original_sensor_x = self.sensor_x.clone()
        self.original_sensor_y = self.sensor_y.clone()

        # 位置网格
        self.location_x = torch.arange(0, self.grid_size, device=self.device)
        self.location_y = torch.arange(0, self.grid_size, device=self.device)

        # 创建位置网格张量用于批量计算
        self.loc_grid_x, self.loc_grid_y = torch.meshgrid(
            self.location_x, self.location_y, indexing='ij'
        )

    def _compute_reach_mask(self):
        """计算每个传感器可以到达的位置掩码"""
        # 扩展维度用于广播计算
        sensor_x_exp = self.sensor_x.view(-1, 1, 1)
        sensor_y_exp = self.sensor_y.view(-1, 1, 1)

        # 计算所有传感器到所有位置的距离
        distances = torch.sqrt(
            (sensor_x_exp - self.loc_grid_x) ** 2 +
            (sensor_y_exp - self.loc_grid_y) ** 2
        )

        # 创建可达位置掩码
        reach_mask = distances <= self.move_limit

        return reach_mask

    def compute_coverage(self, sensor_positions=None):
        """计算当前传感器位置下的覆盖奖励"""
        if sensor_positions is None:
            sensor_x = self.sensor_x
            sensor_y = self.sensor_y
        else:
            sensor_x, sensor_y = sensor_positions

        # 扩展维度用于广播计算
        sensor_x_exp = sensor_x.view(-1, 1)  # (n_sensors, 1)
        sensor_y_exp = sensor_y.view(-1, 1)  # (n_sensors, 1)
        target_x_exp = self.target_x.view(1, -1)  # (1, n_targets)
        target_y_exp = self.target_y.view(1, -1)  # (1, n_targets)

        # 计算所有传感器到所有目标的距离
        distances = torch.sqrt(
            (sensor_x_exp - target_x_exp) ** 2 +
            (sensor_y_exp - target_y_exp) ** 2
        )

        # 计算覆盖矩阵
        coverage_matrix = distances <= self.radius  # (n_sensors, n_targets)

        # 计算每个目标是否被至少一个传感器覆盖
        target_covered = torch.any(coverage_matrix, dim=0)  # (n_targets,)

        # 计算总奖励
        total_reward = torch.sum(self.target_w * target_covered.float())

        return total_reward, target_covered

    def compute_move_cost(self, sensor_idx, new_x, new_y):
        """计算移动成本"""
        current_x = self.sensor_x[sensor_idx]
        current_y = self.sensor_y[sensor_idx]

        move_cost = torch.sqrt((current_x - new_x) ** 2 + (current_y - new_y) ** 2)
        return move_cost

    def compute_action_probabilities(self):
        """计算所有可能动作的概率（奖励提升）"""
        current_reward, _ = self.compute_coverage()
        action_probs = torch.zeros(
            (self.n_sensors, self.grid_size, self.grid_size),
            device=self.device
        )

        # 遍历所有传感器和位置
        for i in range(self.n_sensors):
            # 获取当前传感器的可达位置掩码
            reachable_mask = self.sensor_reach_mask[i]

            # 获取可达位置的坐标
            reachable_indices = torch.nonzero(reachable_mask, as_tuple=True)

            if len(reachable_indices[0]) == 0:
                continue

            # 为每个可达位置计算新的覆盖奖励
            for idx in range(len(reachable_indices[0])):
                j, z = reachable_indices[0][idx], reachable_indices[1][idx]

                # 创建临时传感器位置
                temp_sensor_x = self.sensor_x.clone()
                temp_sensor_y = self.sensor_y.clone()
                temp_sensor_x[i] = j
                temp_sensor_y[i] = z

                # 计算新位置的覆盖奖励
                new_reward, _ = self.compute_coverage((temp_sensor_x, temp_sensor_y))

                # 计算奖励提升
                reward_improvement = max(0.0, new_reward - current_reward)

                action_probs[i, j, z] = reward_improvement

        return action_probs

    def select_best_action(self, action_probs):
        """选择最佳动作"""
        best_value = 0
        best_sensor = -1
        best_x = -1
        best_y = -1

        current_total_cost = self.compute_total_move_cost()

        for i in range(self.n_sensors):
            for j in range(self.grid_size):
                for z in range(self.grid_size):
                    if action_probs[i, j, z] > 0:
                        # 计算移动成本
                        move_cost = self.compute_move_cost(i, j, z)

                        # 检查预算约束
                        if current_total_cost + move_cost <= self.total_budget:
                            if action_probs[i, j, z] > best_value:
                                best_value = action_probs[i, j, z]
                                best_sensor = i
                                best_x = j
                                best_y = z

        return best_value, best_sensor, best_x, best_y

    def compute_total_move_cost(self):
        """计算当前总移动成本"""
        total_cost = 0
        for i in range(self.n_sensors):
            original_x = self.original_sensor_x[i]
            original_y = self.original_sensor_y[i]
            current_x = self.sensor_x[i]
            current_y = self.sensor_y[i]

            total_cost += torch.sqrt(
                (original_x - current_x) ** 2 +
                (original_y - current_y) ** 2
            )

        return total_cost

    def optimize(self, max_iterations=100):
        """执行优化过程"""
        action_history = []
        cost_history = []
        reward_history = []

        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}/{max_iterations}")

            # 计算动作概率
            action_probs = self.compute_action_probabilities()

            # 选择最佳动作
            best_value, best_sensor, best_x, best_y = self.select_best_action(action_probs)

            if best_value == 0:
                print("No improving action found. Stopping optimization.")
                break

            # 执行动作
            self.sensor_x[best_sensor] = best_x
            self.sensor_y[best_sensor] = best_y

            # 更新历史记录
            action_history.append((best_sensor, best_x, best_y))
            current_cost = self.compute_total_move_cost()
            cost_history.append(current_cost.cpu().item())
            reward_history.append(best_value.cpu().item())

            print(f"Action: Sensor {best_sensor} moved to ({best_x}, {best_y})")
            print(f"Reward improvement: {best_value:.2f}")
            print(f"Total cost: {current_cost:.2f}/{self.total_budget}")

        final_reward, _ = self.compute_coverage()
        print(f"Final reward: {final_reward:.2f}")

        return action_history, cost_history, reward_history, final_reward

    def visualize_initial_state(self, save_path="original.png"):
        """可视化初始状态"""
        self._visualize_state(
            self.original_sensor_x.cpu().numpy(),
            self.original_sensor_y.cpu().numpy(),
            "Initial Sensor and Target Distribution",
            save_path
        )

    def visualize_final_state(self, save_path="optimized.png"):
        """可视化最终状态"""
        self._visualize_state(
            self.sensor_x.cpu().numpy(),
            self.sensor_y.cpu().numpy(),
            "Optimized Sensor Distribution",
            save_path,
            show_original_range=True
        )

    def _visualize_state(self, sensor_x, sensor_y, title, save_path, show_original_range=False):
        """可视化状态辅助函数"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制目标
        target_x_np = self.target_x.cpu().numpy()
        target_y_np = self.target_y.cpu().numpy()
        target_w_np = self.target_w.cpu().numpy()

        ax.scatter(target_x_np, target_y_np, color='blue', label='Targets',
                   alpha=0.6, s=target_w_np * 3)

        # 绘制传感器
        ax.scatter(sensor_x, sensor_y, color='red', label='Sensors',
                   marker='*', s=100, alpha=1)

        # 绘制传感器覆盖范围
        centers = np.column_stack((sensor_x, sensor_y))
        for center in centers:
            circle = plt.Circle(center, self.radius, edgecolor='red',
                                fill=False, linestyle='dashed', alpha=0.7)
            ax.add_artist(circle)

        # 如果需要显示原始移动范围
        if show_original_range:
            original_centers = np.column_stack(
                (self.original_sensor_x.cpu().numpy(),
                 self.original_sensor_y.cpu().numpy())
            )
            for center in original_centers:
                circle = plt.Circle(center, self.move_limit, edgecolor='green',
                                    fill=False, linestyle='dashed', alpha=0.3)
                ax.add_artist(circle)

        ax.legend(loc='upper left')
        ax.set_xlim(-100, self.grid_size + 100)
        ax.set_ylim(-100, self.grid_size + 100)
        ax.set_aspect('equal', 'box')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建优化器
    optimizer = SensorCoverageOptimizer(
        n_targets=300,
        n_sensors=30,
        grid_size=1001,
        radius=50,
        move_limit=250,
        total_budget=2000
    )

    print(f"Using device: {optimizer.device}")
    print(f"Number of targets: {optimizer.n_targets}")
    print(f"Number of sensors: {optimizer.n_sensors}")

    # 可视化初始状态
    print("Visualizing initial state...")
    optimizer.visualize_initial_state()

    # 计算初始覆盖奖励
    initial_reward, _ = optimizer.compute_coverage()
    print(f"Initial coverage reward: {initial_reward:.2f}")

    # 执行优化
    print("Starting optimization...")
    action_history, cost_history, reward_history, final_reward = optimizer.optimize(max_iterations=50)

    # 可视化最终状态
    print("Visualizing optimized state...")
    optimizer.visualize_final_state()

    # 打印优化结果摘要
    print("\n" + "=" * 50)
    print("OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"Initial reward: {initial_reward:.2f}")
    print(f"Final reward: {final_reward:.2f}")
    print(f"Improvement: {final_reward - initial_reward:.2f}")
    print(f"Number of moves: {len(action_history)}")
    print(f"Total cost: {cost_history[-1] if cost_history else 0:.2f}/{optimizer.total_budget}")

    # 绘制优化过程
    if reward_history:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(reward_history, 'b-', marker='o')
        plt.title('Reward Improvement per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Reward Improvement')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(cost_history, 'r-', marker='s')
        plt.title('Total Cost per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Total Movement Cost')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("optimization_progress.png", dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    main()