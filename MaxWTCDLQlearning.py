# 导入安装包
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import pickle
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class MaxWTCDLQLearning:
    """
    使用Q-learning强化学习求解MaxWTCDL问题
    集成贪心算法作为启发式指导
    """
    
    def __init__(self, n_targets=300, n_sensors=30, grid_size=1000,
                 radius=50, distance_move_limit=250, distance_total_B=2000,
                 learning_rate=0.1, gamma=0.95, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, seed=42):
        """
        初始化Q-learning求解器
        
        参数:
            n_targets: 目标数量
            n_sensors: 传感器数量
            grid_size: 网格大小
            radius: 传感器感知半径
            distance_move_limit: 单个传感器移动距离限制
            distance_total_B: 总移动距离限制
            learning_rate: Q-learning学习率 (alpha)
            gamma: 折扣因子
            epsilon_start: epsilon-greedy初始探索率
            epsilon_end: epsilon最小值
            epsilon_decay: epsilon衰减率
            seed: 随机种子
        """
        self.n_targets = n_targets
        self.n_sensors = n_sensors
        self.grid_size = grid_size
        self.radius = radius
        self.distance_move_limit = distance_move_limit
        self.distance_total_B = distance_total_B
        
        # Q-learning参数
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q表: Q[state][action] = Q值
        self.Q_table = defaultdict(lambda: defaultdict(float))
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 初始化环境
        self._initialize_environment()
        
    def _initialize_environment(self):
        """初始化环境数据"""
        # 生成目标位置和权重
        target_x = np.random.randint(0, self.grid_size + 1, size=self.n_targets)
        target_y = np.random.randint(0, self.grid_size + 1, size=self.n_targets)
        target_w = np.random.randint(1, 10, size=self.n_targets)
        
        # 生成传感器初始位置
        sensor_x_init = np.random.randint(0, self.grid_size + 1, size=self.n_sensors)
        sensor_y_init = np.random.randint(0, self.grid_size + 1, size=self.n_sensors)
        
        # 转换为PyTorch张量
        self.target_pos = torch.tensor(
            np.column_stack((target_x, target_y)),
            dtype=torch.float32,
            device=device
        )
        
        self.target_w = torch.tensor(
            target_w,
            dtype=torch.float32,
            device=device
        )
        
        self.sensor_pos_init = torch.tensor(
            np.column_stack((sensor_x_init, sensor_y_init)),
            dtype=torch.float32,
            device=device
        )
        
        # 保存numpy版本用于绘图
        self.sensor_pos_init_np = np.column_stack((sensor_x_init, sensor_y_init))
        self.target_pos_np = np.column_stack((target_x, target_y))
        self.target_w_np = target_w
        
        # 预计算传感器可达位置
        self._compute_feasible_actions()
        
    def _compute_feasible_actions(self):
        """预计算每个传感器的可行动作(可达位置)"""
        print("预计算可行动作空间...")
        self.feasible_actions = {}
        
        for sensor_idx in range(self.n_sensors):
            init_pos = self.sensor_pos_init_np[sensor_idx]
            actions = []
            
            # 计算可能的移动范围
            x_min = max(0, int(init_pos[0] - self.distance_move_limit))
            x_max = min(self.grid_size, int(init_pos[0] + self.distance_move_limit))
            y_min = max(0, int(init_pos[1] - self.distance_move_limit))
            y_max = min(self.grid_size, int(init_pos[1] + self.distance_move_limit))
            
            # 使用网格采样减少动作空间 (每10个像素采样一次)
            step = 10
            for x in range(x_min, x_max + 1, step):
                for y in range(y_min, y_max + 1, step):
                    distance = np.sqrt((x - init_pos[0])**2 + (y - init_pos[1])**2)
                    if distance <= self.distance_move_limit:
                        actions.append((x, y, distance))
            
            self.feasible_actions[sensor_idx] = actions
            
        total_actions = sum(len(acts) for acts in self.feasible_actions.values())
        print(f"总可行动作数: {total_actions}")
        
    def _state_to_key(self, sensor_positions):
        """将传感器位置状态转换为可哈希的键"""
        # 四舍五入到最近的10以减少状态空间
        rounded = (sensor_positions.cpu().numpy() / 10).astype(int) * 10
        return tuple(map(tuple, rounded))
    
    def _action_to_key(self, sensor_idx, x, y):
        """将动作转换为可哈希的键"""
        return (sensor_idx, int(x), int(y))
    
    def compute_reward(self, sensor_positions):
        """
        计算当前状态的奖励
        奖励 = 覆盖目标的总权重
        """
        distances = torch.cdist(sensor_positions, self.target_pos)
        covered = (distances <= self.radius).any(dim=0)
        reward = (covered.float() * self.target_w).sum()
        return reward.item()
    
    def compute_total_movement(self, sensor_positions):
        """计算总移动距离"""
        distances = torch.norm(sensor_positions - self.sensor_pos_init, dim=1)
        return distances.sum().item()
    
    def get_valid_actions(self, current_positions, used_budget):
        """
        获取当前状态下的所有有效动作
        考虑总移动距离约束
        """
        valid_actions = []
        current_pos_np = current_positions.cpu().numpy()
        
        for sensor_idx in range(self.n_sensors):
            for x, y, move_dist in self.feasible_actions[sensor_idx]:
                # 创建临时位置
                temp_pos = current_positions.clone()
                temp_pos[sensor_idx] = torch.tensor([x, y], dtype=torch.float32, device=device)
                
                # 计算新的总移动距离
                new_total_cost = self.compute_total_movement(temp_pos)
                
                # 检查是否满足预算约束
                if new_total_cost <= self.distance_total_B:
                    valid_actions.append((sensor_idx, x, y))
        
        return valid_actions
    
    def select_action_epsilon_greedy(self, state_key, valid_actions):
        """
        使用epsilon-greedy策略选择动作
        """
        if len(valid_actions) == 0:
            return None
        
        # Exploration: 随机选择动作
        if np.random.random() < self.epsilon:
            action = valid_actions[np.random.randint(len(valid_actions))]
            return self._action_to_key(*action)
        
        # Exploitation: 选择Q值最大的动作
        q_values = [self.Q_table[state_key][self._action_to_key(*a)] 
                   for a in valid_actions]
        max_q = max(q_values)
        
        # 如果有多个最大Q值,随机选择一个
        best_actions = [valid_actions[i] for i, q in enumerate(q_values) if q == max_q]
        action = best_actions[np.random.randint(len(best_actions))]
        
        return self._action_to_key(*action)
    
    def greedy_policy(self, current_positions, used_budget):
        """
        贪心策略: 选择能带来最大即时奖励增益的动作
        用于Q-learning的启发式指导
        """
        current_reward = self.compute_reward(current_positions)
        valid_actions = self.get_valid_actions(current_positions, used_budget)
        
        if len(valid_actions) == 0:
            return None
        
        best_action = None
        best_gain = -float('inf')
        
        for sensor_idx, x, y in valid_actions:
            # 模拟执行动作
            temp_pos = current_positions.clone()
            temp_pos[sensor_idx] = torch.tensor([x, y], dtype=torch.float32, device=device)
            
            new_reward = self.compute_reward(temp_pos)
            reward_gain = new_reward - current_reward
            
            if reward_gain > best_gain:
                best_gain = reward_gain
                best_action = (sensor_idx, x, y)
        
        return self._action_to_key(*best_action) if best_action else None
    
    def execute_action(self, current_positions, action_key):
        """
        执行动作并返回新状态
        """
        sensor_idx, x, y = action_key
        new_positions = current_positions.clone()
        new_positions[sensor_idx] = torch.tensor([x, y], dtype=torch.float32, device=device)
        return new_positions
    
    def train_episode(self, use_greedy_init=False):
        """
        训练一个episode
        
        参数:
            use_greedy_init: 是否使用贪心策略初始化轨迹
        """
        # 重置环境
        current_positions = self.sensor_pos_init.clone()
        state_key = self._state_to_key(current_positions)
        used_budget = 0
        
        episode_reward = 0
        episode_actions = []
        steps = 0
        
        while True:
            steps += 1
            
            # 获取有效动作
            valid_actions = self.get_valid_actions(current_positions, used_budget)
            
            if len(valid_actions) == 0:
                break
            
            # 选择动作
            if use_greedy_init and steps == 1:
                # 第一步使用贪心策略
                action_key = self.greedy_policy(current_positions, used_budget)
            else:
                # 使用epsilon-greedy策略
                action_key = self.select_action_epsilon_greedy(state_key, valid_actions)
            
            if action_key is None:
                break
            
            # 执行动作
            next_positions = self.execute_action(current_positions, action_key)
            next_state_key = self._state_to_key(next_positions)
            
            # 计算奖励(奖励是增量)
            current_reward = self.compute_reward(current_positions)
            next_reward = self.compute_reward(next_positions)
            reward = next_reward - current_reward
            
            # 更新已使用的预算
            used_budget = self.compute_total_movement(next_positions)
            
            # 获取下一状态的有效动作
            next_valid_actions = self.get_valid_actions(next_positions, used_budget)
            
            # Q-learning更新
            current_q = self.Q_table[state_key][action_key]
            
            if len(next_valid_actions) == 0:
                # 终止状态
                max_next_q = 0
            else:
                # 计算max Q(s', a')
                max_next_q = max([self.Q_table[next_state_key][self._action_to_key(*a)] 
                                 for a in next_valid_actions])
            
            # Q-learning更新公式: Q(s,a) = Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.Q_table[state_key][action_key] = new_q
            
            # 更新状态
            current_positions = next_positions
            state_key = next_state_key
            episode_reward += reward
            episode_actions.append(action_key)
            
            # 如果没有改进,提前终止
            if reward <= 0 and steps > 1:
                break
        
        return episode_reward, episode_actions, steps
    
    def train(self, num_episodes=1000, use_greedy_init=True, verbose=True):
        """
        训练Q-learning模型
        
        参数:
            num_episodes: 训练episode数量
            use_greedy_init: 是否使用贪心策略辅助初始化
            verbose: 是否打印详细信息
        """
        print("=" * 60)
        print("开始Q-learning训练")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(f"学习率 (α): {self.alpha}")
        print(f"折扣因子 (γ): {self.gamma}")
        print(f"初始探索率 (ε): {self.epsilon}")
        print(f"使用贪心初始化: {use_greedy_init}")
        print("=" * 60)
        
        start_time = time.time()
        
        episode_rewards = []
        episode_steps = []
        best_reward = -float('inf')
        best_actions = None
        
        for episode in range(num_episodes):
            # 训练一个episode
            episode_reward, actions, steps = self.train_episode(use_greedy_init)
            
            episode_rewards.append(episode_reward)
            episode_steps.append(steps)
            
            # 更新最佳结果
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_actions = actions
            
            # 衰减epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # 打印进度
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_steps = np.mean(episode_steps[-100:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"平均奖励: {avg_reward:.2f} | "
                      f"平均步数: {avg_steps:.1f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"最佳奖励: {best_reward:.2f}")
        
        end_time = time.time()
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"训练时间: {end_time - start_time:.2f} 秒")
        print(f"最佳episode奖励: {best_reward:.2f}")
        print(f"Q表大小: {len(self.Q_table)} 个状态")
        print("=" * 60)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'best_reward': best_reward,
            'best_actions': best_actions,
            'training_time': end_time - start_time
        }
    
    def evaluate(self, use_learned_policy=True):
        """
        评估策略性能
        
        参数:
            use_learned_policy: True使用学习的策略, False使用纯贪心策略
        """
        print("\n" + "=" * 60)
        print("评估策略性能")
        print("=" * 60)
        
        current_positions = self.sensor_pos_init.clone()
        state_key = self._state_to_key(current_positions)
        used_budget = 0
        
        total_reward = self.compute_reward(current_positions)
        initial_reward = total_reward
        actions_taken = []
        
        print(f"初始奖励: {initial_reward:.2f}")
        
        step = 0
        while True:
            step += 1
            valid_actions = self.get_valid_actions(current_positions, used_budget)
            
            if len(valid_actions) == 0:
                print(f"第 {step} 步: 没有有效动作,终止")
                break
            
            # 选择动作
            if use_learned_policy:
                # 使用学习的Q表 (贪心选择,不探索)
                q_values = [self.Q_table[state_key][self._action_to_key(*a)] 
                           for a in valid_actions]
                max_q = max(q_values)
                best_idx = q_values.index(max_q)
                action = valid_actions[best_idx]
                action_key = self._action_to_key(*action)
            else:
                # 使用贪心策略
                action_key = self.greedy_policy(current_positions, used_budget)
                if action_key is None:
                    break
            
            # 执行动作
            sensor_idx, x, y = action_key
            old_pos = current_positions[sensor_idx].cpu().numpy()
            next_positions = self.execute_action(current_positions, action_key)
            
            # 计算新奖励
            new_reward = self.compute_reward(next_positions)
            reward_gain = new_reward - total_reward
            
            used_budget = self.compute_total_movement(next_positions)
            
            print(f"第 {step} 步: 传感器 {sensor_idx} | "
                  f"({old_pos[0]:.0f}, {old_pos[1]:.0f}) -> ({x}, {y}) | "
                  f"奖励增益: {reward_gain:.2f} | "
                  f"总奖励: {new_reward:.2f} | "
                  f"已用预算: {used_budget:.2f}/{self.distance_total_B}")
            
            # 更新状态
            current_positions = next_positions
            state_key = self._state_to_key(current_positions)
            total_reward = new_reward
            actions_taken.append((sensor_idx, x, y))
            
            # 如果没有改进,终止
            if reward_gain <= 0:
                print(f"第 {step} 步: 无改进,终止")
                break
        
        final_reward = self.compute_reward(current_positions)
        final_cost = self.compute_total_movement(current_positions)
        
        print("\n" + "=" * 60)
        print("评估结果:")
        print("=" * 60)
        print(f"初始奖励: {initial_reward:.2f}")
        print(f"最终奖励: {final_reward:.2f}")
        print(f"奖励提升: {final_reward - initial_reward:.2f}")
        print(f"提升比例: {(final_reward - initial_reward) / initial_reward * 100:.2f}%")
        print(f"总移动距离: {final_cost:.2f} / {self.distance_total_B}")
        print(f"总步数: {step}")
        print("=" * 60)
        
        # 保存最终位置用于绘图
        self.final_positions = current_positions.cpu().numpy()
        
        return {
            'initial_reward': initial_reward,
            'final_reward': final_reward,
            'improvement': final_reward - initial_reward,
            'final_cost': final_cost,
            'actions': actions_taken
        }
    
    def save_model(self, filepath):
        """保存Q表"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.Q_table), f)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载Q表"""
        with open(filepath, 'rb') as f:
            loaded_q = pickle.load(f)
            self.Q_table = defaultdict(lambda: defaultdict(float), loaded_q)
        print(f"模型已从 {filepath} 加载")
    
    def plot_training_progress(self, episode_rewards, save_path=None):
        """绘制训练进度"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制episode奖励
        ax1.plot(episode_rewards, alpha=0.6, label='Episode Reward')
        # 绘制移动平均
        window = 50
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                    linewidth=2, label=f'{window}-Episode Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Progress: Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制奖励分布
        ax2.hist(episode_rewards, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
        ax2.set_xlabel('Episode Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练进度图已保存到: {save_path}")
        
        plt.show()
    
    def plot_result(self, save_path=None):
        """绘制最终结果"""
        if not hasattr(self, 'final_positions'):
            print("请先运行evaluate()方法")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 绘制目标
        ax.scatter(self.target_pos_np[:, 0], self.target_pos_np[:, 1],
                  color='blue', label='Targets', alpha=0.6,
                  s=self.target_w_np * 3)
        
        # 绘制初始传感器位置
        ax.scatter(self.sensor_pos_init_np[:, 0], self.sensor_pos_init_np[:, 1],
                  color='orange', label='Initial Sensors', marker='s', 
                  alpha=0.5, s=100)
        
        # 绘制最终传感器位置
        ax.scatter(self.final_positions[:, 0], self.final_positions[:, 1],
                  color='red', label='Final Sensors', marker='*',
                  alpha=1, s=200)
        
        # 绘制最终覆盖范围
        for i in range(self.n_sensors):
            circle = plt.Circle(self.final_positions[i], self.radius,
                              edgecolor='red', fill=False, linestyle='dashed', alpha=0.5)
            ax.add_artist(circle)
        
        # 绘制移动限制范围
        for i in range(self.n_sensors):
            circle = plt.Circle(self.sensor_pos_init_np[i], self.distance_move_limit,
                              edgecolor='green', fill=False, linestyle='dashed', alpha=0.2)
            ax.add_artist(circle)
        
        # 绘制移动路径
        for i in range(self.n_sensors):
            ax.plot([self.sensor_pos_init_np[i, 0], self.final_positions[i, 0]],
                   [self.sensor_pos_init_np[i, 1], self.final_positions[i, 1]],
                   'g--', alpha=0.5, linewidth=1.5)
            ax.annotate('', xy=self.final_positions[i], 
                       xytext=self.sensor_pos_init_np[i],
                       arrowprops=dict(arrowstyle='->', color='green', 
                                     alpha=0.6, lw=1.5))
        
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(-100, self.grid_size + 100)
        ax.set_ylim(-100, self.grid_size + 100)
        ax.set_aspect('equal', 'box')
        ax.set_title('Q-Learning Result: Sensor Placement Optimization', fontsize=14)
        ax.grid(True, alpha=0.2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果图已保存到: {save_path}")
        
        plt.show()


def compare_greedy_vs_qlearning():
    """
    比较纯贪心算法和Q-learning的性能
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "算法性能对比")
    print("=" * 70)
    
    # 创建Q-learning求解器
    ql_solver = MaxWTCDLQLearning(
        n_targets=300,
        n_sensors=30,
        grid_size=1000,
        radius=50,
        distance_move_limit=250,
        distance_total_B=2000,
        learning_rate=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        seed=42
    )
    
    # 1. 评估纯贪心策略
    print("\n[1] 评估纯贪心策略...")
    greedy_result = ql_solver.evaluate(use_learned_policy=False)
    
    # 2. 训练Q-learning
    print("\n[2] 训练Q-learning模型...")
    training_result = ql_solver.train(
        num_episodes=500,
        use_greedy_init=True,
        verbose=True
    )
    
    # 3. 评估Q-learning策略
    print("\n[3] 评估Q-learning策略...")
    ql_result = ql_solver.evaluate(use_learned_policy=True)
    
    # 4. 比较结果
    print("\n" + "=" * 70)
    print(" " * 25 + "性能对比总结")
    print("=" * 70)
    print(f"{'指标':<30} {'贪心算法':<20} {'Q-Learning':<20}")
    print("-" * 70)
    print(f"{'最终奖励':<30} {greedy_result['final_reward']:<20.2f} {ql_result['final_reward']:<20.2f}")
    print(f"{'奖励提升':<30} {greedy_result['improvement']:<20.2f} {ql_result['improvement']:<20.2f}")
    print(f"{'提升比例 (%)':<30} {greedy_result['improvement']/greedy_result['initial_reward']*100:<20.2f} {ql_result['improvement']/ql_result['initial_reward']*100:<20.2f}")
    print(f"{'移动距离':<30} {greedy_result['final_cost']:<20.2f} {ql_result['final_cost']:<20.2f}")
    print(f"{'执行步数':<30} {len(greedy_result['actions']):<20} {len(ql_result['actions']):<20}")
    print("=" * 70)
    
    improvement = ql_result['final_reward'] - greedy_result['final_reward']
    improvement_pct = (improvement / greedy_result['final_reward'] * 100) if greedy_result['final_reward'] > 0 else 0
    
    if improvement > 0:
        print(f"\n✓ Q-Learning相比贪心算法提升了 {improvement:.2f} ({improvement_pct:.2f}%)")
    elif improvement < 0:
        print(f"\n✗ Q-Learning比贪心算法差 {abs(improvement):.2f} ({abs(improvement_pct):.2f}%)")
    else:
        print(f"\n= Q-Learning与贪心算法性能相当")
    
    print("=" * 70)
    
    # 绘制训练进度
    ql_solver.plot_training_progress(
        training_result['episode_rewards'],
        save_path='result/qlearning_training.png'
    )
    
    # 绘制最终结果
    ql_solver.plot_result(save_path='result/qlearning_result.png')
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    ql_solver.save_model('models/qlearning_model.pkl')
    
    return ql_solver, training_result, greedy_result, ql_result


def main():
    """主函数"""
    # 创建结果目录
    os.makedirs('result', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 运行对比实验
    solver, training_result, greedy_result, ql_result = compare_greedy_vs_qlearning()
    
    return solver, training_result, greedy_result, ql_result


if __name__ == "__main__":
    solver, training_result, greedy_result, ql_result = main()
