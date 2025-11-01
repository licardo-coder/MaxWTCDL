import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import math


class MaxWTCDLSolver:
    def __init__(self, n=100, m=20, p=1000, q=1000, r_s=50, b=100, B=None):
        self.n = n  # 目标数量
        self.m = m  # 传感器数量
        self.p = p  # 位置坐标维度1 (grid size)
        self.q = q  # 位置坐标维度2 (grid size)
        self.r_s = r_s  # 感知半径
        self.b = b  # 单个传感器移动距离约束
        self.B = B  # 总移动距离约束 (如果为None,将在数据生成后设置)

        # 初始化模型
        self.model = gp.Model("MaxWTCDL")

        # 设置Gurobi参数以优化性能
        self._set_gurobi_parameters()

        # 生成模拟数据（在实际应用中替换为真实数据）
        self._generate_synthetic_data()

    def _set_gurobi_parameters(self):
        """设置Gurobi求解器参数"""
        # 启用多线程并行计算
        self.model.setParam('Threads', min(16, gp.GRB.thread_count()))

        # 设置时间限制（秒）
        self.model.setParam('TimeLimit', 3600)  # 1小时

        # 设置内存限制（GB）
        self.model.setParam('MemLimit', 32)

        # 设置MIP间隙容忍度
        self.model.setParam('MIPGap', 0.01)  # 1%的容忍度

        # 启用启发式策略
        self.model.setParam('Heuristics', 0.1)

        # 设置输出级别
        self.model.setParam('OutputFlag', 1)

        # 对于大规模问题，调整节点策略
        self.model.setParam('NodeMethod', 1)  # 使用对偶单纯形法

    def _generate_synthetic_data(self):
        """生成模拟数据 - 在实际应用中替换为真实数据"""
        np.random.seed(42)  # 为了可重复性

        # 生成传感器初始位置
        self.sensor_init_positions = {}
        for i in range(self.m):
            init_x = np.random.randint(0, self.p)
            init_y = np.random.randint(0, self.q)
            self.sensor_init_positions[i] = (init_x, init_y)

        # 生成目标位置和权重
        self.target_positions = {}
        self.w_k = np.zeros(self.n)
        for k in range(self.n):
            target_x = np.random.randint(0, self.p)
            target_y = np.random.randint(0, self.q)
            self.target_positions[k] = (target_x, target_y)
            self.w_k[k] = np.random.randint(1, 10)

        # 为每个传感器计算可行位置集合 G_i (在个体移动约束 b 范围内)
        self.sensor_positions = {}  # 存储每个传感器的可行位置
        self.d_ijz = {}  # 移动成本
        self.cover_sets = {}  # 存储每个位置能覆盖的目标集合 A_ijz

        print("正在生成可行位置和覆盖关系...")
        for i in range(self.m):
            init_x, init_y = self.sensor_init_positions[i]
            positions = []

            # 计算移动范围内的所有网格点
            # 为了减少计算量,只考虑以初始位置为中心的正方形区域
            x_min = max(0, int(init_x - self.b))
            x_max = min(self.p - 1, int(init_x + self.b))
            y_min = max(0, int(init_y - self.b))
            y_max = min(self.q - 1, int(init_y + self.b))

            for j in range(x_min, x_max + 1):
                for z in range(y_min, y_max + 1):
                    # 计算欧几里得距离
                    distance = math.sqrt((j - init_x)**2 + (z - init_y)**2)
                    
                    # 检查是否在个体移动约束内
                    if distance <= self.b:
                        positions.append((j, z))
                        
                        # 记录移动成本 d_ijz
                        self.d_ijz[(i, j, z)] = distance
                        
                        # 计算该位置能覆盖的目标集合 A_ijz
                        covered_targets = set()
                        for k in range(self.n):
                            target_x, target_y = self.target_positions[k]
                            target_distance = math.sqrt((j - target_x)**2 + (z - target_y)**2)
                            
                            # 如果目标在感知半径内,则被覆盖
                            if target_distance <= self.r_s:
                                covered_targets.add(k)
                        
                        self.cover_sets[(i, j, z)] = covered_targets

            self.sensor_positions[i] = positions
            if (i + 1) % 5 == 0:
                print(f"  已处理传感器 {i + 1}/{self.m}, 可行位置数: {len(positions)}")

        print(f"数据生成完成!")

    def build_model(self):
        """构建优化模型"""
        start_time = time.time()

        # 创建变量
        self._create_variables()

        # 添加约束
        self._add_constraints()

        # 设置目标函数
        self._set_objective()

        build_time = time.time() - start_time
        print(f"模型构建完成，耗时: {build_time:.2f}秒")
        print(f"变量数量: {self.model.numVars}")
        print(f"约束数量: {self.model.numConstrs}")

    def _create_variables(self):
        """创建决策变量"""
        # y_k: 目标是否被覆盖
        self.y = self.model.addVars(
            self.n,
            vtype=GRB.BINARY,
            name="y"
        )

        # x_ijz: 传感器是否移动到位置(j,z)
        self.x = {}
        for i in range(self.m):
            for j, z in self.sensor_positions[i]:
                self.x[(i, j, z)] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_{i}_{j}_{z}"
                )

        self.model.update()

    def _add_constraints(self):
        """添加约束条件"""
        # 约束1: 目标覆盖约束
        # y_k <= sum_{i,j,z: t_k in A_{ijz}} x_ijz
        for k in range(self.n):
            cover_vars = []
            for i in range(self.m):
                for j, z in self.sensor_positions[i]:
                    if k in self.cover_sets.get((i, j, z), set()):
                        cover_vars.append(self.x[(i, j, z)])

            if cover_vars:  # 确保至少有一个变量
                self.model.addConstr(
                    self.y[k] <= gp.quicksum(cover_vars),
                    name=f"cover_constraint_{k}"
                )

        # 约束2: 每个传感器最多选择一个位置
        # sum_{j,z: A_ijz in G_i} x_ijz <= 1
        for i in range(self.m):
            position_vars = [
                self.x[(i, j, z)] for j, z in self.sensor_positions[i]
            ]
            if position_vars:
                self.model.addConstr(
                    gp.quicksum(position_vars) <= 1,
                    name=f"sensor_selection_{i}"
                )

        # 约束3: 预算约束
        # sum_{i,j,z} x_ijz * d_ijz <= B
        budget_vars = []
        budget_coeffs = []
        for i in range(self.m):
            for j, z in self.sensor_positions[i]:
                budget_vars.append(self.x[(i, j, z)])
                budget_coeffs.append(self.d_ijz[(i, j, z)])

        # 如果B没有指定,使用所有可能成本之和的30%作为默认值
        if self.B is None:
            total_possible_cost = sum(budget_coeffs)
            self.B = total_possible_cost * 0.3

        if budget_vars:
            self.model.addConstr(
                gp.quicksum(
                    budget_coeffs[idx] * budget_vars[idx]
                    for idx in range(len(budget_vars))
                ) <= self.B,
                name="budget_constraint"
            )

        self.model.update()

    def _set_objective(self):
        """设置目标函数"""
        # max sum_{k=1}^n w_k * y_k
        objective = gp.quicksum(
            self.w_k[k] * self.y[k] for k in range(self.n)
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)

    def solve(self):
        """求解模型"""
        print("开始求解...")
        start_time = time.time()

        try:
            # 优化模型
            self.model.optimize()

            solve_time = time.time() - start_time

            # 输出结果
            self._output_results(solve_time)

        except gp.GurobiError as e:
            print(f"Gurobi错误: {e}")
        except Exception as e:
            print(f"求解过程中出现错误: {e}")

    def _output_results(self, solve_time):
        """输出求解结果"""
        print("\n" + "=" * 50)
        print("求解结果")
        print("=" * 50)

        if self.model.status == GRB.OPTIMAL:
            print(f"找到最优解!")
        elif self.model.status == GRB.TIME_LIMIT:
            print(f"达到时间限制，找到可行解")
        else:
            print(f"求解状态: {self.model.status}")

        print(f"求解时间: {solve_time:.2f}秒")
        print(f"目标函数值: {self.model.objVal:.2f}")
        print(f"MIP Gap: {self.model.MIPGap:.4f}")

        # 输出传感器部署方案
        deployed_sensors = 0
        for i in range(self.m):
            for j, z in self.sensor_positions[i]:
                if abs(self.x[(i, j, z)].x) > 1e-6:  # 检查变量是否被选择
                    print(f"传感器 {i} 部署到位置 ({j}, {z})")
                    deployed_sensors += 1
                    break

        # 输出覆盖目标统计
        covered_targets = sum(1 for k in range(self.n) if abs(self.y[k].x) > 1e-6)
        print(f"部署的传感器数量: {deployed_sensors}")
        print(f"覆盖的目标数量: {covered_targets}/{self.n}")
        print(f"覆盖率: {covered_targets / self.n * 100:.2f}%")

        # 输出预算使用情况
        total_cost = 0
        for i in range(self.m):
            for j, z in self.sensor_positions[i]:
                if abs(self.x[(i, j, z)].x) > 1e-6:
                    total_cost += self.d_ijz[(i, j, z)]

        print(f"总成本: {total_cost:.2f}/{self.B:.2f}")
        print(f"预算使用率: {total_cost / self.B * 100:.2f}%")

    def add_lazy_constraints(self):
        """添加惰性约束（可选的高级功能）"""
        # 这里可以添加问题特定的有效不等式
        # 例如覆盖不等式、背包覆盖不等式等
        pass


def main():
    """主函数"""
    print("MaxWTCDL问题求解器")
    print("=" * 30)

    # 创建求解器实例
    solver = MaxWTCDLSolver(
        n=100,  # 目标数量
        m=20,  # 传感器数量
        p=1000,  # 位置坐标维度1 (grid size)
        q=1000,  # 位置坐标维度2 (grid size)
        r_s=50,  # 感知半径
        b=100,  # 单个传感器移动距离约束
        B=None  # 总移动距离约束 (None表示自动设置为可能成本的30%)
    )

    # 构建并求解模型
    solver.build_model()
    solver.solve()


if __name__ == "__main__":
    main()