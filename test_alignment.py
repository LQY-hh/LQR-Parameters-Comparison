#!/usr/bin/env python3
"""
测试图表对齐效果
"""

import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    """LQR控制器类"""
    def __init__(self, A, B, Q, R):
        """初始化LQR控制器"""
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.K = self._compute_gain()
    
    def _compute_gain(self):
        """计算LQR增益"""
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K
    
    def get_control(self, x):
        """计算控制输入"""
        return -self.K @ x

class SystemSimulator:
    """系统仿真器类"""
    def __init__(self, A, B, controller, dt=0.01, t_span=(0, 10)):
        """初始化系统仿真器"""
        self.A = np.array(A)
        self.B = np.array(B)
        self.controller = controller
        self.dt = dt
        self.t_span = t_span
    
    def simulate(self, x0):
        """执行仿真"""
        t = np.arange(self.t_span[0], self.t_span[1] + self.dt, self.dt)
        x = np.zeros((len(t), len(x0)))
        u = np.zeros((len(t), self.B.shape[1]))
        x[0] = x0
        
        for i in range(1, len(t)):
            u[i-1] = self.controller.get_control(x[i-1])
            x[i] = x[i-1] + (self.A @ x[i-1] + self.B @ u[i-1]) * self.dt
        
        return t, x, u
    
    def calculate_performance_metrics(self, t, x, u):
        """计算性能指标"""
        # 收敛时间
        threshold = 0.01
        convergence_time = None
        for i in range(len(t)):
            if np.max(np.abs(x[i])) < threshold:
                convergence_time = t[i]
                break
        if convergence_time is None:
            convergence_time = t[-1]
        
        # 稳态误差
        steady_state_error = np.max(np.abs(x[-10:].mean(axis=0)))
        
        # 最大控制输入
        max_control_input = np.max(np.abs(u))
        
        # 超调量
        if len(x) > 0:
            max_velocity = np.max(np.abs(x[:, 1]))
            max_position = np.max(np.abs(x[:, 0]))
            initial_position = x[0, 0]
            
            if initial_position != 0:
                position_overshoot = max(max_position / np.abs(initial_position) - 1, 0)
            else:
                position_overshoot = 0
            
            velocity_overshoot = max_velocity
            overshoot = max(position_overshoot, velocity_overshoot)
        else:
            overshoot = 0
        
        return {
            'convergence_time': convergence_time,
            'steady_state_error': steady_state_error,
            'max_control_input': max_control_input,
            'overshoot': overshoot
        }

class TerminalVisualizer:
    """终端可视化器类"""
    def __init__(self):
        """初始化终端可视化器"""
        pass
    
    def display_parameter_comparison(self, comparison_results):
        """展示多组参数的对比结果"""
        # 清屏
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # 显示标题
        print("LQR参数对比分析")
        print("=" * 80)
        
        # 显示对比表格
        print("参数对比表格:")
        print("=" * 80)
        header = f"{'参数组':<15} {'收敛时间(s)':<15} {'稳态误差':<15} {'最大控制输入':<15} {'超调量(%)':<15}"
        print(header)
        print("-" * 80)
        
        for result in comparison_results:
            metrics = result['metrics']
            row = f"{result['name']:<15} {metrics['convergence_time']:<15.3f} {metrics['steady_state_error']:<15.6f} {metrics['max_control_input']:<15.3f} {metrics['overshoot']:<15.2f}"
            print(row)
        print("=" * 80)
        print()
        
        # 显示对比图表
        print("参数对比图表:")
        print("-" * 80)
        
        # 绘制性能指标对比坐标图
        print("性能指标对比坐标图:")
        print("-" * 80)
        
        # 定义性能指标和单位
        metrics = [
            ('convergence_time', '收敛时间 (s)', '越短越好'),
            ('max_control_input', '最大控制输入', '越小越好'),
            ('steady_state_error', '稳态误差', '越小越好'),
            ('overshoot', '超调量 (%)', '越小越好')
        ]
        
        # 为每个参数组定义不同的标记
        markers = ['●', '■', '▲']
        
        for i, (metric_key, metric_name, description) in enumerate(metrics):
            print(f"{i+1}. {metric_name} ({description}):")
            
            # 提取所有参数组的该指标值
            values = [result['metrics'][metric_key] for result in comparison_results]
            max_value = max(values) if max(values) > 0 else 1
            
            # 绘制坐标图
            chart_width = 60
            name_width = 15
            
            # 绘制顶部边界
            print(f"  {'':<{name_width}}┌{'─' * chart_width}┐")
            
            # 绘制数据点
            for j, (result, marker) in enumerate(zip(comparison_results, markers)):
                # 计算X坐标位置，确保在有效范围内
                value = result['metrics'][metric_key]
                x_pos = int((value / max_value) * (chart_width - 1))
                x_pos = max(0, min(x_pos, chart_width - 1))
                
                # 创建数据线，确保数据点和值都正确对齐
                # 使用固定宽度格式化确保对齐
                # 构建字符串
                # 1. 参数组名称部分（固定宽度）
                name_part = "  " + result['name'].ljust(name_width)
                # 2. 左侧边界
                left_border = "│"
                # 3. 图表区域
                chart_part = " " * x_pos + marker + " " * (chart_width - x_pos - 1)
                # 4. 右侧边界和值
                right_part = f"│ {value:.3f}"
                # 5. 拼接所有部分
                line = name_part + left_border + chart_part + right_part
                print(line)
            
            # 绘制底部边界
            print(f"  {'':<{name_width}}└{'─' * chart_width}┘")
            print()
        
        # 显示对比分析
        print("参数对比分析:")
        print("-" * 80)
        
        # 快速收敛 vs 平缓控制
        fast = None
        slow = None
        balanced = None
        
        for result in comparison_results:
            if "高状态权重" in result['name']:
                fast = result['metrics']
            elif "高控制代价" in result['name']:
                slow = result['metrics']
            elif "平衡" in result['name']:
                balanced = result['metrics']
        
        if fast and slow:
            if fast['convergence_time'] < slow['convergence_time']:
                print("• 高状态权重参数的收敛时间更短")
            if fast['max_control_input'] > slow['max_control_input']:
                print("• 高状态权重参数的控制输入更大")
        
        print("• 高控制代价参数的控制输入更平滑，但收敛较慢")
        print("• 平衡型参数在两者之间取得平衡")
        print()
        
        # 最佳选择建议
        print("最佳选择建议:")
        print("- 对响应速度要求高: 选择高状态权重参数")
        print("- 对控制平滑要求高: 选择高控制代价参数")
        print("- 一般情况: 选择平衡型参数")
        print("=" * 80)

# 创建二阶系统模型
def create_second_order_system():
    """创建二阶系统模型"""
    A = [[0, 1],
         [0, 0]]
    B = [[0],
         [1]]
    return A, B

# 主函数
def main():
    """主函数"""
    # 创建系统
    A, B = create_second_order_system()
    
    # 预设参数组
    preset_params = {
        "高状态权重": {
            "Q": np.diag([10.0, 10.0]),
            "R": np.diag([1.0])
        },
        "高控制代价": {
            "Q": np.diag([1.0, 1.0]),
            "R": np.diag([10.0])
        },
        "平衡型": {
            "Q": np.diag([5.0, 5.0]),
            "R": np.diag([5.0])
        }
    }
    
    # 运行所有参数组的仿真
    comparison_results = []
    
    for param_name, params in preset_params.items():
        # 创建LQR控制器
        controller = LQRController(A, B, params["Q"], params["R"])
        
        # 创建仿真器
        simulator = SystemSimulator(A, B, controller)
        
        # 运行仿真
        x0 = np.array([1.0, 0.0])  # 初始状态: [位置, 速度]
        t, x, u = simulator.simulate(x0)
        
        # 计算性能指标
        metrics = simulator.calculate_performance_metrics(t, x, u)
        
        # 存储结果
        comparison_results.append({
            "name": param_name,
            "metrics": metrics,
            "t": t,
            "x": x,
            "u": u,
            "Q": params["Q"],
            "R": params["R"]
        })
    
    # 显示对比结果
    visualizer = TerminalVisualizer()
    visualizer.display_parameter_comparison(comparison_results)

if __name__ == "__main__":
    # 运行主函数
    main()
