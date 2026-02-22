#!/usr/bin/env python3
"""
LQR控制器仿真与参数对比分析
简化版，减少依赖，确保能够正常运行
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

def create_second_order_system():
    """创建二阶系统模型"""
    A = [[0, 1],
         [0, 0]]
    B = [[0],
         [1]]
    return A, B



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
            # 对于位置-速度系统，初始位置为1.0，目标位置为0
            # 计算速度的超调量（速度从0开始，可能会有超调）
            max_velocity = np.max(np.abs(x[:, 1]))
            # 计算位置的超调量（如果系统响应超过初始位置）
            max_position = np.max(np.abs(x[:, 0]))
            initial_position = x[0, 0]
            
            # 综合考虑位置和速度的超调
            if initial_position != 0:
                position_overshoot = max(max_position / np.abs(initial_position) - 1, 0)
            else:
                position_overshoot = 0
            
            # 速度超调（相对于初始速度0）
            velocity_overshoot = max_velocity
            
            # 取两者的最大值作为超调量
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
    
    def display_single_parameter_results(self, t, x, u, metrics, params_name):
        """展示单组参数的控制结果"""
        # 清屏
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # 显示标题
        print(f"LQR控制结果 - {params_name}")
        print("=" * 80)
        
        # 显示性能指标
        print("性能指标:")
        print(f"• 收敛时间: {metrics['convergence_time']:.3f} s")
        print(f"• 稳态误差: {metrics['steady_state_error']:.6f}")
        print(f"• 最大控制输入: {metrics['max_control_input']:.3f}")
        print(f"• 超调量: {metrics['overshoot']:.2f}%")
        print("=" * 80)
        
        # 显示系统信息
        print("系统信息:")
        print(f"• 仿真时间: {t[-1]:.1f}s")
        print(f"• 状态变量: {x.shape[1]}个")
        print(f"• 控制输入: {u.shape[1]}个")
        print(f"• 采样点: {len(t)}个")
        print("=" * 80)
        
        # 显示参数说明
        print("参数说明:")
        if "高状态权重" in params_name:
            print("• Q矩阵较大: 重视状态误差，使系统快速收敛")
            print("• R矩阵较小: 对控制输入限制较少，可能导致控制量较大")
        elif "高控制代价" in params_name:
            print("• Q矩阵较小: 对状态误差要求较低")
            print("• R矩阵较大: 重视控制输入平滑，控制量较小但收敛较慢")
        else:
            print("• Q和R矩阵平衡: 在收敛速度和控制平滑之间取得平衡")
        print("=" * 80)
    
    def display_parameter_comparison(self, comparison_results):
        """
        展示多组参数的对比结果
        """
        # 清屏
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # 显示标题
        print("LQR参数对比分析")
        print("=" * 80)
        
        # 显示对比表格（改进对齐）
        print("参数对比表格:")
        print("=" * 80)
        # 使用更精确的格式化确保对齐
        header = f"{'参数组':<15} {'收敛时间(s)':<15} {'稳态误差':<15} {'最大控制输入':<15} {'超调量(%)':<15}"
        print(header)
        print("-" * 80)
        
        for result in comparison_results:
            metrics = result['metrics']
            # 使用固定宽度和对齐方式
            row = f"{result['name']:<15} {metrics['convergence_time']:<15.3f} {metrics['steady_state_error']:<15.6f} {metrics['max_control_input']:<15.3f} {metrics['overshoot']:<15.2f}"
            print(row)
        print("=" * 80)
        print()
        
        # 显示对比图表
        self._display_comparison_charts(comparison_results)
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
    
    def _display_comparison_charts(self, comparison_results):
        """
        显示对比图表（终端坐标图）
        """
        print("参数对比图表:")
        print("-" * 80)
        
        # 提取数据
        param_names = []
        metrics_data = {}
        
        for result in comparison_results:
            param_name = result['name']
            param_names.append(param_name)
            metrics = result['metrics']
            metrics_data[param_name] = {
                'convergence_time': metrics['convergence_time'],
                'max_control_input': metrics['max_control_input'],
                'steady_state_error': metrics['steady_state_error'],
                'overshoot': metrics['overshoot']
            }
        
        # 绘制性能指标对比坐标图
        self._draw_performance_comparison_chart(param_names, metrics_data)
        print()
        
        # 绘制系统响应曲线对比
        self._draw_system_response_comparison(comparison_results)
        print()
    
    def _draw_performance_comparison_chart(self, param_names, metrics_data):
        """
        绘制性能指标对比坐标图
        """
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
            values = [metrics_data[name][metric_key] for name in param_names]
            max_value = max(values) if max(values) > 0 else 1
            
            # 绘制坐标图
            chart_width = 60
            # 固定参数组名称宽度
            name_width = 15
            
            # 绘制顶部边界
            print("  " + " " * name_width + "┌" + "─" * chart_width + "┐")
            
            # 绘制数据点
            for j, (name, value) in enumerate(zip(param_names, values)):
                # 计算X坐标位置，确保在有效范围内
                x_pos = int((value / max_value) * (chart_width - 1))
                x_pos = max(0, min(x_pos, chart_width - 1))
                
                # 创建数据线，确保数据点和值都正确对齐
                # 使用固定宽度格式化确保对齐
                # 构建字符串
                # 1. 参数组名称部分（固定宽度）
                # 使用ljust确保宽度完全一致
                name_part = "  " + name.ljust(name_width)
                # 2. 左侧边界
                left_border = "│"
                # 3. 图表区域
                chart_part = " " * chart_width
                # 4. 在图表区域中放置标记
                chart_part = chart_part[:x_pos] + markers[j] + chart_part[x_pos+1:]
                # 5. 右侧边界和值
                right_part = f"│ {value:.3f}"
                # 6. 拼接所有部分
                line = name_part + left_border + chart_part + right_part
                print(line)
            
            # 绘制底部边界
            print("  " + " " * name_width + "└" + "─" * chart_width + "┘")
            print()
    
    def _draw_system_response_comparison(self, comparison_results):
        """
        绘制系统响应曲线对比
        """
        print("系统响应曲线对比:")
        print("-" * 80)
        
        # 定义不同参数组的线条样式
        line_styles = ['━', '─', '▬']
        
        # 绘制位置响应曲线
        print("1. 位置响应曲线:")
        self._draw_response_curve(comparison_results, 0, "位置", line_styles)
        print()
        
        # 绘制速度响应曲线
        print("2. 速度响应曲线:")
        self._draw_response_curve(comparison_results, 1, "速度", line_styles)
        print()
    
    def _draw_response_curve(self, comparison_results, state_index, state_name, line_styles):
        """
        绘制特定状态的响应曲线
        """
        chart_width = 70
        chart_height = 15
        # 固定参数组名称宽度
        name_width = 15
        
        # 提取所有响应数据
        all_responses = []
        max_value = 0
        
        for result in comparison_results:
            x = result['x'][:, state_index]
            max_value = max(max_value, max(abs(x)))
            all_responses.append((result['name'], x, result['t']))
        
        # 确保最大值大于0
        if max_value == 0:
            max_value = 1
        
        # 绘制图表框架
        print(f"  {'':<{name_width}}┌{'─' * chart_width}┐")
        
        # 绘制数据曲线
        for i, (name, x, t) in enumerate(all_responses):
            # 采样数据点以适应图表宽度
            sample_points = min(len(x), chart_width)
            step = len(x) // sample_points
            sampled_x = x[::step][:sample_points]
            sampled_t = t[::step][:sample_points]
            
            # 绘制曲线
            # 参数组名称部分（固定宽度）
            name_part = f"  {name:<{name_width}}"
            # 左侧边界
            left_border = "│"
            # 曲线部分
            curve_part = ""
            
            # 为每个时间点创建曲线点
            for j, value in enumerate(sampled_x):
                # 计算Y坐标位置（归一化到图表高度）
                normalized_value = (value + max_value) / (2 * max_value)  # 映射到0-1
                # 使用不同的字符来表示曲线高度
                if normalized_value > 0.8:
                    char = "█"
                elif normalized_value > 0.6:
                    char = "▓"
                elif normalized_value > 0.4:
                    char = "▒"
                elif normalized_value > 0.2:
                    char = "░"
                else:
                    char = " "
                
                # 添加到曲线
                curve_part += char
            
            # 右侧边界
            right_border = "│"
            # 拼接所有部分
            curve = name_part + left_border + curve_part + right_border
            print(curve)
        
        # 绘制图表底部
        print(f"  {'':<{name_width}}└{'─' * chart_width}┘")
        print(f"  {'':<{name_width}}时间 (s) →")

class LQRSimulationApp:
    """
    LQR仿真应用类，用于运行LQR控制仿真和参数对比分析
    """
    
    def __init__(self):
        """
        初始化LQR仿真应用
        """
        self.visualizer = TerminalVisualizer()
        self.system = None
        self.x0 = None
        self.A = None
        self.B = None
    
    def run(self):
        """
        运行应用主界面
        """
        while True:
            # 显示主菜单
            self._show_main_menu()
            
            # 获取用户选择
            choice = input("请输入您的选择 (1-3): ")
            
            if choice == "1":
                # 单组参数运行
                self._run_single_parameter()
            elif choice == "2":
                # 多组参数对比
                self._run_parameter_comparison()
            elif choice == "3":
                # 退出
                print("感谢使用LQR仿真应用！")
                break
            else:
                print("无效的选择，请重新输入。")
                input("按回车键继续...")
    
    def _show_main_menu(self):
        """
        显示主菜单
        """
        # 清屏
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print("LQR控制器仿真与参数对比分析")
        print("=" * 80)
        print("1. 单组参数运行")
        print("2. 多组参数对比")
        print("3. 退出")
        print("=" * 80)
    
    def _select_system(self):
        """
        选择系统模型（仅保留二阶系统）
        """
        # 直接使用二阶系统
        self.A, self.B = create_second_order_system()
        # 转换为numpy数组
        self.A = np.array(self.A)
        self.B = np.array(self.B)
        # 二阶系统初始状态: [位置, 速度]
        self.x0 = np.array([1.0, 0.0])
        print("已选择: 二阶系统 (位置-速度控制)")
    
    def _get_preset_parameters(self):
        """
        获取预设参数组
        
        Returns:
            dict: 预设参数组字典
        """
        # 根据系统维度创建预设参数
        n = len(self.A)  # 状态维度
        m = self.B.shape[1]  # 输入维度
        
        # 预设参数组（统一为五个字名称）
        preset_params = {
            "高状态权重": {
                "Q": np.diag([10.0] * n),
                "R": np.diag([1.0] * m)
            },
            "高控制代价": {
                "Q": np.diag([1.0] * n),
                "R": np.diag([10.0] * m)
            },
            "平衡型参数": {
                "Q": np.diag([5.0] * n),
                "R": np.diag([5.0] * m)
            }
        }
        
        return preset_params
    
    def _run_single_parameter(self):
        """
        单组参数运行
        """
        # 选择系统
        self._select_system()
        
        # 获取预设参数
        preset_params = self._get_preset_parameters()
        
        # 选择参数组
        print("\n请选择参数组:")
        for i, (name, _) in enumerate(preset_params.items(), 1):
            print(f"{i}. {name}")
        
        choice = input("请输入您的选择 (1-3): ")
        
        # 获取选择的参数组
        param_names = list(preset_params.keys())
        if choice in ["1", "2", "3"]:
            selected_param_name = param_names[int(choice) - 1]
            params = preset_params[selected_param_name]
        else:
            print("无效的选择，默认使用平衡型参数。")
            selected_param_name = "平衡型"
            params = preset_params["平衡型"]
        
        # 创建LQR控制器
        controller = LQRController(
            self.A,
            self.B,
            params["Q"],
            params["R"]
        )
        
        # 创建仿真器
        simulator = SystemSimulator(self.A, self.B, controller)
        
        # 运行仿真
        t, x, u = simulator.simulate(self.x0)
        
        # 计算性能指标
        metrics = simulator.calculate_performance_metrics(t, x, u)
        
        # 显示结果
        self.visualizer.display_single_parameter_results(t, x, u, metrics, selected_param_name)
        
        input("按回车键返回主菜单...")
    
    def _run_parameter_comparison(self):
        """
        多组参数对比
        """
        # 选择系统
        self._select_system()
        
        # 获取预设参数
        preset_params = self._get_preset_parameters()
        
        # 运行所有参数组的仿真
        comparison_results = []
        
        for param_name, params in preset_params.items():
            # 创建LQR控制器
            controller = LQRController(
                self.A,
                self.B,
                params["Q"],
                params["R"]
            )
            
            # 创建仿真器
            simulator = SystemSimulator(self.A, self.B, controller)
            
            # 运行仿真
            t, x, u = simulator.simulate(self.x0)
            
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
        self.visualizer.display_parameter_comparison(comparison_results)
        
        input("按回车键返回主菜单...")

if __name__ == "__main__":
    # 运行应用
    app = LQRSimulationApp()
    app.run()