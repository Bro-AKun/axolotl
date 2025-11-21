import torch
import time
import argparse
import sys
from typing import Optional

class GPUMemoryConsumer:
    """
    GPU显存占用程序
    可以指定GPU设备、占用显存大小、计算强度等参数
    """
    
    def __init__(self, gpu_id: int = 0, memory_gb: float = 4.0):
        """
        初始化GPU占用程序
        
        Args:
            gpu_id: 要占用的GPU设备ID
            memory_gb: 要占用的显存大小（GB）
        """
        self.gpu_id = gpu_id
        self.memory_gb = memory_gb
        self.device = None
        self.memory_tensors = []
        
        # 检查GPU可用性
        self._check_gpu_availability()
        
    def _check_gpu_availability(self):
        """检查GPU是否可用"""
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，无法使用GPU")
            sys.exit(1)
            
        if self.gpu_id >= torch.cuda.device_count():
            print(f"❌ GPU {self.gpu_id} 不存在，可用GPU: 0-{torch.cuda.device_count()-1}")
            sys.exit(1)
            
        self.device = torch.device(f'cuda:{self.gpu_id}')
        print(f"✅ 使用GPU {self.gpu_id}: {torch.cuda.get_device_name(self.gpu_id)}")
        
    def occupy_memory(self, memory_gb: Optional[float] = None):
        """
        占用指定大小的显存
        
        Args:
            memory_gb: 显存大小（GB），如果为None则使用初始化值
        """
        if memory_gb is not None:
            self.memory_gb = memory_gb
            
        # 计算需要分配的元素数量（float32，每个4字节）
        bytes_needed = int(self.memory_gb * 1024 * 1024 * 1024)  # 转换为字节
        elements_needed = bytes_needed // 4  # float32每个元素4字节
        
        print(f"🔄 正在分配 {self.memory_gb} GB 显存...")
        
        try:
            # 分配大张量来占用显存
            tensor = torch.randn(elements_needed, dtype=torch.float32, device=self.device)
            self.memory_tensors.append(tensor)
            
            # 确保张量被实际分配（防止延迟分配）
            torch.cuda.synchronize(self.device)
            
            # 获取实际占用显存
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # 转换为GB
            print(f"✅ 成功分配 {allocated:.2f} GB 显存")
            
        except RuntimeError as e:
            print(f"❌ 显存分配失败: {e}")
            # 尝试分配较小的块
            self._allocate_in_chunks(elements_needed)
    
    def _allocate_in_chunks(self, total_elements: int):
        """分块分配显存（当一次性分配失败时使用）"""
        chunk_size = total_elements // 10  # 分成10块
        allocated_elements = 0
        
        while allocated_elements < total_elements:
            try:
                current_chunk = min(chunk_size, total_elements - allocated_elements)
                if current_chunk <= 0:
                    break
                    
                tensor = torch.randn(current_chunk, dtype=torch.float32, device=self.device)
                self.memory_tensors.append(tensor)
                allocated_elements += current_chunk
                
                allocated_gb = (allocated_elements * 4) / (1024**3)
                print(f"📦 已分配: {allocated_gb:.2f} GB")
                
            except RuntimeError:
                print("⚠️  无法分配更多显存，可能已满")
                break
        
        total_allocated = (allocated_elements * 4) / (1024**3)
        print(f"🎯 最终分配: {total_allocated:.2f} GB")
    
    def heavy_computation(self, computation_time: int = 60, batch_size: int = 1024):
        """
        执行大量计算来保持GPU活跃
        
        Args:
            computation_time: 计算持续时间（秒）
            batch_size: 批量大小
        """
        print(f"🔥 开始执行大量计算，持续 {computation_time} 秒...")
        
        # 创建用于计算的张量
        if not self.memory_tensors:
            print("⚠️  没有可用的显存张量，先分配一些显存")
            self.occupy_memory(1.0)  # 分配1GB用于计算
        
        # 使用部分已分配的显存进行计算
        compute_tensor = self.memory_tensors[0][:batch_size * 1000].view(batch_size, -1)
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < computation_time:
            # 执行密集矩阵运算
            a = torch.randn(batch_size, batch_size, device=self.device)
            b = torch.randn(batch_size, batch_size, device=self.device)
            
            # 矩阵乘法（计算密集型）
            c = torch.matmul(a, b)
            
            # 激活函数计算
            d = torch.nn.functional.relu(c)
            
            # 更多的计算操作
            e = torch.nn.functional.softmax(d, dim=1)
            f = torch.nn.functional.log_softmax(e, dim=1)
            
            # 确保计算完成
            torch.cuda.synchronize(self.device)
            
            iteration += 1
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"🔄 已计算 {iteration} 次迭代，用时 {elapsed:.1f} 秒")
    
    def matrix_operations(self, matrix_size: int = 4096, operations: int = 1000):
        """
        执行大规模矩阵运算
        
        Args:
            matrix_size: 矩阵大小 (n x n)
            operations: 操作次数
        """
        print(f"🧮 执行 {operations} 次 {matrix_size}x{matrix_size} 矩阵运算...")
        
        # 创建大矩阵
        a = torch.randn(matrix_size, matrix_size, device=self.device)
        b = torch.randn(matrix_size, matrix_size, device=self.device)
        
        for i in range(operations):
            # 各种矩阵运算
            c = torch.matmul(a, b)                    # 矩阵乘法
            d = torch.inverse(c)                      # 矩阵求逆（计算量很大）
            e = torch.eig(c)                          # 特征值计算
            f = torch.svd(c)                          # 奇异值分解
            
            if i % 100 == 0:
                print(f"📊 已完成 {i}/{operations} 次矩阵运算")
        
        print("✅ 矩阵运算完成")
    
    def neural_network_simulation(self, layers: int = 10, hidden_size: int = 2048):
        """
        模拟神经网络前向传播（计算密集型）
        
        Args:
            layers: 网络层数
            hidden_size: 隐藏层大小
        """
        print(f"🧠 模拟 {layers} 层神经网络，隐藏层大小 {hidden_size}...")
        
        # 创建模拟的网络权重
        weights = []
        biases = []
        
        # 输入层
        input_size = 1024
        current_size = input_size
        
        for i in range(layers):
            # 创建权重矩阵
            w = torch.randn(hidden_size, current_size, device=self.device)
            b = torch.randn(hidden_size, device=self.device)
            weights.append(w)
            biases.append(b)
            current_size = hidden_size
        
        # 输出层
        output_weight = torch.randn(10, current_size, device=self.device)
        output_bias = torch.randn(10, device=self.device)
        
        # 模拟前向传播
        batch_size = 512
        x = torch.randn(batch_size, input_size, device=self.device)
        
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = torch.matmul(x, w.t()) + b
            x = torch.nn.functional.relu(x)  # ReLU激活
            
            if i % 3 == 0:  # 每3层添加归一化
                x = torch.nn.functional.layer_norm(x, (hidden_size,))
        
        # 输出层
        output = torch.matmul(x, output_weight.t()) + output_bias
        output = torch.nn.functional.softmax(output, dim=1)
        
        print("✅ 神经网络模拟完成")
    
    def monitor_gpu_usage(self, monitor_interval: int = 5):
        """监控GPU使用情况"""
        print("\n📊 GPU使用情况监控（Ctrl+C停止）:")
        print("=" * 50)
        
        try:
            while True:
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                cached = torch.cuda.memory_reserved(self.device) / (1024**3)
                utilization = torch.cuda.utilization(self.gpu_id)
                
                print(f"已分配: {allocated:.2f}GB | 缓存: {cached:.2f}GB | 利用率: {utilization}%")
                time.sleep(monitor_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  停止监控")
    
    def cleanup(self):
        """清理显存"""
        print("🧹 清理显存...")
        self.memory_tensors.clear()
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        print(f"✅ 清理完成，剩余显存: {allocated:.2f} GB")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GPU显存占用和计算程序')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID (默认: 0)')
    parser.add_argument('--memory', type=float, default=50.0, help='占用显存大小(GB) (默认: 4.0)')
    parser.add_argument('--time', type=int, default=300, help='计算持续时间(秒) (默认: 300)')
    parser.add_argument('--computation', type=str, choices=['matrix', 'nn', 'mixed'], 
                       default='mixed', help='计算类型 (默认: mixed)')
    
    args = parser.parse_args()
    
    # 创建GPU占用器
    consumer = GPUMemoryConsumer(gpu_id=args.gpu, memory_gb=args.memory)
    
    try:
        # 1. 占用显存
        consumer.occupy_memory()
        
        # 2. 执行计算
        if args.computation == 'matrix':
            consumer.matrix_operations(matrix_size=4096, operations=1000)
        elif args.computation == 'nn':
            consumer.neural_network_simulation(layers=20, hidden_size=4096)
        else:  # mixed
            consumer.heavy_computation(computation_time=args.time)
        
        # 3. 显示监控信息（可选）
        print("\n当前GPU状态:")
        print(f"设备: GPU {args.gpu}")
        print(f"名称: {torch.cuda.get_device_name(args.gpu)}")
        print(f"总显存: {torch.cuda.get_device_properties(args.gpu).total_memory / (1024**3):.1f} GB")
        
        # 保持程序运行
        print(f"\n🔄 程序将持续运行，占用GPU {args.gpu}...")
        print("按 Ctrl+C 退出程序")
        
        # 持续监控
        consumer.monitor_gpu_usage()
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断程序")
    finally:
        # 清理资源
        consumer.cleanup()

if __name__ == "__main__":
    main()