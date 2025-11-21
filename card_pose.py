import torch
import time
import argparse
import sys
import threading
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

class MultiGPUMemoryConsumer:
    """
    多GPU显存占用程序
    可以同时占用多个GPU设备
    """
    
    def __init__(self, gpu_ids: List[int], memory_per_gpu: float = 4.0):
        """
        初始化多GPU占用程序
        
        Args:
            gpu_ids: 要占用的GPU设备ID列表，如 [0, 1, 2]
            memory_per_gpu: 每个GPU要占用的显存大小（GB）
        """
        self.gpu_ids = gpu_ids
        self.memory_per_gpu = memory_per_gpu
        self.consumers = {}
        
        # 检查GPU可用性
        self._check_gpu_availability()
        
    def _check_gpu_availability(self):
        """检查所有GPU是否可用"""
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，无法使用GPU")
            sys.exit(1)
            
        available_gpus = list(range(torch.cuda.device_count()))
        print(f"✅ 可用GPU: {available_gpus}")
        
        for gpu_id in self.gpu_ids:
            if gpu_id not in available_gpus:
                print(f"❌ GPU {gpu_id} 不存在，可用GPU: {available_gpus}")
                sys.exit(1)
                
        print(f"🎯 将占用GPU: {self.gpu_ids}")
        
    def occupy_all_gpus(self):
        """同时占用所有指定的GPU"""
        print(f"🚀 开始同时占用 {len(self.gpu_ids)} 个GPU...")
        
        # 使用线程池并行占用
        with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
            futures = []
            for gpu_id in self.gpu_ids:
                future = executor.submit(self._occupy_single_gpu, gpu_id)
                futures.append(future)
            
            # 等待所有任务完成
            for future in futures:
                future.result()
        
        print("✅ 所有GPU占用完成")
    
    def _occupy_single_gpu(self, gpu_id: int):
        """占用单个GPU"""
        try:
            # 为每个GPU创建独立的consumer
            consumer = SingleGPUConsumer(gpu_id, self.memory_per_gpu)
            consumer.occupy_memory()
            self.consumers[gpu_id] = consumer
            print(f"✅ GPU {gpu_id} 占用成功")
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} 占用失败: {e}")
    
    def heavy_computation_all_gpus(self, computation_time: int = 60):
        """在所有GPU上执行大量计算"""
        print(f"🔥 在所有 {len(self.gpu_ids)} 个GPU上执行计算，持续 {computation_time} 秒...")
        
        threads = []
        for gpu_id, consumer in self.consumers.items():
            thread = threading.Thread(
                target=consumer.heavy_computation,
                args=(computation_time,)
            )
            thread.start()
            threads.append(thread)
            print(f"▶️  GPU {gpu_id} 计算线程启动")
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
    
    def matrix_operations_all_gpus(self, matrix_size: int = 4096, operations: int = 1000):
        """在所有GPU上执行矩阵运算"""
        print(f"🧮 在所有GPU上执行 {operations} 次 {matrix_size}x{matrix_size} 矩阵运算...")
        
        threads = []
        for gpu_id, consumer in self.consumers.items():
            thread = threading.Thread(
                target=consumer.matrix_operations,
                args=(matrix_size, operations)
            )
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
    
    def monitor_all_gpus(self, monitor_interval: int = 5):
        """监控所有GPU使用情况"""
        print(f"\n📊 监控所有 {len(self.gpu_ids)} 个GPU使用情况 (Ctrl+C停止):")
        print("=" * 60)
        
        try:
            while True:
                gpu_status = []
                for gpu_id in self.gpu_ids:
                    if gpu_id in self.consumers:
                        consumer = self.consumers[gpu_id]
                        allocated = torch.cuda.memory_allocated(consumer.device) / (1024**3)
                        utilization = torch.cuda.utilization(gpu_id)
                        gpu_status.append(f"GPU{gpu_id}: {allocated:.1f}GB/{utilization}%")
                
                status_str = " | ".join(gpu_status)
                print(f"📈 {status_str}")
                time.sleep(monitor_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  停止监控")
    
    def cleanup_all_gpus(self):
        """清理所有GPU显存"""
        print("🧹 清理所有GPU显存...")
        for gpu_id, consumer in self.consumers.items():
            try:
                consumer.cleanup()
                print(f"✅ GPU {gpu_id} 清理完成")
            except Exception as e:
                print(f"❌ GPU {gpu_id} 清理失败: {e}")

class SingleGPUConsumer:
    """
    单个GPU占用器
    """
    
    def __init__(self, gpu_id: int, memory_gb: float = 4.0):
        self.gpu_id = gpu_id
        self.memory_gb = memory_gb
        self.device = torch.device(f'cuda:{gpu_id}')
        self.memory_tensors = []
    
    def occupy_memory(self, memory_gb: Optional[float] = None):
        """占用显存"""
        if memory_gb is not None:
            self.memory_gb = memory_gb
            
        bytes_needed = int(self.memory_gb * 1024 * 1024 * 1024)
        elements_needed = bytes_needed // 4
        
        print(f"🔄 GPU {self.gpu_id}: 正在分配 {self.memory_gb} GB 显存...")
        
        try:
            tensor = torch.randn(elements_needed, dtype=torch.float32, device=self.device)
            self.memory_tensors.append(tensor)
            torch.cuda.synchronize(self.device)
            
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            print(f"✅ GPU {self.gpu_id}: 成功分配 {allocated:.2f} GB 显存")
            
        except RuntimeError as e:
            print(f"❌ GPU {self.gpu_id}: 显存分配失败: {e}")
            self._allocate_in_chunks(elements_needed)
    
    def _allocate_in_chunks(self, total_elements: int):
        """分块分配显存"""
        chunk_size = total_elements // 10
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
                print(f"📦 GPU {self.gpu_id}: 已分配 {allocated_gb:.2f} GB")
                
            except RuntimeError:
                print(f"⚠️  GPU {self.gpu_id}: 无法分配更多显存")
                break
        
        total_allocated = (allocated_elements * 4) / (1024**3)
        print(f"🎯 GPU {self.gpu_id}: 最终分配 {total_allocated:.2f} GB")
    
    def heavy_computation(self, computation_time: int = 60, batch_size: int = 1024):
        """执行大量计算"""
        print(f"🔥 GPU {self.gpu_id}: 开始计算，持续 {computation_time} 秒...")
        
        if not self.memory_tensors:
            self.occupy_memory(1.0)
        
        compute_tensor = self.memory_tensors[0][:batch_size * 1000].view(batch_size, -1)
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < computation_time:
            a = torch.randn(batch_size, batch_size, device=self.device)
            b = torch.randn(batch_size, batch_size, device=self.device)
            c = torch.matmul(a, b)
            d = torch.nn.functional.relu(c)
            e = torch.nn.functional.softmax(d, dim=1)
            f = torch.nn.functional.log_softmax(e, dim=1)
            
            torch.cuda.synchronize(self.device)
            
            iteration += 1
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"🔄 GPU {self.gpu_id}: {iteration} 次迭代，用时 {elapsed:.1f} 秒")
    
    def matrix_operations(self, matrix_size: int = 4096, operations: int = 1000):
        """执行矩阵运算"""
        print(f"🧮 GPU {self.gpu_id}: 执行 {operations} 次 {matrix_size}x{matrix_size} 矩阵运算...")
        
        a = torch.randn(matrix_size, matrix_size, device=self.device)
        b = torch.randn(matrix_size, matrix_size, device=self.device)
        
        for i in range(operations):
            c = torch.matmul(a, b)
            d = torch.inverse(c)
            e = torch.eig(c)
            f = torch.svd(c)
            
            if i % 100 == 0:
                print(f"📊 GPU {self.gpu_id}: 完成 {i}/{operations} 次运算")
        
        print(f"✅ GPU {self.gpu_id}: 矩阵运算完成")
    
    def cleanup(self):
        """清理显存"""
        self.memory_tensors.clear()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        print(f"🧹 GPU {self.gpu_id}: 清理完成，剩余 {allocated:.2f} GB")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多GPU显存占用程序')
    parser.add_argument('--gpus', type=str, required=True, 
                       help='要占用的GPU ID列表，用逗号分隔，如: 0,1,2 或 0-3')
    parser.add_argument('--memory', type=float, default=4.0, 
                       help='每个GPU占用显存大小(GB) (默认: 4.0)')
    parser.add_argument('--time', type=int, default=300, 
                       help='计算持续时间(秒) (默认: 300)')
    parser.add_argument('--computation', type=str, choices=['matrix', 'mixed'], 
                       default='mixed', help='计算类型 (默认: mixed)')
    
    args = parser.parse_args()
    
    # 解析GPU ID列表
    gpu_ids = parse_gpu_ids(args.gpus)
    
    if not gpu_ids:
        print("❌ 无效的GPU ID格式")
        print("✅ 正确格式示例:")
        print("   --gpus 0,1,2     # 占用GPU 0,1,2")
        print("   --gpus 0-3       # 占用GPU 0,1,2,3") 
        print("   --gpus 0,2,4     # 占用GPU 0,2,4")
        sys.exit(1)
    
    # 创建多GPU占用器
    consumer = MultiGPUMemoryConsumer(gpu_ids, args.memory)
    
    try:
        # 1. 占用所有GPU显存
        consumer.occupy_all_gpus()
        
        # 2. 执行计算
        if args.computation == 'matrix':
            consumer.matrix_operations_all_gpus(matrix_size=4096, operations=1000)
        else:
            consumer.heavy_computation_all_gpus(computation_time=args.time)
        
        # 3. 显示系统信息
        print("\n" + "="*50)
        print("🎯 系统GPU状态汇总:")
        print("="*50)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name} | 总显存: {total_memory:.1f} GB")
        
        # 4. 持续监控
        print(f"\n🔄 程序持续运行，占用GPU: {gpu_ids}")
        print("按 Ctrl+C 退出程序")
        consumer.monitor_all_gpus()
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断程序")
    finally:
        consumer.cleanup_all_gpus()

def parse_gpu_ids(gpu_str: str) -> List[int]:
    """解析GPU ID字符串"""
    gpu_ids = []
    
    try:
        # 处理范围格式: 0-3
        if '-' in gpu_str:
            start, end = map(int, gpu_str.split('-'))
            gpu_ids = list(range(start, end + 1))
        # 处理列表格式: 0,1,2
        elif ',' in gpu_str:
            gpu_ids = [int(x.strip()) for x in gpu_str.split(',')]
        # 处理单个GPU: 0
        else:
            gpu_ids = [int(gpu_str)]
        
        # 去重并排序
        gpu_ids = sorted(set(gpu_ids))
        return gpu_ids
        
    except ValueError:
        return []

if __name__ == "__main__":
    main()