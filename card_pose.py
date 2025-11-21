import torch
import time
import threading
from typing import List, Optional

class MultiGPUMemoryConsumer:
    """
    多GPU显存占用程序 - 直接运行版本
    占用GPU 1,2,3，每个44GB，持续100000000秒
    """
    
    def __init__(self):
        # 配置参数 - 直接硬编码
        self.gpu_ids = [1, 2, 3]           # 要占用的GPU
        self.memory_per_gpu = 44.0         # 每个GPU占用44GB
        self.total_runtime = 100000000      # 总运行时间100000000秒
        self.consumers = {}
        
        # 检查GPU可用性
        self._check_gpu_availability()
        
    def _check_gpu_availability(self):
        """检查GPU是否可用"""
        print("=" * 60)
        print("🚀 多GPU显存占用程序启动")
        print("=" * 60)
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，无法使用GPU")
            return False
            
        available_gpus = list(range(torch.cuda.device_count()))
        print(f"✅ 系统可用GPU: {available_gpus}")
        print(f"🎯 目标占用GPU: {self.gpu_ids}")
        print(f"💾 每个GPU占用: {self.memory_per_gpu} GB")
        print(f"⏰ 运行时间: {self.total_runtime} 秒 ({self.total_runtime/3600:.1f} 小时)")
        print("=" * 60)
        
        # 检查目标GPU是否存在
        for gpu_id in self.gpu_ids:
            if gpu_id not in available_gpus:
                print(f"❌ 错误: GPU {gpu_id} 不存在！")
                print(f"✅ 可用GPU: {available_gpus}")
                return False
                
        return True
    
    def occupy_all_gpus(self):
        """占用所有指定的GPU显存"""
        print("🔄 开始占用GPU显存...")
        
        threads = []
        for gpu_id in self.gpu_ids:
            thread = threading.Thread(target=self._occupy_single_gpu, args=(gpu_id,))
            thread.start()
            threads.append(thread)
            print(f"▶️  启动GPU {gpu_id} 占用线程")
        
        # 等待所有占用完成
        for thread in threads:
            thread.join()
        
        print("✅ 所有GPU显存占用完成")
    
    def _occupy_single_gpu(self, gpu_id: int):
        """占用单个GPU的显存"""
        try:
            device = torch.device(f'cuda:{gpu_id}')
            
            # 计算需要分配的元素数量
            bytes_needed = int(self.memory_per_gpu * 1024 * 1024 * 1024)
            elements_needed = bytes_needed // 4  # float32每个元素4字节
            
            print(f"🔄 GPU {gpu_id}: 正在分配 {self.memory_per_gpu} GB 显存...")
            
            # 尝试一次性分配大块显存
            memory_tensors = []
            try:
                tensor = torch.randn(elements_needed, dtype=torch.float32, device=device)
                memory_tensors.append(tensor)
                torch.cuda.synchronize(device)
                
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                print(f"✅ GPU {gpu_id}: 成功分配 {allocated:.2f} GB 显存")
                
            except RuntimeError:
                # 如果一次性分配失败，尝试分块分配
                print(f"⚠️  GPU {gpu_id}: 大块分配失败，尝试分块分配...")
                self._allocate_in_chunks(gpu_id, device, elements_needed, memory_tensors)
            
            # 保存占用器
            self.consumers[gpu_id] = {
                'device': device,
                'memory_tensors': memory_tensors,
                'occupied': True
            }
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} 占用失败: {e}")
            self.consumers[gpu_id] = {'occupied': False}
    
    def _allocate_in_chunks(self, gpu_id: int, device, total_elements: int, memory_tensors: list):
        """分块分配显存"""
        chunk_size = total_elements // 20  # 分成20块
        allocated_elements = 0
        
        while allocated_elements < total_elements:
            try:
                current_chunk = min(chunk_size, total_elements - allocated_elements)
                if current_chunk <= 0:
                    break
                    
                tensor = torch.randn(current_chunk, dtype=torch.float32, device=device)
                memory_tensors.append(tensor)
                allocated_elements += current_chunk
                
                allocated_gb = (allocated_elements * 4) / (1024**3)
                if allocated_elements % (chunk_size * 5) == 0:  # 每5块报告一次
                    print(f"📦 GPU {gpu_id}: 已分配 {allocated_gb:.2f} GB")
                
            except RuntimeError as e:
                print(f"⚠️  GPU {gpu_id}: 无法分配更多显存: {e}")
                break
        
        total_allocated = (allocated_elements * 4) / (1024**3)
        print(f"🎯 GPU {gpu_id}: 最终分配 {total_allocated:.2f} GB")
    
    def heavy_computation_all_gpus(self):
        """在所有GPU上执行持续计算"""
        print(f"🔥 在所有GPU上启动持续计算，预计运行 {self.total_runtime} 秒...")
        
        threads = []
        for gpu_id in self.gpu_ids:
            if gpu_id in self.consumers and self.consumers[gpu_id].get('occupied', False):
                thread = threading.Thread(target=self._single_gpu_computation, args=(gpu_id,))
                thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
                thread.start()
                threads.append(thread)
                print(f"▶️  GPU {gpu_id} 计算线程启动")
        
        return threads
    
    def _single_gpu_computation(self, gpu_id: int):
        """单个GPU的持续计算任务"""
        device = torch.device(f'cuda:{gpu_id}')
        start_time = time.time()
        iteration = 0
        
        # 计算配置
        matrix_size = 2048  # 适中的矩阵大小
        batch_size = 512
        
        try:
            while time.time() - start_time < self.total_runtime:
                # 执行密集计算
                self._perform_matrix_operations(device, matrix_size, batch_size)
                
                iteration += 1
                if iteration % 50 == 0:  # 每50次迭代报告一次
                    elapsed = time.time() - start_time
                    remaining = self.total_runtime - elapsed
                    utilization = torch.cuda.utilization(gpu_id)
                    
                    print(f"🔄 GPU {gpu_id}: {iteration} 次迭代 | "
                          f"已运行: {elapsed:.0f}s | 剩余: {remaining:.0f}s | "
                          f"利用率: {utilization}%")
                
                # 短暂休息避免过热
                if iteration % 200 == 0:
                    time.sleep(0.1)
                        
        except Exception as e:
            print(f"❌ GPU {gpu_id} 计算错误: {e}")
    
    def _perform_matrix_operations(self, device, matrix_size: int, batch_size: int):
        """执行矩阵运算来保持GPU活跃"""
        # 大规模矩阵乘法
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
        c = torch.matmul(a, b)
        
        # 激活函数计算
        d = torch.nn.functional.relu(c)
        e = torch.nn.functional.softmax(d, dim=1)
        
        # 更多的数学运算
        f = torch.sin(e) + torch.cos(e)
        g = torch.exp(f) * torch.log(f + 1e-8)
        
        # 确保计算完成
        torch.cuda.synchronize(device)
        
        return g
    
    def monitor_gpus(self):
        """监控所有GPU状态"""
        print("\n📊 开始监控GPU状态...")
        print("=" * 60)
        
        start_time = time.time()
        last_report_time = start_time
        
        try:
            while time.time() - start_time < self.total_runtime:
                current_time = time.time()
                
                # 每30秒报告一次详细状态
                if current_time - last_report_time >= 30:
                    self._print_detailed_status(start_time)
                    last_report_time = current_time
                
                time.sleep(5)  # 每5秒检查一次
                
        except KeyboardInterrupt:
            print("\n⏹️  监控被用户中断")
    
    def _print_detailed_status(self, start_time: float):
        """打印详细的GPU状态"""
        elapsed = time.time() - start_time
        remaining = self.total_runtime - elapsed
        progress = (elapsed / self.total_runtime) * 100
        
        print(f"\n📈 运行状态报告 [已运行: {elapsed:.0f}s | 剩余: {remaining:.0f}s | 进度: {progress:.2f}%]")
        print("-" * 50)
        
        for gpu_id in self.gpu_ids:
            if gpu_id in self.consumers:
                device = torch.device(f'cuda:{gpu_id}')
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                utilization = torch.cuda.utilization(gpu_id)
                
                # 获取GPU温度（如果可用）
                try:
                    temperature = torch.cuda.get_device_properties(gpu_id).temperature
                    temp_str = f" | 温度: {temperature}°C" if temperature else ""
                except:
                    temp_str = ""
                
                print(f"GPU {gpu_id}: {allocated:.1f}GB/{reserved:.1f}GB | 利用率: {utilization}%{temp_str}")
        
        print("-" * 50)
    
    def print_system_info(self):
        """打印系统GPU信息"""
        print("\n💻 系统GPU信息汇总:")
        print("=" * 60)
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3) if i in self.consumers else 0
            
            status = "✅ 已占用" if i in self.gpu_ids else "⚪ 未占用"
            print(f"GPU {i}: {props.name[:30]:30} | {total_memory:5.1f} GB | {allocated:5.1f} GB | {status}")
        
        print("=" * 60)
    
    def cleanup(self):
        """清理所有GPU显存"""
        print("\n🧹 开始清理GPU显存...")
        
        for gpu_id in self.gpu_ids:
            try:
                if gpu_id in self.consumers:
                    # 清空张量列表
                    if 'memory_tensors' in self.consumers[gpu_id]:
                        self.consumers[gpu_id]['memory_tensors'].clear()
                    
                    # 清空GPU缓存
                    torch.cuda.empty_cache()
                    
                    device = torch.device(f'cuda:{gpu_id}')
                    allocated = torch.cuda.memory_allocated(device) / (1024**3)
                    print(f"✅ GPU {gpu_id}: 清理完成，剩余 {allocated:.2f} GB")
                    
            except Exception as e:
                print(f"❌ GPU {gpu_id} 清理失败: {e}")
        
        print("✅ 所有GPU清理完成")

def main():
    """主函数"""
    # 创建多GPU占用器
    consumer = MultiGPUMemoryConsumer()
    
    if not consumer._check_gpu_availability():
        return
    
    try:
        # 1. 占用GPU显存
        consumer.occupy_all_gpus()
        
        # 2. 显示系统信息
        consumer.print_system_info()
        
        # 3. 启动持续计算
        computation_threads = consumer.heavy_computation_all_gpus()
        
        # 4. 显示控制信息
        print(f"\n🎯 程序正在运行，占用GPU {consumer.gpu_ids}")
        print("💡 按 Ctrl+C 可安全退出程序")
        print("⏰ 程序将自动运行直到完成或手动停止")
        print("=" * 60)
        
        # 5. 启动监控
        consumer.monitor_gpus()
        
    except KeyboardInterrupt:
        print("\n⏹️  用户请求停止程序")
    except Exception as e:
        print(f"\n❌ 程序运行错误: {e}")
    finally:
        # 清理资源
        print("\n正在关闭程序...")
        consumer.cleanup()
        print("👋 程序已退出")

if __name__ == "__main__":
    # 直接运行，无需命令行参数
    main()