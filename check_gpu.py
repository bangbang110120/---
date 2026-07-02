import torch

print("=" * 50)
print("PyTorch GPU 检测")
print("=" * 50)

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.current_device()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA 不可用，使用 CPU 训练")
    print("\n可能原因:")
    print("  1. 安装的是 CPU 版本的 PyTorch")
    print("  2. CUDA 版本与 PyTorch 不匹配")
    print("  3. NVIDIA 驱动未安装或版本过低")

print("=" * 50)