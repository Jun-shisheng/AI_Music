"""Quick CUDA/GPU health check."""
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    # simple matmul on GPU
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("matmul OK, sum:", float(c.sum()))
    # allocate/free loop to test stability
    for i in range(5):
        x = torch.randn(1024, 1024, device="cuda")
        del x
        torch.cuda.empty_cache()
    print("alloc/free loop OK")
