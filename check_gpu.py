"""
check_gpu.py — Kiem tra moi truong GPU va chay thu forward pass
Chay: python check_gpu.py
"""

import sys
import subprocess

# ── 1. Python version ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  DGA CNN — GPU Environment Check")
print(f"{'='*60}")
print(f"\n[1] Python : {sys.version}")

# ── 2. PyTorch ───────────────────────────────────────────────────────────────
try:
    import torch
    print(f"[2] PyTorch: {torch.__version__}")
    print(f"    CUDA build version : {torch.version.cuda}")
    print(f"    cuDNN version      : {torch.backends.cudnn.version()}")
except ImportError:
    print("[2] PyTorch: NOT INSTALLED")
    print("    --> Chay: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# ── 3. CUDA availability ─────────────────────────────────────────────────────
print(f"\n[3] CUDA available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("""
    NGUYEN NHAN CO THE:
    a) Chua cai NVIDIA driver   -> tai tai https://www.nvidia.com/drivers
    b) PyTorch build sai CUDA   -> xem buoc 4 de reinstall
    c) GPU khong ho tro CUDA    -> kiem tra ten GPU ben duoi
    """)

# ── 4. GPU info (nvidia-smi) ─────────────────────────────────────────────────
print("[4] nvidia-smi output:")
try:
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,driver_version,memory.total,compute_cap",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        for i, line in enumerate(result.stdout.strip().splitlines()):
            name, driver, mem, cc = [x.strip() for x in line.split(",")]
            print(f"    GPU {i}: {name}")
            print(f"           Driver  : {driver}")
            print(f"           VRAM    : {mem}")
            print(f"           Compute : {cc}")
    else:
        print("    nvidia-smi khong chay duoc — driver chua cai hoac PATH chua co.")
except FileNotFoundError:
    print("    nvidia-smi not found — NVIDIA driver chua duoc cai.")
except subprocess.TimeoutExpired:
    print("    nvidia-smi timeout.")

# ── 5. PyTorch GPU devices ───────────────────────────────────────────────────
print(f"\n[5] torch.cuda.device_count() = {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"    GPU {i}: {p.name}  |  VRAM: {p.total_memory/1024**3:.1f} GB  |  SM: {p.major}.{p.minor}")

# ── 6. Forward pass test ─────────────────────────────────────────────────────
print(f"\n[6] Forward pass test:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Running on: {device}")

try:
    from model import DGACnn
    model = DGACnn().to(device)
    dummy = torch.zeros(32, 1, 8, 73, device=device)

    import time
    # warm-up
    with torch.no_grad():
        _ = model(dummy)

    # benchmark
    N = 100
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            out = model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = (time.perf_counter() - t0) / N * 1000

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model params  : {params:,}")
    print(f"    Input shape   : {dummy.shape}")
    print(f"    Output shape  : {out.shape}")
    print(f"    Avg latency   : {elapsed:.2f} ms / batch (batch=32, x{N} runs)")
    throughput = 32 / (elapsed / 1000)
    print(f"    Throughput    : {throughput:,.0f} samples/sec")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 7. Reinstall command ─────────────────────────────────────────────────────
if not torch.cuda.is_available():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0]
            _, cc = [x.strip() for x in line.split(",")]
            major = int(cc.split(".")[0])
            # Goi y phien ban CUDA phu hop
            if major >= 8:    cuda_tag = "cu121"
            elif major >= 7:  cuda_tag = "cu118"
            else:             cuda_tag = "cu117"
            print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║  GPU phat hien nhung CUDA chua hoat dong             ║
  ║  Chay lenh sau de cai lai PyTorch dung CUDA:         ║
  ║                                                      ║
  ║  pip uninstall torch torchvision -y                  ║
  ║  pip install torch torchvision \\                    ║
  ║      --index-url https://download.pytorch.org/whl/{cuda_tag} ║
  ╚══════════════════════════════════════════════════════╝
            """)
    except Exception:
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║  Cai NVIDIA driver truoc, sau do chay lai script nay ║
  ║  Tai driver: https://www.nvidia.com/drivers          ║
  ╚══════════════════════════════════════════════════════╝
        """)
else:
    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║  GPU san sang! Chay training:                        ║
  ║                                                      ║
  ║  python train.py --train train.csv \\                ║
  ║                  --val   val.csv   \\                ║
  ║                  --test  test.csv                    ║
  ╚══════════════════════════════════════════════════════╝
    """)

print("="*60)
