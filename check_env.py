#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_env.py
============
Kiểm tra cấu hình máy + thư viện phục vụ train.py

Cách dùng:
    python check_env.py

Không cần cài thêm gì — chỉ dùng thư viện standard Python.
(psutil là optional, nếu có sẽ cho thông tin RAM chính xác hơn)
"""

import platform, sys, os, subprocess, multiprocessing

SEP  = "=" * 70
SEP2 = "-" * 70

def run(cmd, shell=True):
    try:
        return subprocess.check_output(
            cmd, shell=shell, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "N/A"

def gb(n_bytes):
    return f"{n_bytes / 1024**3:.1f} GB"

# ── 1. Hệ điều hành ──────────────────────────────────────────────────────────
print(SEP)
print("  HỆ ĐIỀU HÀNH")
print(SEP)
print(f"  OS            : {platform.system()} {platform.release()}")
print(f"  Version       : {platform.version()[:80]}")
print(f"  Machine       : {platform.machine()}")
print(f"  Hostname      : {platform.node()}")
print(f"  Architecture  : {' / '.join(platform.architecture())}")

# ── 2. CPU ────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  CPU")
print(SEP)
print(f"  Logical cores : {multiprocessing.cpu_count()}")

# Windows
if platform.system() == "Windows":
    cpu_w = run('wmic cpu get Name /value').replace("Name=", "").strip()
    cores_w = run('wmic cpu get NumberOfCores /value').replace("NumberOfCores=", "").strip()
    threads_w = run('wmic cpu get ThreadCount /value').replace("ThreadCount=", "").strip()
    freq_w = run('wmic cpu get MaxClockSpeed /value').replace("MaxClockSpeed=", "").strip()
    if cpu_w and cpu_w != "N/A":
        print(f"  Model         : {cpu_w}")
        print(f"  Cores/Threads : {cores_w} / {threads_w}")
        if freq_w and freq_w.isdigit():
            print(f"  Max freq      : {int(freq_w)/1000:.2f} GHz")
# Linux
elif platform.system() == "Linux":
    cpu_l = run("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2")
    phy   = run("grep 'cpu cores'  /proc/cpuinfo | head -1 | cut -d: -f2")
    freq  = run("grep 'cpu MHz'    /proc/cpuinfo | head -1 | cut -d: -f2")
    print(f"  Model         : {cpu_l.strip()}")
    if phy.strip(): print(f"  Physical cores: {phy.strip()}")
    if freq.strip(): print(f"  Current freq  : {float(freq.strip())/1000:.2f} GHz")
# macOS
elif platform.system() == "Darwin":
    print(f"  Model         : {run('sysctl -n machdep.cpu.brand_string')}")
    print(f"  Physical cores: {run('sysctl -n hw.physicalcpu')}")

# ── 3. RAM ────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  RAM")
print(SEP)
try:
    import psutil
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    print(f"  Total         : {gb(vm.total)}")
    print(f"  Available     : {gb(vm.available)}")
    print(f"  Used          : {gb(vm.used)}  ({vm.percent:.1f}%)")
    print(f"  Swap total    : {gb(sw.total)}")
    print(f"  Swap used     : {gb(sw.used)}")
except ImportError:
    if platform.system() == "Windows":
        mem_w = run('wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value')
        print(f"  (wmic)        : {mem_w.replace(chr(10),' ').strip()}")
    else:
        print(f"  free -h       : {run('free -h | grep Mem')}")
    print("  (cài psutil để xem chi tiết hơn: pip install psutil)")

# ── 4. Disk ───────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  DISK")
print(SEP)
try:
    import psutil
    printed = set()
    for part in psutil.disk_partitions(all=False):
        try:
            u = psutil.disk_usage(part.mountpoint)
            if part.device in printed or u.total == 0:
                continue
            printed.add(part.device)
            bar_len = 20
            used_ratio = u.percent / 100
            filled = int(bar_len * used_ratio)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  {part.device:<25} {part.mountpoint:<12} "
                  f"[{bar}] {u.percent:5.1f}%  "
                  f"Total:{u.total/1024**3:.0f}G  Free:{u.free/1024**3:.0f}G")
        except (PermissionError, OSError):
            pass
except ImportError:
    if platform.system() == "Windows":
        print(f"  {run('wmic logicaldisk get DeviceID,Size,FreeSpace /format:list')}")
    else:
        for line in run("df -h").splitlines()[1:]:
            print(f"  {line}")

# ── 5. GPU (NVIDIA) ───────────────────────────────────────────────────────────
print()
print(SEP)
print("  GPU (NVIDIA) — nvidia-smi")
print(SEP)

# Tìm nvidia-smi (Windows để ở chỗ khác)
nvidia_paths = [
    "nvidia-smi",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
]
smi_cmd = None
for p in nvidia_paths:
    try:
        out = subprocess.check_output([p, "--version"], stderr=subprocess.DEVNULL, text=True)
        if out:
            smi_cmd = p
            break
    except Exception:
        pass

if smi_cmd:
    cols = ("name,driver_version,cuda_version,memory.total,memory.free,"
            "memory.used,utilization.gpu,temperature.gpu,power.draw,"
            "clocks.gr,clocks.mem")
    raw = run(f'"{smi_cmd}" --query-gpu={cols} --format=csv,noheader,nounits')
    if raw and raw != "N/A":
        for i, line in enumerate(raw.splitlines()):
            p = [x.strip() for x in line.split(",")]
            if len(p) < 11: continue
            name, drv, cuda_v, mtot, mfree, mused, util, temp, pwr, clk_gr, clk_mem = p[:11]
            print(f"  GPU {i}          : {name}")
            print(f"  Driver        : {drv}    CUDA: {cuda_v}")
            try:
                print(f"  VRAM          : {int(mtot)/1024:.2f} GB total  |  "
                      f"{int(mfree)/1024:.2f} GB free  |  "
                      f"{int(mused)/1024:.2f} GB used")
            except ValueError:
                print(f"  VRAM          : total={mtot} MB  free={mfree} MB  used={mused} MB")
            print(f"  Utilization   : {util}%")
            print(f"  Temperature   : {temp}°C")
            print(f"  Power draw    : {pwr} W")
            print(f"  Clocks        : GPU {clk_gr} MHz  |  MEM {clk_mem} MHz")
            print()
else:
    print("  nvidia-smi không tìm thấy.")
    print("  → Nếu có GPU NVIDIA, thử cài CUDA Toolkit hoặc kiểm tra PATH.")

# CUDA qua PyTorch
print(SEP2)
print("  CUDA qua PyTorch")
print(SEP2)
try:
    import torch
    print(f"  PyTorch ver   : {torch.__version__}")
    print(f"  CUDA avail    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA ver      : {torch.version.cuda}")
        print(f"  cuDNN ver     : {torch.backends.cudnn.version()}")
        n_dev = torch.cuda.device_count()
        for i in range(n_dev):
            prop = torch.cuda.get_device_properties(i)
            vram = prop.total_memory / 1024**3
            has_tc = prop.major >= 7
            has_fp16 = prop.major >= 6
            bf16_ok = torch.cuda.is_bf16_supported() if i == 0 else "?"
            print(f"\n  [Device {i}]")
            print(f"    Name            : {prop.name}")
            print(f"    VRAM            : {vram:.2f} GB")
            print(f"    Compute cap     : {prop.major}.{prop.minor}")
            print(f"    Multiprocessors : {prop.multi_processor_count}")
            print(f"    Tensor Cores    : {'YES ✓' if has_tc else 'NO'}"
                  f"  (Volta = 7.0+, cần cho AMP FP16 nhanh)")
            print(f"    FP16 support    : {'YES ✓' if has_fp16 else 'NO'}")
            print(f"    BF16 support    : {'YES ✓' if bf16_ok else 'NO'}")
    else:
        print()
        print("  ⚠  Không có CUDA. Train sẽ dùng CPU — rất chậm với BERT.")
        print("     Kiểm tra: CUDA Toolkit đã cài? PyTorch có đúng CUDA build?")
        print("     Cài lại: https://pytorch.org/get-started/locally/")
except ImportError:
    print("  torch CHƯA cài  →  pip install torch torchvision --index-url "
          "https://download.pytorch.org/whl/cu121")

# ── 6. Python ─────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  PYTHON")
print(SEP)
print(f"  Version       : {sys.version}")
print(f"  Executable    : {sys.executable}")
print(f"  Prefix        : {sys.prefix}")
# Kiểm tra venv / conda
in_venv  = hasattr(sys, "real_prefix") or (
           hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
in_conda = os.environ.get("CONDA_DEFAULT_ENV", "")
print(f"  Virtual env   : {'YES — ' + sys.prefix if in_venv else 'NO (đang dùng system Python)'}")
if in_conda:
    print(f"  Conda env     : {in_conda}")

# ── 7. Thư viện ──────────────────────────────────────────────────────────────
print()
print(SEP)
print("  THƯ VIỆN (cho train.py)")
print(SEP)

LIBS = [
    # (import_name,   pip_install_name,              loại,      ghi_chú)
    ("torch",         "torch",                       "REQUIRED", "Deep learning core"),
    ("transformers",  "transformers",                "REQUIRED", "BERT / SecureBERT / DistilBERT"),
    ("sklearn",       "scikit-learn",                "REQUIRED", "F1 / AUC / metrics"),
    ("tqdm",          "tqdm",                        "REQUIRED", "Progress bar mỗi epoch"),
    ("numpy",         "numpy",                       "REQUIRED", "Array ops"),
    ("codecarbon",    "codecarbon",                  "OPTIONAL", "Đo điện năng (kWh)"),
    ("psutil",        "psutil",                      "OPTIONAL", "RAM / CPU monitor"),
    ("accelerate",    "accelerate",                  "OPTIONAL", "AMP / multi-GPU"),
    ("tokenizers",    "tokenizers",                  "INFO",     "Cài cùng transformers"),
    ("huggingface_hub","huggingface-hub",             "INFO",     "Cài cùng transformers"),
    ("pandas",        "pandas",                      "OPTIONAL", "Data utils"),
]

any_missing_required = False
rows = []
for imp, pip, kind, note in LIBS:
    try:
        mod = __import__(imp)
        ver = getattr(mod, "__version__", "?")
        rows.append((imp, ver, "✓  OK", kind, note, pip))
    except ImportError:
        rows.append((imp, "—", f"✗  pip install {pip}", kind, note, pip))
        if kind == "REQUIRED":
            any_missing_required = True

# Header
print(f"  {'Thư viện':<22} {'Version':<15} {'Trạng thái':<35} {'Loại':<10} Ghi chú")
print("  " + SEP2)
for imp, ver, status, kind, note, pip in rows:
    color_mark = "  " if "✓" in status else "→ "
    print(f"  {color_mark}{imp:<20} {ver:<15} {status:<35} {kind:<10} {note}")

# ── 8. Khả năng AMP ──────────────────────────────────────────────────────────
print()
print(SEP)
print("  ĐÁNH GIÁ KHẢ NĂNG TĂNG TỐC AMP (Mixed Precision)")
print(SEP)
try:
    import torch
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        vram_gb = prop.total_memory / 1024**3
        cc = f"{prop.major}.{prop.minor}"

        fp16 = prop.major >= 7
        bf16 = torch.cuda.is_bf16_supported()

        print(f"  GPU           : {prop.name}  (CC {cc})")
        print(f"  VRAM          : {vram_gb:.1f} GB")
        print()
        print(f"  FP16 AMP      : {'✓ CÓ THỂ DÙNG' if fp16 else '✗ Không hỗ trợ (CC < 7.0)'}")
        print(f"  BF16 AMP      : {'✓ CÓ THỂ DÙNG (ổn định hơn FP16 cho BERT)' if bf16 else '✗ Không hỗ trợ'}")
        print()
        # Batch size recommendation
        if vram_gb >= 20:
            rec_bs = 128
        elif vram_gb >= 10:
            rec_bs = 64
        elif vram_gb >= 6:
            rec_bs = 32
        else:
            rec_bs = 16
        print(f"  Gợi ý BATCH_SIZE  : {rec_bs}  (với BERT-base, MAX_LEN=64)")
        print(f"  Gợi ý MAX_LEN     : 32  (domain ngắn, tiết kiệm 2x VRAM/tốc độ)")
        if fp16:
            print(f"  Tăng tốc ước tính : ~1.5-2x nếu bật AMP")
        print()

        # Ước tính thời gian train
        # BERT-base: ~110M params, 1 forward+backward ~0.3s/batch (batch=64, len=64, FP32, RTX 3090)
        # DistilBERT: ~66M, SecureBERT: ~125M
        n_train = 109_200  # từ log của anh/chị
        for mname, n_param, factor in [
            ("distilbert-base-uncased", 66, 0.7),
            ("bert-base-uncased",      110, 1.0),
            ("ehsanaghaei/SecureBERT", 125, 1.2),
        ]:
            bs_eff = min(rec_bs, 64)
            steps_per_epoch = n_train // bs_eff
            sec_per_step_fp32 = 0.35 * (vram_gb / 24) ** (-0.5) * factor  # rough
            sec_per_step_amp  = sec_per_step_fp32 * 0.55 if fp16 else sec_per_step_fp32
            ep_fp32 = steps_per_epoch * sec_per_step_fp32
            ep_amp  = steps_per_epoch * sec_per_step_amp
            print(f"  Ước tính / epoch [{mname[:28]:<28}]"
                  f"  FP32: ~{ep_fp32/60:.1f} min  |  AMP: ~{ep_amp/60:.1f} min")
    else:
        print("  Không có CUDA → train trên CPU, rất chậm.")
        print("  Với 109,200 mẫu × BERT-base, 1 epoch CPU ≈ 3-6 giờ.")
except ImportError:
    print("  torch chưa cài.")

# ── 9. Tổng kết & lệnh cài đặt ───────────────────────────────────────────────
print()
print(SEP)
print("  TỔNG KẾT & LỆNH CÀI ĐẶT")
print(SEP)
if any_missing_required:
    missing = [pip for imp, ver, status, kind, note, pip in rows
               if "✗" in status and kind == "REQUIRED"]
    print(f"  ✗ Còn thiếu thư viện REQUIRED: {', '.join(missing)}")
    print()
    print("  Lệnh cài đặt tất cả (chạy trong venv của anh/chị):")
    print()
    print("    # PyTorch với CUDA 12.1 (thay cu121 → cu118 nếu dùng CUDA 11.8):")
    print("    pip install torch torchvision --index-url "
          "https://download.pytorch.org/whl/cu121")
    print()
    print("    # Các thư viện còn lại:")
    print("    pip install transformers scikit-learn tqdm codecarbon "
          "accelerate psutil pandas")
else:
    print("  ✓ Tất cả thư viện REQUIRED đã cài đầy đủ.")
    optional_missing = [pip for imp, ver, status, kind, note, pip in rows
                        if "✗" in status and kind == "OPTIONAL"]
    if optional_missing:
        print(f"  ℹ Thư viện OPTIONAL chưa cài: {', '.join(optional_missing)}")
        print(f"    pip install {' '.join(optional_missing)}")
    print()
    print("  ✓ Sẵn sàng chạy:  python train.py")
print(SEP)

if __name__ == "__main__":
    pass
