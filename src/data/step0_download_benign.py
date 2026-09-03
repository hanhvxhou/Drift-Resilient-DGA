"""
src/data/step0_download_benign.py
──────────────────────────────────
Step 0 (chạy trước step 1): Tải dữ liệu benign (Tranco) cho từng năm 2018-2023
và lưu vào data/raw/benign/tranco_YYYY.csv

CƠ CHẾ HOẠT ĐỘNG:
─────────────────────────────────────────────────────────────────────────────
  Tranco cung cấp list theo ngày từ API/Python package.
  Chiến lược: với mỗi năm YYYY, lấy snapshot ngày 01-07-YYYY (giữa năm)
  → đại diện phân phối tên miền trong năm đó.

  Đặc biệt:
  - Năm 2018: Tranco chưa tồn tại (ra đời 02/2019).
              → Tự động dùng snapshot sớm nhất có thể (2019-03-01) làm proxy.
              → Hoặc người dùng tự cung cấp alexa_2018.csv thủ công.
  - Năm 2019: snapshot từ 2019-07-01 (Tranco đã ổn định sau launch).
  - Năm 2020-2023: snapshot ngày 01-07-YYYY.

LUỒNG XỬ LÝ SAU KHI TẢI:
─────────────────────────────────────────────────────────────────────────────
  1. Tải Top 1M từ Tranco cho ngày đại diện của từng năm.
  2. Lọc: loại domain chứa ký tự không hợp lệ, độ dài < 4 hoặc > 63 ký tự.
  3. Lọc: loại domain có TLD trong blacklist (onion, local, internal...).
  4. Lưu: data/raw/benign/tranco_YYYY.csv (cột duy nhất: "domain").
  5. In thống kê: số domain, ngày snapshot, list_id để ghi vào paper.

YÊU CẦU:
    pip install tranco

Usage:
    python -m src.data.step0_download_benign
    python -m src.data.step0_download_benign --years 2020 2021 2022
    python -m src.data.step0_download_benign --dry-run   # chỉ in kế hoạch
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from src.utils.common import get_logger, load_config

# ── Cấu hình ngày đại diện cho từng năm ──────────────────────────────────────
# Tranco ra đời 2019-02-19. Năm 2018 → dùng snapshot 2019-03-01 làm proxy.
YEAR_TO_DATE: dict[int, str] = {
    2018: "2019-03-01",   # Proxy: Tranco chưa tồn tại năm 2018
    2019: "2019-07-01",
    2020: "2020-07-01",
    2021: "2021-07-01",
    2022: "2022-07-01",
    2023: "2023-07-01",
    2024: "2024-07-01",
    2025: "2025-04-01",   # 2025: chỉ có Q1+Q2, dùng đầu Q2
}

# TLD không hợp lệ làm benign
INVALID_TLDS = {
    "onion", "local", "localhost", "internal", "invalid",
    "example", "test", "home", "lan",
}

# Regex: domain hợp lệ (đơn giản hóa)
DOMAIN_RE = re.compile(r'^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)+$')


# ── Validator ─────────────────────────────────────────────────────────────────
def is_valid_domain(domain: str) -> bool:
    """Trả về True nếu domain có cú pháp hợp lệ và TLD không trong blacklist."""
    if not domain or len(domain) < 4 or len(domain) > 253:
        return False
    parts = domain.lower().split(".")
    tld   = parts[-1]
    if tld in INVALID_TLDS:
        return False
    if not DOMAIN_RE.match(domain.lower()):
        return False
    return True


def clean_domains(domains: list[str]) -> list[str]:
    """Lowercase, strip, bỏ domain không hợp lệ, bỏ trùng."""
    cleaned = [d.strip().lower() for d in domains]
    cleaned = [d for d in cleaned if is_valid_domain(d)]
    # Giữ thứ tự (rank) và loại trùng
    seen: set[str] = set()
    result = []
    for d in cleaned:
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


# ── Downloader ────────────────────────────────────────────────────────────────
def download_year(year: int, out_dir: Path, logger,
                  top_n: int = 1_000_000,
                  retry: int = 3,
                  dry_run: bool = False) -> dict:
    """
    Tải Tranco Top-N cho năm YYYY.
    Trả về dict với metadata để ghi vào paper.
    """
    try:
        from tranco import Tranco  # noqa
    except ImportError:
        raise ImportError("Cài đặt thư viện: pip install tranco")

    date_str = YEAR_TO_DATE[year]
    out_path = out_dir / f"tranco_{year}.csv"

    # Cảnh báo đặc biệt cho 2018
    if year == 2018:
        logger.warning(
            f"  NĂM 2018: Tranco chưa tồn tại (ra đời 02/2019).\n"
            f"  → Dùng snapshot {date_str} làm proxy.\n"
            f"  → Để chính xác hơn: tự cung cấp data/raw/benign/alexa_2018.csv\n"
            f"     (Alexa Top 1M năm 2018, tải từ Internet Archive)."
        )

    if out_path.exists():
        logger.info(f"  {year}: Đã tồn tại → bỏ qua ({out_path.name})")
        existing = pd.read_csv(out_path)
        return {"year": year, "date": date_str, "n_domains": len(existing),
                "list_id": "cached", "out_file": str(out_path)}

    if dry_run:
        logger.info(f"  [DRY-RUN] {year}: sẽ tải Tranco ngày {date_str} (top {top_n:,})")
        return {"year": year, "date": date_str, "n_domains": top_n,
                "list_id": "N/A (dry-run)", "out_file": str(out_path)}

    logger.info(f"  {year}: Đang tải Tranco ngày {date_str} (top {top_n:,}) ...")
    t_client = Tranco(cache=True,
                      cache_dir=str(out_dir.parent / ".tranco_cache"))

    last_err = None
    for attempt in range(1, retry + 1):
        try:
            tranco_list = t_client.list(date=date_str)
            list_id     = tranco_list.list_id
            domains_raw = tranco_list.top(top_n)    # list of strings, ordered by rank
            break
        except Exception as e:
            last_err = e
            logger.warning(f"    Lần thử {attempt}/{retry} thất bại: {e}")
            if attempt < retry:
                time.sleep(5 * attempt)
    else:
        raise RuntimeError(f"Không thể tải Tranco {year} sau {retry} lần: {last_err}")

    # Làm sạch
    before = len(domains_raw)
    domains = clean_domains(domains_raw)
    removed = before - len(domains)
    if removed:
        logger.info(f"    Lọc bỏ {removed:,} domain không hợp lệ / trùng")

    # Lưu
    df = pd.DataFrame({"domain": domains})
    df.to_csv(out_path, index=False)

    logger.info(
        f"  {year}: ✓  {len(df):,} domain  "
        f"| ngày snapshot: {date_str}  "
        f"| list_id: {list_id}  "
        f"| → {out_path.name}"
    )
    return {"year": year, "date": date_str, "n_domains": len(df),
            "list_id": list_id, "out_file": str(out_path)}


# ── Hướng dẫn tải Alexa 2018 thủ công ───────────────────────────────────────
def print_alexa_2018_guide(out_dir: Path) -> None:
    guide = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  HƯỚNG DẪN TẢI DỮ LIỆU BENIGN CHO NĂM 2018 (THỦ CÔNG)            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Tranco chưa tồn tại năm 2018 → 2 lựa chọn:                        ║
║                                                                      ║
║  OPTION A (đơn giản, đủ dùng):                                      ║
║    Script đã dùng snapshot Tranco 2019-03-01 làm proxy.             ║
║    File: {str(out_dir / 'tranco_2018.csv'):52s} ║
║    Ghi chú trong paper: "2018 benign domains sourced from           ║
║    Tranco snapshot 2019-03-01 (earliest available)"                 ║
║                                                                      ║
║  OPTION B (chính xác hơn — Alexa Top 1M từ Internet Archive):      ║
║    1. Truy cập:                                                      ║
║       https://web.archive.org/web/20180701000000*/s3.amazonaws.com/ ║
║       alexa-static/top-1m.csv.zip                                   ║
║    2. Tìm snapshot gần ngày 2018-07-01 nhất                         ║
║    3. Tải file ZIP, giải nén → top-1m.csv                           ║
║       Format: rank,domain  (ví dụ: 1,google.com)                   ║
║    4. Chạy:                                                          ║
║       python -m src.data.step0_download_benign --convert-alexa      ║
║              --alexa-file /path/to/top-1m.csv --year 2018           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(guide)


# ── Convert Alexa CSV sang format chuẩn ──────────────────────────────────────
def convert_alexa(alexa_file: Path, year: int, out_dir: Path, logger) -> None:
    """
    Chuyển đổi Alexa Top-1M CSV (format: rank,domain) sang
    data/raw/benign/tranco_YYYY.csv (format: domain).
    Ghi đè nếu file đã tồn tại.
    """
    logger.info(f"Đang chuyển đổi Alexa file: {alexa_file}")
    df = pd.read_csv(alexa_file, header=None, names=["rank", "domain"])
    df["domain"] = df["domain"].str.strip().str.lower()
    domains = clean_domains(df["domain"].tolist())
    out_path = out_dir / f"tranco_{year}.csv"
    pd.DataFrame({"domain": domains}).to_csv(out_path, index=False)
    logger.info(f"  ✓ {len(domains):,} domain → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(cfg: dict,
        years: list[int] | None = None,
        dry_run: bool = False,
        alexa_file: str | None = None,
        alexa_year: int | None = None) -> None:

    log_dir = Path(cfg["paths"]["results"]) / "logs"
    logger  = get_logger("step0_download_benign", log_dir=log_dir)
    out_dir = Path(cfg["paths"]["benign_raw"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mode: convert Alexa file
    if alexa_file and alexa_year:
        convert_alexa(Path(alexa_file), alexa_year, out_dir, logger)
        return

    target_years = years or list(YEAR_TO_DATE.keys())
    logger.info(f"Sẽ tải Tranco cho các năm: {target_years}")
    logger.info(f"Thư mục lưu: {out_dir}")
    if dry_run:
        logger.info("[DRY-RUN mode] — không tải thực tế")
    logger.info("")

    results = []
    for year in target_years:
        if year not in YEAR_TO_DATE:
            logger.warning(f"  Năm {year} không nằm trong phạm vi 2018-2023, bỏ qua.")
            continue
        try:
            meta = download_year(year, out_dir, logger, dry_run=dry_run)
            results.append(meta)
        except Exception as e:
            logger.error(f"  Năm {year} THẤT BẠI: {e}")

    # In bảng tóm tắt
    logger.info("\n" + "=" * 65)
    logger.info("  TÓM TẮT — dữ liệu benign đã tải")
    logger.info("=" * 65)
    logger.info(f"  {'Năm':>6}  {'Ngày snapshot':>14}  {'Tranco ID':>10}  {'Số domain':>12}")
    logger.info("  " + "-" * 60)
    for r in results:
        logger.info(
            f"  {r['year']:>6}  {r['date']:>14}  {r['list_id']:>10}  {r['n_domains']:>12,}"
        )
    logger.info("=" * 65)

    # Ghi metadata để cite trong paper
    meta_path = out_dir / "benign_metadata.csv"
    pd.DataFrame(results).to_csv(meta_path, index=False)
    logger.info(f"\nMetadata để cite trong paper → {meta_path}")

    # Hướng dẫn 2018 nếu cần
    if 2018 in target_years and not dry_run:
        print_alexa_2018_guide(out_dir)

    logger.info("\nStep 0 complete ✓")
    logger.info("Tiếp theo: python -m src.data.run_pipeline --stub")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 0: Tải dữ liệu benign (Tranco) cho từng năm 2018-2023",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Tải tất cả các năm:
  python -m src.data.step0_download_benign

  # Chỉ tải năm 2021, 2022:
  python -m src.data.step0_download_benign --years 2021 2022

  # Xem kế hoạch mà không tải:
  python -m src.data.step0_download_benign --dry-run

  # Chuyển đổi file Alexa 2018 thủ công:
  python -m src.data.step0_download_benign --convert-alexa \\
         --alexa-file /path/to/top-1m.csv --year 2018
        """
    )
    parser.add_argument("--config",  default=None, help="Đường dẫn config.yaml")
    parser.add_argument("--years",   nargs="+", type=int, default=None,
                        help="Các năm cần tải (mặc định: 2018-2023)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in kế hoạch, không tải thực tế")
    parser.add_argument("--convert-alexa", action="store_true",
                        help="Chuyển đổi file Alexa CSV sang format chuẩn")
    parser.add_argument("--alexa-file",    default=None,
                        help="Đường dẫn file Alexa top-1m.csv")
    parser.add_argument("--year",          type=int, default=None,
                        help="Năm cho file Alexa (dùng với --convert-alexa)")

    args = parser.parse_args()
    cfg  = load_config(args.config)

    if args.convert_alexa:
        if not args.alexa_file or not args.year:
            parser.error("--convert-alexa yêu cầu --alexa-file và --year")
        run(cfg, alexa_file=args.alexa_file, alexa_year=args.year)
    else:
        run(cfg, years=args.years, dry_run=args.dry_run)
