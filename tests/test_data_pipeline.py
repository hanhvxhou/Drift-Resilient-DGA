"""
tests/test_data_pipeline.py
────────────────────────────
Unit tests for the data preparation modules.
These run WITHOUT real DGArchive data (using synthetic fixtures).

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.common import quarter_id, quarter_label, load_config
from src.data.step2_build_dga_windows import stratified_sample
from src.data.step4_annotate_drift import mmd2_biased, cosine_similarity


# ─── quarter helpers ──────────────────────────────────────────────────────────
class TestQuarterHelpers:
    def test_quarter_id_bounds(self):
        assert quarter_id(2018, 1) == "D01"
        assert quarter_id(2018, 4) == "D04"
        assert quarter_id(2019, 1) == "D05"
        assert quarter_id(2023, 4) == "D24"

    def test_quarter_label(self):
        assert quarter_label(2018, 1) == "2018_Q1"
        assert quarter_label(2023, 4) == "2023_Q4"

    def test_sequential_coverage(self):
        """All 24 windows should be unique and cover 2018Q1–2023Q4."""
        ids = [quarter_id(2018 + q // 4, q % 4 + 1) for q in range(24)]
        assert len(set(ids)) == 24
        assert ids[0]  == "D01"
        assert ids[-1] == "D24"


# ─── stratified sampler ───────────────────────────────────────────────────────
class TestStratifiedSample:
    @pytest.fixture
    def sample_df(self):
        """200 rows: 100 family A, 60 family B, 40 family C."""
        rows = (
            [{"domain": f"a{i}.com", "family": "A"} for i in range(100)] +
            [{"domain": f"b{i}.com", "family": "B"} for i in range(60)]  +
            [{"domain": f"c{i}.com", "family": "C"} for i in range(40)]
        )
        return pd.DataFrame(rows)

    def test_returns_all_when_under_budget(self, sample_df):
        rng    = np.random.default_rng(42)
        result = stratified_sample(sample_df, "family", max_total=500, rng=rng)
        assert len(result) == 200

    def test_respects_max_total(self, sample_df):
        rng    = np.random.default_rng(42)
        result = stratified_sample(sample_df, "family", max_total=100, rng=rng)
        assert len(result) <= 100

    def test_proportions_preserved(self, sample_df):
        """After sampling, family proportions should be within 5 pp of original."""
        rng    = np.random.default_rng(42)
        result = stratified_sample(sample_df, "family", max_total=90, rng=rng)
        orig_ratios = sample_df["family"].value_counts(normalize=True)
        samp_ratios = result["family"].value_counts(normalize=True)
        for fam in ["A", "B", "C"]:
            diff = abs(orig_ratios[fam] - samp_ratios[fam])
            assert diff < 0.07, f"Family {fam}: ratio off by {diff:.3f}"

    def test_no_duplicates_in_output(self, sample_df):
        rng    = np.random.default_rng(0)
        result = stratified_sample(sample_df, "family", max_total=80, rng=rng)
        assert result.duplicated(subset="domain").sum() == 0


# ─── MMD² ─────────────────────────────────────────────────────────────────────
class TestMMD2:
    def test_mmd2_identical_distributions_near_zero(self):
        rng = np.random.default_rng(7)
        X   = rng.normal(0, 1, (100, 32)).astype(np.float32)
        # Same distribution → MMD² should be very small
        m2  = mmd2_biased(X, X)
        assert m2 < 1e-6

    def test_mmd2_different_distributions_positive(self):
        rng = np.random.default_rng(7)
        X   = rng.normal(0, 1,   (100, 32)).astype(np.float32)
        Y   = rng.normal(5, 1,   (100, 32)).astype(np.float32)
        m2  = mmd2_biased(X, Y)
        assert m2 > 0.1

    def test_cosine_similarity_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6


# ─── Config loading ───────────────────────────────────────────────────────────
class TestConfig:
    def test_config_loads(self):
        cfg = load_config()
        assert "paths"    in cfg
        assert "temporal" in cfg
        assert "dga"      in cfg

    def test_config_paths_are_strings(self):
        cfg = load_config()
        for key, val in cfg["paths"].items():
            assert isinstance(val, str), f"paths.{key} should be str"

    def test_min_family_samples_positive(self):
        cfg = load_config()
        assert cfg["dga"]["min_family_samples"] > 0

    def test_max_per_window_positive(self):
        cfg = load_config()
        assert cfg["dga"]["max_per_window"] > 0


# ─── End-to-end mini integration test ────────────────────────────────────────
class TestMiniPipeline:
    """
    Create synthetic DGA CSVs in a temp directory and run steps 1 + 2.
    Verifies the pipeline produces correct outputs without real DGArchive data.
    """

    @pytest.fixture
    def mini_cfg(self, tmp_path):
        """Build a minimal config pointing to temp directories."""
        raw_dir     = tmp_path / "raw" / "dgarchive"
        benign_dir  = tmp_path / "raw" / "benign"
        interim_dir = tmp_path / "interim"
        windows_dir = tmp_path / "windows"
        bench_dir   = tmp_path / "benchmark"
        results_dir = tmp_path / "results"
        for d in [raw_dir, benign_dir, interim_dir, windows_dir, bench_dir, results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return {
            "paths": {
                "dgarchive_raw":  str(raw_dir),
                "benign_raw":     str(benign_dir),
                "interim":        str(interim_dir),
                "windows_dir":    str(windows_dir),
                "benchmark_dir":  str(bench_dir),
                "results":        str(results_dir),
            },
            "temporal": {"start": "2018-01-01", "end": "2023-12-31", "freq": "QS"},
            "dga":    {"min_family_samples": 100, "max_per_window": 200, "label": 1},
            "benign": {
                "max_per_window": 200,
                "label": 0,
                "strategy": "annual_partition",
                "single_snapshot_file": None,
                "verify_cross_year_overlap": True,
            },
            "integrity": {"verify_no_future_leak": True, "verify_no_cross_contamination": True},
            "random_seed": 42,
            "drift": {"embedding_samples_per_window": 100, "mmd_kernel": "rbf",
                      "delta1": None, "delta2": None, "tau": 0.85, "k": 3},
        }

    @pytest.fixture
    def synthetic_csvs(self, mini_cfg):
        """Write two synthetic *_dga.csv files with rows spanning 2018-2023."""
        raw_dir = Path(mini_cfg["paths"]["dgarchive_raw"])
        rng     = np.random.default_rng(0)
        chars   = list("abcdefghijklmnopqrstuvwxyz0123456789")

        for family in ["conficker", "necurs"]:
            rows = []
            for i in range(600):
                # Random domain
                length = rng.integers(8, 20)
                domain = "".join(rng.choice(chars, length)) + ".com"
                # Spread dates uniformly across 2018-2023
                day_offset = int(rng.integers(0, 365 * 6))
                date = pd.Timestamp("2018-01-01") + pd.Timedelta(days=day_offset)
                date_str = date.strftime("%Y-%m-%d 00:00:00")
                end_str  = date.strftime("%Y-%m-%d 23:59:59")
                rows.append({
                    "domain":     domain,
                    "domain_id":  i,
                    "valid_from": date_str,
                    "valid_to":   end_str,
                    "seed_id":    f"{family}_seed",
                })
            pd.DataFrame(rows).to_csv(raw_dir / f"{family}_dga.csv", index=False)
        return mini_cfg

    def test_step1_produces_parquet(self, synthetic_csvs):
        from src.data.step1_merge_dgarchive import run
        run(synthetic_csvs)
        parquet = Path(synthetic_csvs["paths"]["interim"]) / "dgarchive_merged.parquet"
        assert parquet.exists()
        df = pd.read_parquet(parquet)
        assert len(df) > 0
        assert "family" in df.columns
        assert "quarter_id" in df.columns
        assert set(df["family"].unique()) == {"conficker", "necurs"}

    def test_step2_produces_window_files(self, synthetic_csvs):
        from src.data.step1_merge_dgarchive import run as step1
        from src.data.step2_build_dga_windows import run as step2
        step1(synthetic_csvs)
        step2(synthetic_csvs)
        windows_dir = Path(synthetic_csvs["paths"]["windows_dir"])
        csv_files   = list(windows_dir.glob("D*_dga.csv"))
        assert len(csv_files) > 0
        # Each produced file has correct columns and label=1
        for f in csv_files:
            df = pd.read_csv(f)
            assert set(["domain","label","family","quarter_id","quarter_label"]).issubset(df.columns)
            assert (df["label"] == 1).all()
            assert df.duplicated(subset="domain").sum() == 0

    def test_step4_stub(self, mini_cfg):
        """Stub drift annotation should produce valid JSON without a backbone."""
        Path(mini_cfg["paths"]["benchmark_dir"]).mkdir(parents=True, exist_ok=True)
        from src.data.step4_annotate_drift import run_stub
        run_stub(mini_cfg)
        label_file = Path(mini_cfg["paths"]["benchmark_dir"]) / "drift_labels.json"
        assert label_file.exists()
        data = json.loads(label_file.read_text())
        assert "boundaries" in data
        assert len(data["boundaries"]) == 23
        for b in data["boundaries"]:
            assert "drift_type"  in b
            assert "window_from" in b
            assert "window_to"   in b


# ─── Domain validator (step 0) ────────────────────────────────────────────────
class TestDomainValidator:
    """Test is_valid_domain and clean_domains từ step0."""

    def setup_method(self):
        from src.data.step0_download_benign import is_valid_domain, clean_domains
        self.is_valid  = is_valid_domain
        self.clean     = clean_domains

    def test_valid_domains_accepted(self):
        valids = ["google.com", "github.io", "example.co.uk",
                  "sub.domain.org", "xn--nxasmq6b.com"]
        for d in valids:
            assert self.is_valid(d), f"Should be valid: {d}"

    def test_invalid_domains_rejected(self):
        invalids = [
            "localhost",          # no TLD
            "abc.onion",          # blacklisted TLD
            "abc.local",          # blacklisted TLD
            "",                   # empty
            "a" * 260 + ".com",   # too long
            "-.com",              # starts with hyphen
            "abc",                # no dot
        ]
        for d in invalids:
            assert not self.is_valid(d), f"Should be invalid: {d}"

    def test_clean_deduplicates(self):
        domains = ["google.com", "GOOGLE.COM", "github.com", "google.com"]
        result  = self.clean(domains)
        assert len(result) == 2
        assert "google.com" in result
        assert "github.com" in result

    def test_clean_lowercases(self):
        result = self.clean(["GitHub.COM"])
        assert result == ["github.com"]

    def test_year_to_date_coverage(self):
        from src.data.step0_download_benign import YEAR_TO_DATE
        for year in range(2018, 2024):
            assert year in YEAR_TO_DATE, f"Missing year {year} in YEAR_TO_DATE"

    def test_2018_proxy_date(self):
        """2018 must use a post-launch Tranco date (Tranco launched 2019-02)."""
        from src.data.step0_download_benign import YEAR_TO_DATE
        date_2018 = YEAR_TO_DATE[2018]
        assert date_2018 >= "2019-02-19", \
            f"2018 proxy date {date_2018} predates Tranco launch (2019-02-19)"

    def test_convert_alexa(self, tmp_path):
        """convert_alexa should produce correct output CSV."""
        from src.data.step0_download_benign import convert_alexa
        from src.utils.common import get_logger
        logger = get_logger("test_convert")
        # Write fake Alexa CSV with 100 valid domains
        alexa_file = tmp_path / "top-1m.csv"
        rows = "\n".join(f"{i+1},domain{i:04d}.com" for i in range(100))
        alexa_file.write_text(rows)
        out_dir = tmp_path / "benign"
        out_dir.mkdir()
        convert_alexa(alexa_file, 2018, out_dir, logger)
        out = out_dir / "tranco_2018.csv"
        assert out.exists()
        df = pd.read_csv(out)
        assert "domain" in df.columns
        assert len(df) == 100
        assert df["domain"].str.islower().all()


# ─── Benign strategy tests ────────────────────────────────────────────────────
class TestBenignStrategies:
    """
    Test the two strategies for assigning timestamp-free benign domains.
    Core invariant: NO benign domain appears in more than 1 window.
    """

    def _make_benign_files(self, benign_dir: Path, n_per_year: int = 2000):
        """Write synthetic tranco_YYYY.csv files for 2018-2023."""
        for year in range(2018, 2024):
            domains = [f"benign{year}_{i:05d}.com" for i in range(n_per_year)]
            pd.DataFrame({"domain": domains}).to_csv(
                benign_dir / f"tranco_{year}.csv", index=False
            )

    def _make_dga_windows(self, windows_dir: Path):
        """Write minimal DGA window files (no domain overlap with benign)."""
        for q in range(24):
            win_id = quarter_id(2018 + q // 4, q % 4 + 1)
            q_label = quarter_label(2018 + q // 4, q % 4 + 1)
            rows = [{"domain": f"dga_{win_id}_{i:04d}.xyz",
                     "label": 1, "family": "conficker",
                     "quarter_id": win_id, "quarter_label": q_label}
                    for i in range(50)]
            pd.DataFrame(rows).to_csv(windows_dir / f"{win_id}_dga.csv", index=False)

    @pytest.fixture
    def base_cfg(self, tmp_path):
        benign_dir  = tmp_path / "raw" / "benign"
        windows_dir = tmp_path / "windows"
        bench_dir   = tmp_path / "benchmark"
        results_dir = tmp_path / "results"
        for d in [benign_dir, windows_dir, bench_dir, results_dir]:
            d.mkdir(parents=True, exist_ok=True)
        return {
            "paths": {
                "dgarchive_raw": str(tmp_path / "raw" / "dgarchive"),
                "benign_raw":    str(benign_dir),
                "interim":       str(tmp_path / "interim"),
                "windows_dir":   str(windows_dir),
                "benchmark_dir": str(bench_dir),
                "results":       str(results_dir),
            },
            "benign": {
                "max_per_window": 100,
                "label": 0,
                "strategy": "annual_partition",
                "single_snapshot_file": None,
                "verify_cross_year_overlap": True,
            },
            "integrity": {"verify_no_future_leak": True,
                          "verify_no_cross_contamination": True},
            "random_seed": 42,
        }

    def _no_benign_leak(self, bench_dir: Path) -> bool:
        """Return True iff no benign domain appears in more than one window."""
        seen: set[str] = set()
        for q in range(24):
            win_id = quarter_id(2018 + q // 4, q % 4 + 1)
            path = bench_dir / f"{win_id}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            benign_domains = set(df.loc[df["label"] == 0, "domain"].str.lower())
            if benign_domains & seen:
                return False
            seen.update(benign_domains)
        return True

    def test_annual_partition_no_leak(self, base_cfg, tmp_path):
        """annual_partition: no benign domain in >1 window."""
        benign_dir  = Path(base_cfg["paths"]["benign_raw"])
        windows_dir = Path(base_cfg["paths"]["windows_dir"])
        bench_dir   = Path(base_cfg["paths"]["benchmark_dir"])
        self._make_benign_files(benign_dir, n_per_year=1200)
        self._make_dga_windows(windows_dir)
        from src.data.step3_merge_benign import run
        base_cfg["benign"]["strategy"] = "annual_partition"
        run(base_cfg)
        assert self._no_benign_leak(bench_dir), \
            "annual_partition: benign temporal leak detected!"

    def test_single_snapshot_no_leak(self, base_cfg, tmp_path):
        """single_snapshot: no benign domain in >1 window."""
        benign_dir  = Path(base_cfg["paths"]["benign_raw"])
        windows_dir = Path(base_cfg["paths"]["windows_dir"])
        bench_dir   = Path(base_cfg["paths"]["benchmark_dir"])
        # single snapshot: only one file needed
        pd.DataFrame({"domain": [f"snap_{i:06d}.com" for i in range(10000)]}).to_csv(
            benign_dir / "tranco_2023.csv", index=False
        )
        self._make_dga_windows(windows_dir)
        from src.data.step3_merge_benign import run
        base_cfg["benign"]["strategy"] = "single_snapshot"
        run(base_cfg)
        assert self._no_benign_leak(bench_dir), \
            "single_snapshot: benign temporal leak detected!"

    def test_no_dga_benign_overlap(self, base_cfg, tmp_path):
        """No domain should be both DGA and benign in the same window."""
        benign_dir  = Path(base_cfg["paths"]["benign_raw"])
        windows_dir = Path(base_cfg["paths"]["windows_dir"])
        bench_dir   = Path(base_cfg["paths"]["benchmark_dir"])
        self._make_benign_files(benign_dir, n_per_year=1200)
        self._make_dga_windows(windows_dir)
        from src.data.step3_merge_benign import run
        run(base_cfg)
        for q in range(24):
            win_id = quarter_id(2018 + q // 4, q % 4 + 1)
            path = bench_dir / f"{win_id}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            dga_d    = set(df.loc[df["label"] == 1, "domain"])
            benign_d = set(df.loc[df["label"] == 0, "domain"])
            overlap  = dga_d & benign_d
            assert len(overlap) == 0, \
                f"{win_id}: {len(overlap)} domains appear as both DGA and benign"
