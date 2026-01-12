import faulthandler
import os
import platform
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def diagnostics_enabled() -> bool:
    return _truthy_env("OPEN_DILOCO_DIAGNOSTICS") or _truthy_env("DILOCO_DIAGNOSTICS")


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class DiskUsage:
    total: int
    used: int
    free: int


def _disk_usage(path: str) -> DiskUsage | None:
    try:
        st = os.statvfs(path)
    except (FileNotFoundError, PermissionError, OSError):
        return None
    total = st.f_frsize * st.f_blocks
    free = st.f_frsize * st.f_bavail
    used = total - (st.f_frsize * st.f_bfree)
    return DiskUsage(total=total, used=used, free=free)


def _sum_glob_sizes(path: str, patterns: list[str]) -> int | None:
    base = Path(path)
    if not base.exists():
        return None
    total = 0
    try:
        for pat in patterns:
            for p in base.glob(pat):
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
        return total
    except OSError:
        return None


def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    step = 1024.0
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < step or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= step
    return f"{size:.2f}B"


def _get_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _get_gpu_mem() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception:
        return None


def _get_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    versions["python"] = sys.version.replace("\n", " ")
    versions["platform"] = platform.platform()
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        pass
    try:
        import hivemind

        versions["hivemind"] = getattr(hivemind, "__version__", "unknown")
    except Exception:
        pass
    try:
        import transformers

        versions["transformers"] = getattr(transformers, "__version__", "unknown")
    except Exception:
        pass
    return versions


def _get_diag_log_path() -> Path | None:
    # If a directory is set, we write one file per PID to make log collection easy.
    diag_dir = os.environ.get("OPEN_DILOCO_DIAG_DIR") or os.environ.get("DILOCO_DIAG_DIR")
    if not diag_dir:
        return None
    out_dir = Path(diag_dir).expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return out_dir / f"open_diloco_diag_{os.getpid()}.log"


def enable_faulthandler() -> None:
    """Enable Python fatal-signal tracebacks.

    This is specifically useful for SIGBUS/SIGSEGV style crashes where Python never
    gets a chance to raise an exception.
    """

    log_path = _get_diag_log_path()
    if log_path is not None:
        try:
            f = open(log_path, "a", buffering=1)
            faulthandler.enable(file=f, all_threads=True)
        except OSError:
            faulthandler.enable(all_threads=True)
    else:
        faulthandler.enable(all_threads=True)

    # Allow on-demand stack dumps: `kill -USR1 <pid>`
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass


def diag_snapshot(stage: str, extra: dict[str, Any] | None = None) -> None:
    if not diagnostics_enabled():
        return

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    pid = os.getpid()
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")

    shm_usage = _disk_usage("/dev/shm")
    torch_shm_bytes = _sum_glob_sizes("/dev/shm", patterns=["torch_*", "pymp-*"])

    rss_bytes = _get_rss_bytes()
    gpu_mem = _get_gpu_mem()
    versions = _get_versions()

    fields: dict[str, Any] = {
        "ts": ts,
        "pid": pid,
        "rank": rank,
        "local_rank": local_rank,
        "stage": stage,
        "rss": _format_bytes(rss_bytes),
        "shm_total": _format_bytes(shm_usage.total) if shm_usage else "n/a",
        "shm_used": _format_bytes(shm_usage.used) if shm_usage else "n/a",
        "shm_free": _format_bytes(shm_usage.free) if shm_usage else "n/a",
        "shm_torch_files": _format_bytes(torch_shm_bytes),
    }
    if gpu_mem is not None:
        fields["gpu_free"] = _format_bytes(gpu_mem[0])
        fields["gpu_total"] = _format_bytes(gpu_mem[1])
    if extra:
        fields.update(extra)

    msg = " | ".join(f"{k}={v}" for k, v in fields.items())

    log_path = _get_diag_log_path()
    if log_path is not None:
        try:
            with open(log_path, "a", buffering=1) as f:
                f.write(msg + "\n")
                if _truthy_env("OPEN_DILOCO_DIAG_VERSIONS"):
                    f.write("versions=" + str(versions) + "\n")
        except OSError:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
    else:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


def enforce_min_shm_free_gib() -> None:
    """Optionally fail fast if /dev/shm is too full.

    Set OPEN_DILOCO_SHM_MIN_FREE_GIB to an integer (GiB).
    """

    min_free_gib = _safe_int_env("OPEN_DILOCO_SHM_MIN_FREE_GIB", 0)
    if min_free_gib <= 0:
        return
    usage = _disk_usage("/dev/shm")
    if usage is None:
        return
    min_free_bytes = int(min_free_gib * 1024**3)
    if usage.free < min_free_bytes:
        raise RuntimeError(
            f"/dev/shm too full: free={_format_bytes(usage.free)} < min={min_free_gib}GiB; "
            "set a bigger /dev/shm or clear stale torch_* files"
        )


def start_periodic_snapshots(interval_s: float = 10.0, label: str | None = None) -> None:
    if not diagnostics_enabled():
        return

    if interval_s <= 0:
        return

    thread_label = label or "periodic"

    def _loop() -> None:
        while True:
            try:
                diag_snapshot(f"{thread_label}")
            except Exception:
                pass
            time.sleep(interval_s)

    t = threading.Thread(target=_loop, name="open_diloco_diag", daemon=True)
    t.start()
