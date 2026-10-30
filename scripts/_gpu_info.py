"""
Shared GPU-identification helper for this project's Chrono::FSI/SPH
scripts (added 2026-09-18, ahead of first access to a third real machine
-- a college RTX 3090 desktop, separate from both the original laptop
(no CUDA) and the second machine, RTX 5060 -- see the project notes' "RTX 3090
machine note").

Every real GPU comparison this project has ever made (Colab Tesla T4
vs. the RTX 5060, Entries 46 vs. 57) only worked because the actual
device name was checked and logged at the time, not assumed from which
machine a session happened to be on. This module does the same check
programmatically instead of relying on someone remembering to state it
by hand: it asks the CUDA runtime directly which device is actually
present, so every real run's log unambiguously says which physical GPU
produced it. Never hardcode a GPU name anywhere else in this project's
scripts -- import and call log_gpu_info() instead.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_gpu_name() -> str:
    """Returns the real, currently-active CUDA device name, e.g.
    'NVIDIA GeForce RTX 3090' or 'Tesla T4' -- queried from the runtime,
    never guessed from which machine/session this happens to be. Returns
    a clearly-labeled placeholder string (not an exception) if no CUDA
    device is visible, since callers here run in headless/logging
    contexts, not error-handling ones."""
    try:
        import torch
    except ImportError:
        return "UNKNOWN (torch not importable)"
    if not torch.cuda.is_available():
        return "NONE (torch.cuda.is_available() is False -- no GPU visible to this process)"
    return torch.cuda.get_device_name(0)


def log_gpu_info(logger_: "logging.Logger | None" = None) -> str:
    """Logs (and returns) the real detected GPU name at INFO level. Call
    this once near the start of any script that actually runs Chrono::FSI,
    so every real output file/log this project produces is traceable to
    the specific physical hardware that made it -- exactly the same
    discipline the development log already applies by hand (e.g. Entry 46's
    'Tesla T4', Entry 57's 'RTX 5060'), just automatic instead of manual."""
    log = logger_ or logger
    name = get_gpu_name()
    log.info("Detected GPU: %s", name)
    return name
