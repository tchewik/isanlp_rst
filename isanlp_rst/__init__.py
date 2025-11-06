"""Convenience helpers for working with RST structures.

This module exposes the RST viewer implementation (ported from ``rstviewer``)
directly inside :mod:`isanlp_rst`.  Additionally, it provides utilities for
serialising ``isanlp.annotation_rst.DiscourseUnit`` trees back into the
``.rs3`` format understood by the viewer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import transformers
import warnings
from pathlib import Path
from typing import Any
from typing import IO, Awaitable, Dict, Optional, Union

from .rstviewer import RenderedRST
from .rstviewer import main as _rst_main

try:  # pragma: no cover - dependency is optional in tests
    from isanlp.annotation_rst import DiscourseUnit
except Exception:  # pragma: no cover - fall back when isanlp is unavailable
    DiscourseUnit = None  # type: ignore[misc]

logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message=r"The new embeddings will be initialized from a multivariate normal distribution",
)
warnings.filterwarnings(
    "ignore",
    message=r"dropout option adds dropout after all but last recurrent layer",
    module=r"torch\.nn\.modules\.rnn",
)

__all__ = [
    "RenderedRST",
    "render",
    "to_html",
    "to_png",
    "to_pdf",
]

PathLike = Union[str, os.PathLike]


def render(rs3_source: Union[PathLike, bytes, IO[str], IO[bytes]], *,
           display_inline: bool = True, colab: bool = False) -> RenderedRST:
    """Render an RST tree and, optionally, display it inline.

    This is a light-weight proxy around :func:`isanlp_rst.rstviewer.main.render`.
    """

    return _rst_main.render(rs3_source, display_inline=display_inline, colab=colab)


def to_html(rs3_path: PathLike, html_path: Optional[PathLike] = None, *,
            user: str = "temp_user", project: str = "rstviewer_temp") -> str:
    """Convert an ``.rs3`` file into HTML.

    Parameters
    ----------
    rs3_path:
        Path to the source ``.rs3`` file.
    html_path:
        Optional destination path. When provided the resulting HTML is written
        to this location. The HTML string is returned in all cases.
    user, project:
        Passed through to :func:`isanlp_rst.rstviewer.main.rs3tohtml` to maintain
        compatibility with the viewer's expectations.
    """

    html_str = _rst_main.rs3tohtml(os.fspath(rs3_path), user=user, project=project)
    if html_path is not None:
        Path(html_path).write_text(html_str, encoding="utf-8")
    else:
        return html_str


def to_png(rs3_path: PathLike, png_path: Optional[PathLike] = None, *,
           base64_encoded: bool = False, device_scale_factor: int = 2,
           timeout_ms: int = 10_000) -> Union[bytes, str, None]:
    """Render an ``.rs3`` file to PNG (works in both sync and async environments)."""

    # If there's no running loop, use the fast sync path.
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return _rst_main.rs3topng(
            os.fspath(rs3_path),
            png_filepath=os.fspath(png_path) if png_path is not None else None,
            base64_encoded=base64_encoded,
            device_scale_factor=device_scale_factor,
            timeout_ms=timeout_ms,
        )

    # Running inside an event loop (e.g., Jupyter) → use the async renderer via a worker.
    coro = _rst_main.rs3topng_async(
        os.fspath(rs3_path),
        png_filepath=os.fspath(png_path) if png_path is not None else None,
        base64_encoded=base64_encoded,
        device_scale_factor=device_scale_factor,
        timeout_ms=timeout_ms,
    )
    return _run_coro_sync_result(coro)


def to_pdf(rs3_path: PathLike, pdf_path: PathLike, *,
           device_scale_factor: int = 2, viewport_width: int = 1600,
           viewport_height: int = 1000, timeout_ms: int = 10_000,
           margin_px: int = 12) -> None:
    """Render an ``.rs3`` file to PDF.

    The viewer exposes only an asynchronous PDF renderer; this helper makes
    it convenient to call from synchronous contexts (including notebooks). When
    an event loop is already running the rendering is executed in a worker
    thread so that the current loop does not need to be interrupted.
    """

    coro = _rst_main.rs3topdf_async(
        os.fspath(rs3_path),
        os.fspath(pdf_path),
        device_scale_factor=device_scale_factor,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        timeout_ms=timeout_ms,
        margin_px=margin_px,
    )

    _run_coro_sync_result(coro)


def _run_coro_sync_result(coro: Awaitable[T]) -> T:
    """Execute `coro` to completion and return its result, regardless of asyncio state."""
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop → run directly
        return asyncio.run(coro)

    # Running loop → run in a worker thread
    result: Dict[str, Any] = {"exc": None, "value": None}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            result["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if result["exc"] is not None:
        raise result["exc"]
    return result["value"]  # type: ignore[return-value]
