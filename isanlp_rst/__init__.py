"""Convenience helpers for working with RST structures.

This module exposes the RST viewer implementation (ported from ``rstviewer``)
directly inside :mod:`isanlp_rst`.  Additionally, it provides utilities for
serialising ``isanlp.annotation_rst.DiscourseUnit`` trees back into the
``.rs3`` format understood by the viewer.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Awaitable, Dict, List, Optional, Union, TYPE_CHECKING
import xml.etree.ElementTree as ET

from .rstviewer import main as _rst_main
from .rstviewer import RenderedRST

try:  # pragma: no cover - dependency is optional in tests
    from isanlp.annotation_rst import DiscourseUnit
except Exception:  # pragma: no cover - fall back when isanlp is unavailable
    DiscourseUnit = None  # type: ignore[misc]

__all__ = [
    "RenderedRST",
    "render",
    "to_html",
    "to_png",
    "to_pdf",
    "write_discourse_unit_as_rs3",
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
    return html_str


def to_png(rs3_path: PathLike, png_path: Optional[PathLike] = None, *,
           base64_encoded: bool = False, device_scale_factor: int = 2,
           timeout_ms: int = 10_000) -> Union[bytes, str, None]:
    """Render an ``.rs3`` file to PNG.

    This delegates to :func:`isanlp_rst.rstviewer.main.rs3topng`. All keyword arguments are
    forwarded verbatim.
    """

    return _rst_main.rs3topng(
        os.fspath(rs3_path),
        png_filepath=os.fspath(png_path) if png_path is not None else None,
        base64_encoded=base64_encoded,
        device_scale_factor=device_scale_factor,
        timeout_ms=timeout_ms,
    )


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

    _run_coro_sync(coro)


def _run_coro_sync(coro: Awaitable[None]) -> None:
    """Execute ``coro`` to completion regardless of the current asyncio state."""

    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return

    # When a loop is already running (e.g. Jupyter) fall back to executing the
    # coroutine inside a dedicated thread.
    result: Dict[str, Optional[BaseException]] = {"exc": None}

    def _runner() -> None:
        try:
            asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            result["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if result["exc"] is not None:
        raise result["exc"]


# ---------------------------------------------------------------------------
# RS3 serialisation from DiscourseUnit
# ---------------------------------------------------------------------------


@dataclass
class _RS3Node:
    node_id: int
    kind: str  # "segment" or "group"
    text: Optional[str] = None
    group_type: Optional[str] = None
    parent: Optional[int] = None
    relname: str = "span"


def _get_children(node: "DiscourseUnit") -> List["DiscourseUnit"]:
    children: List["DiscourseUnit"] = []
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    if left is not None:
        children.append(left)
    if right is not None:
        children.append(right)
    extra = getattr(node, "children", None)
    if extra:
        children.extend(extra)
    return children


def _is_leaf(node: "DiscourseUnit") -> bool:
    return len(_get_children(node)) == 0


def _collect_nodes(root: "DiscourseUnit") -> Tuple[Dict[int, _RS3Node], Dict[int, int]]:
    nodes: Dict[int, _RS3Node] = {}
    id_map: Dict[int, int] = {}
    edu_counter = 0
    group_counter = 0

    def add_edus(node: "DiscourseUnit") -> None:
        nonlocal edu_counter
        if _is_leaf(node):
            edu_counter += 1
            node_id = edu_counter
            nodes[node_id] = _RS3Node(
                node_id=node_id,
                kind="segment",
                text=(getattr(node, "text", "") or ""),
            )
            id_map[id(node)] = node_id
        else:
            for child in _get_children(node):
                add_edus(child)

    def add_groups(node: "DiscourseUnit") -> None:
        nonlocal group_counter
        if _is_leaf(node):
            return

        for child in _get_children(node):
            add_groups(child)

        group_counter += 1
        node_id = edu_counter + group_counter
        nuc = (getattr(node, "nuclearity", "") or "").upper()
        group_type = "multinuc" if nuc == "NN" or len(_get_children(node)) > 2 else "span"
        nodes[node_id] = _RS3Node(node_id=node_id, kind="group", group_type=group_type)
        id_map[id(node)] = node_id

    add_edus(root)
    add_groups(root)
    return nodes, id_map


def _assign_structure(root: "DiscourseUnit", nodes: Dict[int, _RS3Node],
                      id_map: Dict[int, int]) -> Dict[str, str]:
    relations: Dict[str, str] = {"span": "rst"}

    def visit(node: "DiscourseUnit", parent: Optional["DiscourseUnit"]) -> None:
        node_id = id_map[id(node)]
        node_info = nodes[node_id]

        if parent is None:
            node_info.parent = 0
            node_info.relname = "span"
        else:
            parent_id = id_map[id(parent)]
            node_info.parent = parent_id

            nuc = (getattr(parent, "nuclearity", "") or "").upper()
            rel = getattr(parent, "relation", None) or "span"
            children = _get_children(parent)
            child_rel_type = "rst"

            if nuc == "NN" or len(children) > 2:
                # Multinuclear nodes: children share the same relation label.
                node_info.relname = rel
                child_rel_type = "multinuc"
            else:
                is_left_child = children and node is children[0]
                if nuc == "SN":
                    is_nucleus = not is_left_child
                elif nuc == "NS":
                    is_nucleus = is_left_child
                else:
                    # Default: treat the first child as the nucleus
                    is_nucleus = is_left_child

                if is_nucleus:
                    node_info.relname = "span"
                else:
                    node_info.relname = rel

            relname = (node_info.relname or "span").strip() or "span"
            node_info.relname = relname
            relations.setdefault(relname, child_rel_type)

        for child in _get_children(node):
            visit(child, node)

    visit(root, None)
    return relations


def _build_xml(nodes: Dict[int, _RS3Node], relations: Dict[str, str]) -> ET.Element:
    rst_el = ET.Element("rst")
    header_el = ET.SubElement(rst_el, "header")
    rels_el = ET.SubElement(header_el, "relations")

    for relname in sorted(relations):
        rel_type = relations[relname]
        ET.SubElement(rels_el, "rel", {"name": relname, "type": rel_type})

    body_el = ET.SubElement(rst_el, "body")
    for node_id in sorted(nodes):
        info = nodes[node_id]
        attrs = {"id": str(info.node_id)}
        if info.parent is not None:
            attrs["parent"] = str(info.parent)
        if info.relname:
            attrs["relname"] = info.relname

        if info.kind == "segment":
            seg_el = ET.SubElement(body_el, "segment", attrs)
            seg_el.text = info.text or ""
        else:
            attrs["type"] = info.group_type or "span"
            ET.SubElement(body_el, "group", attrs)

    _indent_xml(rst_el)
    return rst_el


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + "  " * level
    children = list(element)
    if children:
        if element.tag != "segment":
            if not element.text or not element.text.strip():
                element.text = indent + "  "
        for child in children:
            _indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent + "  "
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = indent
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def write_discourse_unit_as_rs3(tree: "DiscourseUnit", output_path: PathLike) -> Path:
    """Serialise ``tree`` into the RS3 format and write it to ``output_path``."""

    if DiscourseUnit is None:  # pragma: no cover - defensive runtime guard
        raise ImportError(
            "isanlp.annotation_rst.DiscourseUnit is required to serialise RS3 trees."
        )

    nodes, id_map = _collect_nodes(tree)
    relations = _assign_structure(tree, nodes, id_map)
    xml_root = _build_xml(nodes, relations)

    output_path = Path(output_path)
    ET.ElementTree(xml_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _discourse_unit_to_rs3(self: "DiscourseUnit", output_path: PathLike) -> Path:
    return write_discourse_unit_as_rs3(self, output_path)


if DiscourseUnit is not None and not hasattr(DiscourseUnit, "to_rs3"):
    DiscourseUnit.to_rs3 = _discourse_unit_to_rs3  # type: ignore[attr-defined]


if TYPE_CHECKING:  # pragma: no cover - typing aid only
    # ``DiscourseUnit`` is optional at runtime; make mypy aware of the helper.
    class _DiscourseUnitProto:
        def to_rs3(self, output_path: PathLike) -> Path: ...
