"""JSON serialisation helpers for RST trees produced by ``isanlp_rst``.

The parser returns ``isanlp.annotation_rst.DiscourseUnit`` trees. Downstream
consumers commonly need a JSON-compatible representation of the tree to
cache, transmit, or visualise without holding the live object graph.

This module provides:

- :func:`tree_to_dict` — recursive serialiser preserving the public
  attributes of each ``DiscourseUnit`` (id, relation, nuclearity, start,
  end, text, proba) plus children (left, right) when present.
- :func:`tree_from_dict` — round-trip inverse, using ``DiscourseUnit``
  from the parent ``isanlp`` package if available, otherwise returning
  the dict unchanged.

Both functions are pure and have no model dependencies — suitable for
use in tests, caching layers, and report generators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# Attributes preserved during serialisation. Order matters only for stable
# JSON output (deterministic field order helps caching).
_PUBLIC_ATTRS: tuple[str, ...] = (
    "id",
    "relation",
    "nuclearity",
    "start",
    "end",
    "text",
    "proba",
)


def tree_to_dict(node: Any) -> dict[str, Any]:
    """Serialise an RST tree node (``DiscourseUnit``-like) to a plain dict.

    Recursively walks ``left`` and ``right`` children. Returns an empty dict
    when the input is ``None``. Missing attributes are silently omitted from
    the result rather than included as ``None`` — this keeps cached JSON
    compact and makes round-trip equality stable.

    Args:
        node: The root ``DiscourseUnit`` (or any object exposing the
            ``id``, ``relation``, ``nuclearity``, ``start``, ``end``,
            ``text``, ``proba``, ``left``, ``right`` attributes). Pass
            ``parser(text)['rst'][0]`` directly.

    Returns:
        A JSON-serialisable dict. Children appear under ``"left"`` and
        ``"right"`` keys, recursively serialised.

    Examples:
        >>> result = parser("Some text. More text.")
        >>> from isanlp_rst.utils.serialization import tree_to_dict
        >>> serialised = tree_to_dict(result['rst'][0])
        >>> import json
        >>> json.dumps(serialised)  # works
    """
    if node is None:
        return {}

    out: dict[str, Any] = {}
    for attr in _PUBLIC_ATTRS:
        value = getattr(node, attr, None)
        if value is not None:
            out[attr] = value

    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    if left is not None:
        out["left"] = tree_to_dict(left)
    if right is not None:
        out["right"] = tree_to_dict(right)

    return out


def tree_from_dict(data: dict[str, Any]) -> Any:
    """Inverse of :func:`tree_to_dict`.

    Reconstructs a ``DiscourseUnit`` tree from a previously serialised dict
    if the parent ``isanlp`` package is importable; otherwise returns the
    dict unchanged (allowing dict-shaped consumers to keep working).

    Args:
        data: Output of :func:`tree_to_dict`.

    Returns:
        A ``DiscourseUnit`` tree if ``isanlp.annotation_rst`` is available,
        otherwise the input dict.
    """
    if not data:
        return None

    try:
        from isanlp.annotation_rst import DiscourseUnit
    except Exception:
        return data

    kwargs = {attr: data[attr] for attr in _PUBLIC_ATTRS if attr in data}
    node = DiscourseUnit(**kwargs)
    if "left" in data:
        node.left = tree_from_dict(data["left"])
    if "right" in data:
        node.right = tree_from_dict(data["right"])
    return node


__all__ = ["tree_from_dict", "tree_to_dict"]
