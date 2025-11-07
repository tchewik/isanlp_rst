from __future__ import annotations

import sys
import types
from importlib import import_module
from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Iterable, List, Optional, Sequence, Tuple


def str2bool(value):
    """Robust string-to-bool conversion used in configs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


class BasePredictor(ABC):
    """A mixin-style base with shared tokenization, batching and offset utils.    """

    tokenizer = None

    @staticmethod
    def divide_chunks(_list: Sequence, n: int) -> Iterable[Sequence]:
        """Yield chunks of size `n` from `_list` (handles empty lists)."""
        if _list:
            for i in range(0, len(_list), n):
                yield _list[i : min(i + n, len(_list))]
        else:
            yield _list

    @staticmethod
    def build_offset_converter_from_words(
        text: str,
        tokens: Sequence[str],
        token_offsets: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[List[int], List[int]]:
        """Build offset converter from word tokens and optional (start, end) pairs.

        If `token_offsets` is omitted, a best-effort alignment is performed.
        Returns two lists: `positions` (flattened space of tokenized text) and
        `originals` (mapped indices in the original text).
        """
        if token_offsets is None:
            token_offsets = BasePredictor._guess_token_offsets(text, tokens)

        positions: List[int] = []
        originals: List[int] = []
        cursor = 0

        for idx, (tok, (start, end)) in enumerate(zip(tokens, token_offsets)):
            token_text = tok or ""
            for _ in range(len(token_text)):
                positions.append(cursor)
                originals.append(start)
                start += 1
                cursor += 1
            positions.append(cursor)
            originals.append(end)
            if idx != len(tokens) - 1:
                cursor += 1

        if not positions:
            positions = [0]
            originals = [0]

        return positions, originals

    @staticmethod
    def build_offset_converter_from_razdel(
        tokens,
    ) -> Tuple[List[int], List[int]]:
        """Build offset converter from a list of `razdel.Token` objects."""
        positions: List[int] = []
        originals: List[int] = []
        cursor = 0

        for idx, token in enumerate(tokens):
            token_text = token.text
            for char_idx in range(len(token_text)):
                positions.append(cursor)
                originals.append(token.start + char_idx)
                cursor += 1
            positions.append(cursor)
            originals.append(token.stop)
            if idx != len(tokens) - 1:
                cursor += 1

        if not positions:
            positions = [0]
            originals = [0]

        return positions, originals

    @staticmethod
    def _map_offset(value: int, positions: List[int], originals: List[int]) -> int:
        if not positions:
            return value
        index = bisect_right(positions, value) - 1
        if index < 0:
            index = 0
        elif index >= len(originals):
            index = len(originals) - 1
        return originals[index]

    def remap_tree_offsets(
        self,
        unit,
        positions: List[int],
        originals: List[int],
        original_text: str,
    ) -> None:
        """Recursively remap `.start`/`.end` of leaf/internal nodes to original text.

        Mutates `unit` in-place and updates `unit.text` accordingly.
        """
        left = getattr(unit, "left", None)
        right = getattr(unit, "right", None)

        if left is not None:
            self.remap_tree_offsets(left, positions, originals, original_text)
        if right is not None:
            self.remap_tree_offsets(right, positions, originals, original_text)

        if left is None and right is None:
            unit.start = self._map_offset(unit.start, positions, originals)
            unit.end = self._map_offset(unit.end, positions, originals)
        else:
            unit.start = left.start if left is not None else self._map_offset(unit.start, positions, originals)
            unit.end = right.end if right is not None else self._map_offset(unit.end, positions, originals)

        unit.text = original_text[unit.start : unit.end]

    @staticmethod
    def _guess_token_offsets(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
        """Best-effort alignment of already-tokenized `tokens` to raw `text`.

        This is used when external word tokens are supplied without
        character-level offsets. It walks forward to find each token.
        """
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        for token in tokens:
            if not token:
                offsets.append((cursor, cursor))
                continue

            start = cursor
            while start <= len(text) and text[start : start + len(token)] != token:
                start += 1
                if start >= len(text):
                    start = cursor
                    break
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return offsets

    @staticmethod
    def _recount_spans(word_offsets, subword_offsets, word_span_boundaries):
        """ Given word span boundaries, recount for subwords. """
        subword_span_boundaries = [0]

        for w_end in word_span_boundaries:
            final_char = word_offsets[w_end][1]
            for i in range(1, len(subword_offsets)):
                if subword_offsets[i][0] < subword_offsets[i][1]:
                    if subword_offsets[i][0] >= final_char:
                        # Fixes LUKE segmentation
                        if i - 1 in subword_span_boundaries:
                            subword_span_boundaries.append(i)
                        else:
                            subword_span_boundaries.append(i - 1)
                        break

        if not len(subword_offsets) - 1 in subword_span_boundaries:
            subword_span_boundaries.append(len(subword_offsets) - 1)

        return subword_span_boundaries[1:]

    @staticmethod
    def _collect_leaf_texts(unit, acc: List[str]) -> None:
        left = getattr(unit, 'left', None)
        right = getattr(unit, 'right', None)

        if left is None and right is None:
            acc.append(unit.text)
            return

        if left is not None:
            BasePredictor._collect_leaf_texts(left, acc)
        if right is not None:
            BasePredictor._collect_leaf_texts(right, acc)

