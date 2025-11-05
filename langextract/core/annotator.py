"""Utilities for annotation that add Japanese sentence support."""

# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from absl import logging

from langextract.core import tokenizer

try:  # pragma: no cover - optional dependency
  from fast_bunkai import Bunkai as _BunkaiCls  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
  _BunkaiCls = None

try:  # pragma: no cover - optional dependency
  from fast_bunkai import BunkaiFactory as _BunkaiFactory  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
  _BunkaiFactory = None

_BUNKAI_SPLITTER: Any | None = None


def _get_bunkai_splitter() -> Any | None:
  """Returns a cached instance of the fast-bunkai sentence splitter."""

  global _BUNKAI_SPLITTER

  if _BUNKAI_SPLITTER is not None:
    return _BUNKAI_SPLITTER

  splitter: Any | None = None

  if _BunkaiCls is not None:
    try:
      splitter = _BunkaiCls()  # type: ignore[call-arg]
    except TypeError:  # Some versions require a factory to create the splitter.
      splitter = None
    except Exception as error:  # pragma: no cover - defensive logging
      logging.warning("Failed to initialize fast-bunkai splitter: %s", error)
      splitter = None

  if splitter is None and _BunkaiFactory is not None:
    try:
      factory = _BunkaiFactory()
      if hasattr(factory, "create"):
        splitter = factory.create()
    except Exception as error:  # pragma: no cover - defensive logging
      logging.warning("Failed to initialize fast-bunkai factory: %s", error)

  if splitter is None:
    logging.info(
        "fast-bunkai is not available; falling back to rule-based splitting."
    )

  _BUNKAI_SPLITTER = splitter
  return _BUNKAI_SPLITTER


def _call_splitter(splitter: Any, text: str) -> Iterable[Any] | None:
  """Invokes the fast-bunkai splitter with broad compatibility."""

  if splitter is None:
    return None

  if hasattr(splitter, "split"):
    return splitter.split(text)
  if callable(splitter):  # pragma: no branch - typical fast-bunkai API
    return splitter(text)
  if hasattr(splitter, "__call__"):
    return splitter.__call__(text)
  return None


def _sentence_text(sentence: Any) -> str:
  """Extracts the sentence text from the splitter output."""

  if isinstance(sentence, str):
    return sentence

  for attr in ("text", "surface", "sentence", "content"):
    value = getattr(sentence, attr, None)
    if value:
      return value() if callable(value) else value

  return str(sentence)


def _sentence_span(sentence: Any) -> tuple[int | None, int | None] | None:
  """Extracts the (start, end) span from the splitter output when available."""

  for attr in ("span", "range", "offset"):
    span = getattr(sentence, attr, None)
    if span is None:
      continue
    if callable(span):
      try:
        span = span()
      except TypeError:
        continue
    if isinstance(span, Sequence) and len(span) >= 2:
      start, end = span[0], span[1]
      return (int(start) if start is not None else None, int(end) if end is not None else None)

  start = getattr(sentence, "start", None)
  end = getattr(sentence, "end", None)
  if start is not None or end is not None:
    return (
        int(start) if start is not None else None,
        int(end) if end is not None else None,
    )

  return None


def _split_with_bunkai(text: str) -> list[tuple[int, int]]:
  """Splits text into character spans using fast-bunkai when available."""

  splitter = _get_bunkai_splitter()
  if splitter is None:
    return []

  results = _call_splitter(splitter, text)
  if not results:
    return []

  spans: list[tuple[int, int]] = []
  cursor = 0

  for item in results:
    sentence_text = _sentence_text(item)
    if not sentence_text:
      continue

    span = _sentence_span(item)
    if span is not None:
      start, end = span
    else:
      start = text.find(sentence_text, cursor)
      if start == -1:
        start = cursor
      end = start + len(sentence_text)

    if start is None or end is None:
      continue

    start = max(start, cursor)
    end = max(end, start)

    spans.append((start, end))
    cursor = end

  spans.sort(key=lambda span: span[0])
  return spans


def _find_start_token_index(
    tokens: Sequence[tokenizer.Token], char_pos: int
) -> int:
  """Finds the token index that contains or follows the character position."""

  for idx, token in enumerate(tokens):
    start = token.char_interval.start_pos
    end = token.char_interval.end_pos
    if char_pos <= start:
      return idx
    if start <= char_pos < end:
      return idx
  return len(tokens)


def _find_end_token_index(
    tokens: Sequence[tokenizer.Token], char_pos: int
) -> int:
  """Finds the token index marking the end of the character position."""

  for idx, token in enumerate(tokens):
    start = token.char_interval.start_pos
    end = token.char_interval.end_pos
    if char_pos <= start:
      return idx
    if start < char_pos <= end:
      return idx + 1
  return len(tokens)


def _default_sentence_intervals(
    tokenized_text: tokenizer.TokenizedText,
) -> list[tokenizer.TokenInterval]:
  """Falls back to the rule-based sentence segmentation."""

  text = tokenized_text.text
  tokens = tokenized_text.tokens
  intervals: list[tokenizer.TokenInterval] = []
  start = 0
  token_count = len(tokens)

  while start < token_count:
    sentence_range = tokenizer.find_sentence_range(text, tokens, start)
    if sentence_range.end_index <= start:
      break
    intervals.append(sentence_range)
    start = sentence_range.end_index

  if not intervals and token_count:
    intervals.append(tokenizer.TokenInterval(0, token_count))

  return intervals


def get_sentence_token_intervals(
    tokenized_text: tokenizer.TokenizedText,
) -> list[tokenizer.TokenInterval]:
  """Returns token intervals for sentences using fast-bunkai when available."""

  tokens = tokenized_text.tokens
  if not tokens:
    return []

  spans = _split_with_bunkai(tokenized_text.text)
  if not spans:
    return _default_sentence_intervals(tokenized_text)

  intervals: list[tokenizer.TokenInterval] = []
  last_end_index = 0

  for idx, (start_char, end_char) in enumerate(spans):
    start_index = _find_start_token_index(tokens, start_char)
    end_index = _find_end_token_index(tokens, end_char)

    if idx == 0 and start_index > 0:
      start_index = 0
    if start_index < last_end_index:
      start_index = last_end_index
    if end_index <= start_index:
      end_index = start_index

    end_index = max(end_index, start_index)
    if end_index <= last_end_index:
      continue

    intervals.append(
        tokenizer.TokenInterval(start_index=start_index, end_index=end_index)
    )
    last_end_index = end_index

  token_count = len(tokens)
  if last_end_index < token_count:
    intervals.append(
        tokenizer.TokenInterval(start_index=last_end_index, end_index=token_count)
    )

  return intervals or _default_sentence_intervals(tokenized_text)

