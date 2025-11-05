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

"""Tokenization utilities for text with SudachiPy support."""

from collections.abc import Sequence, Set
import dataclasses
import enum
import re
import unicodedata

from absl import logging

from langextract.core import debug_utils
from langextract.core import exceptions

try:  # pragma: no cover - optional dependency
  from sudachipy import dictionary as _sudachi_dictionary  # type: ignore
  from sudachipy import tokenizer as _sudachi_tokenizer_module  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
  _sudachi_dictionary = None
  _sudachi_tokenizer_module = None

__all__ = [
    "BaseTokenizerError",
    "InvalidTokenIntervalError",
    "SentenceRangeError",
    "CharInterval",
    "TokenInterval",
    "TokenType",
    "Token",
    "TokenizedText",
    "tokenize",
    "tokens_text",
    "find_sentence_range",
]


_SUDACHI_TOKENIZER = None


class BaseTokenizerError(exceptions.LangExtractError):
  """Base class for all tokenizer-related errors."""


class InvalidTokenIntervalError(BaseTokenizerError):
  """Error raised when a token interval is invalid or out of range."""


class SentenceRangeError(BaseTokenizerError):
  """Error raised when the start token index for a sentence is out of range."""


@dataclasses.dataclass
class CharInterval:
  """Represents a range of character positions in the original text.

  Attributes:
    start_pos: The starting character index (inclusive).
    end_pos: The ending character index (exclusive).
  """

  start_pos: int
  end_pos: int


@dataclasses.dataclass
class TokenInterval:
  """Represents an interval over tokens in tokenized text.

  The interval is defined by a start index (inclusive) and an end index
  (exclusive).

  Attributes:
    start_index: The index of the first token in the interval.
    end_index: The index one past the last token in the interval.
  """

  start_index: int = 0
  end_index: int = 0


class TokenType(enum.IntEnum):
  """Enumeration of token types produced during tokenization.

  Attributes:
    WORD: Represents an alphabetical word token.
    NUMBER: Represents a numeric token.
    PUNCTUATION: Represents punctuation characters.
    ACRONYM: Represents an acronym or slash-delimited abbreviation.
  """

  WORD = 0
  NUMBER = 1
  PUNCTUATION = 2
  ACRONYM = 3


@dataclasses.dataclass
class Token:
  """Represents a token extracted from text.

  Each token is assigned an index and classified into a type (word, number,
  punctuation,
  or acronym). The token also records the range of characters (its CharInterval)
  that
  correspond to the substring from the original text. Additionally, it tracks
  whether it
  follows a newline.

  Attributes:
    index: The position of the token in the sequence of tokens.
    token_type: The type of the token, as defined by TokenType.
    char_interval: The character interval within the original text that this
      token spans.
    first_token_after_newline: True if the token immediately follows a newline
      or carriage return.
  """

  index: int
  token_type: TokenType
  char_interval: CharInterval = dataclasses.field(
      default_factory=lambda: CharInterval(0, 0)
  )
  first_token_after_newline: bool = False


@dataclasses.dataclass
class TokenizedText:
  """Holds the result of tokenizing a text string.

  Attributes:
    text: The original text that was tokenized.
    tokens: A list of Token objects extracted from the text.
  """

  text: str
  tokens: list[Token] = dataclasses.field(default_factory=list)


# Regex patterns for tokenization.
_LETTERS_PATTERN = r"[A-Za-z]+"
_DIGITS_PATTERN = r"[0-9]+"
_SYMBOLS_PATTERN = r"[^A-Za-z0-9\s]+"
_END_OF_SENTENCE_PATTERN = re.compile(r"[.?!]$")
_SLASH_ABBREV_PATTERN = r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+"

_TOKEN_PATTERN = re.compile(
    rf"{_SLASH_ABBREV_PATTERN}|{_LETTERS_PATTERN}|{_DIGITS_PATTERN}|{_SYMBOLS_PATTERN}"
)
_WORD_PATTERN = re.compile(rf"(?:{_LETTERS_PATTERN}|{_DIGITS_PATTERN})\Z")

# Known abbreviations that should not count as sentence enders.
# TODO: This can potentially be removed given most use cases
# are larger context.
_KNOWN_ABBREVIATIONS = frozenset({"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St."})


def _get_sudachi_tokenizer():
  """Returns a cached SudachiPy tokenizer instance when available."""

  global _SUDACHI_TOKENIZER

  if _SUDACHI_TOKENIZER is not None:
    return _SUDACHI_TOKENIZER

  if _sudachi_dictionary is None:
    return None

  try:
    tokenizer = _sudachi_dictionary.Dictionary().create()
  except Exception as error:  # pragma: no cover - defensive logging
    logging.warning("Failed to initialize SudachiPy tokenizer: %s", error)
    tokenizer = None

  _SUDACHI_TOKENIZER = tokenizer
  return _SUDACHI_TOKENIZER


def _get_sudachi_split_mode():
  """Retrieves a SudachiPy split mode compatible across versions."""

  if _sudachi_tokenizer_module is None:
    return None

  for mode_name in ("C", "B", "A"):
    try:
      return getattr(_sudachi_tokenizer_module.Tokenizer.SplitMode, mode_name)
    except AttributeError:
      continue
  return None


def _call_sudachi_tokenize(tokenizer_obj, text: str):
  """Invokes SudachiPy tokenization with broad signature support."""

  if tokenizer_obj is None:
    return None

  try:
    return tokenizer_obj.tokenize(text)
  except TypeError:
    pass
  except Exception as error:  # pragma: no cover - defensive logging
    logging.warning("SudachiPy tokenize(text) failed: %s", error)
    return None

  split_mode = _get_sudachi_split_mode()
  if split_mode is None:
    return None

  for args in ((split_mode, text), (text, split_mode)):
    try:
      return tokenizer_obj.tokenize(*args)
    except TypeError:
      continue
    except Exception as error:  # pragma: no cover - defensive logging
      logging.warning("SudachiPy tokenize%r failed: %s", args, error)
      return None
  return None


def _sudachi_surface(morpheme) -> str:
  """Extracts the surface form from a SudachiPy morpheme."""

  if morpheme is None:
    return ""

  for attr in ("surface", "dictionary_form", "normalized_form"):
    value = getattr(morpheme, attr, None)
    if value is None:
      continue
    if callable(value):
      try:
        value = value()
      except TypeError:
        continue
    if isinstance(value, str):
      return value
  return str(morpheme)


def _sudachi_span(morpheme) -> tuple[int | None, int | None]:
  """Extracts the (start, end) indices from a SudachiPy morpheme."""

  start = None
  end = None

  for attr in ("begin", "start"):
    value = getattr(morpheme, attr, None)
    if value is None:
      continue
    if callable(value):
      try:
        value = value()
      except TypeError:
        continue
    try:
      start = int(value)
      break
    except (TypeError, ValueError):
      continue

  for attr in ("end", "stop"):
    value = getattr(morpheme, attr, None)
    if value is None:
      continue
    if callable(value):
      try:
        value = value()
      except TypeError:
        continue
    try:
      end = int(value)
      break
    except (TypeError, ValueError):
      continue

  return (start, end)


def _sudachi_part_of_speech(morpheme) -> Sequence[str] | None:
  """Returns the part-of-speech tuple for a SudachiPy morpheme."""

  if morpheme is None:
    return None

  pos = getattr(morpheme, "part_of_speech", None)
  if pos is None:
    return None

  if callable(pos):
    try:
      pos = pos()
    except TypeError:
      return None

  if isinstance(pos, Sequence):
    return pos
  return None


def _is_punctuation_text(text: str) -> bool:
  """Determines if the entire text is composed of punctuation characters."""

  if not text:
    return False
  return all(unicodedata.category(char).startswith("P") for char in text)


def _classify_token(morpheme, text: str) -> TokenType:
  """Determines the TokenType for a SudachiPy morpheme."""

  pos = _sudachi_part_of_speech(morpheme)
  if pos and pos[0] == "記号":
    return TokenType.PUNCTUATION

  if re.fullmatch(_DIGITS_PATTERN, text):
    return TokenType.NUMBER
  if re.fullmatch(_SLASH_ABBREV_PATTERN, text):
    return TokenType.ACRONYM
  if _is_punctuation_text(text):
    return TokenType.PUNCTUATION

  return TokenType.WORD


def _tokenize_with_sudachi(text: str) -> list[Token] | None:
  """Tokenizes text with SudachiPy when available."""

  tokenizer_obj = _get_sudachi_tokenizer()
  morphemes = _call_sudachi_tokenize(tokenizer_obj, text)
  if not morphemes:
    return None

  tokens: list[Token] = []
  previous_end = 0

  for token_index, morpheme in enumerate(morphemes):
    surface = _sudachi_surface(morpheme)
    if not surface:
      continue

    start_pos, end_pos = _sudachi_span(morpheme)
    if start_pos is None or end_pos is None:
      start_pos = text.find(surface, previous_end)
      if start_pos == -1:
        start_pos = previous_end
      end_pos = start_pos + len(surface)

    if start_pos < previous_end:
      start_pos = previous_end
      end_pos = max(end_pos, start_pos)

    first_after_newline = False
    if token_index > 0:
      gap = text[previous_end:start_pos]
      if "\n" in gap or "\r" in gap:
        first_after_newline = True

    tokens.append(
        Token(
            index=token_index,
            token_type=_classify_token(morpheme, surface),
            char_interval=CharInterval(start_pos=start_pos, end_pos=end_pos),
            first_token_after_newline=first_after_newline,
        )
    )
    previous_end = end_pos

  return tokens if tokens else None


def _tokenize_with_regex(text: str) -> list[Token]:
  """Fallback regex-based tokenization matching the original behavior."""

  tokens: list[Token] = []
  previous_end = 0

  for token_index, match in enumerate(_TOKEN_PATTERN.finditer(text)):
    start_pos, end_pos = match.span()
    matched_text = match.group()

    first_after_newline = False
    if token_index > 0:
      gap = text[previous_end:start_pos]
      if "\n" in gap or "\r" in gap:
        first_after_newline = True

    if re.fullmatch(_DIGITS_PATTERN, matched_text):
      token_type = TokenType.NUMBER
    elif re.fullmatch(_SLASH_ABBREV_PATTERN, matched_text):
      token_type = TokenType.ACRONYM
    elif _WORD_PATTERN.fullmatch(matched_text):
      token_type = TokenType.WORD
    else:
      token_type = TokenType.PUNCTUATION

    tokens.append(
        Token(
            index=token_index,
            token_type=token_type,
            char_interval=CharInterval(start_pos=start_pos, end_pos=end_pos),
            first_token_after_newline=first_after_newline,
        )
    )
    previous_end = end_pos

  return tokens


@debug_utils.debug_log_calls
def tokenize(text: str) -> TokenizedText:
  """Splits text into tokens using SudachiPy when available."""

  sudachi_tokens = _tokenize_with_sudachi(text)

  tokenized = TokenizedText(text=text)
  if sudachi_tokens is not None:
    tokenized.tokens = sudachi_tokens
    return tokenized

  tokenized.tokens = _tokenize_with_regex(text)
  return tokenized


def tokens_text(
    tokenized_text: TokenizedText,
    token_interval: TokenInterval,
) -> str:
  """Reconstructs the substring of the original text spanning a given token interval.

  Args:
    tokenized_text: A TokenizedText object containing token data.
    token_interval: The interval specifying the range [start_index, end_index)
      of tokens.

  Returns:
    The exact substring of the original text corresponding to the token
    interval.

  Raises:
    InvalidTokenIntervalError: If the token_interval is invalid or out of range.
  """
  if (
      token_interval.start_index < 0
      or token_interval.end_index > len(tokenized_text.tokens)
      or token_interval.start_index >= token_interval.end_index
  ):

    raise InvalidTokenIntervalError(
        f"Invalid token interval. start_index={token_interval.start_index}, "
        f"end_index={token_interval.end_index}, "
        f"total_tokens={len(tokenized_text.tokens)}."
    )

  start_token = tokenized_text.tokens[token_interval.start_index]
  end_token = tokenized_text.tokens[token_interval.end_index - 1]
  return tokenized_text.text[
      start_token.char_interval.start_pos : end_token.char_interval.end_pos
  ]


def _is_end_of_sentence_token(
    text: str,
    tokens: Sequence[Token],
    current_idx: int,
    known_abbreviations: Set[str] = _KNOWN_ABBREVIATIONS,
) -> bool:
  """Checks if the punctuation token at `current_idx` ends a sentence.

  A token is considered a sentence terminator and is not part of a known
  abbreviation. Only searches the text corresponding to the current token.

  Args:
    text: The entire input text.
    tokens: The sequence of Token objects.
    current_idx: The current token index to check.
    known_abbreviations: Abbreviations that should not count as sentence enders
      (e.g., "Dr.").

  Returns:
    True if the token at `current_idx` ends a sentence, otherwise False.
  """
  current_token_text = text[
      tokens[current_idx]
      .char_interval.start_pos : tokens[current_idx]
      .char_interval.end_pos
  ]
  if _END_OF_SENTENCE_PATTERN.search(current_token_text):
    if current_idx > 0:
      prev_token_text = text[
          tokens[current_idx - 1]
          .char_interval.start_pos : tokens[current_idx - 1]
          .char_interval.end_pos
      ]
      if f"{prev_token_text}{current_token_text}" in known_abbreviations:
        return False
    return True
  return False


def _is_sentence_break_after_newline(
    text: str,
    tokens: Sequence[Token],
    current_idx: int,
) -> bool:
  """Checks if there's a newline before the next token and if that next token starts uppercase.

  This is a heuristic for determining sentence boundaries. It favors terminating
  a sentence prematurely over missing a sentence boundary, and will terminate a
  sentence early if the first line ends with new line and the second line begins
  with a capital letter.

  Args:
    text: The entire input text.
    tokens: The sequence of Token objects.
    current_idx: The current token index.

  Returns:
    True if a newline is found between current_idx and current_idx+1, and
    the next token (if any) begins with an uppercase character.
  """
  if current_idx + 1 >= len(tokens):
    return False

  gap_text = text[
      tokens[current_idx]
      .char_interval.end_pos : tokens[current_idx + 1]
      .char_interval.start_pos
  ]
  if "\n" not in gap_text:
    return False

  next_token_text = text[
      tokens[current_idx + 1]
      .char_interval.start_pos : tokens[current_idx + 1]
      .char_interval.end_pos
  ]
  return bool(next_token_text) and next_token_text[0].isupper()


def find_sentence_range(
    text: str,
    tokens: Sequence[Token],
    start_token_index: int,
) -> TokenInterval:
  """Finds a 'sentence' interval from a given start index.

  Sentence boundaries are defined by:
    - punctuation tokens in _END_OF_SENTENCE_PATTERN
    - newline breaks followed by an uppercase letter
    - not abbreviations in _KNOWN_ABBREVIATIONS (e.g., "Dr.")

  This favors terminating a sentence prematurely over missing a sentence
  boundary, and will terminate a sentence early if the first line ends with new
  line and the second line begins with a capital letter.

  Args:
    text: The original text.
    tokens: The tokens that make up `text`.
    start_token_index: The token index from which to begin the sentence.

  Returns:
    A TokenInterval representing the sentence range [start_token_index, end). If
    no sentence boundary is found, the end index will be the length of
    `tokens`.

  Raises:
    SentenceRangeError: If `start_token_index` is out of range.
  """
  if start_token_index < 0 or start_token_index >= len(tokens):
    raise SentenceRangeError(
        f"start_token_index={start_token_index} out of range. "
        f"Total tokens: {len(tokens)}."
    )

  i = start_token_index
  while i < len(tokens):
    if tokens[i].token_type == TokenType.PUNCTUATION:
      if _is_end_of_sentence_token(text, tokens, i, _KNOWN_ABBREVIATIONS):
        return TokenInterval(start_index=start_token_index, end_index=i + 1)
    if _is_sentence_break_after_newline(text, tokens, i):
      return TokenInterval(start_index=start_token_index, end_index=i + 1)
    i += 1

  return TokenInterval(start_index=start_token_index, end_index=len(tokens))
