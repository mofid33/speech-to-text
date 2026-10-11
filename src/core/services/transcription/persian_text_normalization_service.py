import re
import unicodedata

from src.infrastructure.di.inject import inject


@inject
class PersianTextNormalizationService:
    _ZWNJ = "\u200c"

    _CHARACTER_TRANSLATION = str.maketrans(
        {
            "\u064a": "\u06cc",  # Arabic Yeh -> Persian Yeh
            "\u0649": "\u06cc",  # Alef Maksura -> Persian Yeh
            "\u0643": "\u06a9",  # Arabic Kaf -> Persian Kaf
            "\u0640": "",  # Tatweel
            "\u200b": "",  # Zero Width Space
            "\u200d": "",  # Zero Width Joiner
            "\u200e": "",  # Left-to-Right Mark
            "\u200f": "",  # Right-to-Left Mark
            "\u202a": "",  # Left-to-Right Embedding
            "\u202b": "",  # Right-to-Left Embedding
            "\u202c": "",  # Pop Directional Formatting
            "\u202d": "",  # Left-to-Right Override
            "\u202e": "",  # Right-to-Left Override
            "\u2060": "",  # Word Joiner
            "\u2066": "",  # Left-to-Right Isolate
            "\u2067": "",  # Right-to-Left Isolate
            "\u2068": "",  # First Strong Isolate
            "\u2069": "",  # Pop Directional Isolate
            "\ufeff": "",  # BOM / Zero Width No-Break Space
        }
    )

    _HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[^\S\n]+")
    _SPACES_AROUND_NEWLINE_PATTERN = re.compile(r" *\n *")
    _SPACES_AROUND_ZWNJ_PATTERN = re.compile(r" *\u200c *")
    _MULTIPLE_ZWNJ_PATTERN = re.compile(r"\u200c{2,}")
    _EXCESS_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")

    _VERB_PREFIX_PATTERN = re.compile(
        r"(?<![\w\u200c])(ن?می) +(?=\S)"
    )

    _PLURAL_SUFFIX_PATTERN = re.compile(
        r"(?<=\S) +(های|ها)(?=(?:\s|$|[،؛:,.!?؟]))"
    )

    def __init__(self):
        pass

    def normalize_text(
        self,
        text: str,
    ) -> str:
        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "Persian text normalization expects a string"
            )

        normalized = unicodedata.normalize(
            "NFKC",
            text,
        )

        normalized = normalized.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        normalized = normalized.translate(
            self._CHARACTER_TRANSLATION
        )

        normalized = (
            self._HORIZONTAL_WHITESPACE_PATTERN.sub(
                " ",
                normalized,
            )
        )

        normalized = (
            self._SPACES_AROUND_NEWLINE_PATTERN.sub(
                "\n",
                normalized,
            )
        )

        normalized = (
            self._SPACES_AROUND_ZWNJ_PATTERN.sub(
                self._ZWNJ,
                normalized,
            )
        )

        normalized = (
            self._VERB_PREFIX_PATTERN.sub(
                rf"\1{self._ZWNJ}",
                normalized,
            )
        )

        normalized = (
            self._PLURAL_SUFFIX_PATTERN.sub(
                rf"{self._ZWNJ}\1",
                normalized,
            )
        )

        normalized = (
            self._MULTIPLE_ZWNJ_PATTERN.sub(
                self._ZWNJ,
                normalized,
            )
        )

        normalized = "\n".join(
            line.strip()
            for line in normalized.split(
                "\n"
            )
        )

        normalized = (
            self._EXCESS_BLANK_LINES_PATTERN.sub(
                "\n\n",
                normalized,
            )
        )

        return normalized.strip()