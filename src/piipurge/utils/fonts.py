import io
import sys
import csv
import fitz
import logging
import platform
import subprocess
from .. import consts
from pathlib import Path
from functools import cache
from difflib import SequenceMatcher
import matplotlib.font_manager as fm
from fontTools.ttLib import TTFont, TTLibFileIsCollectionError


logging.basicConfig(format=consts.LOG_FORMAT, level=logging.INFO)


DEFAULT_PYMUPDF_FONTS = [
    "Times-Roman",
    "Times-Bold",
    "Times-Italic",
    "Times-BoldItalic",
    "Helvetica",
    "Helvetica-Bold",
    "Helvetica-Oblique",
    "Helvetica-BoldOblique",
    "Courier",
    "Courier-Bold",
    "Courier-Oblique",
    "Courier-BoldOblique",
    "Symbol",
    "ZapfDingbats",
]
DEFAULT_TEXT_MEASUREMENT_FONT = "helv"
DEFAULT_PYMUPDF_FONT = {
    "normal": "Segoe UI",
    "italic": "Segoe UI Italic",
    "bold": "Segoe UI Bold",
}
DEFAULT_FONT_SIZE = 11


@cache
def get_system_fonts_metadata():
    font_paths = fm.findSystemFonts(fontext="ttf")
    metadata = [get_font_metadata(p) for p in font_paths]
    metadata = [f for f in metadata if f["family"] != "Unknown"]

    return metadata


def get_replacement_fonts():
    fonts = {}

    with open(consts.REPLACEMENT_FONTS_PATH) as fh:
        csv_reader = csv.reader(
            fh,
            delimiter=",",
        )
        _ = next(csv_reader)
        for row in csv_reader:
            fonts[row[0]] = row[1:]

    return fonts


def get_font_metadata(font_path):
    try:
        metadata = {}
        font = TTFont(font_path)
        for record in font["name"].names:
            name = record.toUnicode()
            metadata[record.nameID] = name
    except TTLibFileIsCollectionError:
        print(
            f"(fonts_utils.get_font_metadata): Failed to retrieve font metadata for font at path '{font_path}."
        )

    return {
        "font_path": font_path,
        "family": metadata.get(1, "Unknown"),
        "sub_family": metadata.get(2, "Unknown"),
        "full_name": metadata.get(4, "Unknown"),
        "version": metadata.get(5, "Unknown"),
    }


@cache
def get_matching_font(font_name, similarity_threshold=0.75):
    system_fonts_metadata = get_system_fonts_metadata()
    replacement_fonts = get_replacement_fonts()
    fonts_to_search = [font_name, *(replacement_fonts.get(font_name, []))]
    seq_match = SequenceMatcher()

    matches = []

    for idx, font in enumerate(fonts_to_search):
        seq_match.set_seq2(font)
        for font_metadata in system_fonts_metadata:
            seq_match.set_seq1(font_metadata["full_name"])
            similarity_score = seq_match.ratio()
            if similarity_score > similarity_threshold:
                # the higher ranking fonts should be the ones at the
                # beginning of the list.
                font_weight = len(fonts_to_search) + 1 - idx
                matches.append((font_metadata, font_weight * similarity_score))

    if matches:
        matches = sorted(matches, key=lambda i: i[1], reverse=True)
        return matches[0][0]

    default_font_name = (
        DEFAULT_PYMUPDF_FONT["bold"]
        if "bold" in font_name
        else (
            DEFAULT_PYMUPDF_FONT["italic"]
            if "italic" in font_name
            else DEFAULT_PYMUPDF_FONT["normal"]
        )
    )
    default_font_metadata = get_font_metadata(get_font_path(default_font_name))

    return default_font_metadata


def get_font_path(font_name):
    """Find the system path of a given font by name."""
    if sys.platform == "win32":
        font_dirs = [Path("C:/Windows/Fonts")]
    elif sys.platform == "darwin":  # macOS
        font_dirs = [
            Path("~/Library/Fonts").expanduser(),
            Path("/System/Library/Fonts/Supplemental"),
            Path("/Library/Fonts"),
        ]
    else:  # Linux
        font_dirs = [
            Path("~/.fonts").expanduser(),
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
        ]

    for font_dir in font_dirs:
        if font_dir.exists():
            for font_path in font_dir.rglob("*.ttf"):
                font_metadata = get_font_metadata(font_path)
                if (
                    font_name == f"{font_metadata['family']}"
                    or font_name
                    == f"{font_metadata["family"]}-{font_metadata["sub_family"]}"
                ):
                    return str(font_path)

    return None


def load_font(font_name):
    font_path = get_font_path(font_name)
    if font_path:
        with open(font_path, "rb") as fh:
            return io.BytesIO(fh.read())

    return None


def get_text_length(text, font_name=DEFAULT_PYMUPDF_FONT, font_size=DEFAULT_FONT_SIZE):
    if (
        font_name not in DEFAULT_PYMUPDF_FONTS
        or font_name in DEFAULT_PYMUPDF_FONT.values()
        or not get_font_path(font_name)
    ):
        font_name = DEFAULT_TEXT_MEASUREMENT_FONT

    return sum(
        [fitz.get_text_length(c, fontname=font_name, fontsize=font_size) for c in text]
    )


def get_windows_default_font():
    try:
        import winreg

        reg_key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
        )
        i = 0
        fonts = []
        while True:
            try:
                font_name, font_file, _ = winreg.EnumValue(reg_key, i)
                fonts.append((font_name, font_file))
                i += 1
            except OSError:
                break
        winreg.CloseKey(reg_key)

        # Look for "Segoe UI" or "Segoe UI Variable"
        for font_name, font_file in fonts:
            if "Segoe UI" in font_name:
                return font_name
        return "Unknown"
    except ImportError:
        return "winreg module not available (not running on Windows)"
    except Exception as e:
        return f"Windows font detection error: {e}"


def get_linux_default_font():
    try:
        output = (
            subprocess.check_output(["fc-match", "--format=%{family}\n"])
            .decode()
            .strip()
        )
        return output
    except FileNotFoundError:
        return "'fc-match' not found — install fontconfig"
    except Exception as e:
        return f"Linux font detection error: {e}"


def get_system_default_font():
    os_platform = platform.system().lower()

    if os_platform == "windows":
        return get_windows_default_font()
    elif os_platform == "linux":
        return get_linux_default_font()

    logging.error(f"Unsupported OS platform {platform.system()}.")
    raise f"Unsupported OS platform {platform.system()}."
