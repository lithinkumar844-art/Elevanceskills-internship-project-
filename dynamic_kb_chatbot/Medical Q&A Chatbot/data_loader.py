"""
data_loader.py — Parses MedQuAD XML files into a flat pandas DataFrame.

Each row:  question | answer | focus | qtype | source | url
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Question types that have rich answers (skip "susceptibility" etc. if needed)
SKIP_EMPTY_ANSWERS = True


def parse_xml_file(path: Path) -> list[dict]:
    """Parse one MedQuAD XML file into a list of QA dicts."""
    records = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        source = root.attrib.get("source", "")
        url    = root.attrib.get("url", "")
        focus  = (root.findtext("Focus") or "").strip()

        for pair in root.findall(".//QAPair"):
            q_el = pair.find("Question")
            a_el = pair.find("Answer")

            if q_el is None or a_el is None:
                continue

            question = (q_el.text or "").strip()
            answer   = (a_el.text or "").strip()
            qtype    = q_el.attrib.get("qtype", "general")

            if SKIP_EMPTY_ANSWERS and len(answer) < 30:
                continue

            records.append({
                "question": question,
                "answer":   answer,
                "focus":    focus,
                "qtype":    qtype,
                "source":   source,
                "url":      url,
            })

    except ET.ParseError as e:
        logger.warning("XML parse error in %s: %s", path, e)
    except Exception as e:
        logger.warning("Error loading %s: %s", path, e)

    return records


def load_dataset(data_dir: str | Path, max_files: int = 0) -> pd.DataFrame:
    """
    Recursively find all XML files under data_dir and parse them.

    Args:
        data_dir:  Root folder containing the MedQuAD subfolders
        max_files: 0 = load all; N = stop after N files (for quick testing)

    Returns:
        DataFrame with columns: question, answer, focus, qtype, source, url
    """
    data_dir = Path(data_dir)
    xml_files = sorted(data_dir.rglob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {data_dir}")

    if max_files:
        xml_files = xml_files[:max_files]

    all_records: list[dict] = []
    for fp in xml_files:
        all_records.extend(parse_xml_file(fp))

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["question"])
    df = df.reset_index(drop=True)

    logger.info("Loaded %d QA pairs from %d files.", len(df), len(xml_files))
    return df
