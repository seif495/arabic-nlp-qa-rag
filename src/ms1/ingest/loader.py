import os
from pathlib import Path
from typing import List, Optional
import csv

from src.common.paths import resolve_ms1_paths, MS1Paths

# Load all 13 .txt files into a list.
def load_transcripts(paths: Optional[MS1Paths] = None) -> List[dict]:
    """
    Load all transcript .txt files from the external data directory.
    
    Args:
        paths: MS1Paths object. If None, resolves paths automatically.
    
    Returns:
        List of transcript records with filename, video_id, and content.
    
    Raises:
        FileNotFoundError: If transcripts directory does not exist.
        ValueError: If no .txt files are found.
    """
    if paths is None:
        paths = resolve_ms1_paths()
    
    dir_path = paths.data_external / "transcripts"
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {dir_path}")
    
    # Find all .txt files, sorted for determinism
    txt_files = sorted(dir_path.glob("*.txt"))
    
    if not txt_files:
        raise ValueError(f"No .txt files found in {dir_path}")
    
    transcripts = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            transcripts.append({
                "filename": file_path.name,
                "video_id": file_path.stem,
                "content": f.read()
            })

    return transcripts

# Load all 13 .csv files into a list of dicts.
def load_qa_files(paths: Optional[MS1Paths] = None) -> List[dict]:
    """
    Load all QA .csv files from the external data directory into structured records.
    
    Args:
        paths: MS1Paths object. If None, resolves paths automatically.
    
    Returns:
        List of dictionaries representing QA records.
    
    Raises:
        FileNotFoundError: If qa directory does not exist.
        ValueError: If no .csv files are found.
    """
    if paths is None:
        paths = resolve_ms1_paths()
    
    dir_path = paths.data_external / "qa"
    
    if not dir_path.exists():
        raise FileNotFoundError(f"QA directory not found: {dir_path}")
    
    csv_files = sorted(dir_path.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No .csv files found in {dir_path}")
    
    
    qa_records = []
    for file_path in csv_files:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["source_file"] = file_path.name
                    row["source_video_id"] = file_path.stem
                    qa_records.append(row)
    
    return qa_records

""" Check that each transcript has a matching QA file.
    This is done by removing the | character and collapsing whitespace in both the transcript
    filename (stem) and the video_title column in the QA CSVs, then checking for matches.
"""
def _normalize_title(title: str) -> str:
    """Normalize title by removing | and collapsing whitespace."""
    import re
    # Remove pipe character, collapse multiple spaces into one, strip edges
    normalized = title.replace("|", " ")
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def validate_pairing(paths: Optional[MS1Paths] = None) -> dict:
    """
    Validate that each transcript file has a corresponding QA entry via video_title.
    
    Matches transcript filenames (stems) against the video_title column in QA CSVs.
    Normalizes both by removing '|' and collapsing whitespace.
    
    Args:
        paths: MS1Paths object. If None, resolves paths automatically.
    
    Returns:
        Dictionary mapping transcript video_ids to their matched QA video_titles.
    
    Raises:
        ValueError: If transcripts and QA video_titles don't match.
    """
    if paths is None:
        paths = resolve_ms1_paths()
    
    transcripts_path = paths.data_external / "transcripts"
    qa_path = paths.data_external / "qa"
    
    # Get transcript filenames (stems), normalized
    transcript_ids = {_normalize_title(f.stem) for f in transcripts_path.glob("*.txt")}
    
    # Extract unique video_titles from QA CSVs, normalized
    qa_video_titles = set()
    for csv_file in qa_path.glob("*.csv"):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "video_title" in row and row["video_title"].strip():
                    qa_video_titles.add(_normalize_title(row["video_title"]))
                    break  # Only need one per file (assuming all rows have same title)
    
    # Check for mismatches
    missing_qa = transcript_ids - qa_video_titles
    missing_transcripts = qa_video_titles - transcript_ids
    
    errors = []
    if missing_qa:
        errors.append(f"Transcripts with no matching QA video_title: {sorted(missing_qa)}")
    if missing_transcripts:
        errors.append(f"QA video_titles with no matching transcript: {sorted(missing_transcripts)}")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    # Return the mapping for downstream use
    return {vid: vid for vid in transcript_ids}

# Check for missing/malformed data and report errors.
def validate_records(qa_records: List[dict]) -> None:
    """
    Validate QA records for missing or malformed data.
    
    Args:
        qa_records: List of QA record dictionaries to validate.
    
    Raises:
        ValueError: If records have missing required fields or malformed data.
    """
    if not qa_records:
        raise ValueError("No QA records to validate")
    
    required_fields = {"question", "answer"}
    errors = []
    
    for idx, record in enumerate(qa_records):
        missing_fields = required_fields - set(record.keys())
        if missing_fields:
            errors.append(f"Record {idx}: missing fields {sorted(missing_fields)}")
        
        for field in required_fields:
            value = record.get(field, "").strip()
            if not value:
                errors.append(f"Record {idx}: field '{field}' is empty or whitespace")
    
    if errors:
        raise ValueError(f"QA record validation failed:\n" + "\n".join(errors))