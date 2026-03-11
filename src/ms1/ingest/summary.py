"""
Ingest summary generation for MS1.

Runs all loaders and validators, then saves standardized output artifacts.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.common.paths import resolve_ms1_paths, MS1Paths, make_ms1_output_filename
from src.ms1.ingest.loader import (
    load_transcripts,
    load_qa_files,
    validate_pairing,
    validate_records,
)


def generate_ingest_summary(paths: Optional[MS1Paths] = None) -> dict:
    """
    Run all loaders and validators, then save summary artifacts.
    
    Outputs are saved to experiments/ms1/ with proper naming convention:
    - ms1_ingest_loader_summary_v001.json
    - ms1_ingest_dataset_inventory_v001.md
    - ms1_ingest_validation_summary_v001.md
    
    Args:
        paths: MS1Paths object. If None, resolves paths automatically.
    
    Returns:
        Summary dictionary with all stats and validation results.
    """
    if paths is None:
        paths = resolve_ms1_paths()
    
    output_path = paths.experiments_ms1
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "transcripts": {},
        "qa": {},
        "validation": {},
        "errors": []
    }
    
    # --- Load transcripts ---
    try:
        transcripts = load_transcripts(paths)
        summary["transcripts"] = {
            "file_count": len(transcripts),
            "files": [t["filename"] for t in transcripts],
            "total_characters": sum(len(t["content"]) for t in transcripts),
            "status": "success"
        }
    except Exception as e:
        summary["transcripts"]["status"] = "failed"
        summary["errors"].append(f"Transcript loading: {str(e)}")
        transcripts = []
    
    # --- Load QA files ---
    try:
        qa_records = load_qa_files(paths)
        # Get unique source files
        source_files = sorted(set(r["source_file"] for r in qa_records))
        summary["qa"] = {
            "file_count": len(source_files),
            "files": source_files,
            "total_records": len(qa_records),
            "status": "success"
        }
    except Exception as e:
        summary["qa"]["status"] = "failed"
        summary["errors"].append(f"QA loading: {str(e)}")
        qa_records = []
    
    # --- Validate pairing ---
    try:
        validate_pairing(paths)
        summary["validation"]["pairing"] = "pass"
    except ValueError as e:
        summary["validation"]["pairing"] = "fail"
        summary["errors"].append(f"Pairing: {str(e)}")
    
    # --- Validate records ---
    try:
        validate_records(qa_records)
        summary["validation"]["records"] = "pass"
    except ValueError as e:
        summary["validation"]["records"] = "fail"
        summary["errors"].append(f"Records: {str(e)}")
    
    # --- Save outputs with correct naming convention ---
    
    # 1. ms1_ingest_loader_summary_v001.json
    summary_filename = make_ms1_output_filename("ingest", "loader_summary", 1, "json")
    summary_file = output_path / summary_filename
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 2. ms1_ingest_dataset_inventory_v001.md
    inventory_filename = make_ms1_output_filename("ingest", "dataset_inventory", 1, "md")
    inventory_file = output_path / inventory_filename
    with open(inventory_file, "w", encoding="utf-8") as f:
        f.write("# Dataset Inventory\n\n")
        f.write(f"Generated: {summary['generated_at']}\n\n")
        
        f.write("## Transcripts\n\n")
        f.write("| # | Filename | Video ID |\n")
        f.write("|---|----------|----------|\n")
        if "files" in summary["transcripts"]:
            for i, fname in enumerate(summary["transcripts"]["files"], 1):
                vid = Path(fname).stem
                f.write(f"| {i} | {fname} | {vid} |\n")
        
        f.write("\n## QA Files\n\n")
        f.write("| # | Filename | Video ID |\n")
        f.write("|---|----------|----------|\n")
        if "files" in summary["qa"]:
            for i, fname in enumerate(summary["qa"]["files"], 1):
                vid = Path(fname).stem
                f.write(f"| {i} | {fname} | {vid} |\n")
        
        f.write(f"\n## Summary\n\n")
        f.write(f"- Total transcripts: {summary['transcripts'].get('file_count', 0)}\n")
        f.write(f"- Total QA files: {summary['qa'].get('file_count', 0)}\n")
        f.write(f"- Total QA records: {summary['qa'].get('total_records', 0)}\n")
    
    # 3. ms1_ingest_validation_summary_v001.md
    validation_filename = make_ms1_output_filename("ingest", "validation_summary", 1, "md")
    validation_file = output_path / validation_filename
    with open(validation_file, "w", encoding="utf-8") as f:
        f.write("# Validation Summary\n\n")
        f.write(f"Generated: {summary['generated_at']}\n\n")
        
        f.write("## Check Results\n\n")
        f.write("| Check | Result |\n")
        f.write("|-------|--------|\n")
        f.write(f"| Transcript loading | {summary['transcripts'].get('status', 'N/A')} |\n")
        f.write(f"| QA loading | {summary['qa'].get('status', 'N/A')} |\n")
        f.write(f"| Pairing validation | {summary['validation'].get('pairing', 'N/A')} |\n")
        f.write(f"| Record validation | {summary['validation'].get('records', 'N/A')} |\n")
        
        if summary["errors"]:
            f.write("\n## Errors\n\n")
            for err in summary["errors"]:
                f.write(f"- {err}\n")
        else:
            f.write("\n## Status\n\n")
            f.write("All validations passed successfully.\n")
    
    print(f"Saved: {summary_file}")
    print(f"Saved: {inventory_file}")
    print(f"Saved: {validation_file}")
    
    return summary


# Allow running directly: python -m src.ms1.ingest.summary
if __name__ == "__main__":
    generate_ingest_summary()
