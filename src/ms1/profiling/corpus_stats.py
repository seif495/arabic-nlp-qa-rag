"""
Corpus statistics and distributions for MS1-ANALYSIS-01.

Computes character/token lengths, descriptive statistics, and generates ASCII histograms.
Outputs a JSON summary and a Markdown report.
"""

import json
import statistics
from typing import List, Dict, Any

from src.common.paths import resolve_ms1_paths, make_ms1_output_filename
from src.ms1.ingest.loader import load_transcripts, load_qa_files


def compute_text_stats(texts: List[str]) -> Dict[str, Any]:
    """Calculates summary statistics (count, min, max, mean, median, std) for characters and tokens."""
    if not texts:
        return {}

    char_lengths = [len(t) for t in texts]
    token_lengths = [len(t.split()) for t in texts]

    def get_stats(lengths: List[int]) -> Dict[str, float]:
        return {
            "count": len(lengths),
            "min": float(min(lengths)),
            "max": float(max(lengths)),
            "mean": float(statistics.mean(lengths)),
            "median": float(statistics.median(lengths)),
            "std_dev": float(statistics.stdev(lengths) if len(lengths) > 1 else 0.0),
        }

    return {"characters": get_stats(char_lengths), "tokens": get_stats(token_lengths)}


def generate_ascii_histogram(data: List[int], bins: int = 10) -> str:
    """Generates a simple ASCII histogram for data arrays without requiring external plotting libraries."""
    if not data:
        return "No data for histogram."

    min_val, max_val = min(data), max(data)
    if min_val == max_val:
        return f"All values are exactly {min_val}"

    bin_width = (max_val - min_val) / bins
    histogram = [0] * bins

    for val in data:
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx == bins:
            bin_idx -= 1  # Include the absolute maximum in the last bin
        histogram[bin_idx] += 1

    max_count = max(histogram)
    lines = []

    for i in range(bins):
        bin_start = min_val + i * bin_width
        bin_end = min_val + (i + 1) * bin_width
        # Scale bar up to 40 characters maximum
        bar_length = int(40 * histogram[i] / max_count) if max_count > 0 else 0
        bar = "█" * bar_length
        lines.append(f"{bin_start:6.1f} - {bin_end:6.1f} | {histogram[i]:4d} | {bar}")

    return "\n".join(lines)


def run_corpus_analysis(paths=None, *, verbose: bool = True) -> None:
    """Main function to generate analysis artifacts."""
    if paths is None:
        paths = resolve_ms1_paths()

    if verbose:
        print("Loading data...")
    transcripts = load_transcripts(paths)
    qa_records = load_qa_files(paths)

    # Extract raw texts
    transcript_contents = [t["content"] for t in transcripts]
    questions = [r["question"] for r in qa_records if "question" in r]
    answers = [r["answer"] for r in qa_records if "answer" in r]

    if verbose:
        print("Computing lengths and statistics...")
    stats = {
        "transcripts": compute_text_stats(transcript_contents),
        "questions": compute_text_stats(questions),
        "answers": compute_text_stats(answers),
    }

    output_dir = paths.experiments_ms1
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw JSON
    stats_filename = make_ms1_output_filename("profiling", "corpus_stats", 1, "json")
    stats_path = output_dir / stats_filename
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Compatibility artifact for Alice ticket expectation
    legacy_stats_path = output_dir / "corpus_stats.json"
    with open(legacy_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Calculate array lengths for the visual histograms
    t_tokens = [len(t.split()) for t in transcript_contents]
    q_tokens = [len(q.split()) for q in questions]
    a_tokens = [len(a.split()) for a in answers]

    # Generate the Markdown Report exactly as Charly mandated
    md_filename = make_ms1_output_filename("profiling", "corpus_summary", 1, "md")
    md_path = output_dir / md_filename

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Corpus Distribution Analysis\n\n")

        f.write("## Summary Statistics Table (Tokens)\n\n")
        f.write("| Component | Count | Min | Max | Mean | Median | Std Dev |\n")
        f.write("|-----------|-------|-----|-----|------|--------|---------|\n")
        for key in ["transcripts", "questions", "answers"]:
            ts = stats[key]["tokens"]
            f.write(
                f"| {key.capitalize()} | {ts['count']} | {ts['min']} | {ts['max']} | {ts['mean']:.1f} | {ts['median']} | {ts['std_dev']:.1f} |\n"
            )

        f.write("\n## Summary Statistics Table (Characters)\n\n")
        f.write("| Component | Count | Min | Max | Mean | Median | Std Dev |\n")
        f.write("|-----------|-------|-----|-----|------|--------|---------|\n")
        for key in ["transcripts", "questions", "answers"]:
            cs = stats[key]["characters"]
            f.write(
                f"| {key.capitalize()} | {cs['count']} | {cs['min']} | {cs['max']} | {cs['mean']:.1f} | {cs['median']} | {cs['std_dev']:.1f} |\n"
            )

        f.write("\n## Histograms (Token Lengths)\n\n")
        f.write("### Transcripts\n```text\n")
        f.write(generate_ascii_histogram(t_tokens))
        f.write("\n```\n\n### Questions\n```text\n")
        f.write(generate_ascii_histogram(q_tokens))
        f.write("\n```\n\n### Answers\n```text\n")
        f.write(generate_ascii_histogram(a_tokens))
        f.write("\n```\n\n")

        f.write("## Short Observation Notes\n\n")
        f.write(
            "- Auto-generated report; add interpretation notes in the notebook export step.\n"
        )
    if verbose:
        print(f"Stats saved: {stats_path}")
        print(f"Stats saved: {legacy_stats_path}")
        print(f"Markdown report (with histograms) saved: {md_path}")
        print(
            "DONE! Please review the markdown report and fill in the 'Short Observation Notes' section."
        )


if __name__ == "__main__":
    run_corpus_analysis()
