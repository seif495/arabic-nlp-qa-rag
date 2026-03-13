# Corpus Distribution Analysis

## Summary Statistics Table (Tokens)

| Component | Count | Min | Max | Mean | Median | Std Dev |
|-----------|-------|-----|-----|------|--------|---------|
| Transcripts | 13 | 4329.0 | 9393.0 | 6875.2 | 6195.0 | 1595.7 |
| Questions | 3890 | 3.0 | 9.0 | 6.1 | 6.0 | 1.4 |
| Answers | 3890 | 1.0 | 10.0 | 4.9 | 5.0 | 1.6 |

## Summary Statistics Table (Characters)

| Component | Count | Min | Max | Mean | Median | Std Dev |
|-----------|-------|-----|-----|------|--------|---------|
| Transcripts | 13 | 26431.0 | 59094.0 | 42638.0 | 38574.0 | 9810.5 |
| Questions | 3890 | 14.0 | 38.0 | 29.5 | 31.0 | 6.0 |
| Answers | 3890 | 3.0 | 52.0 | 27.9 | 28.0 | 8.2 |

## Histograms (Token Lengths)

### Transcripts
```text
4329.0 - 4835.4 |    1 | █████████████
4835.4 - 5341.8 |    1 | █████████████
5341.8 - 5848.2 |    2 | ██████████████████████████
5848.2 - 6354.6 |    3 | ████████████████████████████████████████
6354.6 - 6861.0 |    0 | 
6861.0 - 7367.4 |    1 | █████████████
7367.4 - 7873.8 |    1 | █████████████
7873.8 - 8380.2 |    1 | █████████████
8380.2 - 8886.6 |    1 | █████████████
8886.6 - 9393.0 |    2 | ██████████████████████████
```

### Questions
```text
   3.0 -    3.6 |   30 | 
   3.6 -    4.2 |  750 | ██████████████████████
   4.2 -    4.8 |    0 | 
   4.8 -    5.4 |  210 | ██████
   5.4 -    6.0 |    0 | 
   6.0 -    6.6 | 1350 | ████████████████████████████████████████
   6.6 -    7.2 | 1260 | █████████████████████████████████████
   7.2 -    7.8 |    0 | 
   7.8 -    8.4 |    0 | 
   8.4 -    9.0 |  290 | ████████
```

### Answers
```text
   1.0 -    1.9 |   11 | 
   1.9 -    2.8 |  128 | █████
   2.8 -    3.7 |  674 | ███████████████████████████████
   3.7 -    4.6 |  855 | ███████████████████████████████████████
   4.6 -    5.5 |  858 | ████████████████████████████████████████
   5.5 -    6.4 |  666 | ███████████████████████████████
   6.4 -    7.3 |  434 | ████████████████████
   7.3 -    8.2 |  195 | █████████
   8.2 -    9.1 |   64 | ██
   9.1 -   10.0 |    5 | 
```

## Short Observation Notes


<!-- AUTO-NOTES-START -->
*(Auto-generated from notebook metrics. Re-run export cell to refresh.)*
- **Outliers**: Transcript token length spans from 4329 to 9393, while estimated video duration spans from 21:42 to 51:01.
- **Short/Long samples**: Shortest video is **تاج محل  الدحيح** (21:42); longest is **كيف تحولت روسيا إلى إمبراطورية؟  الدحيح** (51:01).
- **Imbalance**: Transcript mean length is 6875.23 tokens versus QA means of 6.08 (questions) and 4.94 (answers).
- **General shape**: Questions and answers are short-form (roughly 3–9 and 1–10 tokens), which supports concise retrieval units compared with long transcript contexts.
<!-- AUTO-NOTES-END -->

<!-- AUTO-DURATION-START -->
## Duration Analysis Addendum (Auto-Generated)

- Generated from notebook on: 2026-03-13 20:03:13
- Videos analyzed: 13
- Average video length: 35:20
- Median video length: 31:28
- Min/Max video length: 21:42 / 51:01
- Average words per minute (rough): 194.97
- Mean transcript tokens: 6875.23
- Mean question tokens: 6.08
- Mean answer tokens: 4.94

### Shortest 3 Videos
- تاج محل  الدحيح — 21:42
- هل Citizen Kane أفضل فيلم في التاريخ؟  الدحيح — 28:09
- الساموراي  الدحيح — 28:57

### Longest 3 Videos
- كيف تحولت روسيا إلى إمبراطورية؟  الدحيح — 51:01
- جون كينيدي  الدحيح — 45:55
- منابع النيل  الدحيح — 43:59

### Video ID Mapping

| ID | Video Title | Duration |
|---|---|---|
| V01 | تاج محل  الدحيح | 21:42 |
| V02 | هل Citizen Kane أفضل فيلم في التاريخ؟  الدحيح | 28:09 |
| V03 | الساموراي  الدحيح | 28:57 |
| V04 | مصير الأرض و الشمس و كل شيء  الدحيح | 29:58 |
| V05 | الأخطبوط  الدحيح | 30:32 |
| V06 | أعظم طائرة حربية  الدحيح | 30:38 |
| V07 | كيف تنقل جبل وزنه 30 طن قبل أن يغرق؟  الدحيح | 31:28 |
| V08 | فيزياء و فلسفة الحركة  الدحيح | 36:38 |
| V09 | كيف تسيطر على عقول البشر؟  الدحيح | 37:18 |
| V10 | معركة ذي قار  الدحيح | 43:05 |
| V11 | منابع النيل  الدحيح | 43:59 |
| V12 | جون كينيدي  الدحيح | 45:55 |
| V13 | كيف تحولت روسيا إلى إمبراطورية؟  الدحيح | 51:01 |

<!-- AUTO-DURATION-END -->
