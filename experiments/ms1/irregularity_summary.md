# Pattern and Noise Detection Summary

Generated: 2026-03-13T22:01:00.816414

## Dataset Coverage

- Transcript files: 13
- QA records: 3890
- Text units analyzed: 23148

## Irregularity Frequency Table

| Category                  | Count | Percentage of Text Units |
| ------------------------- | ----: | -----------------------: |
| orthographic_variation    | 15953 |                   68.92% |
| formatting_artifacts      | 11204 |                   48.40% |
| code_switching            |   966 |                    4.17% |
| punctuation_inconsistency |   158 |                    0.68% |
| repetition_emphasis       |   119 |                    0.51% |

## Example Irregularity Showcase

### orthographic_variation

- Count: 15953 (68.92% of text units)

| Signal               | Count |
| -------------------- | ----: |
| variant_token_family | 13361 |
| mixed_alif_forms     |  7363 |
| diacritics_presence  |  5304 |
| ya_alef_maksura_mix  |  2110 |
| tatweel_presence     |   982 |

1. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 3
   - **Signal:** mixed_alif_forms
   - **Reason:** Multiple alif letter forms co-occur in the same text unit.
   - **Excerpt:** 4.238: عملنا أفجر طيارة في تاريخ "أمريكا".
2. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 4
   - **Signal:** diacritics_presence
   - **Reason:** Arabic diacritics detected.
   - **Excerpt:** 6.184: أنا متحمس جدًا من امبارح،
3. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 4
   - **Signal:** mixed_alif_forms
   - **Reason:** Multiple alif letter forms co-occur in the same text unit.
   - **Excerpt:** 6.184: أنا متحمس جدًا من امبارح،
4. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 5
   - **Signal:** diacritics_presence
   - **Reason:** Arabic diacritics detected.
   - **Excerpt:** 8.308: ها، ورّيني!
5. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 6
   - **Signal:** mixed_alif_forms
   - **Reason:** Multiple alif letter forms co-occur in the same text unit.
   - **Excerpt:** 9.435: أقدم لحضرتك فخر الطيران الأمريكي

### formatting_artifacts

- Count: 11204 (48.40% of text units)

| Signal           | Count |
| ---------------- | ----: |
| timestamp_prefix | 11109 |
| bullet_prefix    |    80 |
| symbol_clutter   |    22 |
| multi_space      |     9 |

1. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 1
   - **Signal:** timestamp_prefix
   - **Reason:** Timestamp-like prefix detected.
   - **Excerpt:** 0.0: سيادة الكولونيل، صبرك في محله،
2. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 2
   - **Signal:** timestamp_prefix
   - **Reason:** Timestamp-like prefix detected.
   - **Excerpt:** 3.076: مبروك علينا،
3. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 3
   - **Signal:** timestamp_prefix
   - **Reason:** Timestamp-like prefix detected.
   - **Excerpt:** 4.238: عملنا أفجر طيارة في تاريخ "أمريكا".
4. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 4
   - **Signal:** timestamp_prefix
   - **Reason:** Timestamp-like prefix detected.
   - **Excerpt:** 6.184: أنا متحمس جدًا من امبارح،
5. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 5
   - **Signal:** timestamp_prefix
   - **Reason:** Timestamp-like prefix detected.
   - **Excerpt:** 8.308: ها، ورّيني!

### code_switching

- Count: 966 (4.17% of text units)

| Signal               | Count |
| -------------------- | ----: |
| latin_token_presence |   966 |

1. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 7
   - **Signal:** latin_token_presence
   - **Reason:** Latin token(s) detected: F-35
   - **Excerpt:** 12.205: الـF-35.
2. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 21
   - **Signal:** latin_token_presence
   - **Reason:** Latin token(s) detected: Target
   - **Excerpt:** تـTarget أطفال بالتحديد يعني...
3. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 35
   - **Signal:** latin_token_presence
   - **Reason:** Latin token(s) detected: Model
   - **Excerpt:** 63.501: اللي هو بيفكر يشتري الـModel دا يعني.
4. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 50
   - **Signal:** latin_token_presence
   - **Reason:** Latin token(s) detected: Sensor
   - **Excerpt:** 89.125: هل فيها Sensor رَكنة؟
5. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 51
   - **Signal:** latin_token_presence
   - **Reason:** Latin token(s) detected: Sensor
   - **Excerpt:** 91.081: Sensor رَكنة؟!

### punctuation_inconsistency

- Count: 158 (0.68% of text units)

| Signal                    | Count |
| ------------------------- | ----: |
| ellipsis_pattern          |   154 |
| repeated_punctuation      |   154 |
| mixed_punctuation_systems |     3 |
| space_before_punctuation  |     1 |

1. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 15
   - **Signal:** repeated_punctuation
   - **Reason:** Repeated punctuation marks detected.
   - **Excerpt:** 29.108: كمبيونر نسـ....
2. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 15
   - **Signal:** ellipsis_pattern
   - **Reason:** Ellipsis punctuation pattern found.
   - **Excerpt:** 29.108: كمبيونر نسـ....
3. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 18
   - **Signal:** repeated_punctuation
   - **Reason:** Repeated punctuation marks detected.
   - **Excerpt:** 34.878: مش عارف، يا فندم، بصراحة...
4. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 18
   - **Signal:** ellipsis_pattern
   - **Reason:** Ellipsis punctuation pattern found.
   - **Excerpt:** 34.878: مش عارف، يا فندم، بصراحة...
5. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 21
   - **Signal:** repeated_punctuation
   - **Reason:** Repeated punctuation marks detected.
   - **Excerpt:** تـTarget أطفال بالتحديد يعني...

### repetition_emphasis

- Count: 119 (0.51% of text units)

| Signal                 | Count |
| ---------------------- | ----: |
| repeated_word_sequence |   119 |

1. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 45
   - **Signal:** repeated_word_sequence
   - **Reason:** Repeated word sequence detected.
   - **Excerpt:** 80.211: لأ، لأ، ما تخافش، ما فيهاش.
2. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 147
   - **Signal:** repeated_word_sequence
   - **Reason:** Repeated word sequence detected.
   - **Excerpt:** 256.895: خُد بالك انت، انت اللي سايق!
3. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 374
   - **Signal:** repeated_word_sequence
   - **Reason:** Repeated word sequence detected.
   - **Excerpt:** إنها تشيل 4 صواريخ جو جو،
4. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 485
   - **Signal:** repeated_word_sequence
   - **Reason:** Repeated word sequence detected.
   - **Excerpt:** 803.602: ولكن Actually, Actually، يا عزيزي،
5. **Location:** transcript_line | أعظم طائرة حربية الدحيح | أعظم طائرة حربية الدحيح.txt | line 569
   - **Signal:** repeated_word_sequence
   - **Reason:** Repeated word sequence detected.
   - **Excerpt:** 941.007: هنا، الصاروخ أرض جو، مش جو جو،

## Orthographic Variant Family Highlights

| Normalized Form | Surface Forms    | Unit Count |
| --------------- | ---------------- | ---------: |
| علي             | على, علي         |       1108 |
| كان             | كأن, كان         |        829 |
| الجمله          | الجملة, الجمله   |        664 |
| العباره         | العبارة, العباره |        661 |
| انا             | أنا, إنا, انا    |        383 |
| انت             | أنت, انت         |        381 |
| حاجه            | حاجة, حاجه       |        307 |
| ايه             | إيه, اية, ايه    |        284 |
| فيه             | فئة, فئه, فيه    |        275 |
| ابو             | أبو, ابو         |        250 |
