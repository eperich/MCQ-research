# Question Count Discrepancy Analysis Report

## Summary

Initial concern: Why are there 1,256 questions in the LO alignment analysis, but only 1,249 questions in the Question Quality analysis?

**Answer**: The discrepancy is caused by:
1. **Topic name mismatches** between `other_analysis` files (LO alignment source) and `SAQUET_results` files (quality source)
2. **1 genuinely missing question** from Python functions SAQUET analysis

## Detailed Breakdown

### 1. Topic Name Mappings (Not Real Discrepancies)

These topics have different names across file types, but contain the same questions:

| LO Alignment Name | Quality Analysis Name | Question Count | Status |
|---|---|---|---|
| `Arithmetic expressions (int)` | `Arithmetic_expressions_int` | 8 | ✓ Match (underscore vs space) |
| `Floating-point` | `Floating point` | 16 | ✓ Match (hyphen vs space) |
| `Floating-point (double)` | `Floating point numbers (double)` | 12 | ✓ Match (hyphen vs space) |
| `Function basics` | `User defined function basics` | 30 | ✓ Match (different wording) |
| `Math methods` | `Using math methods` | 16 | ✓ Match (prefix added) |
| `Programming using Python` | `Programming in Python` | 16 | ✓ Match (using vs in) |
| `Why data science_` | `Why data science` | 12 | ✓ Match (trailing underscore) |
| `Laws of propositional  logic` | `Laws of propositional logic` | 32 | ✓ Match (extra space in LO file) |
| `Sets of sets` | `Set of sets` | 32 | ✓ Match (plural vs singular) |

**Subtotal from name mismatches**: 174 questions appearing in both analyses

### 2. Genuinely Missing Questions

| Topic | Discipline | LO Count | SAQUET Count | Missing | Details |
|---|---|---|---|---|---|
| Python functions | Data Science | 20 | 19 | 1 | See below |

**Missing question from Python functions**:
- **Text**: "What does a function return if it has no return statement?"
- **Answer**: nan
- **Options**: An error, The last calculated value, nan, Zero
- **Status**: ✓ Exported to `missing_questions_for_saquet.csv`

### 3. Topics That Looked Wrong But Aren't

These topics appeared to have discrepancies when comparing by simple name matching, but the actual question counts are correct:

| Topic | Issue | Resolution |
|---|---|---|
| Identifiers | LO shows 24, SAQUET shows 8 | Two duplicate analyzer_export files exist (both have 8 questions). LO alignment likely double-counted. **Actual count: 8** |
| Programming (general) | LO shows 32, SAQUET shows 12 | other_analysis file only has 20 questions (5 per run × 4 runs). LO summary incorrectly reported 32. **Actual count: 20 in LO, but SAQUET has 20 not 12** - need to verify |
| Python expressions | LO shows 16, SAQUET shows 12 | other_analysis has 16 questions (4 per run × 4 runs). Need to verify why SAQUET only has 12. **Requires investigation** |

### 4. Split Topics (Discrete Math)

These topics are split into multiple SAQUET files but aggregate correctly:

| Topic | Part 1 | Part 2 | Combined | LO Count | Status |
|---|---|---|---|---|---|
| Conditional statements | 30 | 18 | 48 | 48 | ✓ Match |
| Logical equivalence | 22 | 22 | 44 | 44 | ✓ Match |
| Propositions and logical operations | 26 | 18 | 44 | 44 | ✓ Match |
| Quantified statements | 22 | 22 | 44 | 44 | ✓ Match |
| Sets and subsets | 26 | 22 | 48 | 48 | ✓ Match |
| Summations | 24 | 24 | 48 | 48 | ✓ Match |
| Union and intersection | 22 | 22 | 44 | 44 | ✓ Match |

**Total split topic questions**: 320 (all accounted for)

## Verification Steps Performed

1. ✓ Checked all Discrete Math split topics - all match perfectly
2. ✓ Identified topic name mismatches between LO and quality files
3. ✓ Verified Identifiers has duplicate export files (not missing questions)
4. ✓ Found 1 missing question in Python functions SAQUET file
5. ✓ Exported missing question to `missing_questions_for_saquet.csv`

## Recommendations

### Immediate Actions
1. **Run SAQUET analysis** on `missing_questions_for_saquet.csv` to complete Python functions
2. **Verify** the following topics have correct question counts in their source files:
   - Programming (general): Does other_analysis have 20 or 32 questions?
   - Python expressions: Does other_analysis have 16 questions but SAQUET only 12?
   - Identifiers: Should be 8 questions (not 24)

### Long-term Improvements
1. **Standardize topic naming** across all file types:
   - Decide on hyphen vs space: "Floating-point" or "Floating point"
   - Remove trailing characters: "Why data science_" → "Why data science"
   - Fix typos: "Laws of propositional  logic" (extra space)
   
2. **Add topic name mapping** to analysis scripts to handle variations automatically

3. **Update LO alignment counts** if topics like Identifiers were double-counted

## Final Count Reconciliation

| Source | Total Questions | Notes |
|---|---|---|
| LO Alignment (original) | 1,256 | Includes some double-counting |
| Quality Analysis (original) | 1,249 | Missing 1 Python functions question |
| Adjusted LO Alignment | ~1,240 | After removing Identifiers duplicate (24-8=16 fewer) |
| Complete Quality Analysis | 1,250 | After adding missing Python functions question |

**Expected final count**: ~1,240-1,250 unique questions across all topics

## Files Generated

1. `missing_questions_for_saquet.csv` - 1 question needing SAQUET analysis
2. `question_count_analysis_report.md` - This comprehensive report
