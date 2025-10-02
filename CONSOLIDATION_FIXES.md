# Consolidation Fixes - Addressing Over-Consolidation Issues

**Date**: 2025-10-01
**Issue**: Pipeline v10 produced only 3 intent and 3 sentiment categories (over-consolidated)
**Goal**: Generate 5-7 UNIQUE intents and 4-6 UNIQUE sentiments with no overlapping definitions

---

## Changes Made to curator.py

### 1. Updated Pydantic Model Constraints (Lines 41-42)

**Before:**
```python
intent_categories: List[ConsolidatedCategory] = Field(..., min_items=3, max_items=5, description="EXACTLY 3-5 distinct intent categories")
sentiment_categories: List[ConsolidatedCategory] = Field(..., min_items=3, max_items=4, description="EXACTLY 3-4 distinct sentiment categories")
```

**After:**
```python
intent_categories: List[ConsolidatedCategory] = Field(..., min_items=5, max_items=7, description="EXACTLY 5-7 distinct, UNIQUE intent categories")
sentiment_categories: List[ConsolidatedCategory] = Field(..., min_items=4, max_items=6, description="EXACTLY 4-6 distinct, UNIQUE sentiment categories")
```

**Impact**: Pydantic validation will now REJECT taxonomies with fewer than 5 intents or 4 sentiments.

---

### 2. Updated LLM Consolidation Prompt (Lines 959-1005)

#### 2.1 Changed Targets

**Before:**
```
- 3-5 Intent Categories that capture distinct customer purposes
- 3-4 Sentiment Categories that reflect meaningful emotional tones
```

**After:**
```
- EXACTLY 5-7 Intent Categories that capture distinct customer purposes
- EXACTLY 4-6 Sentiment Categories that reflect meaningful emotional tones
```

#### 2.2 Added UNIQUENESS and DISCRETENESS Requirements

**New Section:**
```
**CRITICAL: Each category must be UNIQUE and DISCRETE**
- Categories must represent fundamentally different communication patterns
- NO overlapping definitions - each category must be clearly distinct
- Categories must require different operational responses
- Avoid creating categories that are merely variations of each other
```

#### 2.3 Added Minimum Coverage Threshold

**New Principle:**
```
5. **Minimum Coverage**: Each final category should represent at least 5% of total emails (avoid rare edge cases)
```

**Impact**: Categories with < 5% coverage should be merged or discarded.

#### 2.4 Strengthened Intent Guidelines

**Added:**
```
- Each final category must be DISCRETE - no partial overlaps allowed
- Test: "Can I clearly explain when to use Category A vs Category B?" If not, merge them
```

#### 2.5 Strengthened Sentiment Guidelines

**Added:**
```
- Each final sentiment must be UNIQUE - no gradient variations of the same emotion
- Avoid creating multiple levels of the same sentiment (e.g., "Frustrated" and "Very Frustrated" should be ONE category)
- Preserve distinctions that require fundamentally different handling approaches (e.g., cooperative vs hostile)
```

#### 2.6 Added Uniqueness Validation Checklist

**New Section:**
```
**Uniqueness Validation:**
For each pair of final categories, ask:
1. "Are these fundamentally different communication patterns?" → If NO, merge them
2. "Would agents respond differently to these categories?" → If NO, merge them
3. "Can an email clearly belong to only ONE of these?" → If NO, merge them
4. "Do these meet the 5% minimum coverage threshold?" → If NO, merge or discard
```

#### 2.7 Updated Critical Instructions

**Added:**
```
- EXACTLY 5-7 intents and 4-6 sentiments - no more, no less
- Each category must represent at least 5% of total emails
- NO overlapping or ambiguous category boundaries
```

---

### 3. Fixed Labeling Guide Examples (Lines 361-415)

**Problem**: Labeling guide was generating placeholder examples like "Customer email requesting..."

**Before:**
```python
examples = intent_data.get('examples', [])
for example in examples:
    guide += f"- \"{example}\"\n"
```

**After:**
```python
# Try to get real email examples first, fall back to key indicators
real_examples = intent_data.get('real_email_examples', [])
if real_examples:
    for example in real_examples[:3]:
        content = example.get('content', '')
        if content:
            guide += f"- \"{content}\"\n"
else:
    # Fall back to examples from taxonomy (not placeholders)
    examples = intent_data.get('examples', [])
    for example in examples[:3]:
        # Skip generic placeholder examples
        if not ('Customer email requesting' in example or
               'Communication regarding' in example or
               'Email about' in example):
            guide += f"- \"{example}\"\n"
```

**Impact**: Labeling guide will now show:
1. **Real email snippets** from `real_email_examples` (preferred), OR
2. **Actual indicators** from taxonomy.yaml, OR
3. **Nothing** if only placeholders are available (better than misleading examples)

---

### 4. Updated Consolidation Trigger Threshold (Line 624)

**Before:**
```python
if self.client and (len(intent_categories) > 5 or len(sentiment_categories) > 4):
```

**After:**
```python
if self.client and (len(intent_categories) > 7 or len(sentiment_categories) > 6):
```

**Impact**: Consolidation will only trigger if we have MORE than 7 intents or 6 sentiments (previously triggered at >5 and >4). This allows the system to keep more granular categories when appropriate.

---

## Expected Outcomes in Next Pipeline Run

### Taxonomy Quality

| Metric | Before (v10) | Expected After |
|--------|--------------|----------------|
| **Intent categories** | 3 | 5-7 |
| **Sentiment categories** | 3 | 4-6 |
| **Min coverage per category** | None (1.4% observed) | 5% enforced |
| **Category overlap** | Possible | Prevented by validation |
| **Real examples in guide** | No (placeholders) | Yes |

### Category Characteristics

**Intents (5-7 categories):**
- Each represents a DISTINCT customer purpose
- No overlapping definitions
- Each requires different operational response
- Each represents ≥5% of emails
- Clear decision boundaries between categories

**Sentiments (4-6 categories):**
- Each represents a UNIQUE emotional tone
- No gradient variations (e.g., no "Frustrated" + "Very Frustrated")
- Each requires fundamentally different handling
- Each represents ≥5% of emails
- Clear distinction between cooperative, neutral, and negative tones

### Example Valid Taxonomy

**Good Intent Categories (Discrete):**
1. Payment Inquiry (22%) - Asking about payment status
2. Invoice Correction (15%) - Requesting invoice changes
3. Documentation Request (12%) - Needing W9, PO, etc.
4. Payment Commitment (18%) - Confirming payment will be made
5. Dispute Resolution (20%) - Challenging charges or errors
6. Account Update (8%) - Changing billing contacts/details
7. Service Cancellation (5%) - Terminating services

**Good Sentiment Categories (Unique):**
1. Cooperative Professional (40%) - Helpful, factual communication
2. Frustrated Escalation (25%) - Expressing dissatisfaction, urgency
3. Neutral Administrative (20%) - Purely procedural, no emotion
4. Apologetic Conciliatory (10%) - Acknowledging issues, seeking resolution
5. Urgent Concern (5%) - Time-sensitive, worried tone

**Bad Examples (Would be rejected):**
- ❌ "Frustrated" + "Very Frustrated" (gradient variations)
- ❌ "Payment Inquiry" + "Payment Status Check" (overlapping definitions)
- ❌ "Account Management" at 1.4% coverage (below 5% threshold)
- ❌ Categories with duplicate key indicators

---

## Testing the Changes

### Validation Checklist

After running the pipeline, verify:

1. **Count Check:**
   - [ ] 5-7 intent categories (not 3-5)
   - [ ] 4-6 sentiment categories (not 3-4)

2. **Coverage Check:**
   - [ ] Each intent category ≥ 5% coverage
   - [ ] Each sentiment category ≥ 5% coverage
   - [ ] Total coverage ≥ 60% (sum of all categories)

3. **Uniqueness Check:**
   - [ ] No overlapping intent definitions
   - [ ] No overlapping sentiment definitions
   - [ ] No duplicate key indicators across categories

4. **Discreteness Check:**
   - [ ] Can clearly explain when to use each category
   - [ ] Each category requires different agent response
   - [ ] No gradient variations (e.g., "Mild Frustrated" vs "Frustrated")

5. **Example Quality Check:**
   - [ ] Labeling guide has real email examples (not "Customer email requesting...")
   - [ ] taxonomy.yaml has real email snippets in examples field

---

## Running the Updated Pipeline

```bash
# Run with the fixed consolidation logic
uv run python run_pipeline.py --input source_data/litera_raw_emails_v3_fixed.json --dataset-name litera_test_v11
```

**Expected Output Files:**
- `outputs/litera_test_v11/taxonomy.yaml` - 5-7 intents, 4-6 sentiments, ≥5% coverage each
- `outputs/litera_test_v11/taxonomy_labeling_guide.md` - Real email examples
- `outputs/litera_test_v11/curation_summary.json` - Validation metrics

---

## Rollback Plan

If the changes cause issues, revert with:

```bash
git checkout pipeline/curator.py
```

The specific changes are isolated to `curator.py` only, making rollback safe.

---

## Summary

**What Changed:**
- Consolidation targets: 3-5 intents → 5-7 intents, 3-4 sentiments → 4-6 sentiments
- Added 5% minimum coverage threshold
- Strengthened uniqueness and discreteness requirements in LLM prompt
- Fixed labeling guide to use real email examples
- Updated consolidation trigger threshold

**Why:**
- Previous taxonomy (v10) was over-consolidated with only 3 categories each
- Categories had 1.4%-1.6% coverage (too rare for production)
- Labeling guide had placeholder examples (unusable for human labeling)
- Over-consolidation loses business granularity

**Expected Result:**
- 5-7 unique, discrete intent categories with ≥5% coverage each
- 4-6 unique, discrete sentiment categories with ≥5% coverage each
- Real email examples in labeling guide
- Production-ready taxonomy for Phase 2 (manual labeling + validation)

---

**Status**: ✅ Ready for testing with next pipeline run
**File Modified**: `pipeline/curator.py` only
**Breaking Changes**: None (backwards compatible)
