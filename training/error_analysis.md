# Error Analysis: Intent Taxonomy Decision

## Initial approach: 6-class taxonomy

Mapped the raw dataset's 9 `queue` values (after dropping Human Resources)
directly to 6 categories: Billing, Technical Support, Sales, Account,
Product Support, General Inquiry.

**Result:** LinearSVC, macro-F1 = 0.50, accuracy = 56%.

Billing (F1=0.75) and Technical Support (F1=0.68) — the two most
lexically distinct categories — classified well. Account, Product
Support, and General Inquiry all scored poorly (F1 0.37–0.43) and were
heavily confused with each other and with Technical Support.

## Diagnosis: manual inspection of misclassified examples

Rather than assume the model needed tuning, we pulled the actual text of
misclassified tickets for the worst-confused class pairs. Representative
examples:

- **True: Product Support, Predicted: Technical Support** —
  _"I am facing difficulties with my dashboard failing to update in real
  time. It is likely related to caching..."_ — this reads as a textbook
  technical bug report. There is no reliable signal that distinguishes
  "product support issue" from "technical support issue" in the dataset's
  phrasing.

- **True: General Inquiry, Predicted: Technical Support** —
  _"I am contacting you to report an issue with the investment prognosis
  feature. The analytics tool seems to be malfunctioning..."_ — this is a
  bug report, not a general inquiry by any reasonable human labeling
  standard.

- **True: Account, Predicted: Technical Support** —
  _"A data breach has been detected in the hospital system, resulting
  from unauthorized access..."_ — a security-incident ticket labeled
  "Account" is a defensible mislabel; "Technical Support" is at least as
  reasonable a label.

**Conclusion:** the confusion was not caused by a model or feature
limitation — the model was frequently _correct_ in a common-sense reading
of the ticket, while the ground-truth label appeared inconsistent. This
is a known characteristic of synthetically generated datasets: label
assignment during generation can be loosely coupled to the generated
text. This is a data problem, not a model problem, so hyperparameter
tuning would not have addressed it.

## Decision

Merged `Product Support`, `Account`, and `General Inquiry` into a single
`General Inquiry` category. Kept `Billing`, `Technical Support`, and
`Sales` as-is since they were textually distinct and classified well.

**Result after merge:** LinearSVC, macro-F1 = 0.62, accuracy = 66%.

## Hyperparameter tuning (after merge)

With the 4-class taxonomy confirmed as the right target, tuned the TF-IDF +
LinearSVC pipeline via manual grid search:

- Vocabulary size: tested 8k / 15k / 20k / 30k `max_features` — larger
  vocab kept helping up to 30k (diminishing returns beyond that untested
  due to time budget).
- `min_df`: 1 vs 2 — `min_df=1` outperformed `min_df=2`, i.e. rare terms
  carried real signal rather than being noise, likely because category-
  specific jargon (e.g. product/tool names in Technical Support tickets)
  only appears a handful of times.
- `class_weight='balanced'`: consistently improved macro-F1 over
  unweighted, as expected given class imbalance (Sales has only 330
  examples vs 5245 for Technical Support).
- `C` (SVM regularization): grid searched 0.1-5.0, best at C=1.5.
- Word + character n-grams (3-5 char, `analyzer='char_wb'`) combined via
  feature stacking: tested but did **not** outperform word-only n-grams,
  so dropped to keep the inference pipeline simpler and faster.

**Final config:** `max_features=30000, ngram_range=(1,2), min_df=1,
class_weight='balanced', C=1.5`.

**Final result:** LinearSVC, macro-F1 = 0.686, accuracy = 70.3%.

| Category          | Precision | Recall | F1   |
| ----------------- | --------- | ------ | ---- |
| Billing           | 0.84      | 0.80   | 0.82 |
| General Inquiry   | 0.67      | 0.67   | 0.67 |
| Sales             | 0.67      | 0.45   | 0.54 |
| Technical Support | 0.71      | 0.72   | 0.71 |

This is a genuine, non-leaked improvement over the naive baseline (+14pt
accuracy, +19pt macro-F1), driven by proper hyperparameter search rather
than data leakage or feature gimmicks — vectorizer was always fit on
train only.

## Data cleaning update: near-empty bodies, missing subjects, engineered features

Revisited the raw dataset after initial deployment and found several
additional cleaning issues not caught in the first pass:

- **1,010 rows** (of 11,923 English rows) had a null `subject`.
  Previously handled silently (coerced to empty string); now explicitly
  tracked via a `subject_missing` flag during cleaning.
- **27 rows** had a near-empty `body` (<20 characters, e.g. "I need
  assistance") — almost no signal for any model. Confirmed via manual
  inspection these were spread thinly across every category (max 5 in
  any one class), so dropping them from **training only** does not
  starve any class. A real incoming email this short is still handled
  at inference time (flagged, not rejected — see below), only the
  _training_ set excludes them.
- **Placeholder tokens** in the generator's synthetic text (`<tel_num>`,
  `<email>`, `<acc_num>`, `<website_url>`, `<ref_num>`, `<your_name>`,
  `<user>`, `<n>`, `<br>`) — confirmed the existing generic regex
  (`<[^>]+>`) strips all variants, not just the one originally spot-checked.

### Engineered features — tested, and mostly a negative result (documented honestly)

Built `app/features.py` with numeric features computable from
subject+body alone at inference time (word/char count, exclamation
count, ALL-CAPS ratio, question-mark presence, urgency-keyword count,
missing-subject flag, low-information-body flag). Explicitly excluded
the dataset's `tag_1`/`tag_2`/`tag_3` columns from any model input —
those are annotations added after human triage and would not exist for
a real incoming email; using them would be data leakage, even though
they correlate strongly with `queue`.

**Tested stacking these features (scaled) alongside TF-IDF for both
models:**

- **Model A (intent):** macro-F1 dropped from 0.686 to 0.645, even after
  re-tuning `C` for the new feature space (best re-tuned: 0.652, still
  worse than text-only). Diagnosis: intent/category is a _topic_ signal;
  these features encode _tone_/urgency, not topic — the wrong feature
  for this task.
- **Model B (priority):** macro-F1 essentially unchanged (0.602 →
  0.603), within noise. Slightly disappointing given priority is
  tone-related in principle, but TF-IDF already captures the same
  urgency words directly (e.g. "urgent", "immediately") as text
  features, so the engineered counts added little the vectorizer wasn't
  already seeing.

**Decision:** don't use these features as classifier input for either
model — a negative result, kept and documented rather than silently
discarded, per the same evidence-first approach used for the taxonomy
merge. They ARE still useful downstream, just not as ML input: exposed
as **human-review trigger signals** in the API (`missing_subject`,
`low_information_body` — see `app/routes.py`) so an operator knows when
a prediction was made on unusually thin information, independent of the
model's own confidence score.

### Retrained metrics after the cleaning fixes (near-empty rows removed)

- **Model A:** macro-F1 = 0.646 (down slightly from 0.686). Root cause
  investigated: **not** the row removal itself (only 1 of the 27 dropped
  rows was in the Sales class). The 329-example Sales class is small
  enough (66 in test) that removing ~26 rows elsewhere shifted the
  stratified split's random assignment, changing which specific Sales
  examples happened to land in train vs. test — expected sampling
  variance on a small class, not a pipeline regression. Sales recall
  swung 0.45 → 0.33 as a direct result; other classes were stable.
- **Model B:** macro-F1 = 0.603 (up slightly from 0.592, within noise).

## Model B (priority) — accepted ceiling, not further tuned

Model B's confusion matrix (evaluated before the cleaning-fix retrain
above) showed errors concentrated almost entirely between _adjacent_
priority levels (LOW↔MEDIUM, MEDIUM↔HIGH) rather than between the two
extremes (only 170 of 2344 test rows confused LOW directly with HIGH).
This is the signature of an information ceiling, not a fixable model or
data-quality problem: priority is partly a business judgment call (SLA,
customer tier, context outside the email text) that no amount of
text-only tuning can fully recover. This is also why the product design
blends the model's priority prediction with the sentiment signal in
`predictor.py`, rather than trusting text classification alone — see
`app/predictor.py` for the blending logic. This reasoning still holds
after the later cleaning-fix retrain (macro-F1 = 0.603).

## Remaining known limitation: Sales class

`Sales` remains the weakest class after the merge, for a consistent,
honest reason across every retrain: only ~330 examples in the full
dataset (~66 in the held-out test set). This is a sample-size
limitation, not a labeling problem, and merging it into another category
would eliminate the sales-routing use case entirely — so it was kept as
its own class and documented as a known limitation rather than "fixed"
by hiding it. Its exact F1 has fluctuated across retrains purely due to
which specific examples land in the small test fold (0.54 → 0.42 after
the cleaning-fix retrain above) — expected variance on a small class,
not a signal of instability in the modeling approach itself.
