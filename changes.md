# Changes.md — Aligning Project with N-Gram Plagiarism Detection Goals

## 📌 Purpose of This Document

This document provides a **comprehensive gap analysis and step-by-step transformation plan** to align the current plagiarism detection system with the **expected academic objective: N-gram based plagiarism detection using the PAN-09 dataset**.

---

# 🧠 1. What the Project is SUPPOSED to Be

## Expected Core Goal

Build and evaluate a **plagiarism detection system using N-grams**, including:

* Understanding plagiarism types:

  * Exact copy
  * Partial plagiarism
  * Paraphrasing
* Implementing:

  * N-gram extraction
  * Similarity measures (Jaccard, Cosine, Overlap)
* Running experiments on PAN-09 dataset
* Evaluating performance (Precision, Recall, F1)
* Analyzing strengths & weaknesses

---

# ⚙️ 2. What the Current System Actually Is

## Current Architecture (Simplified)

```
Input → Sentence Transformer → Semantic Similarity
       → N-gram overlap (minor role)
       → Stylometry (burstiness)
       → Combined scoring → Output
```

## Key Observations

### ✅ Strengths

* Uses **semantic embeddings (SentenceTransformer)**
* Includes **TF-IDF baseline**
* Performs **sentence-level matching**
* Supports **PDF/DOCX/TXT**
* Generates **detailed reports**
* Uses **N-grams (but only as secondary feature)**

---

# ❌ 3. Core Misalignment

## Fundamental Problem

The system is:

> ❌ NOT an N-gram-based plagiarism detector
> ✅ It is a **hybrid AI-driven forensic system**

---

## Why This is a Problem

The professor expects:

| Requirement                     | Status    |
| ------------------------------- | --------- |
| Pure N-gram pipeline            | ❌ Missing |
| Jaccard similarity              | ❌ Missing |
| Overlap coefficient             | ❌ Missing |
| Controlled experiments (vary N) | ❌ Missing |
| Evaluation metrics              | ❌ Missing |
| Analysis of preprocessing       | ❌ Missing |

---

# 🔍 4. Detailed Issues

## 🔴 Issue 1: N-grams Are Not the Core

### Current:

* N-grams used only for **overlap visualization**
* Final decision depends mostly on **semantic similarity**

### Required:

* N-grams must be the **primary detection mechanism**

---

## 🔴 Issue 2: No Standard Similarity Metrics

Missing:

* Jaccard similarity
* Overlap coefficient

Currently:

* Custom overlap ratio only

---

## 🔴 Issue 3: No Multi-N Analysis

Current:

```
n = 3 (fixed)
```

Expected:

```
n ∈ {1, 2, 3, 4}
```

---

## 🔴 Issue 4: No Experimental Framework

Missing:

* Comparison across N values
* Controlled evaluation
* Result tables

---

## 🔴 Issue 5: No Evaluation Metrics

Missing:

* Precision
* Recall
* F1-score

---

## 🔴 Issue 6: PAN-09 Dataset Not Fully Utilized

Current:

* Used as source index

Missing:

* Use of **ground truth annotations**
* Proper evaluation against labeled plagiarism cases

---

## 🔴 Issue 7: No Preprocessing Experiments

Missing comparison of:

* Raw text vs cleaned text
* Stopword removal impact

---

## 🔴 Issue 8: No Failure Analysis

Missing:

* Demonstration of failure in:

  * Paraphrasing
  * Synonym replacement

---

# 🚀 5. Required Changes (Step-by-Step)

---

## ✅ STEP 1: Build a Baseline N-Gram Model

Create a new module:

```
ngram_baseline.py
```

### Must include:

#### 1. Preprocessing

* Lowercase
* Remove punctuation
* Tokenization

#### 2. N-gram Generation

```
unigram, bigram, trigram, 4-gram
```

---

## ✅ STEP 2: Implement Similarity Functions

### Jaccard Similarity

```
|A ∩ B| / |A ∪ B|
```

### Overlap Coefficient

```
|A ∩ B| / min(|A|, |B|)
```

### Optional:

* Cosine similarity (TF-IDF)

---

## ✅ STEP 3: Build Comparison Pipeline

For each suspicious document:

```
For each source document:
    Generate N-grams
    Compute similarity
    Store highest match
```

---

## ✅ STEP 4: Add Threshold-Based Detection

```
if similarity > threshold:
    plagiarism = True
```

Test multiple thresholds:

* 0.3
* 0.5
* 0.7

---

## ✅ STEP 5: Run Multi-N Experiments

For:

```
N = 1, 2, 3, 4
```

Record:

* Accuracy
* Observations

---

## ✅ STEP 6: Preprocessing Experiments

Compare:

| Variant   | Description                    |
| --------- | ------------------------------ |
| Raw       | No cleaning                    |
| Lowercase | Basic cleaning                 |
| Cleaned   | Remove punctuation + stopwords |

---

## ✅ STEP 7: Evaluation Using PAN-09

Use:

* Ground truth labels

Compute:

* Precision
* Recall
* F1-score

---

## ✅ STEP 8: Failure Case Analysis

### Must include:

#### Case 1: Exact Copy

* High similarity

#### Case 2: Paraphrasing

* Low similarity

Example:

```
Original: "He is very happy"
Paraphrased: "He feels joyful"
→ N-gram fails
```

---

## ✅ STEP 9: Compare With Current System

### Add Section:

| Feature              | N-gram Model | Current System |
| -------------------- | ------------ | -------------- |
| Exact match          | ✅            | ✅              |
| Paraphrase detection | ❌            | ✅              |
| Complexity           | Low          | High           |
| Accuracy             | Medium       | High           |

---

## ✅ STEP 10: Integrate Advanced Methods (Already Present)

Your system already supports:

* TF-IDF + Cosine similarity ✅
* Sentence-level detection ✅
* Semantic similarity ✅

👉 You must:

* Present these as **enhancements over baseline**

---

# 📊 6. Required Analysis Section

You MUST answer:

---

## 🔹 Does Bigram Work Better Than Unigram?

Expected:

* Yes — captures phrase-level similarity

---

## 🔹 What Happens When N Increases?

| N  | Effect      |
| -- | ----------- |
| 1  | Too general |
| 2  | Balanced    |
| 3  | Strong      |
| 4+ | Too strict  |

---

## 🔹 Does Preprocessing Improve Accuracy?

Expected:

* Yes — removes noise
* But excessive cleaning may remove useful signals

---

## 🔹 Where Does System Fail?

### N-gram fails in:

* Paraphrasing
* Synonyms
* Sentence reordering

---

# 🧩 7. Final Expected Architecture

```
                ┌──────────────┐
                │ PAN Dataset  │
                └──────┬───────┘
                       │
         ┌─────────────▼─────────────┐
         │ Preprocessing Variants    │
         └─────────────┬─────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ N-gram Generation (1–4)     │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ Similarity (Jaccard, etc.)  │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ Threshold Decision          │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ Evaluation Metrics          │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ Comparison with Advanced    │
        │ (TF-IDF + Semantic Model)  │
        └─────────────────────────────┘
```

---

# 🧠 8. Key Takeaway

## Current Status:

* You built a **high-end plagiarism system**

## Missing:

* Basic **academic experimental framework**

---

## Final Alignment Goal

Transform your project into:

> “A structured study of N-gram plagiarism detection, supported by experiments, evaluation, and comparison with advanced methods.”

---

# 🚀 9. Immediate Action Plan

### Phase 1 (Critical)

* Build baseline N-gram system
* Implement similarity metrics

### Phase 2

* Run experiments (vary N, preprocessing)

### Phase 3

* Evaluate using PAN dataset

### Phase 4

* Compare with your current system

### Phase 5

* Document everything

---

# ✅ Conclusion

Your project is **technically strong but academically misaligned**.

To score well:

* You must **demonstrate understanding**
* Not just **build a powerful system**

---

**End of Changes.md**
