import joblib
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# ================= 1. LOAD MODEL =================
print("🚀 Loading Forensic Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
source_index = joblib.load('forensic_index.pkl')

def get_ngrams(text, n=3):
    """Extracts a set of n-grams from text."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    words = list(text.lower().split())
    return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

def get_burstiness(text):
    lengths = [len(s.split()) for s in re.split(r'[.!?]+', text) if len(s.split()) > 0]
    return np.std(lengths) if len(lengths) > 1 else 0

print("✅ MODEL READY.")

# ================= 2. INTERACTIVE LOOP =================
while True:
    user_input = input("\n👉 Enter text to check: ")
    if user_input.lower() == 'exit': break

    print("🔍 Running N-Gram & Semantic Analysis...")

    # A. Semantic Vector Logic
    user_vec = model.encode([user_input], show_progress_bar=False)[0]
    user_ngrams = get_ngrams(user_input)
    user_burst = get_burstiness(user_input)

    # B. Comparison
    all_matches = []
    for item in source_index:
        # Calculate Semantic Similarity
        sem_sim = np.dot(user_vec, item['vector']) / (np.linalg.norm(user_vec) * np.linalg.norm(item['vector']))
        
        # Calculate Exact N-Gram Overlap (Innovation: Lexical matching)
        source_ngrams = get_ngrams(item['text'])
        common_ngrams = user_ngrams.intersection(source_ngrams)
        ngram_sim = len(common_ngrams) / len(user_ngrams) if user_ngrams else 0
        
        all_matches.append((item['filename'], sem_sim, ngram_sim, list(common_ngrams)))

    # Sort by highest combined score
    all_matches.sort(key=lambda x: (x[1] + x[2])/2, reverse=True)
    best_file, s_sim, n_sim, matched_ngrams = all_matches[0]

    # ================= 3. ENHANCED REPORT =================
    print("\n" + "="*50)
    print("🔬 TRIPLE-SIGNAL PLAGIARISM REPORT")
    print("="*50)
    print(f"Matched File:     {best_file}")
    print(f"Semantic Match:   {float(s_sim) * 100:.1f}% (Meaning)")
    print(f"N-Gram Match:     {float(n_sim) * 100:.1f}% (Exact Words)")
    print(f"Author Rhythm:    {float(user_burst):.2f} (Burstiness)")
    
    # SHOW THE N-GRAMS HERE
    if matched_ngrams:
        print("\n📝 MATCHING N-GRAMS DETECTED:")
        # Show top 5 matching sequences
        ngram_list = list(matched_ngrams)
        for gram in ngram_list[:5]:
            print(f"  - \"{' '.join(gram)}\"")
    
    print("-" * 50)
    # Threshold alignment for exact brackets
    s_percent = s_sim * 100
    if s_percent < 30:
        print("✅ DECISION: LOW PLAGIARISM RISK (0-30%)")
    elif s_percent < 70:
        print("⚠️ DECISION: MEDIUM PLAGIARISM RISK (30-70%)")
    else:
        print("❌ DECISION: HIGH PLAGIARISM RISK (70-100%)")
    print("="*50)