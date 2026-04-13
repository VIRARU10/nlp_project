import string
import math
from collections import Counter
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except Exception:
        # Fallback if download fails
        return {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "are", "was", "were", "be", "been"}

STOP_WORDS = get_stopwords()
PUNCTUATION = set(string.punctuation)

def preprocess_text(text: str, mode: str = 'cleaned') -> list:
    """
    Preprocesses text according to the specified mode.
    Modes:
      - 'raw': Basic split, no modification.
      - 'lowercase': Lowercase + basic split.
      - 'cleaned': Lowercase, remove punctuation, remove stopwords.
    """
    if not text:
        return []
        
    if mode == 'raw':
        return text.split()
    elif mode == 'lowercase':
        return text.lower().split()
    elif mode == 'cleaned':
        # Apply standard cleaning
        text = text.lower()
        # Basic tokenization
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
            
        # Remove punctuation and stopwords
        cleaned_tokens = [
            token for token in tokens 
            if token not in PUNCTUATION and token not in STOP_WORDS and token.isalnum()
        ]
        return cleaned_tokens
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

def generate_ngrams(tokens: list, n: int) -> list:
    """Generates a list of n-grams from a list of tokens."""
    if n <= 0:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def lcs_length(list_a: list, list_b: list) -> int:
    """Computes the Longest Common Subsequence between two token lists."""
    if not list_a or not list_b:
        return 0
    m, n = len(list_a), len(list_b)
    # Use two rows to save memory
    prev_row = [0] * (n + 1)
    curr_row = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list_a[i - 1] == list_b[j - 1]:
                curr_row[j] = prev_row[j - 1] + 1
            else:
                curr_row[j] = max(prev_row[j], curr_row[j - 1])
        prev_row = curr_row[:]
        
    return curr_row[n]

def jaccard_similarity(ngrams_a: list, ngrams_b: list) -> float:
    """Calculates Jaccard Similarity using Multi-sets (Counters)."""
    if not ngrams_a and not ngrams_b:
        return 0.0
        
    count_a = Counter(ngrams_a)
    count_b = Counter(ngrams_b)
    
    intersection = sum((count_a & count_b).values())
    union = sum((count_a | count_b).values())
    
    return intersection / union if union > 0 else 0.0

def overlap_coefficient(ngrams_a: list, ngrams_b: list) -> float:
    """Calculates Overlap Coefficient using Multi-sets (Counters)."""
    if not ngrams_a or not ngrams_b:
        return 0.0
        
    count_a = Counter(ngrams_a)
    count_b = Counter(ngrams_b)
    
    intersection = sum((count_a & count_b).values())
    min_len = min(len(ngrams_a), len(ngrams_b))
    
    return intersection / min_len if min_len > 0 else 0.0

def cosine_similarity(ngrams_a: list, ngrams_b: list) -> float:
    """Calculates Cosine Similarity using frequency profiles (Counters)."""
    if not ngrams_a or not ngrams_b:
        return 0.0
        
    count_a = Counter(ngrams_a)
    count_b = Counter(ngrams_b)
    
    intersection_keys = set(count_a.keys()).intersection(set(count_b.keys()))
    
    dot_product = sum(count_a[ngram] * count_b[ngram] for ngram in intersection_keys)
    
    magnitude_a = math.sqrt(sum(val ** 2 for val in count_a.values()))
    magnitude_b = math.sqrt(sum(val ** 2 for val in count_b.values()))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
        
    return dot_product / (magnitude_a * magnitude_b)

def get_similarity_score(text_a: str, text_b: str, n: int = 3, mode: str = 'cleaned', metric: str = 'overlap', use_lcs_fallback: bool = False) -> float:
    """End-to-end pipeline to compute similarity between two raw strings."""
    tokens_a = preprocess_text(text_a, mode)
    tokens_b = preprocess_text(text_b, mode)
    
    if not tokens_a or not tokens_b:
        return 0.0

    ngrams_a = generate_ngrams(tokens_a, n)
    ngrams_b = generate_ngrams(tokens_b, n)
    
    score = 0.0
    if metric == 'jaccard':
        score = jaccard_similarity(ngrams_a, ngrams_b)
    elif metric == 'overlap':
        score = overlap_coefficient(ngrams_a, ngrams_b)
    elif metric == 'cosine':
        score = cosine_similarity(ngrams_a, ngrams_b)
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    # LCS Fallback: if pure N-grams are broken by light obfuscation, check if 
    # the underlying token sequence is still highly aligned.
    if use_lcs_fallback and score < 0.1:
        # Calculate LCS overlap. Overlap by length of the smaller sequence
        lcs_len = lcs_length(tokens_a, tokens_b)
        min_len = min(len(tokens_a), len(tokens_b))
        lcs_score = lcs_len / min_len if min_len > 0 else 0.0
        # If LCS score is significantly high but N-gram score is zero, take LCS - small penalty
        # because ngrams imply order constraint.
        if lcs_score > 0.4:
            score = max(score, lcs_score * 0.8)

    return score
