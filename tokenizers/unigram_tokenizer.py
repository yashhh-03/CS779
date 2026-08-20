import argparse
import math
import re
import time
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple

NEG_INF = float("-inf") ## log(0)
_WORD_RE = re.compile(r"\S+") ## re for any sequence of non-whitespace

# -------------------------
# Corpus loading (optimized)
# -------------------------
def load_training_data(train_path: str) -> Counter:
    wc = Counter()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            wc.update(_WORD_RE.findall(line))
    return wc

def build_seed_vocab(word_counts: Counter, max_sub_len: int = 6, min_sub_count: int = 2, seed_size: int = 8000) -> Tuple[List[str], Dict[str, int]]:
    subfreq = Counter() ## Count how often every possible substring appears
    for word, cnt in word_counts.items():
        L = len(word)
        for i in range(L):
            # Limit substring length for efficiency
            for l in range(1, min(max_sub_len, L - i) + 1):
                sub = word[i:i + l]
                subfreq[sub] += cnt

    # Get all characters
    chars = set()
    for word in word_counts:
        chars.update(word)
    
    # Select top tokens efficiently
    candidates = []
    for tok, freq in subfreq.items():
        if freq >= min_sub_count:
            candidates.append((freq, tok))
    
    candidates.sort(reverse=True)
    tokens = [tok for _, tok in candidates[:seed_size]]
    
    # Add all characters
    for ch in chars:
        if ch not in tokens:
            tokens.append(ch)
    
    token_counts = {tok: subfreq.get(tok, 1) for tok in tokens}
    return tokens, token_counts

# -------------------------
# Optimized Trie with caching
# -------------------------
class FastTrie:
    def __init__(self, tokens: List[str]):
        self.root = {}
        self.token_lengths = set()
        self.token_indices = {}
        
        for idx, tok in enumerate(tokens): # Building tries here
            self.token_indices[tok] = idx
            self.token_lengths.add(len(tok))
            node = self.root
            for ch in tok:
                node = node.setdefault(ch, {})
            if "_toks" not in node:
                node["_toks"] = []
            node["_toks"].append(idx)
    
    def get_matches(self, word: str, pos: int) -> List[Tuple[int, int]]:
        """Get all matching tokens starting at position pos"""
        matches = []
        node = self.root
        max_len = min(len(word) - pos, max(self.token_lengths))
        
        for j in range(pos, pos + max_len):
            ch = word[j]
            if ch not in node:
                break
            node = node[ch]
            if "_toks" in node:
                for tid in node["_toks"]:
                    matches.append((tid, j - pos + 1))
        return matches

# -------------------------
# Optimized EM Step
# -------------------------
def logsumexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

def em_step_fast(word_counts: Counter, tokens: List[str], log_probs: List[float], trie: FastTrie) -> Tuple[List[float], float]:
    expected = [0.0] * len(tokens) 
    corpus_ll = 0.0 # total log-likelihood of the data
    
    # Precompute log probabilities for efficiency
    log_probs_arr = log_probs
    
    for word, wcount in word_counts.items():
        if wcount == 0:
            continue
            
        L = len(word)
        if L == 0:
            continue
            
        # Forward pass
        log_alpha = [NEG_INF] * (L + 1)
        log_alpha[0] = 0.0
        
        for i in range(L):
            if log_alpha[i] == NEG_INF:
                continue
                
            matches = trie.get_matches(word, i)
            for tid, l in matches:
                j = i + l
                if j > L:
                    continue
                val = log_alpha[i] + log_probs_arr[tid]
                log_alpha[j] = logsumexp(log_alpha[j], val)
        
        log_Pw = log_alpha[L]
        if log_Pw == NEG_INF:
            continue
            
        corpus_ll += wcount * log_Pw
        
        # Backward pass
        log_beta = [NEG_INF] * (L + 1)
        log_beta[L] = 0.0
        
        for i in range(L - 1, -1, -1):
            matches = trie.get_matches(word, i)
            for tid, l in matches:
                j = i + l
                if j > L:
                    continue
                if log_beta[j] == NEG_INF:
                    continue
                val = log_probs_arr[tid] + log_beta[j]
                log_beta[i] = logsumexp(log_beta[i], val)
        
        # Collect expectations
        for i in range(L):
            if log_alpha[i] == NEG_INF:
                continue
                
            matches = trie.get_matches(word, i)
            for tid, l in matches:
                j = i + l
                if j > L or log_beta[j] == NEG_INF:
                    continue
                contrib = math.exp(log_alpha[i] + log_probs_arr[tid] + log_beta[j] - log_Pw)
                expected[tid] += wcount * contrib
    
    return expected, corpus_ll

def normalize_expected(expected: List[float]) -> List[float]:
    total = sum(expected)
    if total > 0:
        return [e / total for e in expected]
    return [1.0 / len(expected)] * len(expected)

def prune_tokens_fast(tokens: List[str], expected_counts: List[float], keep_eta: float, min_keep_tokens: Set[str]) -> Tuple[List[str], List[float]]:
    n = len(tokens)
    keep_k = max(int(n * keep_eta), len(min_keep_tokens))
    
    # Create list of indices to keep
    indices = list(range(n))
    indices.sort(key=lambda i: expected_counts[i], reverse=True)
    
    keep_indices = set(indices[:keep_k])
    # Always keep single character tokens
    for i, tok in enumerate(tokens):
        if len(tok) == 1 and tok in min_keep_tokens:
            keep_indices.add(i)
    
    new_tokens = [tokens[i] for i in sorted(keep_indices)]
    new_probs = [expected_counts[i] for i in sorted(keep_indices)]
    
    total = sum(new_probs)
    if total > 0:
        new_probs = [p / total for p in new_probs]
    else:
        new_probs = [1.0 / len(new_probs)] * len(new_probs)
    
    return new_tokens, new_probs

# -------------------------
# Optimized Viterbi
# -------------------------
def viterbi_segment_fast(word: str, tokens: List[str], log_probs: List[float], trie: FastTrie) -> List[str]:
    L = len(word)
    if L == 0:
        return []
    
    dp = [NEG_INF] * (L + 1)
    back = [None] * (L + 1)
    dp[0] = 0.0
    
    for i in range(L):
        if dp[i] == NEG_INF:
            continue
            
        matches = trie.get_matches(word, i)
        for tid, l in matches:
            j = i + l
            if j > L:
                continue
            val = dp[i] + log_probs[tid]
            if val > dp[j]:
                dp[j] = val
                back[j] = (i, tid)
    
    # Fallback: character-level segmentation
    if dp[L] == NEG_INF:
        return list(word)
    
    # Reconstruct segmentation
    segs = []
    pos = L
    while pos > 0:
        prev, tid = back[pos]
        segs.append(tokens[tid])
        pos = prev
    
    return list(reversed(segs))

# -------------------------
# Optimized Tokenizer Training
# -------------------------
def train_unigram_tokenizer_fast(word_counts: Counter, vocab_size: int, eta: float = 0.6, seed_size: int = 5000, em_iterations: int = 1) -> Tuple[List[str], Dict]:
    print("Building seed vocabulary...")
    tokens, token_counts = build_seed_vocab(word_counts, seed_size=seed_size)
    probs = [token_counts[tok] / sum(token_counts.values()) for tok in tokens]
    single_chars = {t for t in tokens if len(t) == 1}
    
    print(f"Initial vocabulary size: {len(tokens)}")
    print(f"Target vocabulary size: {vocab_size}")
    
    iteration = 0
    while len(tokens) > vocab_size:
        iteration += 1
        print(f"Iteration {iteration}: Current vocab size = {len(tokens)}")
        
        trie = FastTrie(tokens)
        log_probs = [math.log(max(p, 1e-300)) for p in probs]
        
        # Single EM iteration for speed
        expected, ll = em_step_fast(word_counts, tokens, log_probs, trie)
        print(f"  Log likelihood: {ll:.2f}")
        
        probs = normalize_expected(expected)
        tokens, probs = prune_tokens_fast(tokens, expected, keep_eta=eta, min_keep_tokens=single_chars)
    
    print(f"Final vocabulary size: {len(tokens)}")
    tokenizer = {
        "tokens": tokens,
        "probs": probs,
        "trie": FastTrie(tokens),
        "max_token_len": max(len(t) for t in tokens) if tokens else 0
    }
    return tokens, tokenizer

# -------------------------
# Optimized Tokenization
# -------------------------
def tokenize_text_fast(text: str, tokenizer: Dict) -> List[str]:
    tokens = tokenizer["tokens"]
    log_probs = [math.log(max(p, 1e-300)) for p in tokenizer["probs"]]
    trie = tokenizer["trie"]
    
    out = []
    words = _WORD_RE.findall(text)
    total_words = len(words)
    
    for i, word in enumerate(words):
        if i % 1000 == 0:
            print(f"Tokenizing: {i}/{total_words} words processed")
            
        segs = viterbi_segment_fast(word, tokens, log_probs, trie)
        if segs:
            segs[0] = "\u2581" + segs[0]
        out.extend(segs)
    
    return out

def detokenize_tokens_fast(tokens: List[str]) -> str:
    result = []
    current_word = []
    
    for tok in tokens:
        if tok.startswith("\u2581"):
            if current_word:
                result.append(''.join(current_word))
                current_word = []
            result.append(" ")
            tok = tok[1:]
        current_word.append(tok)
    
    if current_word:
        result.append(''.join(current_word))
    
    return "".join(result).strip()

# -------------------------
# Save Helpers
# -------------------------
def save_vocab(vocab: List[str], rollno: str, vocab_size: int):
    fname = f"{rollno}_assignment2_unigram_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def save_tokens(tokens: List[str], rollno: str):
    fname = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text: str, rollno: str):
    fname = f"{rollno}_assignment2_unigram_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251110086"
    
    start_time = time.time()
    
    print("Loading training data...")
    word_counts = load_training_data(args.train)
    print(f"Loaded {len(word_counts)} unique words")
    
    print("Training tokenizer...")
    vocab, tokenizer = train_unigram_tokenizer_fast(word_counts, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)
    
    print("Loading input text...")
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    print("Tokenizing...")
    tokens = tokenize_text_fast(sample_text, tokenizer)
    save_tokens(tokens, rollno)
    
    print("Detokenizing...")
    detok_text = detokenize_tokens_fast(tokens)
    save_detokenized(detok_text, rollno)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
