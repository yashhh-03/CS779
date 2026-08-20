import argparse
import unicodedata
from collections import Counter
import re
import time

# -------------------------
# Data Loading & Unicode Normalization
# -------------------------
def load_training_data(train_path):
    """Load corpus with UTF-8 encoding."""
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def normalize_text_nfkc(text):
    """Apply NKFC Unicode normalization as required."""
    return unicodedata.normalize("NFKC", text)

def add_whitespace_markers(text):
    """Add whitespace markers (U+2581) to text."""
    words = re.findall(r'\S+', text)
    if not words:
        return ""
    return words[0] + ''.join('▁' + word for word in words[1:])

# -------------------------
# FAST Frequency-Based Tokenizer (No BPE)
# -------------------------
def train_fast_tokenizer(text, vocab_size):
    """
    Ultra-fast frequency-based tokenizer that runs in O(n) time.
    Uses substring frequency counting instead of slow BPE merges.
    """
    print("Building frequency statistics...")
    start_time = time.time()
    
    # Start with character vocabulary
    vocab = set(text)
    
    # Sample text for frequency counting (1MB max for speed)
    sample_size = min(len(text), 1000000)
    sample_text = text[:sample_size]
    
    # Count most frequent substrings of length 2-4
    substring_freq = Counter()
    max_len = 4  # Short substrings for speed
    
    for length in range(2, max_len + 1):
        for i in range(len(sample_text) - length + 1):
            substring = sample_text[i:i+length]
            substring_freq[substring] += 1
    
    # Add most frequent substrings to vocabulary
    most_frequent = substring_freq.most_common(vocab_size - len(vocab))
    for substring, freq in most_frequent:
        if freq >= 2:  # Only add reasonably frequent substrings
            vocab.add(substring)
    
    # Add reserved tokens in specified order
    reserved_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    for token in reserved_tokens:
        vocab.add(token)
    
    # Add byte fallback tokens
    for b in range(256):
        vocab.add(f"<0x{b:02x}>")
    
    # Deterministic vocabulary ordering
    vocab_list = []
    vocab_list.extend(reserved_tokens)
    vocab_list.extend(sorted([f"<0x{b:02x}>" for b in range(256)]))
    vocab_list.extend(sorted([token for token in vocab 
                            if token not in reserved_tokens and not token.startswith("<0x")]))
    
    print(f"Vocabulary built in {time.time() - start_time:.1f}s")
    return vocab_list, {"vocab": set(vocab_list)}

# -------------------------
# Greedy Longest-Match Tokenization (FAST)
# -------------------------
def tokenize_fast(text, tokenizer):
    """Very fast greedy longest-match tokenization."""
    vocab = tokenizer["vocab"]
    
    # Precompute max token length for efficiency
    max_token_len = 0
    for token in vocab:
        max_token_len = max(max_token_len, len(token))
    
    tokens = []
    i = 0
    n = len(text)
    
    while i < n:
        # Find longest matching token (greedy)
        found = False
        for length in range(min(max_token_len, n - i), 0, -1):
            candidate = text[i:i+length]
            if candidate in vocab:
                tokens.append(candidate)
                i += length
                found = True
                break
        
        if not found:
            # Byte fallback for unknown characters
            char = text[i]
            try:
                for b in char.encode('utf-8'):
                    tokens.append(f"<0x{b:02x}>")
            except:
                tokens.append("<unk>")
            i += 1
    
    return tokens

def detokenize_fast(tokens):
    """Fast deterministic detokenization."""
    text = ""
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        if token == "▁":
            text += " "
        elif token.startswith("<0x") and token.endswith(">"):
            # Handle byte sequences
            byte_tokens = []
            while i < len(tokens) and tokens[i].startswith("<0x") and tokens[i].endswith(">"):
                try:
                    byte_val = int(tokens[i][3:-1], 16)
                    byte_tokens.append(byte_val)
                except:
                    pass
                i += 1
            i -= 1
            
            if byte_tokens:
                try:
                    text += bytes(byte_tokens).decode('utf-8', errors='replace')
                except:
                    # Fallback: output byte tokens as-is
                    for bt in byte_tokens:
                        text += f"<0x{bt:02x}>"
        elif token in ["<pad>", "<s>", "</s>", "<unk>"]:
            # Skip special tokens
            pass
        else:
            text += token
        
        i += 1
    
    return text

# -------------------------
# Save Functions
# -------------------------
def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_sp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

# -------------------------
# Main with Performance Optimization
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251110086"
    total_start = time.time()
    
    # Phase 1: Load and normalize
    print("Phase 1: Loading and normalizing data...")
    train_text = load_training_data(args.train)
    train_text = normalize_text_nfkc(train_text)
    train_text = add_whitespace_markers(train_text)

    
    # Phase 2: Train tokenizer (FAST)
    print("Phase 2: Training tokenizer...")
    train_start = time.time()
    vocab, tokenizer = train_fast_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)
    train_time = time.time() - train_start
    print(f"  Trained in {train_time:.1f}s")
    
    # Phase 3: Process input text
    print("Phase 3: Processing input text...")
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    sample_text = normalize_text_nfkc(sample_text)
    sample_text = add_whitespace_markers(sample_text)
    
    # Phase 4: Tokenize (FAST)
    print("Phase 4: Tokenizing...")
    tokenize_start = time.time()
    tokens = tokenize_fast(sample_text, tokenizer)
    save_tokens(tokens, rollno)
    tokenize_time = time.time() - tokenize_start
    print(f"  Tokenized in {tokenize_time:.1f}s")
    
    # Phase 5: Detokenize
    print("Phase 5: Detokenizing...")
    detok_start = time.time()
    detok_text = detokenize_fast(tokens)
    # Remove space markers for final output
    detok_text = detok_text.replace("▁", " ")
    save_detokenized(detok_text, rollno)
    detok_time = time.time() - detok_start
    
         
            
            
