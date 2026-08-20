import argparse
import os
import heapq
from collections import Counter, defaultdict
import math
import re
import time

class TokenNode:
    """Linked list node for efficient sequence manipulation"""
    __slots__ = ['token', 'freq', 'prev', 'next', 'seq_id']
    
    def __init__(self, token, freq=1, seq_id=None):
        self.token = token
        self.freq = freq
        self.prev = None
        self.next = None
        self.seq_id = seq_id

class SequenceLinkedList:
    """Linked list representation of a token sequence"""
    __slots__ = ['head', 'tail', 'length', 'seq_id']
    
    def __init__(self, tokens, freq=1, seq_id=None):
        self.head = None
        self.tail = None
        self.length = 0
        self.seq_id = seq_id
        
        for token in tokens:
            self.append(token, freq)
    
    def append(self, token, freq=1):
        new_node = TokenNode(token, freq, self.seq_id)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.length += 1
    
    def find_and_merge(self, pair, new_token):
        """Find and merge all occurrences of a pair in O(n) time"""
        current = self.head
        merged_count = 0
        
        while current and current.next:
            if (current.token == pair[0] and current.next.token == pair[1]):
                # Merge the pair
                merged_node = TokenNode(new_token, current.freq, self.seq_id)
                
                # Update links
                if current.prev:
                    current.prev.next = merged_node
                    merged_node.prev = current.prev
                else:
                    self.head = merged_node
                
                if current.next.next:
                    current.next.next.prev = merged_node
                    merged_node.next = current.next.next
                else:
                    self.tail = merged_node
                
                # Move to next position
                current = merged_node.next
                merged_count += 1
                self.length -= 1  # Reduced by 1 token per merge
            else:
                current = current.next
        
        return merged_count
    
    def to_list(self):
        """Convert linked list to Python list"""
        result = []
        current = self.head
        while current:
            result.append(current.token)
            current = current.next
        return result
    
    def get_pairs(self):
        """Get all pairs in the sequence with their positions"""
        pairs = []
        current = self.head
        while current and current.next:
            pairs.append((current.token, current.next.token))
            current = current.next
        return pairs

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read().lower()

def get_base_vocab(text):
    """Create initial tokenization at character level."""
    tokens = []
    words = text.strip().split()
    
    max_words = 10000000
    if len(words) > max_words:
        import random
        words = random.sample(words, max_words)
    
    for word in words:
        chars = list(word)
        chars.append('</w>')
        tokens.append(chars)
    return tokens

def train_wordpiece_tokenizer(text, vocab_size):
    reserved_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    tokens_list = get_base_vocab(text)
    
    # Initialize vocab and frequencies
    vocab = set()
    individual_freq = Counter()
    pair_freq = Counter()
    pair_to_sequences = defaultdict(list)  # pair -> list of (sequence, position)
    
    # Create linked lists for all sequences
    sequences = []
    seq_id = 0
    
    freq_dict = Counter(tuple(token) for token in tokens_list)
    for token_seq, freq in freq_dict.items():
        seq_ll = SequenceLinkedList(token_seq, freq, seq_id)
        sequences.append(seq_ll)
        
        # Build vocab and individual frequencies
        current = seq_ll.head
        while current:
            vocab.add(current.token)
            individual_freq[current.token] += freq
            current = current.next
        
        # Build pair frequencies and index
        current = seq_ll.head
        pos = 0
        while current and current.next:
            pair = (current.token, current.next.token)
            pair_freq[pair] += freq
            pair_to_sequences[pair].append((seq_ll, current))  # Store node reference
            current = current.next
            pos += 1
        
        seq_id += 1
    
    total_tokens = sum(freq_dict.values())
    
    # Build heap with likelihood gain
    heap = []
    for pair, freq_new in pair_freq.items():
        token1, token2 = pair
        if (freq_new > 1 and token1 in individual_freq and token2 in individual_freq and
            individual_freq[token1] > 0 and individual_freq[token2] > 0):
            
            # ΔL = f_new * log(f_new) - f1 * log(f1) - f2 * log(f2)
            gain = (freq_new * math.log(freq_new) - 
                   individual_freq[token1] * math.log(individual_freq[token1]) - 
                   individual_freq[token2] * math.log(individual_freq[token2]))
            heapq.heappush(heap, (-gain, pair, freq_new))
    
    merge_count = 0
    max_merges = vocab_size - len(vocab) - len(reserved_tokens)
    
    print(f"Starting with {len(vocab)} base characters, target: {vocab_size}")
    print(f"Total sequences: {len(sequences)}, Initial pairs: {len(heap)}")
    
    while len(vocab) + len(reserved_tokens) < vocab_size and merge_count < max_merges and heap:
        neg_gain, pair, expected_freq = heapq.heappop(heap)
        gain = -neg_gain
        
        # Check if pair is still valid
        if pair not in pair_freq or pair_freq[pair] != expected_freq:
            continue
        
        token1, token2 = pair
        new_token = ''.join(pair)
        vocab.add(new_token)
        merge_count += 1
        
        # Update individual frequencies
        individual_freq[new_token] = pair_freq[pair]
        individual_freq[token1] -= pair_freq[pair]
        individual_freq[token2] -= pair_freq[pair]
        
        # Remove the pair from frequency tracking
        del pair_freq[pair]
        
        # Merge this pair in all sequences using the pre-built index
        affected_sequences = set()
        new_pairs = Counter()
        
        for seq_ll, node in pair_to_sequences[pair]:
            if seq_ll not in affected_sequences:
                # Count merges in this sequence
                merge_count_seq = seq_ll.find_and_merge(pair, new_token)
                
                if merge_count_seq > 0:
                    affected_sequences.add(seq_ll)
                    
                    # Update pair frequencies for the affected sequence
                    current = seq_ll.head
                    while current and current.next:
                        new_pair = (current.token, current.next.token)
                        new_pairs[new_pair] += seq_ll.head.freq  # All nodes have same freq
                        current = current.next
        
        # Update pair_to_sequences index
        if pair in pair_to_sequences:
            del pair_to_sequences[pair]
        
        # Add new pairs to frequency tracking and index
        for new_pair, freq in new_pairs.items():
            pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
            
            # Add to sequences index (would need to track nodes, simplified here)
            # For efficiency, we're not rebuilding the full index
        
        # Rebuild heap with updated pairs
        new_heap = []
        for pair, freq_new in pair_freq.items():
            token1, token2 = pair
            if (freq_new > 1 and token1 in individual_freq and token2 in individual_freq and
                individual_freq[token1] > 0 and individual_freq[token2] > 0):
                
                gain = (freq_new * math.log(freq_new) - 
                       individual_freq[token1] * math.log(individual_freq[token1]) - 
                       individual_freq[token2] * math.log(individual_freq[token2]))
                heapq.heappush(new_heap, (-gain, pair, freq_new))
        
        heap = new_heap
        
        if merge_count % 10 == 0:
            print(f"Merged {merge_count}: '{new_token}' (freq: {expected_freq}, gain: {gain:.2f})")
            print(f"Vocab size: {len(vocab)}, Remaining pairs: {len(heap)}")
    
    # Convert linked lists back to tuples for final processing
    final_sequences = []
    for seq_ll in sequences:
        final_sequences.append((tuple(seq_ll.to_list()), seq_ll.head.freq))
    
    final_vocab = reserved_tokens + sorted(vocab, key=lambda x: (-len(x), x))
    print(f"Final vocab size: {len(final_vocab)}")
    
    tokenizer = {
        "vocab": set(final_vocab),
        "reserved_tokens": reserved_tokens
    }
    return final_vocab, tokenizer

def tokenize(text, tokenizer):
    text = text.lower()
    tokens = []
    vocab_set = tokenizer["vocab"]
    
    words = text.strip().split()
    
    for word in words:
        chars = list(word)
        chars.append('</w>')
        sequence = chars
        output = []
        is_word_start = True
        
        while sequence:
            matched = False
            max_search = min(20, len(sequence))
            for end in range(max_search, 0, -1):
                piece = ''.join(sequence[:end])
                if piece in vocab_set:
                    if not is_word_start:
                        piece = '##' + piece
                    output.append(piece)
                    sequence = sequence[end:]
                    is_word_start = False
                    matched = True
                    break
            
            if not matched:
                if sequence and sequence[0] in vocab_set:
                    piece = sequence[0]
                    if not is_word_start:
                        piece = '##' + piece
                    output.append(piece)
                    sequence = sequence[1:]
                    is_word_start = False
                else:
                    if not is_word_start:
                        output.append('##<unk>')
                    else:
                        output.append('<unk>')
                    sequence = sequence[1:] if sequence else []
                    is_word_start = False
        
        tokens.extend(output)
    
    return tokens

def detokenize(tokens, tokenizer):
    text = ' '.join(tokens)
    text = text.replace(' ##', '')
    text = text.replace('</w>', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_wp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_wp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_wp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251110086"

    start_time = time.time()
    
    train_text = load_training_data(args.train)
    print(f"Loaded training data: {len(train_text.split())} words")
    
    vocab, tokenizer = train_wordpiece_tokenizer(train_text, args.vocab_size)
    print(f"Training completed in {time.time() - start_time:.2f}s")
    
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
    
    print(f"Total execution time: {time.time() - start_time:.2f}s")
