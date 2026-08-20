import argparse
from collections import defaultdict, Counter
import heapq
import re

# ------------------------
# Symbol and SymbolPair structures
# ------------------------
class Symbol:
	
    def __init__(self, start_pos, end_pos, prev_idx=-1, next_idx=-1, text=""):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.prev = prev_idx
        self.next = next_idx
        self.text = text
    
    def __repr__(self):
        return f"Symbol('{self.text}', {self.start_pos}:{self.end_pos})"

class SymbolPair:
    def __init__(self, left_idx, right_idx, score, pair_text):
        self.left = left_idx
        self.right = right_idx
        self.score = score
        self.pair_text = pair_text
        self.heap_score = -score
    
    def __lt__(self, other):
        return self.heap_score < other.heap_score
    
    def __repr__(self):
        return f"SymbolPair({self.left}->{self.right}, '{self.pair_text}', score={self.score})"

# Symbol List with linked list operations

class SymbolList:
#it can find a potential merge candidate between two symbols"
    def __init__(self):
        self.symbols = []
    
    def from_text(self, text, word_boundaries):
        self.symbols = []
        
        for word_start, word_end in word_boundaries:
            word_text = text[word_start:word_end]
            word_symbols = []
            prev_idx = -1
            
            for i, char in enumerate(word_text):
                symbol_idx = len(self.symbols)
                symbol = Symbol(word_start + i, word_start + i + 1, prev_idx, -1, char)
                
                if prev_idx != -1:
                    self.symbols[prev_idx].next = symbol_idx
                
                self.symbols.append(symbol)
                word_symbols.append(symbol_idx)
                prev_idx = symbol_idx
            
            if word_symbols:
                eow_idx = len(self.symbols)
                eow_symbol = Symbol(word_end, word_end + 1, prev_idx, -1, "</w>")
                if prev_idx != -1:
                    self.symbols[prev_idx].next = eow_idx
                self.symbols.append(eow_symbol)
    
    def get_symbol(self, idx):
        if 0 <= idx < len(self.symbols):
            return self.symbols[idx]
        return None
    
    def merge_symbols(self, left_idx, right_idx):
        left_symbol = self.get_symbol(left_idx)
        right_symbol = self.get_symbol(right_idx)
        
        if not left_symbol or not right_symbol:
            return None
        
        merged_text = left_symbol.text + right_symbol.text
        merged_symbol = Symbol(
            left_symbol.start_pos,
            right_symbol.end_pos,
            left_symbol.prev,
            right_symbol.next,
            merged_text
        )
        
        if right_symbol.next != -1:
            self.symbols[right_symbol.next].prev = left_idx
        
        self.symbols[left_idx] = merged_symbol
        self.symbols[right_idx] = None
        
        return merged_symbol
    
    def get_valid_symbols(self):
        return [(i, symbol) for i, symbol in enumerate(self.symbols) if symbol is not None]
    
    def __len__(self):
        return len(self.symbols)

# ------------------------
# BPE Tokenizer with Priority Queue
# ------------------------
class PriorityQueueBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merge_priorities = {}
        self.final_merges = {}
    
    def maybe_add_pair(self, left_idx, right_idx, symbol_list, agenda):
        if left_idx == -1 or right_idx == -1: #"--> checking if symbol can be add to merge queue" 
            return
        
        left_symbol = symbol_list.get_symbol(left_idx)
        right_symbol = symbol_list.get_symbol(right_idx)
        
        if not left_symbol or not right_symbol:
            return
        
        if left_symbol.next != right_idx:
            return
        
        pair_text = left_symbol.text + right_symbol.text
        score = self.merge_priorities.get(pair_text, -1)
        
        if score > -1:
            symbol_pair = SymbolPair(left_idx, right_idx, score, pair_text)
            heapq.heappush(agenda, symbol_pair)
    
    def train(self, text, vocab_size):
        print("Starting BPE training with priority queue...")
        
        reserved_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        self.vocab = {token: i for i, token in enumerate(reserved_tokens)}
        
        text = text.strip()
        words = re.findall(r'\b\w+\b', text) #"--> extracting word from text"
        # If the dataset is too large we can sample most frequent words for efficiency 
        if len(words) > 100000:
            word_counter = Counter(words)
            total_count = sum(word_counter.values())
            cumulative = 0
            sampled_words = []
            for word, count in word_counter.most_common():
                sampled_words.append(word)
                cumulative += count
                if cumulative >= 0.99 * total_count:
                    break
            words = sampled_words

        # calculating word boundaries
        word_boundaries = []
        pos = 0
        sample_text = ' '.join(words)
        
        for word in words:
            word_start = sample_text.find(word, pos)
            if word_start == -1:
                continue
            word_end = word_start + len(word)
            word_boundaries.append((word_start, word_end))
            pos = word_end
        
        symbol_list = SymbolList() #--> create symbol list from text
        symbol_list.from_text(sample_text, word_boundaries)
         #Count frequency of all adjacent symbol pairs
        pair_counts = Counter()
        for i, symbol in enumerate(symbol_list.symbols):
            if symbol and symbol.next != -1:
                next_symbol = symbol_list.get_symbol(symbol.next)
                if next_symbol:
                    pair_text = symbol.text + next_symbol.text
                    pair_counts[pair_text] += 1
        #Assign priority to most frequ pairs
        merge_id = 0
        for pair_text, count in pair_counts.most_common():
            if count >= 5 and merge_id < (vocab_size - len(reserved_tokens)):  # Relaxed threshold
                self.merge_priorities[pair_text] = merge_id
                self.final_merges[pair_text] = count
                merge_id += 1
        
         #Initialize priority queue with valid symbol pairs
        agenda = []
        valid_symbols = symbol_list.get_valid_symbols()
        
        for i, (idx, symbol) in enumerate(valid_symbols[:-1]):
            if symbol.next != -1:
                self.maybe_add_pair(idx, symbol.next, symbol_list, agenda)
        
        heapq.heapify(agenda)

        
        merges_applied = 0
        max_merges = vocab_size - len(reserved_tokens)  # Use full available space
        
        while agenda and merges_applied < max_merges:
            symbol_pair = heapq.heappop(agenda)
            
            left_symbol = symbol_list.get_symbol(symbol_pair.left)
            right_symbol = symbol_list.get_symbol(symbol_pair.right)
            
            if (not left_symbol or not right_symbol or left_symbol.next != symbol_pair.right):
                continue
            #here i am merging the symbol
            merged_symbol = symbol_list.merge_symbols(symbol_pair.left, symbol_pair.right)
            
            if merged_symbol:
                self.final_merges[merged_symbol.text] = symbol_pair.score
                
                self.maybe_add_pair(
                    merged_symbol.prev, 
                    symbol_pair.left, 
                    symbol_list, 
                    agenda
                )
                self.maybe_add_pair(
                    symbol_pair.left, 
                    merged_symbol.next, 
                    symbol_list, 
                    agenda
                )
                
                merges_applied += 1
                
        
        final_vocab = list(reserved_tokens)
        
        seen_symbols = set()
        for idx, symbol in symbol_list.get_valid_symbols():
            if symbol.text and symbol.text not in seen_symbols:
                final_vocab.append(symbol.text)
                seen_symbols.add(symbol.text)
        
        for merge_text, frequency in sorted(self.final_merges.items(), key=lambda x: x[1], reverse=True):
            if merge_text not in seen_symbols and len(final_vocab) < vocab_size:
                final_vocab.append(merge_text)
                seen_symbols.add(merge_text)
        
        if len(final_vocab) < vocab_size:
            all_chars = set()
            for word in words:
                all_chars.update(word)
            for char in sorted(all_chars):
                if char not in seen_symbols and len(final_vocab) < vocab_size:
                    final_vocab.append(char)
                    seen_symbols.add(char)
        
        return final_vocab[:vocab_size], self.final_merges
    
    def tokenize_word(self, word, merges):
        tokens = [char for char in word] + ['</w>']
        
        for merge_text, _ in sorted(merges.items(), key=lambda x: x[1], reverse=True):
            if len(merge_text) < 2:
                continue
                
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i + 1] == merge_text:
                    new_tokens.append(merge_text)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def tokenize_text(self, text, merges):
        tokens = []
        words = re.findall(r'\b\w+\b', text.strip())
        

        for i, word in enumerate(words):
            
            word_tokens = self.tokenize_word(word, merges)
            tokens.extend(word_tokens)
        
        return tokens

# ------------------------
# File I/O
# ------------------------
def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            clean_token = token.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            f.write(clean_token + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            clean_tok = tok.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            f.write(clean_tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

def detokenize(tokens):
    text = ' '.join(tokens)  # First join with spaces
    # Replace </w> with nothing, but preserve word boundaries
    text = text.replace(' ', '')
    #text = text.replace(' </w>', ' ')  # Handle space + </w>
    text = text.replace('</w>', ' ')    # Handle any remaining </w>
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251110086"

    print(f"Loading training data from {args.train}...")
    train_text = load_training_data(args.train)
    
    
    tokenizer = PriorityQueueBPETokenizer()
    
    print("Training BPE...")
    vocab, merges = tokenizer.train(train_text, args.vocab_size)
    

    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        input_text = f.read()
    
    tokens = tokenizer.tokenize_text(input_text, merges)
    save_tokens(tokens, rollno)
    
    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)
    
    print("Done!")

