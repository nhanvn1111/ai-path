# File: 05_tokenizer.py
# BPE + WordPiece Tokenizer tu Scratch - Week 5-7
#
# TAI SAO CAN TOKENIZER?
# Model khong hieu text, chi hieu so. Tokenizer chuyen text -> so:
# "Hello world" -> [15496, 995]
#
# TAI SAO KHONG DUNG TUNG KY TU?
# - Char-level: "Hello" -> [H, e, l, l, o] = 5 tokens -> sequence dai, train cham
# - Word-level: "Hello" -> [Hello] = 1 token -> vocab qua lon, khong xu ly duoc tu moi
# - BPE/WordPiece: "unhappiness" -> ["un", "happi", "ness"] = 3 tokens
#   -> balance giua vocab size va sequence length
#   -> xu ly duoc tu moi bang cach chia nho thanh subwords
#
# GPT dung BPE, BERT dung WordPiece.

import re
import os
import time
import urllib.request
from collections import Counter


# ============ BPE TOKENIZER ============

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer

    Y tuong:
    1. Bat dau voi characters
    2. Lien tuc merge cap xuat hien nhieu nhat
    3. Stop khi dat vocab_size

    Vi du:
    "low lower lowest"
    -> ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', ...]
    -> merge ('l', 'o') -> 'lo'
    -> merge ('lo', 'w') -> 'low'
    """

    # Bai tap 4: Special tokens
    SPECIAL_TOKENS = ['<pad>', '<unk>', '<s>', '</s>', '[CLS]', '[SEP]', '[MASK]']

    def __init__(self, vocab_size=1000):
        """
        vocab_size: so luong tokens toi da trong vocab
                    Nho (1000) = nhieu subword, sequence dai nhung vocab nho
                    Lon (50000, GPT-2 dung 50257) = it subword, sequence ngan
                    Trade-off: vocab lon -> lookup table lon (ton memory)
                               vocab nho -> sequence dai (ton compute)
        """
        self.vocab_size = vocab_size
        self.merges = {}          # {('l','o'): 'lo'} - cac cap da merge
        self.vocab = {}           # {'lo': 5} - token -> id
        self.inverse_vocab = {}   # {5: 'lo'} - id -> token

    def get_pairs(self, word):
        pairs = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs

    def merge_pair(self, tokens, pair, merged):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, texts, verbose=True):
        """
        texts:   list cac string de hoc vocab, vd ["hello world", "hello there"]
                 BPE dem frequency cua tung cap ky tu, merge cap nhieu nhat
        verbose: True = in tien do merge
        """
        # Buoc 1: Tokenize thanh characters + end of word marker
        words = Counter()
        for text in texts:
            for word in text.split():
                word_tokens = tuple(list(word) + ['</w>'])
                words[word_tokens] += 1

        if verbose:
            unique_chars = set(c for w in words for c in w)
            print(f"  Initial vocabulary: {len(unique_chars)} characters")
            print(f"  Unique words: {len(words)}")

        # Buoc 2: Iteratively merge pairs
        num_merges = max(0, self.vocab_size - 256 - len(self.SPECIAL_TOKENS) - 1)

        for i in range(num_merges):
            pair_counts = Counter()
            for word, freq in words.items():
                pairs = self.get_pairs(list(word))
                for pair, count in pairs.items():
                    pair_counts[pair] += count * freq

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            merged = best_pair[0] + best_pair[1]

            new_words = Counter()
            for word, freq in words.items():
                new_word = tuple(self.merge_pair(list(word), best_pair, merged))
                new_words[new_word] += freq
            words = new_words

            self.merges[best_pair] = merged

            if verbose and (i + 1) % 100 == 0:
                print(f"  Merge {i + 1}/{num_merges}: {best_pair} -> '{merged}'")

        # Buoc 3: Build vocabulary
        all_tokens = set()
        for word in words:
            all_tokens.update(word)

        for i in range(256):
            all_tokens.add(chr(i))

        all_tokens.add('</w>')

        # Bai tap 4: Add special tokens
        for st in self.SPECIAL_TOKENS:
            all_tokens.add(st)

        for idx, token in enumerate(sorted(all_tokens)):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

        if verbose:
            print(f"  Final vocabulary size: {len(self.vocab)}")

    def encode(self, text):
        """
        text: string can encode, vd "Hello world"
        Return: list token ids, vd [15, 42, 3, 99]
        """
        tokens = []
        for word in text.split():
            word_tokens = list(word) + ['</w>']

            for pair, merged in self.merges.items():
                word_tokens = self.merge_pair(word_tokens, pair, merged)

            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab.get('<unk>', 0))

        return tokens

    def decode(self, ids):
        """
        ids: list token ids, vd [15, 42, 3, 99]
        Return: string goc, vd "Hello world"
        Nguoc lai cua encode(): id -> token -> ghep lai thanh text
        """
        tokens = [self.inverse_vocab.get(i, '<unk>') for i in ids]
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

    def encode_with_special(self, text, add_cls=False, add_sep=False):
        """
        Encode voi special tokens (BERT style)

        text:    string can encode
        add_cls: True = them [CLS] o dau (BERT can [CLS] cho classification)
        add_sep: True = them [SEP] o cuoi (BERT can [SEP] de danh dau cuoi cau)
        """
        ids = []
        if add_cls:
            ids.append(self.vocab['[CLS]'])
        ids.extend(self.encode(text))
        if add_sep:
            ids.append(self.vocab['[SEP]'])
        return ids

    def encode_pair(self, text_a, text_b):
        """Bai tap 4: Encode sentence pair (BERT style)"""
        ids = [self.vocab['[CLS]']]
        ids.extend(self.encode(text_a))
        ids.append(self.vocab['[SEP]'])
        ids.extend(self.encode(text_b))
        ids.append(self.vocab['[SEP]'])
        return ids


# ============ BAI TAP 2: WORDPIECE TOKENIZER ============

class WordPieceTokenizer:
    """
    WordPiece Tokenizer (BERT style)

    Khac voi BPE:
    - BPE: merge pair co frequency cao nhat
    - WordPiece: merge pair maximize likelihood cua corpus

    Score = freq(ab) / (freq(a) * freq(b))

    Prefix '##' cho subword tokens (khong phai bat dau word)
    Vi du: "playing" -> ["play", "##ing"]
    """

    SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    def __init__(self, vocab_size=1000):
        """
        vocab_size: giong BPE - so tokens toi da trong vocab
                    BERT-base dung 30522 tokens
        """
        self.vocab_size = vocab_size
        self.vocab = {}           # token -> id
        self.inverse_vocab = {}   # id -> token

    def train(self, texts, verbose=True):
        """
        texts:   list string de hoc vocab
        verbose: True = in tien do

        Khac BPE: WordPiece merge pair co SCORE cao nhat (khong phai frequency)
        Score = freq(ab) / (freq(a) * freq(b))
        -> Uu tien merge pair hiem nhung thuong xuat hien cung nhau
        """
        # Buoc 1: Split thanh words va dem frequency
        word_freq = Counter()
        for text in texts:
            for word in text.lower().split():
                word = re.sub(r'[^\w]', '', word)
                if word:
                    word_freq[word] += 1

        # Buoc 2: Split moi word thanh characters, them ## prefix
        word_splits = {}
        for word in word_freq:
            chars = list(word)
            split = [chars[0]] + ['##' + c for c in chars[1:]]
            word_splits[word] = split

        # Build initial vocab tu tat ca characters
        token_freq = Counter()
        for word, freq in word_freq.items():
            for token in word_splits[word]:
                token_freq[token] += freq

        # Buoc 3: Iteratively merge (WordPiece scoring)
        num_merges = self.vocab_size - len(token_freq) - len(self.SPECIAL_TOKENS)

        if verbose:
            print(f"  Initial tokens: {len(token_freq)}")
            print(f"  Target merges: {max(0, num_merges)}")

        for i in range(max(0, num_merges)):
            # Tinh pair scores
            pair_freq = Counter()
            for word, freq in word_freq.items():
                tokens = word_splits[word]
                for j in range(len(tokens) - 1):
                    pair_freq[(tokens[j], tokens[j + 1])] += freq

            if not pair_freq:
                break

            # WordPiece score: freq(ab) / (freq(a) * freq(b))
            best_pair = None
            best_score = -1
            for pair, freq in pair_freq.items():
                fa = token_freq.get(pair[0], 1)
                fb = token_freq.get(pair[1], 1)
                score = freq / (fa * fb)
                if score > best_score:
                    best_score = score
                    best_pair = pair

            if best_pair is None:
                break

            # Merge
            a, b = best_pair
            if b.startswith('##'):
                merged = a + b[2:]
            else:
                merged = a + b

            # Update word_splits
            for word in word_splits:
                tokens = word_splits[word]
                new_tokens = []
                j = 0
                while j < len(tokens):
                    if j < len(tokens) - 1 and tokens[j] == a and tokens[j + 1] == b:
                        new_tokens.append(merged)
                        j += 2
                    else:
                        new_tokens.append(tokens[j])
                        j += 1
                word_splits[word] = new_tokens

            # Update token freq
            token_freq[merged] = pair_freq[best_pair]

            if verbose and (i + 1) % 100 == 0:
                print(f"  Merge {i + 1}: {best_pair} -> '{merged}' (score={best_score:.4f})")

        # Build vocab
        all_tokens = set()
        for word in word_splits:
            all_tokens.update(word_splits[word])
        for st in self.SPECIAL_TOKENS:
            all_tokens.add(st)

        for idx, token in enumerate(sorted(all_tokens)):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

        self._word_splits = word_splits

        if verbose:
            print(f"  Final vocabulary size: {len(self.vocab)}")

    def encode(self, text):
        tokens = []
        for word in text.lower().split():
            word = re.sub(r'[^\w]', '', word)
            if not word:
                continue

            # Greedy longest-match (giong BERT)
            chars = list(word)
            i = 0
            sub_tokens = []
            while i < len(chars):
                # Tim longest match
                end = len(chars)
                found = False
                while end > i:
                    candidate = ''.join(chars[i:end])
                    if i > 0:
                        candidate = '##' + candidate
                    if candidate in self.vocab:
                        sub_tokens.append(candidate)
                        found = True
                        break
                    end -= 1
                if not found:
                    sub_tokens.append('[UNK]')
                    i += 1
                else:
                    i = end

            tokens.extend(sub_tokens)

        return [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]

    def decode(self, ids):
        tokens = [self.inverse_vocab.get(i, '[UNK]') for i in ids]
        text = ''
        for token in tokens:
            if token in self.SPECIAL_TOKENS:
                continue
            if token.startswith('##'):
                text += token[2:]
            else:
                text += ' ' + token
        return text.strip()


# ============ HELPER ============

def download_shakespeare():
    """Download Shakespeare text cho training"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakespeare.txt")

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return f.read()

    print("  Downloading Shakespeare...")
    try:
        response = urllib.request.urlopen(url)
        text = response.read().decode('utf-8')
        with open(cache_path, 'w') as f:
            f.write(text)
        return text
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


# ============ MAIN ============
if __name__ == "__main__":
    print("=" * 60)
    print("TEST CO BAN - BPE Tokenizer")
    print("=" * 60)

    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog runs quickly through the forest",
        "a quick brown dog jumps over a lazy fox",
        "the forest has many brown trees",
        "foxes and dogs are quick animals",
    ] * 100

    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus)

    test_text = "the quick fox"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\n  Original: '{test_text}'")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  '{decoded}'")
    assert decoded == test_text, f"Decode mismatch: '{decoded}' != '{test_text}'"
    print("  Encode/Decode: OK")

    print(f"\n  First 10 merges:")
    for i, (pair, merged) in enumerate(list(tokenizer.merges.items())[:10]):
        print(f"    {pair} -> '{merged}'")

    # ============ BAI TAP 4: Special tokens ============
    print("\n" + "=" * 60)
    print("BAI TAP 4: Special Tokens")
    print("=" * 60)

    # Check special tokens in vocab
    for st in BPETokenizer.SPECIAL_TOKENS:
        assert st in tokenizer.vocab, f"Missing special token: {st}"
    print(f"  Special tokens in vocab: {BPETokenizer.SPECIAL_TOKENS}")

    # BERT-style encoding
    encoded_cls = tokenizer.encode_with_special("the quick fox", add_cls=True, add_sep=True)
    print(f"  [CLS] + 'the quick fox' + [SEP]: {encoded_cls}")
    assert tokenizer.inverse_vocab[encoded_cls[0]] == '[CLS]'
    assert tokenizer.inverse_vocab[encoded_cls[-1]] == '[SEP]'
    print("  CLS/SEP positions: OK")

    # Sentence pair
    pair_encoded = tokenizer.encode_pair("the fox", "the dog")
    pair_tokens = [tokenizer.inverse_vocab[i] for i in pair_encoded]
    print(f"  Sentence pair tokens: {pair_tokens[:8]}...")
    assert pair_tokens[0] == '[CLS]'
    sep_count = sum(1 for t in pair_tokens if t == '[SEP]')
    assert sep_count == 2, f"Expected 2 [SEP], got {sep_count}"
    print("  Sentence pair format: OK")

    # [MASK] token
    mask_id = tokenizer.vocab['[MASK]']
    print(f"  [MASK] token id: {mask_id}")
    print("  OK")

    # ============ BAI TAP 1: Train tren dataset lon hon ============
    print("\n" + "=" * 60)
    print("BAI TAP 1: Train tren dataset lon (Shakespeare)")
    print("=" * 60)

    shakespeare = download_shakespeare()

    if shakespeare:
        lines = [line.strip() for line in shakespeare.split('\n') if len(line.strip()) > 10]
        print(f"  Corpus: {len(lines)} lines, {len(shakespeare)} chars")

        # Train BPE voi vocab lon hon
        bpe_large = BPETokenizer(vocab_size=500)
        t0 = time.time()
        bpe_large.train(lines)
        t_train = time.time() - t0
        print(f"  Training time: {t_train:.2f}s")

        # Test
        test_sentences = [
            "to be or not to be that is the question",
            "all the world is a stage",
            "the lady doth protest too much",
        ]

        print(f"\n  Compression ratio:")
        for sent in test_sentences:
            encoded = bpe_large.encode(sent)
            decoded = bpe_large.decode(encoded)
            chars = len(sent)
            tokens = len(encoded)
            print(f"    '{sent}'")
            print(f"      {chars} chars -> {tokens} tokens (ratio: {chars / tokens:.1f}x)")
            assert decoded == sent, f"Decode failed: '{decoded}'"
        print("  All encode/decode: OK")
    else:
        print("  Skipped (download failed)")

    # ============ BAI TAP 2: WordPiece Tokenizer ============
    print("\n" + "=" * 60)
    print("BAI TAP 2: WordPiece Tokenizer (BERT style)")
    print("=" * 60)

    wp = WordPieceTokenizer(vocab_size=300)
    wp.train(corpus)

    test_text = "the quick fox"
    wp_encoded = wp.encode(test_text)
    wp_decoded = wp.decode(wp_encoded)
    print(f"\n  Original: '{test_text}'")
    print(f"  Encoded:  {wp_encoded}")
    print(f"  Decoded:  '{wp_decoded}'")

    # Show subword tokenization
    test_words = ["quickly", "foxes", "jumping", "animals"]
    print(f"\n  Subword examples:")
    for word in test_words:
        ids = wp.encode(word)
        tokens = [wp.inverse_vocab[i] for i in ids]
        print(f"    '{word}' -> {tokens}")

    # ============ BAI TAP 3: Compare voi HuggingFace ============
    print("\n" + "=" * 60)
    print("BAI TAP 3: Compare voi HuggingFace tokenizers")
    print("=" * 60)

    try:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE as HFBPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Train HuggingFace BPE
        hf_tokenizer = HFTokenizer(HFBPE(unk_token="<unk>"))
        hf_tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=300,
            special_tokens=["<unk>", "<pad>", "</w>"]
        )

        t0 = time.time()
        hf_tokenizer.train_from_iterator(corpus, trainer)
        t_hf = time.time() - t0

        # Train ours
        ours = BPETokenizer(vocab_size=300)
        t0 = time.time()
        ours.train(corpus, verbose=False)
        t_ours = time.time() - t0

        print(f"\n  {'Metric':<25} {'Ours':<15} {'HuggingFace'}")
        print("  " + "-" * 55)
        print(f"  {'Vocab size':<25} {len(ours.vocab):<15} {hf_tokenizer.get_vocab_size()}")
        print(f"  {'Training time':<25} {t_ours:<15.4f} {t_hf:.4f}")

        # Compare encoding lengths
        test_texts = [
            "the quick brown fox jumps over the lazy dog",
            "foxes and dogs are quick animals",
        ]
        print(f"\n  Token count comparison:")
        for text in test_texts:
            our_len = len(ours.encode(text))
            hf_len = len(hf_tokenizer.encode(text).ids)
            print(f"    '{text[:40]}...'")
            print(f"      Ours: {our_len} tokens, HF: {hf_len} tokens")

        print(f"\n  -> HuggingFace nhanh hon vi implement bang Rust")
        print(f"  -> Nhung thuat toan giong nhau (BPE)")

    except ImportError:
        print("  tokenizers chua cai. Chay: pip install tokenizers")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 5-7 (Bai 05):
# [x] Build BPE tokenizer tu scratch
#     -> BPETokenizer class: train() hoc vocab, encode()/decode() chuyen doi text <-> ids
#        BPE = Byte Pair Encoding: bat dau tu tung ky tu, merge cap xuat hien nhieu nhat
#        VD: "low low low" -> ["l","o","w"] -> merge "l"+"o" = "lo" -> ["lo","w"] -> ...
#        GPT-2 dung BPE voi vocab_size=50257
