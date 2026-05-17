import numpy as np
from typing import Dict, List, Optional, Tuple
import hashlib

class Atlas:
    """
    The Reffelt Atlas: A semantic dictionary that maps words to 'Nonces' (integers).
    Words with similar meanings/definitions are assigned nonces that are numerically close.
    This allows for O(1) similarity checks using simple integer subtraction.
    """
    def __init__(self):
        self.word_to_nonce: Dict[str, int] = {}
        self.nonce_to_word: Dict[int, str] = {}
        self.categories: Dict[str, Tuple[int, int]] = {}
        self._next_category_start = 1000
        self._spacing = 10000 # Space between categories

    def register_category(self, name: str):
        """Reserves a range of nonces for a semantic category."""
        if name in self.categories:
             return self.categories[name][0]
        start = self._next_category_start
        end = start + self._spacing
        self.categories[name] = (start, end)
        self._next_category_start += self._spacing
        return start

    def add_word(self, word: str, category: str, sub_index: int = 0):
        """
        Registers a word.
        'category': High-level semantic group (e.g., 'noun.animal').
        'sub_index': Fine-grained position (e.g., 'canine' vs 'feline').
        """
        if category not in self.categories:
            self.register_category(category)
        
        base, _ = self.categories[category]
        # Generate nonce: Base Category + Sub Index + Hash(word)%100 (for collision avoidance)
        # In a real system, sub_index would be a float embedding mapped to int.
        # Here we simulate it.
        
        # Simple clustering:
        nonce = base + sub_index
        
        # Ensure uniqueness
        while nonce in self.nonce_to_word:
            nonce += 1
            
        self.word_to_nonce[word] = nonce
        self.nonce_to_word[nonce] = word
        
    def get_nonce(self, word: str) -> Optional[int]:
        return self.word_to_nonce.get(word)
        
    def get_word(self, nonce: int) -> Optional[str]:
        return self.nonce_to_word.get(nonce)

    def is_similar(self, word_a: str, word_b: str, threshold: int = 2000) -> bool:
        """Checks if two words are semantically related via nonce distance."""
        n1 = self.get_nonce(word_a)
        n2 = self.get_nonce(word_b)
        if n1 is None or n2 is None:
            return False
        return abs(n1 - n2) < threshold

    def get_similar_words(self, seed_word: str, threshold: int = 2000) -> List[str]:
        """Finds all words in the Atlas 'close' to the seed word."""
        seed_nonce = self.get_nonce(seed_word)
        if seed_nonce is None:
            return []
            
        # This is O(N) linear scan.
        # Real implementation would use a sorted list or B-Tree for O(log N) range query.
        matches = []
        for nonce, word in self.nonce_to_word.items():
            if abs(nonce - seed_nonce) < threshold:
                matches.append(word)
        return matches

    def populate_basic_english(self):
        """Populates the Atlas with a small, testable subset of English."""
        # Animals (Start 1000)
        self.add_word("animal", "noun.animal", 0)
        self.add_word("dog", "noun.animal", 100)
        self.add_word("puppy", "noun.animal", 105)
        self.add_word("wolf", "noun.animal", 120)
        self.add_word("cat", "noun.animal", 500)
        self.add_word("kitten", "noun.animal", 505)
        self.add_word("lion", "noun.animal", 550)
        self.add_word("tiger", "noun.animal", 560)
        self.add_word("fish", "noun.animal", 1000)
        self.add_word("shark", "noun.animal", 1050)
        
        # Technology (Start 11000)
        self.add_word("technology", "noun.tech", 0)
        self.add_word("computer", "noun.tech", 100)
        self.add_word("laptop", "noun.tech", 105)
        self.add_word("phone", "noun.tech", 200)
        self.add_word("tablet", "noun.tech", 205)
        self.add_word("internet", "noun.tech", 500)
        self.add_word("wifi", "noun.tech", 505)
        self.add_word("software", "noun.tech", 800)
        self.add_word("app", "noun.tech", 805)
        
        # Nature (Start 21000)
        self.add_word("nature", "noun.nature", 0)
        self.add_word("tree", "noun.nature", 100)
        self.add_word("forest", "noun.nature", 150)
        self.add_word("flower", "noun.nature", 300)
        self.add_word("rose", "noun.nature", 305)
        self.add_word("sun", "noun.nature", 800)
        self.add_word("moon", "noun.nature", 850)
        
        # Actions (Start 31000)
        self.add_word("run", "verb.motion", 100)
        self.add_word("walk", "verb.motion", 120)
        self.add_word("jump", "verb.motion", 150)
        self.add_word("eat", "verb.consumption", 500)
        self.add_word("drink", "verb.consumption", 550)

    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump({
                "words": self.word_to_nonce,
                "categories": self.categories
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Atlas':
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        atlas = cls()
        atlas.word_to_nonce = data["words"]
        atlas.categories = data["categories"]
        atlas.nonce_to_word = {v: k for k, v in atlas.word_to_nonce.items()}
        return atlas
