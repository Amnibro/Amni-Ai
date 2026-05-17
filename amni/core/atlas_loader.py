import json
from amni.core.atlas import Atlas

class AtlasLoader:
    @staticmethod
    def load_common_english(atlas: Atlas):
        """
        Populates the Atlas with ~1000 common English words grouped by approximate category.
        In a real scenario, this would load from WordNet or a frequency list.
        """
        # Define some broad categories and words
        categories = {
            "noun.animal": [
                "dog", "cat", "fish", "bird", "lion", "tiger", "bear", "wolf", "fox", "rabbit",
                "mouse", "rat", "horse", "cow", "pig", "sheep", "goat", "chicken", "duck",
                "eagle", "hawk", "shark", "whale", "dolphin", "snake", "frog", "lizard",
                "ant", "bee", "spider", "butterfly", "worm", "monkey", "elephant", "zebra"
            ],
            "noun.food": [
                "apple", "banana", "orange", "grape", "fruit", "vegetable", "carrot", "potato",
                "tomato", "bread", "cheese", "meat", "chicken", "beef", "pork", "fish",
                "rice", "pasta", "pizza", "burger", "sandwich", "soup", "salad", "cake",
                "cookie", "chocolate", "candy", "water", "juice", "milk", "coffee", "tea"
            ],
            "noun.tech": [
                "computer", "laptop", "phone", "tablet", "screen", "keyboard", "mouse",
                "internet", "wifi", "web", "site", "app", "software", "program", "code",
                "data", "file", "folder", "drive", "disk", "memory", "processor", "chip",
                "battery", "camera", "video", "audio", "music", "game", "robot", "machine"
            ],
            "noun.body": [
                "head", "hair", "face", "eye", "ear", "nose", "mouth", "tooth", "tongue",
                "neck", "arm", "hand", "finger", "thumb", "leg", "foot", "toe", "skin",
                "bone", "blood", "heart", "brain", "stomach", "muscle", "back", "body"
            ],
            "noun.nature": [
                "tree", "flower", "grass", "plant", "leaf", "root", "forest", "wood",
                "mountain", "hill", "river", "lake", "ocean", "sea", "beach", "sand",
                "sky", "cloud", "rain", "snow", "wind", "storm", "sun", "moon", "star",
                "space", "earth", "world", "fire", "water", "ice", "rock", "stone"
            ],
            "verb.motion": [
                "run", "walk", "jump", "fly", "swim", "crawl", "climb", "fall", "roll",
                "turn", "spin", "move", "go", "come", "leave", "stay", "stop", "start"
            ],
            "verb.perception": [
                "see", "hear", "smell", "taste", "touch", "feel", "look", "listen", "watch"
            ],
            "adj.color": [
                "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown",
                "black", "white", "gray", "gold", "silver", "light", "dark"
            ]
        }

        count = 0
        for cat, words in categories.items():
            # Spread sub-indices
            for i, word in enumerate(words):
                # Sub-index spacing to allow insertions
                sub_index = i * 10
                atlas.add_word(word, cat, sub_index)
                count += 1
                
        print(f"Loaded {count} words into Atlas.")
