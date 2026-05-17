from typing import List
from amni.core.atlas import Atlas

class ContextMatcher:
    def __init__(self, atlas: Atlas):
        self.atlas = atlas

    def encode(self, prompt: str) -> List[int]:
        """
        Converts a text prompt into a list of active semantic nonces.
        """
        # Remove punctuation?
        import re
        clean_prompt = re.sub(r'[^\w\s]', '', prompt.lower())
        words = clean_prompt.split()
        
        nonces = []
        for w in words:
            # Simple direct lookup
            n = self.atlas.get_nonce(w)
            if n is not None:
                nonces.append(n)
            
        return list(set(nonces))
