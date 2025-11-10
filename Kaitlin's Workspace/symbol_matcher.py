import json
from collections import defaultdict


class ArasaacMatcher:
    def __init__(self, arasaac_data_path):

        with open(arasaac_data_path, 'r', encoding='utf-8') as f: # Do NOT forget encoding. It makes Windows so sad.
            self.symbols = json.load(f)

        # Build indices.
        self.synset_to_symbols = self._build_synset_index()
        self.keyword_to_symbols = self._build_keyword_index()

    def _build_synset_index(self):
        """Create reverse index: synset --> list of symbol IDs."""
        index = defaultdict(list)

        for symbol in self.symbols:
            synsets = symbol.get('synsets', [])
            symbol_id = symbol['_id']

            for synset in synsets:
                # Store both with and without POS suffix for flexibility.
                index[synset].append(symbol_id) # With PoS

                base_synset = synset.split('-')[0]
                index[base_synset].append(symbol_id) # Without PoS

        return dict(index)

    def _build_keyword_index(self):
        """Create reverse index: keyword --> list of symbol IDs."""
        index = defaultdict(list)

        for symbol in self.symbols:
            symbol_id = symbol['_id']
            keywords = symbol.get('keywords', [])

            # Extract keyword strings from keyword objects.
            for kw_obj in keywords:
                keyword = kw_obj.get('keyword', '').lower()
                if keyword:
                    index[keyword].append(symbol_id)

                    # Also index plural form if available.
                    plural = kw_obj.get('plural', '').lower()
                    if plural:
                        index[plural].append(symbol_id)

        return dict(index)

    def match_synsets_to_symbols(self, concept_to_synsets):
        """Match concepts to ARASAAC symbols via synsets."""
        concept_to_symbols = {}

        for concept, synsets in concept_to_synsets.items():
            matched_symbols = set()

            for synset in synsets:
                # Convert WordNet synset name to offset format
                # e.g., 'playground.n.01' → need to look up offset
                # For now, try direct match and keyword fallback

                # Try synset lookup (if synsets were stored as names)
                symbols = self.synset_to_symbols.get(synset, [])
                matched_symbols.update(symbols)

            # Fallback: try keyword matching if no synset match
            if not matched_symbols:
                keyword_matches = self.keyword_to_symbols.get(concept.lower(), [])
                matched_symbols.update(keyword_matches)

            if matched_symbols:
                concept_to_symbols[concept] = list(matched_symbols)

        return concept_to_symbols

    def match_synset_offsets_to_symbols(self, synset_offsets):
        """Direct matching using WordNet offset IDs."""
        matched_symbols = set()

        for offset in synset_offsets:
            symbols = self.synset_to_symbols.get(offset, [])
            matched_symbols.update(symbols)

        return list(matched_symbols)

    def get_symbol_metadata(self, symbol_id):
        """Get full metadata for a symbol."""
        for symbol in self.symbols:
            if symbol['_id'] == symbol_id:
                # Add image URL.
                symbol['image_url'] = f"https://static.arasaac.org/pictograms/{symbol_id}/{symbol_id}_500.png"

                # Extract clean keyword list.
                symbol['keyword_list'] = [
                    kw['keyword'] for kw in symbol.get('keywords', [])
                ]

                return symbol
        return None

    def search_by_keyword(self, keyword):
        """Search for symbols by keyword. Return list of symbol metadata."""
        keyword = keyword.lower()
        symbol_ids = self.keyword_to_symbols.get(keyword, [])
        return [self.get_symbol_metadata(sid) for sid in symbol_ids]