from collections import Counter, defaultdict


class BoardGenerator:
    def __init__(self, arasaac_matcher):
        self.matcher = arasaac_matcher

        # Core vocabulary that should ALWAYS be included
        self.core_vocabulary = [
            'i', 'you', 'me', 'want', 'like', 'need', 'help',
            'yes', 'no', 'more', 'stop', 'go', 'come',
            'good', 'bad', 'happy', 'sad'
        ]

    def rank_symbols(self, concept_to_symbols, caption):
        """Rank symbols by relevance."""

        #  Current scoring factors:
        #  1. Frequency in caption
        #  2. Part of speech priority
        #  3. Penalty for generic


        symbol_scores = Counter()
        symbol_to_concepts = defaultdict(list)

        # PoS priority weights - may need to tweak this. Start testing with default.
        pos_weights = {
            'nouns': 1.0,
            'verbs': 0.9,
            'adjectives': 0.7,
            'adverbs': 0.6
        }

        for concept, symbols in concept_to_symbols.items():
            # Determine PoS.
            pos_category = 'nouns'
            weight = pos_weights.get(pos_category, 0.5)

            # Count concept frequency in caption.
            freq = caption.lower().count(concept)

            for symbol_id in symbols:
                symbol_scores[symbol_id] += weight * (1 + freq)
                symbol_to_concepts[symbol_id].append(concept)

        # # Core vocabulary boost - Leaning toward fixed core vocabulary and so this likely not needed anymore.
        # for symbol_id in symbol_scores:
        #     symbol = self.matcher.get_symbol_metadata(symbol_id)
        #     if symbol:
        #         keywords = symbol.get('keywords', [])
        #         if any(kw.lower() in self.core_vocabulary for kw in keywords):
        #             symbol_scores[symbol_id] *= 2.0

        # Penalize symbols that match too many concepts (may be too generic)
        for symbol_id, concepts in symbol_to_concepts.items():
            if len(concepts) > 5:
                symbol_scores[symbol_id] *= 0.5

        return symbol_scores.most_common()

    def generate_board(self, image_path, max_symbols=24):
        """Returns list of symbol metadata for board."""
        # Step 1: Caption
        caption = self.captioner.caption_image(image_path)

        # Step 2: Extract concepts
        concepts = self.extractor.extract_concepts(caption)

        # Step 3: Map to synsets
        concept_to_synsets = self.synset_mapper.get_all_synsets_for_concepts(concepts)

        # Step 4: Match to ARASAAC
        concept_to_symbols = self.matcher.match_synsets_to_symbols(concept_to_synsets)

        # Step 5: Rank and select
        ranked_symbols = self.rank_symbols(concept_to_symbols, caption)

        # Step 6: Select top symbols + ensure core vocabulary
        board_symbols = []

        # Add core vocabulary first.
        for core_word in self.core_vocabulary:  # ALL core words
            matches = self.matcher.keyword_to_symbols.get(core_word, [])
            if matches:
                board_symbols.append(matches[0])

        # Add top-ranked scene-specific symbols.
        for symbol_id, score in ranked_symbols:
            if symbol_id not in board_symbols:
                board_symbols.append(symbol_id)
            if len(board_symbols) >= max_symbols:
                break

        # Get full metadata.
        board = [self.matcher.get_symbol_metadata(sid) for sid in board_symbols]

        return {
            'caption': caption,
            'concepts': concepts,
            'board': board
        }