from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet'), nltk.download('omw-1.4') # Download once.

class SynsetMapper:
    def __init__(self):
        # PoS tag mapping from spaCy to WordNet
        self.pos_map = {
            'nouns': wn.NOUN,
            'verbs': wn.VERB,
            'adjectives': wn.ADJ,
            'adverbs': wn.ADV
        }

    def get_synsets(self, concept, pos_tag):
        """Get all WordNet synsets for a concept."""
        wn_pos = self.pos_map.get(pos_tag, wn.NOUN)
        synsets = wn.synsets(concept, pos=wn_pos)

        # Return synset names (ID format)
        return [syn.name() for syn in synsets]

    def get_all_synsets_for_concepts(self, concepts_dict):
        """Map all concepts to synsets."""
        concept_to_synsets = {}

        for pos_category, concept_list in concepts_dict.items():
            if pos_category == 'noun_chunks': # Handle compound concepts separately.
                continue

            for concept in concept_list:
                synsets = self.get_synsets(concept, pos_category)
                if synsets:
                    concept_to_synsets[concept] = synsets

        return concept_to_synsets

    def expand_with_hypernyms(self, synset_ids, depth=1):
        """Expand synsets with hypernyms."""
        expanded = set(synset_ids)

        for synset_id in synset_ids:
            synset = wn.synset(synset_id)

            # Get hypernyms up to depth.
            for _ in range(depth):
                hypernyms = synset.hypernyms()
                if not hypernyms:
                    break
                synset = hypernyms[0]  # Take most common parent
                expanded.add(synset.name())

        return list(expanded)