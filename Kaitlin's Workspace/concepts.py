import spacy
from collections import defaultdict
from spacy.cli import download
download("en_core_web_sm")

class ConceptExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        # PoS tags
        self.target_pos = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'}

    def extract_concepts(self, caption):
        """Extract parts of speech (PoS) from caption."""
        doc = self.nlp(caption.lower())

        concepts = defaultdict(list)

        for token in doc:
            # Skip stopwords and very short words.
            if token.is_stop or len(token.text) < 3:
                continue

            # Get lemmatized form (base form).
            lemma = token.lemma_

            if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                concepts['nouns'].append(lemma)
            elif token.pos_ == 'VERB':
                # Avoid helping verbs.
                if lemma not in ['be', 'have', 'do', 'will', 'would', 'could', 'should']:
                    concepts['verbs'].append(lemma)
            elif token.pos_ == 'ADJ':
                concepts['adjectives'].append(lemma)
            elif token.pos_ == 'ADV':
                concepts['adverbs'].append(lemma)

        # Extract noun chunks for compound concepts.
        concepts['noun_chunks'] = [
            chunk.text.lower() for chunk in doc.noun_chunks
            if len(chunk.text.split()) <= 3  # Max 3 words (for now?)
        ]

        # Remove duplicates.
        for key in concepts:
            concepts[key] = list(dict.fromkeys(concepts[key]))

        return dict(concepts)