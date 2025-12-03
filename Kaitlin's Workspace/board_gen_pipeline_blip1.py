import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import sys
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

start_time = time.time()

## Configuration ##

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages.

current_directory = Path.cwd()
parent_directory = current_directory.parent

# Test Paths
ARASAAC_DATA_PATH = './data/arasaac_pictograms_complete_20251106_130529.json'
TEST_IMAGE_PATH = './test_images/cleaned_test_images/airport/clean_airport_002.jpg'

# Model
BLIP_MODEL_NAME = 'Salesforce/blip-image-captioning-base'

## Classes ##

class ArasaacMatcher:
    '''Handles all ARASAAC symbol matching.'''
    
    def __init__(self, data_path):
        print(f'Loading ARASAAC data from {data_path}...')
        with open(data_path, 'r', encoding='utf-8') as f:
            self.symbols = json.load(f)
        print(f'Loaded {len(self.symbols)} symbols\n')
        
        self.synset_to_symbols = self.build_synset_index()
        self.keyword_to_symbols = self.build_keyword_index()
    
    def build_synset_index(self):
        index = defaultdict(list)
        for symbol in self.symbols:
            for synset in symbol.get('synsets', []):
                index[synset].append(symbol['_id'])
        return dict(index)
    
    def build_keyword_index(self):
        index = defaultdict(list)
        for symbol in self.symbols:
            for kw_obj in symbol.get('keywords', []):
                keyword = kw_obj.get('keyword', '').lower()
                if keyword:
                    index[keyword].append(symbol['_id'])
                    plural = kw_obj.get('plural', '').lower()
                    if plural:
                        index[plural].append(symbol['_id'])
        return dict(index)
    
    def get_symbol_metadata(self, symbol_id):
        for symbol in self.symbols:
            if symbol['_id'] == symbol_id:
                symbol['image_url'] = f'https://static.arasaac.org/pictograms/{symbol_id}/{symbol_id}_500.png'
                symbol['keyword_list'] = [kw['keyword'] for kw in symbol.get('keywords', [])]
                return symbol
        return None

    def search_compound(self, compound):
        '''Search for compound noun symbols.'''

        # Try exact match first (with underscores or spaces)
        exact_matches = []

        for symbol in self.symbols:
            keywords = [kw.get('keyword', '').lower() for kw in symbol.get('keywords', [])]

            # Check if compound appears in keywords
            if compound in keywords or compound.replace(' ', '_') in keywords:
                exact_matches.append(symbol['_id'])

        if exact_matches:
            return exact_matches

        # Fallback: search for symbols containing both words
        words = compound.split()
        matching_symbols = defaultdict(int)

        for word in words:
            word_symbols = self.keyword_to_symbols.get(word, [])
            for sid in word_symbols:
                matching_symbols[sid] += 1

        # Return symbols that match multiple words
        return [sid for sid, count in matching_symbols.items() if count >= 2]

class SceneCaptioner:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME, use_fast=False)
        self.model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME,
            dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)

    def caption_image(self, image_path, prompt=None):
        '''Generate detailed caption for AAC board generation.'''

        image = Image.open(image_path).convert('RGB')

        # Prompts to refine output.
        ## Natural language interrogation.
        prompts = [
            'What do you see?:',
            # 'Question: What actions are happening in this image? Answer: ',
            # 'Question: What is the generic setting of this image? Answer: ?',
            # 'Question: What objects are visible? Answer:',
        ]

        captions = []

        for p in prompts:
            if p:
                inputs = self.processor(images=image, text=p, return_tensors='pt').to(
                    self.device, torch.float16 if self.device == 'cuda' else torch.float32
                )
            else:
                inputs = self.processor(images=image, return_tensors='pt').to(
                    self.device, torch.float16 if self.device == 'cuda' else torch.float32
                )

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                min_length=60,
                num_beams=5, # 2 - greedy fast; 5 - default ; 8 - thoughtful
                length_penalty=1.2,  # Encourage longer outputs
                no_repeat_ngram_size=3  # Avoid repetition
            )

            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if caption.startswith(p):
                caption = caption[len(p):].strip()

            print(f'Prompt: {p}')
            print(f'Caption: {caption}')

            captions.append(caption)

        return captions
    
    def get_unique_words(self, captions, core_vocab=None, min_length=3):
        ''' Extract set of unique words from all captions, excluding core vocabulary and small words.'''

        import re

        # Default to empty set if no core vocab provided.
        if core_vocab is None:
            core_vocab = set()
        else:
            core_vocab = set(word.lower() for word in core_vocab)

        unique_words = set()

        for caption in captions:
            # Clean and split caption.
            words = re.findall(r'\b[a-zA-Z]+\b', caption.lower())

            for word in words:
                # Filter out: core vocab, small words, common stopwords.
                if (word not in core_vocab and
                        len(word) >= min_length and
                        word not in {'the', 'and', 'this', 'that', 'with', 'from', 'are', 'was', 'were'}):
                    unique_words.add(word)

        return unique_words


class SceneClassifier:
    '''Classify the general scene/context from captions.'''

    def __init__(self):
        # Define scene keywords and their contexts
        self.scene_patterns = {
            'airport': {
                'keywords': ['airport', 'terminal', 'plane', 'flight', 'luggage', 'baggage', 'passenger'],
                'min_matches': 2
            },
            'playground': {
                'keywords': ['playground', 'slide', 'swing', 'children', 'play', 'park', 'monkey bars'],
                'min_matches': 2
            },
            'restaurant': {
                'keywords': ['restaurant', 'table', 'food', 'eating', 'menu', 'waiter', 'dining'],
                'min_matches': 2
            },
            'hospital': {
                'keywords': ['hospital', 'doctor', 'nurse', 'patient', 'medical', 'clinic', 'exam room'],
                'min_matches': 2
            },
            'school': {
                'keywords': ['classroom', 'school', 'teacher', 'student', 'desk', 'board', 'learning'],
                'min_matches': 2
            },
            'home': {
                'keywords': ['living room', 'kitchen', 'bedroom', 'bathroom', 'house', 'couch', 'bed'],
                'min_matches': 2
            },
            'store': {
                'keywords': ['store', 'shop', 'shopping', 'checkout', 'cart', 'aisle', 'cashier'],
                'min_matches': 2
            }
        }

    def classify_scene(self, captions, concepts):
        '''Determine the most likely scene context.'''

        # Combine all text for analysis.
        all_text = ' '.join(captions).lower()
        all_concepts = []
        for concept_list in concepts.values():
            all_concepts.extend([c.lower() for c in concept_list])

        scene_scores = {}

        for scene_type, scene_info in self.scene_patterns.items():
            score = 0
            matches = []

            for keyword in scene_info['keywords']:
                if keyword in all_text or keyword in all_concepts:
                    score += 1
                    matches.append(keyword)

            if score >= scene_info['min_matches']:
                scene_scores[scene_type] = {
                    'score': score,
                    'matches': matches
                }

        # Return the highest scoring scene
        if scene_scores:
            best_scene = max(scene_scores.items(), key=lambda x: x[1]['score'])
            return best_scene[0], best_scene[1]

        return None, None


class ContextVocabulary:
    '''Provides context-appropriate vocabulary for different scenes.'''

    def __init__(self):
        self.context_words = {
            'airport': {
                'core': [
                    # Navigation/Places
                    'gate', 'security', 'check-in', 'baggage claim', 'customs',
                    'departure', 'arrival', 'terminal', 'ticket counter',

                    # Actions
                    'board', 'fly', 'travel', 'wait', 'check luggage', 'show passport',

                    # Objects
                    'ticket', 'boarding pass', 'passport', 'suitcase', 'carry-on',
                    'plane', 'luggage cart',

                    # People
                    'pilot', 'flight attendant', 'passenger', 'security officer',

                    # States/Feelings
                    'delayed', 'on-time', 'early', 'late', 'lost', 'found',
                    'nervous', 'excited',

                    # Common needs
                    'bathroom', 'food', 'water', 'help', 'where', 'when'
                ],
                'priority': 'high'  # Always include these
            },

            'playground': {
                'core': [
                    # Equipment
                    'slide', 'swing', 'monkey bars', 'seesaw', 'sandbox', 'climbing frame',

                    # Actions
                    'play', 'run', 'climb', 'swing', 'slide', 'push', 'jump',

                    # Social
                    'friend', 'share', 'take turns', 'wait',

                    # Safety
                    'careful', 'hurt', 'help', 'stop',

                    # Feelings
                    'fun', 'happy', 'tired', 'thirsty', 'hungry'
                ],
                'priority': 'high'
            },

            'restaurant': {
                'core': [
                    # Places
                    'table', 'menu', 'bathroom', 'exit',

                    # People
                    'waiter', 'waitress', 'server', 'chef',

                    # Actions
                    'order', 'eat', 'drink', 'pay', 'wait',

                    # Foods (common)
                    'water', 'milk', 'juice', 'pizza', 'hamburger', 'fries',
                    'salad', 'dessert',

                    # States
                    'hungry', 'thirsty', 'full', 'hot', 'cold',

                    # Needs
                    'more', 'done', 'help', 'napkin', 'fork', 'spoon'
                ],
                'priority': 'high'
            },

            'hospital': {
                'core': [
                    # Places
                    'waiting room', 'exam room', 'emergency room', 'reception',

                    # People
                    'doctor', 'nurse', 'patient',

                    # Actions
                    'hurt', 'pain', 'sick', 'help', 'wait', 'examine',

                    # Body parts
                    'head', 'arm', 'leg', 'stomach', 'throat',

                    # Needs
                    'medicine', 'bandage', 'water', 'bathroom',

                    # Feelings
                    'scared', 'nervous', 'better', 'worse'
                ],
                'priority': 'high'
            },

            'school': {
                'core': [
                    'teacher', 'student', 'classroom', 'desk', 'board',
                    'read', 'write', 'learn', 'listen', 'question',
                    'book', 'pencil', 'paper', 'computer',
                    'bathroom', 'lunch', 'recess', 'help'
                ],
                'priority': 'high'
            }
        }

    def get_context_vocabulary(self, scene_type):
        '''Get relevant vocabulary for a scene type.'''

        if scene_type in self.context_words:
            return self.context_words[scene_type]['core']

        return []

class ConceptExtractor:
    '''Extract concepts using spaCy NLP.'''
    
    def __init__(self):
        print('Loading spaCy model...')
        import spacy
        self.nlp = spacy.load('en_core_web_sm')
        print('✓ spaCy loaded\n')

    def extract_concepts(self, caption):
        doc = self.nlp(caption.lower())

        concepts = defaultdict(list)

        # Define common AAC-relevant compound patterns
        aac_compound_patterns = [
            # Travel/Airport
            'baggage claim', 'security checkpoint', 'boarding pass', 'gate area',
            'luggage cart', 'airport terminal', 'check-in counter', 'departure gate',

            # Medical
            'waiting room', 'exam room', 'doctor office', 'emergency room',

            # School
            'lunch room', 'class room', 'school bus', 'play ground',

            # Food
            'ice cream', 'peanut butter', 'hot dog', 'french fries',

            # General
            'living room', 'bed room', 'bath room', 'front door'
        ]

        # Extract compounds from noun chunks
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            words = chunk_text.split()

            # Remove leading articles
            if words and words[0] in ['the', 'a', 'an', 'some', 'many']:
                words = words[1:]

            if len(words) >= 2:
                compound = ' '.join(words)

                # Check if it matches known AAC patterns
                if compound in aac_compound_patterns:
                    concepts['compounds'].append(compound)
                # Or if it's a noun + noun pattern
                elif all(doc[i].pos_ in ['NOUN', 'PROPN'] for i in range(len(doc))
                         if doc[i].text in words):
                    concepts['compounds'].append(compound)

        # Extract individual tokens
        for token in doc:
            if token.is_stop or len(token.text) < 3:
                continue

            lemma = token.lemma_

            if token.pos_ in ['NOUN', 'PROPN']:
                concepts['nouns'].append(lemma)
            elif token.pos_ == 'VERB':
                if lemma not in ['be', 'have', 'do', 'will', 'would', 'could', 'should']:
                    concepts['verbs'].append(lemma)
            elif token.pos_ == 'ADJ':
                concepts['adjectives'].append(lemma)
            elif token.pos_ == 'ADV':
                concepts['adverbs'].append(lemma)

        # Remove duplicates
        for key in concepts:
            concepts[key] = list(dict.fromkeys(concepts[key]))

        return dict(concepts)


class SynsetMapper:
    '''Map concepts to WordNet synsets.'''
    
    def __init__(self):
        print('Loading WordNet...')
        from nltk.corpus import wordnet as wn
        self.wn = wn
        self.pos_map = {
            'nouns': wn.NOUN,
            'verbs': wn.VERB,
            'adjectives': wn.ADJ,
            'adverbs': wn.ADV
        }
        print('✓ WordNet loaded\n')
    
    def get_synsets(self, concept, pos_tag):
        wn_pos = self.pos_map.get(pos_tag, self.wn.NOUN)
        synsets = self.wn.synsets(concept, pos=wn_pos)
        
        result = {'names': [], 'offsets': []}
        
        for syn in synsets:
            result['names'].append(syn.name())
            offset = str(syn.offset()).zfill(8)
            pos_suffix = syn.pos()
            result['offsets'].append(f'{offset}-{pos_suffix}')
        
        return result

    def get_all_synsets_for_concepts(self, concepts_dict):
        concept_to_synsets = {}

        # Handle compounds FIRST
        for compound in concepts_dict.get('compounds', []):
            # Try the compound as-is
            synsets = self.get_synsets(compound.replace(' ', '_'), 'nouns')

            if synsets['offsets']:
                concept_to_synsets[compound] = synsets
            else:
                # Fallback: try each word separately but mark as compound
                words = compound.split()
                combined_synsets = {'names': [], 'offsets': []}
                for word in words:
                    word_synsets = self.get_synsets(word, 'nouns')
                    combined_synsets['names'].extend(word_synsets['names'])
                    combined_synsets['offsets'].extend(word_synsets['offsets'])

                if combined_synsets['offsets']:
                    concept_to_synsets[compound] = combined_synsets

        # Then handle regular concepts
        for pos_category, concept_list in concepts_dict.items():
            if pos_category == 'compounds':  # Skip, already handled
                continue

            for concept in concept_list:
                synsets = self.get_synsets(concept, pos_category)
                if synsets['offsets']:
                    concept_to_synsets[concept] = synsets

        return concept_to_synsets


class BoardGenerator:
    '''Complete AAC board generation pipeline.'''

    def __init__(self, matcher, captioner, extractor, synset_mapper):
        self.matcher = matcher
        self.captioner = captioner
        self.extractor = extractor
        self.synset_mapper = synset_mapper

        self.core_vocabulary = [
            'i', 'you', 'me', 'they', 'want', 'like', 'need', 'help',
            'yes', 'no', 'more', 'stop', 'go', 'come',
            'good', 'bad', 'happy', 'sad'
        ]

    def get_best_symbol(self, symbol_ids, concept):
        '''Select the most appropriate symbol from multiple options.'''

        if not symbol_ids:
            return None

        if len(symbol_ids) == 1:
            return symbol_ids[0]

        best_id = None
        best_score = -1

        for sid in symbol_ids:
            metadata = self.matcher.get_symbol_metadata(sid)
            if not metadata:
                continue

            score = 0
            keywords = [kw.lower() for kw in metadata.get('keyword_list', [])]

            if concept.lower() in keywords:
                score += 10

            score -= len(keywords) * 0.1

            categories = metadata.get('categories', [])
            if 'basic' in categories or 'core' in categories:
                score += 5

            if score > best_score:
                best_score = score
                best_id = sid

        return best_id if best_id else symbol_ids[0]

    def match_concepts(self, concept_to_synsets):
        '''Match concepts to ARASAAC symbols via synsets.'''

        concept_to_symbols = {}

        for concept, synset_data in concept_to_synsets.items():
            matched_symbols = set()

            # Check if it's a compound (has spaces)
            if ' ' in concept:
                # Try compound-specific search
                compound_matches = self.matcher.search_compound(concept)
                matched_symbols.update(compound_matches)

            # Try synset matching.
            for offset in synset_data['offsets']:
                symbols = self.matcher.synset_to_symbols.get(offset, [])
                matched_symbols.update(symbols)

            # Fallback: keyword matching.
            if not matched_symbols:
                keyword_matches = self.matcher.keyword_to_symbols.get(concept.lower(), [])
                matched_symbols.update(keyword_matches)

            if matched_symbols:
                best_symbol = self.get_best_symbol(list(matched_symbols), concept)
                if best_symbol:
                    concept_to_symbols[concept] = [best_symbol]

        return concept_to_symbols

    def match_keywords(self, keywords):
        '''Direct keyword matching to ARASAAC symbols.'''

        keyword_to_symbols = {}

        for keyword in keywords:
            matched_symbols = self.matcher.keyword_to_symbols.get(keyword.lower(), [])

            if matched_symbols:
                best_symbol = self.get_best_symbol(matched_symbols, keyword)
                if best_symbol:
                    keyword_to_symbols[keyword] = [best_symbol]

        return keyword_to_symbols

    def generate_board(self, image_path, max_symbols=64):
        '''Complete pipeline: image → AAC board.'''

        print(f'Generating AAC Board for: {image_path}')

        # Caption image
        print('\nGenerating caption...')
        captions = self.captioner.caption_image(image_path)
        combined_caption = ' '.join(captions)
        print(f'Captions generated: {len(captions)} captions')

        # Extract unique keywords
        print('\nExtracting keywords...')
        keywords = self.captioner.get_unique_words(captions, self.core_vocabulary, min_length=3)
        print(f'  Found {len(keywords)} unique keywords: {list(keywords)[:10]}...')

        # Extract concepts from all captions
        print('\nExtracting concepts...')
        all_concepts = {'nouns': [], 'verbs': [], 'adjectives': [], 'adverbs': []}
        for caption in captions:
            concepts = self.extractor.extract_concepts(caption)
            for key in all_concepts:
                all_concepts[key].extend(concepts.get(key, []))

        # Remove duplicates
        for key in all_concepts:
            all_concepts[key] = list(dict.fromkeys(all_concepts[key]))

        print(f'  Nouns: {all_concepts.get("nouns", [])[:5]}')
        print(f'  Verbs: {all_concepts.get("verbs", [])[:5]}')
        print(f'  Adjectives: {all_concepts.get("adjectives", [])[:5]}')

        # Map to synsets
        print('\nMapping to WordNet synsets...')
        concept_to_synsets = self.synset_mapper.get_all_synsets_for_concepts(all_concepts)
        print(f'  Mapped {len(concept_to_synsets)} concepts to synsets')

        # Match to ARASAAC symbols
        print('\nMatching to ARASAAC symbols...')
        concept_to_symbols = self.match_concepts(concept_to_synsets)
        print(f'  Matched {len(concept_to_symbols)} concepts to symbols via synsets')

        # Add direct keyword matches
        print('\nAdding direct keyword matches...')
        keyword_matches = self.match_keywords(keywords)
        print(f'  Matched {len(keyword_matches)} additional keywords directly')

        # Merge both matching strategies
        all_matches = {**concept_to_symbols, **keyword_matches}

        # Build board (with context awareness)
        print('\nBuilding board...')
        board = self.build_board(
            all_matches,
            all_concepts,
            context_words=[],  # No context words yet - we'll add this later
            caption=combined_caption,
            max_symbols=max_symbols
        )
        print(f'  Final board: {len(board)} symbols')

        return {
            'captions': captions,
            'combined_caption': combined_caption,
            'concepts': all_concepts,
            'keywords': list(keywords),
            'board': board
        }

    def build_board(self, concept_to_symbols, image_concepts, context_words=None, caption='', max_symbols=64):
        '''Rank and select symbols with optional context awareness.

        Args:
            concept_to_symbols: Dict mapping concepts to symbol IDs
            image_concepts: Dict of concepts by POS from image
            context_words: Optional list of context-relevant words
            caption: Combined caption text for frequency analysis
            max_symbols: Maximum number of symbols to return
        '''

        if context_words is None:
            context_words = []

        # Map concepts to POS
        concept_to_pos = {}
        for pos_category, concept_list in image_concepts.items():
            for concept in concept_list:
                concept_to_pos[concept] = pos_category

        symbol_scores = Counter()
        symbol_to_concepts = {}

        # POS weights
        pos_weights = {
            'nouns': 1.0,
            'verbs': 0.9,
            'adjectives': 0.7,
            'adverbs': 0.6
        }

        for concept, symbol_ids in concept_to_symbols.items():
            # Check if concept is from image or context.
            in_image = concept in [c for concepts in image_concepts.values() for c in concepts]
            in_context = concept.lower() in [w.lower() for w in context_words]

            # Base score
            freq = caption.lower().count(concept.lower())
            pos_category = concept_to_pos.get(concept, 'nouns')
            weight = pos_weights.get(pos_category, 0.5)

            # Boost image concepts (things actually visible).
            if in_image:
                weight *= 1.5  # 50% boost for visible items

            # Boost context words (situationally relevant).
            elif in_context:
                weight *= 1.2  # 20% boost for context vocabulary

            for symbol_id in symbol_ids:
                symbol_scores[symbol_id] += weight + (freq * 0.5)

                # Track which concepts this symbol represents.
                if symbol_id not in symbol_to_concepts:
                    symbol_to_concepts[symbol_id] = []
                symbol_to_concepts[symbol_id].append(concept)

        # Build board with deduplication.
        board_symbol_ids = []
        represented_concepts = set()

        for symbol_id, score in symbol_scores.most_common():
            symbol_concepts = symbol_to_concepts.get(symbol_id, [])

            # Skip if concept already represented.
            if any(concept in represented_concepts for concept in symbol_concepts):
                continue

            board_symbol_ids.append(symbol_id)
            represented_concepts.update(symbol_concepts)

            if len(board_symbol_ids) >= max_symbols:
                break

        return [self.matcher.get_symbol_metadata(sid) for sid in board_symbol_ids]

    def display_board(self, result):
        '''Pretty print the board.'''

        print('\n' + '=' * 80)
        print('FINAL AAC BOARD')
        print('=' * 80)
        print(f'\nScenario: \'{result["combined_caption"]}\'\n')
        print(f'Keywords extracted: {len(result.get("keywords", []))}')
        if result.get('keywords'):
            print(f'  {", ".join(list(result["keywords"])[:15])}...\n')
        print(f'Board contains {len(result["board"])} symbols:\n')

        for i, symbol in enumerate(result['board'], 1):
            keywords = ', '.join(symbol['keyword_list'][:3])
            print(f'{i:2d}. {keywords:40s}')
            print(f'    {symbol["image_url"]}\n')


def main():
    '''Run the complete Image-to-Board workflow.'''
    
    # Confirm symbol data file exists.
    if not Path(ARASAAC_DATA_PATH).exists():
        print(f'Error: Data file not found at {ARASAAC_DATA_PATH}')
        print('Please update ARASAAC_DATA_PATH in this script.')
        sys.exit(1)
    
    try:
        # Initialize components.
        print('Initializing...\n')
        matcher = ArasaacMatcher(ARASAAC_DATA_PATH)
        captioner = SceneCaptioner()
        extractor = ConceptExtractor()
        synset_mapper = SynsetMapper()
        generator = BoardGenerator(matcher, captioner, extractor, synset_mapper)
        

        print('Ready to begin.')

        
        # Option 1: Generate from image.
        if Path(TEST_IMAGE_PATH).exists(): # Update to user input selection after testing.
            result = generator.generate_board(TEST_IMAGE_PATH)
            generator.display_board(result)
        
        # Option 2: Generate from text.
        else:

            while True:
                text = input('\nEnter a scenario description: ').strip()
                
                # Skip image captioning, and use text directly.
                print(f'\nProcessing: \'{text}\'')
                concepts = extractor.extract_concepts(text)
                concept_to_synsets = synset_mapper.get_all_synsets_for_concepts(concepts)
                concept_to_symbols = generator.match_concepts(concept_to_synsets)
                board = generator.build_board(concept_to_symbols, concepts, text, 64)
                
                result = {'caption': [text], 'combined_caption': text, 'concepts': concepts, 'board': board}
                generator.display_board(result)

        # Option 3: Generate storyboard.
    
    except KeyboardInterrupt:
        print('\n\nStopped by user.')
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Script took {elapsed_time:.2f} seconds")
    print(f"Script took {elapsed_time / 60:.2f} minutes")


if __name__ == '__main__':
    main()
