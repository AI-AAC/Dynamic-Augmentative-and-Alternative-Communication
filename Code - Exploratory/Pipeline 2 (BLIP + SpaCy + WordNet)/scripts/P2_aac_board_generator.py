'''
Dynamic AAC Board Generator
Combines computer vision and NLP to generate context-appropriate AAC boards.

Authors: Kaitlin Moore & Matthew Yurkunas
Course: 95-891 Introduction to Artificial Intelligence
Date: December 2025

Usage:
    streamlit run aac_board_generator.py
'''

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from gtts import gTTS
from nltk.corpus import wordnet
from PIL import Image
import spacy
import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration



# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages.

BLIP_MODEL_NAME = 'Salesforce/blip-image-captioning-base'

# Update this path to your ARASAAC data file location.
current_directory = Path.cwd()
parent_directory = current_directory.parent
ARASAAC_DATA_PATH = f'{parent_directory}/data/arasaac_pictograms_complete_20251106_130529.json'


# ARASAAC Symbol Matching
class ArasaacMatcher:
    '''Handles all ARASAAC symbol matching operations.'''

    def __init__(self, data_path):
        '''Load ARASAAC data and build indexes.'''

        print(f'Loading ARASAAC data from {data_path}...')
        with open(data_path, 'r', encoding='utf-8') as f:
            self.symbols = json.load(f)
        print(f'Loaded {len(self.symbols)} symbols\n')

        self.synset_to_symbols = self._build_synset_index()
        self.keyword_to_symbols = self._build_keyword_index()

    def _build_synset_index(self):
        '''Build index mapping synset IDs to symbol IDs.'''

        index = defaultdict(list)
        for symbol in self.symbols:
            for synset in symbol.get('synsets', []):
                index[synset].append(symbol['_id'])
        return dict(index)

    def _build_keyword_index(self):
        '''Build index mapping keywords to symbol IDs.'''

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
        '''Get full metadata for a symbol by ID.'''

        for symbol in self.symbols:
            if symbol['_id'] == symbol_id:
                symbol['image_url'] = (
                    f'https://static.arasaac.org/pictograms/'
                    f'{symbol_id}/{symbol_id}_500.png'
                )
                symbol['keyword_list'] = [
                    kw['keyword'] for kw in symbol.get('keywords', [])
                ]
                return symbol
        return None

    def search_compound(self, compound):
        '''Search for compound noun symbols.'''

        # Try exact match first (with underscores or spaces).
        exact_matches = []

        for symbol in self.symbols:
            keywords = [
                kw.get('keyword', '').lower() for kw in symbol.get('keywords', [])
            ]

            # Check if compound appears in keywords.
            if compound in keywords or compound.replace(' ', '_') in keywords:
                exact_matches.append(symbol['_id'])

        if exact_matches:
            return exact_matches

        # Fallback: search for symbols containing both words.
        words = compound.split()
        matching_symbols = defaultdict(int)

        for word in words:
            word_symbols = self.keyword_to_symbols.get(word, [])
            for sid in word_symbols:
                matching_symbols[sid] += 1

        # Return symbols that match multiple words.
        return [sid for sid, count in matching_symbols.items() if count >= 2]


# Image Captioning
class SceneCaptioner:
    '''Generate captions from images using BLIP.'''

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        '''Initialize BLIP model.'''

        print('Loading BLIP model...')
        self.device = device
        self.processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME, use_fast=False)
        self.model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)
        print('BLIP loaded\n')

    def caption_image(self, image_path, prompt=None):
        '''Generate detailed caption for AAC board generation.'''

        image = Image.open(image_path).convert('RGB')

        # Prompts to refine output.
        prompts = [
            'What do you see?:',
        ]

        captions = []

        for p in prompts:
            if p:
                inputs = self.processor(images=image, text=p, return_tensors='pt').to(
                    self.device,
                    torch.float16 if self.device == 'cuda' else torch.float32
                )
            else:
                inputs = self.processor(images=image, return_tensors='pt').to(
                    self.device,
                    torch.float16 if self.device == 'cuda' else torch.float32
                )

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                min_length=60,
                num_beams=5,
                length_penalty=1.2,
                no_repeat_ngram_size=3
            )

            caption = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            if caption.startswith(p):
                caption = caption[len(p):].strip()

            captions.append(caption)

        return captions


# Concept Extraction
class ConceptExtractor:
    '''Extract concepts using spaCy NLP.'''

    def __init__(self):
        '''Load spaCy model.'''

        print('Loading spaCy model...')
        self.nlp = spacy.load('en_core_web_sm')
        print('spaCy loaded\n')

    def extract_concepts(self, caption):
        '''Extract nouns, verbs, adjectives, adverbs, and compounds from text.'''

        doc = self.nlp(caption.lower())

        concepts = defaultdict(list)

        # Extract compounds from noun chunks.
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            words = chunk_text.split()

            # Remove leading articles.
            if words and words[0] in ['the', 'a', 'an', 'some', 'many']:
                words = words[1:]

            if len(words) >= 2:
                # Check if all tokens in the chunk are nouns.
                chunk_tokens = [token for token in chunk if token.text in words]
                if all(token.pos_ in ['NOUN', 'PROPN'] for token in chunk_tokens):
                    compound = ' '.join(words)
                    concepts['compounds'].append(compound)

        # Extract individual tokens.
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

        # Remove duplicates while preserving order.
        for key in concepts:
            concepts[key] = list(dict.fromkeys(concepts[key]))

        return dict(concepts)


# Synset Mapping
class SynsetMapper:
    '''Map concepts to WordNet synsets.'''

    def __init__(self):
        '''Initialize WordNet.'''

        print('Loading WordNet...')
        self.wn = wordnet
        self.pos_map = {
            'nouns': wordnet.NOUN,
            'verbs': wordnet.VERB,
            'adjectives': wordnet.ADJ,
            'adverbs': wordnet.ADV
        }
        print('WordNet loaded\n')

    def get_synsets(self, concept, pos_tag):
        '''Get synsets for a concept with given POS.'''

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
        '''Get synsets for all concepts in a dictionary.'''

        concept_to_synsets = {}

        # Handle compounds first.
        for compound in concepts_dict.get('compounds', []):
            # Try the compound as-is.
            synsets = self.get_synsets(compound.replace(' ', '_'), 'nouns')

            if synsets['offsets']:
                concept_to_synsets[compound] = synsets
            else:
                # Fallback: try each word separately.
                words = compound.split()
                combined_synsets = {'names': [], 'offsets': []}
                for word in words:
                    word_synsets = self.get_synsets(word, 'nouns')
                    combined_synsets['names'].extend(word_synsets['names'])
                    combined_synsets['offsets'].extend(word_synsets['offsets'])

                if combined_synsets['offsets']:
                    concept_to_synsets[compound] = combined_synsets

        # Then handle regular concepts.
        for pos_category, concept_list in concepts_dict.items():
            if pos_category == 'compounds':
                continue

            for concept in concept_list:
                synsets = self.get_synsets(concept, pos_category)
                if synsets['offsets']:
                    concept_to_synsets[concept] = synsets

        return concept_to_synsets


# Board Generator
class BoardGenerator:
    '''Complete AAC board generation with WordNet expansion.'''

    def __init__(self, arasaac_matcher, captioner=None, extractor=None, synset_mapper=None):
        '''Initialize with required components.'''

        self.matcher = arasaac_matcher
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

            # Exact match bonus.
            if concept.lower() in keywords:
                score += 10

            # Prefer symbols with fewer keywords (more specific).
            score -= len(keywords) * 0.1

            # Prefer basic/core categories.
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

            # Check if it's a compound (has spaces).
            if ' ' in concept:
                compound_matches = self.matcher.search_compound(concept)
                matched_symbols.update(compound_matches)

            # Try synset matching.
            for offset in synset_data.get('offsets', []):
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

    def expand_via_wordnet(self, image_concepts, current_matches, represented_concepts, min_needed):
        '''Expand vocabulary using WordNet hypernyms when below minimum.'''

        expanded = {}
        added_count = 0

        # Collect all original concepts to avoid re-adding them.
        original_concepts = set()
        for concept_list in image_concepts.values():
            if isinstance(concept_list, list):
                original_concepts.update([c.lower() for c in concept_list])

        # Also track what we've already expanded to avoid duplicates.
        already_tried = set(represented_concepts) | original_concepts

        # Expand nouns first (most concrete), then verbs.
        pos_to_expand = [
            ('nouns', wordnet.NOUN),
            ('verbs', wordnet.VERB),
            ('adjectives', wordnet.ADJ)
        ]

        for pos_category, wn_pos in pos_to_expand:
            if added_count >= min_needed:
                break

            concepts = image_concepts.get(pos_category, [])
            if not isinstance(concepts, list):
                continue

            for concept in concepts:
                if added_count >= min_needed:
                    break

                synsets = wordnet.synsets(concept, pos=wn_pos)[:2]

                for syn in synsets:
                    if added_count >= min_needed:
                        break

                    # Try hypernyms (broader terms: dog -> animal).
                    for hypernym in syn.hypernyms()[:3]:
                        if added_count >= min_needed:
                            break

                        word = hypernym.lemmas()[0].name().replace('_', ' ').lower()

                        if word in already_tried or len(word) < 3:
                            continue

                        already_tried.add(word)

                        matches = self.matcher.keyword_to_symbols.get(word, [])
                        if matches:
                            best = self.get_best_symbol(matches, word)
                            if best:
                                expanded[word] = [best]
                                added_count += 1
                                print(f"'    Expanded '{concept}' -> '{word}'")

                    # Try hyponyms (narrower terms: animal -> dog, cat).
                    for hyponym in syn.hyponyms()[:2]:
                        if added_count >= min_needed:
                            break

                        word = hyponym.lemmas()[0].name().replace('_', ' ').lower()

                        if word in already_tried or len(word) < 3:
                            continue

                        already_tried.add(word)

                        matches = self.matcher.keyword_to_symbols.get(word, [])
                        if matches:
                            best = self.get_best_symbol(matches, word)
                            if best:
                                expanded[word] = [best]
                                added_count += 1
                                print(f"'    Expanded '{concept}' -> '{word}'")

                    # Try similar words (sister terms via shared hypernym).
                    for lemma in syn.lemmas()[:3]:
                        if added_count >= min_needed:
                            break

                        for related in lemma.derivationally_related_forms()[:2]:
                            if added_count >= min_needed:
                                break

                            word = related.name().replace('_', ' ').lower()

                            if word in already_tried or len(word) < 3:
                                continue

                            already_tried.add(word)

                            matches = self.matcher.keyword_to_symbols.get(word, [])
                            if matches:
                                best = self.get_best_symbol(matches, word)
                                if best:
                                    expanded[word] = [best]
                                    added_count += 1
                                    print(f"    Expanded '{concept}' -> '{word}' (related)")

        if expanded:
            print(f'  WordNet expansion added {len(expanded)} concepts')

        return expanded

    def build_board(self, concept_to_symbols, image_concepts, caption='', max_symbols=64, min_symbols=16):
        '''Build board with minimum symbol guarantee via WordNet expansion.'''

        # Step 1: Map concepts to POS for weighting.
        concept_to_pos = {}
        for pos_category, concept_list in image_concepts.items():
            if isinstance(concept_list, list):
                for concept in concept_list:
                    concept_to_pos[concept] = pos_category

        # Step 2: Score and rank symbols.
        symbol_scores = Counter()
        symbol_to_concepts = {}

        pos_weights = {
            'nouns': 1.0,
            'verbs': 0.9,
            'adjectives': 0.7,
            'adverbs': 0.6
        }

        for concept, symbol_ids in concept_to_symbols.items():
            # Base score from frequency.
            freq = caption.lower().count(concept.lower()) if caption else 0
            pos_category = concept_to_pos.get(concept, 'nouns')
            weight = pos_weights.get(pos_category, 0.5)

            for symbol_id in symbol_ids:
                symbol_scores[symbol_id] += weight + (freq * 0.5)

                # Track which concepts this symbol represents.
                if symbol_id not in symbol_to_concepts:
                    symbol_to_concepts[symbol_id] = []
                symbol_to_concepts[symbol_id].append(concept)

        # Step 3: Build board with deduplication.
        board_symbol_ids = []
        represented_concepts = set()

        for symbol_id, score in symbol_scores.most_common():
            if symbol_id in board_symbol_ids:
                continue

            symbol_concepts = symbol_to_concepts.get(symbol_id, [])

            # Skip if concept already represented.
            if any(concept in represented_concepts for concept in symbol_concepts):
                continue

            board_symbol_ids.append(symbol_id)
            represented_concepts.update(symbol_concepts)

            if len(board_symbol_ids) >= max_symbols:
                break

        # Step 4: Expand via WordNet if below minimum.
        if len(board_symbol_ids) < min_symbols:
            needed = min_symbols - len(board_symbol_ids)
            print(f'  Below minimum ({len(board_symbol_ids)}/{min_symbols}), expanding via WordNet...')

            expanded_matches = self.expand_via_wordnet(
                image_concepts,
                concept_to_symbols,
                represented_concepts,
                needed
            )

            # Add expanded symbols to board.
            for concept, symbol_ids in expanded_matches.items():
                if len(board_symbol_ids) >= min_symbols:
                    break
                if concept in represented_concepts:
                    continue

                for sid in symbol_ids:
                    if sid not in board_symbol_ids:
                        board_symbol_ids.append(sid)
                        represented_concepts.add(concept)
                        break

        # Step 5: Get full metadata.
        board = []
        for sid in board_symbol_ids:
            metadata = self.matcher.get_symbol_metadata(sid)
            if metadata:
                board.append(metadata)

        print(f'  Final board: {len(board)} symbols (min: {min_symbols}, max: {max_symbols})')

        return board

    def generate_board(self, image_path, max_symbols=64, min_symbols=16):
        '''Complete pipeline: image -> AAC board.'''

        print(f'Generating AAC Board for: {image_path}')

        # Caption image.
        print('\nGenerating caption...')
        captions = self.captioner.caption_image(image_path)
        combined_caption = ' '.join(captions) if isinstance(captions, list) else captions
        print(f'Caption: {combined_caption[:100]}...')

        # Extract concepts.
        print('\nExtracting concepts...')
        concepts = self.extractor.extract_concepts(combined_caption)
        print(f'  Nouns: {concepts.get('nouns', [])[:5]}')
        print(f'  Verbs: {concepts.get('verbs', [])[:5]}')

        # Map to synsets.
        print('\nMapping to WordNet synsets...')
        concept_to_synsets = self.synset_mapper.get_all_synsets_for_concepts(concepts)
        print(f'  Mapped {len(concept_to_synsets)} concepts')

        # Match to ARASAAC.
        print('\nMatching to ARASAAC symbols...')
        concept_to_symbols = self.match_concepts(concept_to_synsets)
        print(f'  Matched {len(concept_to_symbols)} concepts')

        # Build board.
        print('\nBuilding board...')
        board = self.build_board(
            concept_to_symbols,
            concepts,
            caption=combined_caption,
            max_symbols=max_symbols,
            min_symbols=min_symbols
        )

        return {
            'captions': captions if isinstance(captions, list) else [captions],
            'combined_caption': combined_caption,
            'concepts': concepts,
            'board': board
        }

    def generate_board_from_text(self, text, max_symbols=64, min_symbols=16):
        '''Generate board from text description (no image).'''

        print(f'Generating AAC Board from text: "{text[:50]}..."')

        # Extract concepts directly from text.
        print('\nExtracting concepts...')
        concepts = self.extractor.extract_concepts(text)
        print(f'  Nouns: {concepts.get('nouns', [])[:5]}')
        print(f'  Verbs: {concepts.get('verbs', [])[:5]}')

        # Map to synsets.
        print('\nMapping to WordNet synsets...')
        concept_to_synsets = self.synset_mapper.get_all_synsets_for_concepts(concepts)
        print(f'  Mapped {len(concept_to_synsets)} concepts')

        # Match to ARASAAC.
        print('\nMatching to ARASAAC symbols...')
        concept_to_symbols = self.match_concepts(concept_to_synsets)
        print(f'  Matched {len(concept_to_symbols)} concepts')

        # Build board.
        print('\nBuilding board...')
        board = self.build_board(
            concept_to_symbols,
            concepts,
            caption=text,
            max_symbols=max_symbols,
            min_symbols=min_symbols
        )

        return {
            'captions': [text],
            'combined_caption': text,
            'concepts': concepts,
            'board': board
        }

    def display_board(self, result):
        '''Pretty print the board to console.'''

        print('\n' + '=' * 80)
        print('FINAL AAC BOARD')
        print('=' * 80)
        print(f'\nScenario: \'{result['combined_caption']}\'\n')
        print(f'Board contains {len(result['board'])} symbols:\n')

        for i, symbol in enumerate(result['board'], 1):
            keywords = ', '.join(symbol['keyword_list'][:3])
            print(f'{i:2d}. {keywords:40s}')
            print(f'    {symbol['image_url']}\n')


# Streamlit Application
def run_streamlit_app():
    '''Run the Streamlit web application.'''

    st.set_page_config(page_title='AAC Board Generator', layout='wide')

    # Initialize session state.
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    # Cache component loading.
    @st.cache_resource
    def load_components():
        matcher = ArasaacMatcher(str(ARASAAC_DATA_PATH))
        captioner = SceneCaptioner()
        extractor = ConceptExtractor()
        synset_mapper = SynsetMapper()
        generator = BoardGenerator(matcher, captioner, extractor, synset_mapper)
        return generator

    generator = load_components()

    # App Title.
    st.title('Dynamic AAC Board Generator')
    st.write('Generate communication boards from images or text descriptions')

    # Sidebar for input selection and image display.
    with st.sidebar:
        st.header('Controls')

        input_type = st.radio('Input Type:', ['Upload Image', 'Text Description'])

        # Board size controls.
        st.subheader('Board Settings')
        min_symbols = st.slider('Minimum symbols:', 8, 32, 16, 4)
        max_symbols = st.slider('Maximum symbols:', 24, 64, 48, 4)

        # Show uploaded image in sidebar if it exists.
        if st.session_state.uploaded_image:
            st.image(
                st.session_state.uploaded_image,
                caption='Current Image',
                use_container_width=True
            )

        # Clear board button.
        if st.session_state.result:
            if st.button('Clear Board'):
                st.session_state.result = None
                st.session_state.uploaded_image = None
                st.rerun()

    # Main content area.
    if input_type == 'Upload Image':
        uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

        if uploaded_file:
            # Store image in session state.
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image

            if st.button('Generate Board', type='primary'):
                with st.spinner('Generating AAC board...'):
                    # Save temp file.
                    image.save('temp_image.jpg')
                    st.session_state.result = generator.generate_board(
                        'temp_image.jpg',
                        max_symbols=max_symbols,
                        min_symbols=min_symbols
                    )

                    st.success(f'Generated {len(st.session_state.result['board'])} symbols!')

    else:  # Text Description.
        text_input = st.text_area(
            'Enter scenario description:',
            placeholder='e.g., Children playing at the playground with slides and swings',
            height=100
        )

        if st.button('Generate Board', type='primary') and text_input:
            with st.spinner('Generating AAC board...'):
                # Use the text-to-board method.
                st.session_state.result = generator.generate_board_from_text(
                    text_input,
                    max_symbols=max_symbols,
                    min_symbols=min_symbols
                )

                st.session_state.uploaded_image = None  # Clear image when using text.
                st.success(f'Generated {len(st.session_state.result['board'])} symbols!')

    # Display Board.
    if st.session_state.result:
        result = st.session_state.result

        st.markdown('---')
        st.subheader('Generated AAC Board')
        st.write(f'**Scene:** {result['combined_caption']}')

        # Display controls.
        num_symbols = st.slider(
            'Number of symbols to display:',
            8, 64,
            min(24, len(result['board'])),
            4
        )
        cols_per_row = st.slider('Symbols per row:', 4, 8, 6, 1)

        board = result['board'][:num_symbols]
        st.write(f'Displaying {len(board)} of {len(result['board'])} total symbols')

        # Render board grid.
        for row_start in range(0, len(board), cols_per_row):
            cols = st.columns(cols_per_row)

            for idx, col in enumerate(cols):
                symbol_idx = row_start + idx
                if symbol_idx < len(board):
                    symbol = board[symbol_idx]

                    with col:
                        # Display image.
                        st.image(symbol['image_url'], use_container_width=True)

                        # Keyword label.
                        keyword = symbol['keyword_list'][0] if symbol.get('keyword_list') else 'N/A'
                        st.markdown(f'**{keyword}**', unsafe_allow_html=True)

                        # Audio button.
                        if st.button('Play', key=f'audio_{symbol_idx}'):
                            try:
                                tts = gTTS(text=keyword, lang='en', slow=False)

                                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                                    tts.save(fp.name)
                                    st.audio(fp.name, format='audio/mp3', autoplay=True)

                            except Exception as e:
                                st.warning(f'Could not generate audio for "{keyword}"')

    else:
        # Welcome message when no board is generated.
        st.info('Upload an image or enter a text description to generate an AAC board.')


# Main Entry Point
if __name__ == '__main__':
    run_streamlit_app()
