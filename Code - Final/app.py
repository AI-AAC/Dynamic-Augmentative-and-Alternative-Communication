import streamlit as st
from gtts import gTTS
import tempfile
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import os

## Configuration ##
st.set_page_config(page_title="AAC Board Generator with Core Vocabulary", layout="wide")  # Widened layout

# Paths
CSV_PATH = "./data/ARASAAC_symbols.csv"

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'core_vocab_symbols' not in st.session_state:
    st.session_state.core_vocab_symbols = None


# Cache so it only loads once.
@st.cache_resource
def load_models():
    """Load BLIP and sentence transformer models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load BLIP for image captioning
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        blip_loaded = True
    except Exception as e:
        st.error(f"Could not load BLIP model: {e}")
        blip_processor = None
        blip_model = None
        blip_loaded = False
    
    # Load sentence transformer
    try:
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        sentence_model_loaded = True
    except Exception as e:
        st.error(f"Could not load sentence transformer: {e}")
        sentence_model = None
        sentence_model_loaded = False
    
    return {
        'blip_processor': blip_processor,
        'blip_model': blip_model,
        'blip_loaded': blip_loaded,
        'sentence_model': sentence_model,
        'sentence_model_loaded': sentence_model_loaded,
        'device': device
    }


@st.cache_data
def load_csv_data():
    """Load and preprocess CSV data."""
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates(subset=['primary_keyword'])
    return df


@st.cache_data
def precompute_embeddings(df, _sentence_model):
    """Pre-compute keyword embeddings."""
    keywords = df["primary_keyword"].tolist()
    embeddings = _sentence_model.encode(keywords, show_progress_bar=False, convert_to_numpy=True)
    df = df.copy()
    df["keyword_emb"] = [emb for emb in embeddings]
    return df


def get_core_vocabulary_symbols(df):
    """Get core vocabulary symbols using hardcoded synset IDs from CSV."""
    # Hardcoded synset IDs for core vocabulary (looked up from CSV)
    core_vocab_synsets = {
        'i': '06841868-n',
        'you': '',  # Empty synset
        'want': '01829179-v',
        'like': '01781131-v',
        'need': '01191258-v',
        'help': '00081834-v',
        'yes': '07218560-n',
        'no': '07219764-n',
        'more': '00099891-r',
        'stop': '08534954-n',
        'go': '01839438-v',
        'come': '01853188-v',
        'good': '01133477-a',
        'bad': '01129296-a',
        'happy': '01151786-a',
        'sad': '01364779-a',
        'please': '00010428-r',
        'who': '01379820-a',
        'how': '',  # Empty synset
        'can': '02950393-n',
        'make': '01622033-v',
        'see': '02133754-v',
        'get': '02531751-v',
        'have': '02209474-v'
    }
    
    core_symbols = []
    
    # Look up each synset in the dataframe
    for word, synset_id in core_vocab_synsets.items():
        if synset_id:
            # Match by synset
            matches = df[df['synset'] == synset_id]
        else:
            # For empty synsets, match by primary_keyword (case-insensitive, single word)
            matches = df[
                (df['primary_keyword'].str.lower() == word.lower()) &
                (~df['primary_keyword'].str.contains(r'\s', regex=True, na=False))
            ]
        
        if not matches.empty:
            row = matches.iloc[0]
            core_symbols.append({
                'synset': row['synset'],
                'pictogram_id': row['pictogram_id'],
                'primary_keyword': row['primary_keyword'],
                'image_url': row['image_url'],
                'keyword_list': [row['primary_keyword']],
                'is_core': True
            })
        else:
            print(f"Warning: Could not find symbol for core word: {word} (synset: {synset_id})")
    
    return core_symbols


def caption_image(image, models):
    """Generate caption from image using BLIP."""
    if not models['blip_loaded']:
        raise Exception("BLIP model not loaded")
    
    image = image.convert("RGB")
    inputs = models['blip_processor'](images=image, return_tensors="pt").to(models['device'])
    
    with torch.no_grad():
        out = models['blip_model'].generate(**inputs, max_length=30, num_beams=5, repetition_penalty=1.15)
    
    caption = models['blip_processor'].decode(out[0], skip_special_tokens=True).strip()
    return caption


def text_to_board(text, df_with_embeddings, sentence_model, top_n=20):
    """Convert text to ranked symbols using sentence transformers."""
    # Encode input text
    text_emb = sentence_model.encode(text, convert_to_numpy=True)
    
    # Calculate similarity scores (cosine similarity)
    df_with_embeddings = df_with_embeddings.copy()
    df_with_embeddings["score"] = df_with_embeddings["keyword_emb"].apply(
        lambda emb: np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb)))
    
    # Sort by score and get top N
    df_sorted = df_with_embeddings.sort_values("score", ascending=False)
    top_results = df_sorted.head(top_n * 2)  # Get more to account for duplicates
    
    # Extract unique symbols (prefer highest scoring)
    symbols = []
    seen_synsets = set()
    for _, row in top_results.iterrows():
        synset = row['synset']
        if synset not in seen_synsets:
            symbols.append({
                'synset': synset,
                'pictogram_id': row['pictogram_id'],
                'primary_keyword': row['primary_keyword'],
                'image_url': row['image_url'],
                'keyword_list': [row['primary_keyword']],  # Format for compatibility
                'score': row['score'],
                'is_core': False
            })
            seen_synsets.add(synset)
            if len(symbols) >= top_n:
                break
    
    return symbols


def display_symbol_grid(symbols, cols_per_row=6, section_title="", key_prefix=""):
    """Display symbols in a grid layout."""
    if not symbols:
        return
    
    if section_title:
        st.subheader(section_title)
    
    for row_start in range(0, len(symbols), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for idx, col in enumerate(cols):
            symbol_idx = row_start + idx
            if symbol_idx < len(symbols):
                symbol = symbols[symbol_idx]
                
                with col:
                    # Display image
                    st.image(symbol['image_url'], use_container_width=True)
                    
                    # Keyword label
                    keyword = symbol['keyword_list'][0] if symbol['keyword_list'] else symbol.get('primary_keyword', 'N/A')
                    st.markdown(f"**{keyword}**", unsafe_allow_html=True)
                    
                    # Audio button
                    if st.button(f"🔊", key=f"{key_prefix}_audio_{symbol_idx}"):
                        try:
                            tts = gTTS(text=keyword, lang='en', slow=False)
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                                tts.save(fp.name)
                                st.audio(fp.name, format='audio/mp3', autoplay=True)
                            
                        except Exception as e:
                            st.warning(f"Could not generate audio for '{keyword}'")


# Load models and data
models = load_models()
df = load_csv_data()

# Pre-compute embeddings if sentence model is loaded
if models['sentence_model_loaded']:
    df_with_embeddings = precompute_embeddings(df, models['sentence_model'])
else:
    df_with_embeddings = None

# Load core vocabulary symbols once
if st.session_state.core_vocab_symbols is None:
    with st.spinner("Loading core vocabulary..."):
        st.session_state.core_vocab_symbols = get_core_vocabulary_symbols(df)

# App Title
st.title("Dynamic AAC Board Generator with Core Vocabulary")
st.write("Generate communication boards from images or text descriptions with built-in core vocabulary")

# Sidebar for input selection AND image display
with st.sidebar:
    st.header("⚙️ Controls")

    input_type = st.radio("Input Type:", ["Upload Image", "Text Description"])
    
    # Specificity slider (Matt's feature)
    specificity = st.slider("Specificity (number of symbols):", 10, 30, 20, 1)
    
    # Core vocabulary display options
    st.markdown("---")
    st.header("📋 Core Vocabulary")
    show_core_vocab = st.checkbox("Show Core Vocabulary", value=True)
    core_cols = st.slider("Core vocab symbols per row:", 4, 10, 6, 1)

    # Show uploaded image in sidebar if it exists.
    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Current Image", use_container_width=True)

    # Clear board button.
    if st.session_state.result:
        if st.button("🗑️ Clear Board"):
            st.session_state.result = None
            st.session_state.uploaded_image = None
            st.rerun()

# Main content area
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        # Store image in session state.
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image

        if st.button("Generate Board", type="primary"):
            if not models['blip_loaded']:
                st.error("BLIP model not available. Please install transformers.")
            elif not models['sentence_model_loaded']:
                st.error("Sentence transformer model not available.")
            else:
                with st.spinner("Generating AAC board..."):
                    # Generate caption from image using BLIP
                    caption = caption_image(image, models)
                    st.info(f"Generated caption: {caption}")
                    
                    # Convert caption to board using sentence transformers
                    board = text_to_board(caption, df_with_embeddings, models['sentence_model'], top_n=specificity)
                    
                    st.session_state.result = {
                        'captions': [caption],
                        'combined_caption': caption,
                        'board': board
                    }
                    st.success(f"Generated {len(board)} symbols!")

else:  # Text Description
    text_input = st.text_area("Enter scenario description:",
                              placeholder="e.g., Children playing at the playground",
                              height=100)

    if st.button("Generate Board", type="primary") and text_input:
        if not models['sentence_model_loaded']:
            st.error("Sentence transformer model not available.")
        else:
            with st.spinner("Generating AAC board..."):
                # Use text-to-board pipeline with sentence transformers
                board = text_to_board(text_input, df_with_embeddings, models['sentence_model'], top_n=specificity)

                st.session_state.result = {
                    'captions': [text_input],
                    'combined_caption': text_input,
                    'board': board
                }
                st.session_state.uploaded_image = None  # Clear image when using text.
                st.success(f"Generated {len(board)} symbols!")

# Display Core Vocabulary (always shown if enabled)
if show_core_vocab and st.session_state.core_vocab_symbols:
    st.markdown("---")
    st.header("📋 Core Vocabulary")
    st.write("Essential words for communication - always available")
    
    display_symbol_grid(
        st.session_state.core_vocab_symbols,
        cols_per_row=core_cols,
        key_prefix="core"
    )

# Display Dynamic Board - use session_state result.
if st.session_state.result:
    result = st.session_state.result

    st.markdown("---")
    st.subheader("🎯 Context-Specific Symbols")
    st.write(f"**Scene:** {result['combined_caption']}")

    # Show more symbols with responsive columns
    num_symbols = st.slider("Number of symbols to display:", 8, len(result['board']), min(24, len(result['board'])), 4)
    cols_per_row = st.slider("Symbols per row:", 4, 8, 6, 1)

    board = result['board'][:num_symbols]
    st.write(f"Displaying {len(board)} of {len(result['board'])} total symbols")

    display_symbol_grid(
        board,
        cols_per_row=cols_per_row,
        key_prefix="dynamic"
    )

else:
    # Welcome message
    if not show_core_vocab or not st.session_state.core_vocab_symbols:
        st.info("Upload an image or enter a text description to generate an AAC board")
    else:
        st.info("Upload an image or enter a text description to generate context-specific symbols. Core vocabulary is shown above.")

