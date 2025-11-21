import streamlit as st
from gtts import gTTS
import tempfile
from pathlib import Path
from PIL import Image

## Configuration ##
st.set_page_config(page_title="AAC Board Generator", layout="wide")  # Widened layout

current_directory = Path.cwd()
parent_directory = current_directory.parent

from board_gen_pipeline_blip1 import (
    ArasaacMatcher,
    SceneCaptioner,
    ConceptExtractor,
    SynsetMapper,
    BoardGenerator
)

ARASAAC_DATA_PATH = f'./data/arasaac_pictograms_complete_20251106_130529.json'

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None


# Cache so it only loads once.
@st.cache_resource
def load_components():
    matcher = ArasaacMatcher(str(ARASAAC_DATA_PATH))
    captioner = SceneCaptioner()
    extractor = ConceptExtractor()
    synset_mapper = SynsetMapper()
    generator = BoardGenerator(matcher, captioner, extractor, synset_mapper)
    return generator


generator = load_components()

# App Title
st.title("Dynamic AAC Board Generator")
st.write("Generate communication boards from images or text descriptions")

# Sidebar for input selection AND image display
with st.sidebar:
    st.header("⚙️ Controls")

    input_type = st.radio("Input Type:", ["Upload Image", "Text Description"])

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
            with st.spinner("Generating AAC board..."):
                # Save temp file.
                image.save("temp_image.jpg")
                st.session_state.result = generator.generate_board("temp_image.jpg")
                st.success(f"Generated {len(st.session_state.result['board'])} symbols!")

else:  # Text Description
    text_input = st.text_area("Enter scenario description:",
                              placeholder="e.g., Children playing at the playground",
                              height=100)

    if st.button("Generate Board", type="primary") and text_input:
        with st.spinner("Generating AAC board..."):
            # Use text-to-board pipeline.
            concepts = generator.extractor.extract_concepts(text_input)
            concept_to_synsets = generator.synset_mapper.get_all_synsets_for_concepts(concepts)
            concept_to_symbols = generator.match_concepts(concept_to_synsets)
            board = generator.build_board(concept_to_symbols, concepts, text_input, 64)

            st.session_state.result = {
                'captions': [text_input],
                'combined_caption': text_input,
                'concepts': concepts,
                'board': board
            }
            st.session_state.uploaded_image = None  # Clear image when using text.
            st.success(f"Generated {len(st.session_state.result['board'])} symbols!")

# Display Board - use session_state result.
if st.session_state.result:
    result = st.session_state.result

    st.markdown("---")
    st.subheader("Generated AAC Board")
    st.write(f"**Scene:** {result['combined_caption']}")

    # Show more symbols with responsive columns
    num_symbols = st.slider("Number of symbols to display:", 8, 64, 24, 4)
    cols_per_row = st.slider("Symbols per row:", 4, 8, 6, 1)

    board = result['board'][:num_symbols]
    st.write(f"Displaying {len(board)} of {len(result['board'])} total symbols")

    for row_start in range(0, len(board), cols_per_row):
        cols = st.columns(cols_per_row)

        for idx, col in enumerate(cols):
            symbol_idx = row_start + idx
            if symbol_idx < len(board):
                symbol = board[symbol_idx]

                with col:
                    # Display image
                    st.image(symbol['image_url'], use_container_width=True)

                    # Keyword label
                    keyword = symbol['keyword_list'][0] if symbol['keyword_list'] else 'N/A'
                    st.markdown(f"**{keyword}**", unsafe_allow_html=True)

                    # Audio button
                    if st.button(f"🔊", key=f"audio_{symbol_idx}"):
                        try:
                            tts = gTTS(text=keyword, lang='en', slow=False)

                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                                tts.save(fp.name)
                                st.audio(fp.name, format='audio/mp3', autoplay=True)

                        except Exception as e:
                            st.warning(f"Could not generate audio for '{keyword}'")

else:
    # Welcome message
    st.info("Upload an image or enter a text description to generate an AAC board")