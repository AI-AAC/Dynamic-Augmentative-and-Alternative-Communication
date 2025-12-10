# Dynamic AAC Board Generator

A Dynamic Augmentative and Alternative Communication (AAC) Board Generator that uses computer vision and natural language processing to automatically create context-appropriate communication boards from images or text descriptions.

## Authors

- Kaitlin Moore
- Matthew Yurkunas

## Course

95-891 Introduction to Artificial Intelligence | Carnegie Mellon University | Fall 2025

## Overview

Over 5 million Americans rely on AAC devices, yet current digital boards lack situationally relevant vocabulary and require extensive manual customization. This project addresses that gap by using AI to generate contextually relevant AAC boards on-the-fly.

The system uses:
- **BLIP** for image captioning
- **spaCy** for natural language processing and concept extraction
- **WordNet** for semantic mapping and vocabulary expansion
- **ARASAAC** pictogram database (13,500+ symbols) for symbol matching

## Installation

### 1. Clone or download the repository

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download required language models

Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

Download NLTK WordNet data:
```python
import nltk
nltk.download('wordnet')
```

### 4. Configure the data path

Update the `ARASAAC_DATA_PATH` variable in `aac_board_generator.py` to point to your ARASAAC JSON data file:

```python
ARASAAC_DATA_PATH = 'path/to/your/arasaac_pictograms_complete_20251106_130529.json'
```

## Usage

Run the Streamlit application:

```bash
streamlit run P2_aac_board_generator.py
```

Or alternatively:

```bash
python -m streamlit run aac_board_generator.py
```

The application will open in your web browser (typically at `http://localhost:8501`).

### Features

- **Image to Board**: Upload an image and the system generates a communication board with relevant vocabulary based on the scene.
- **Text to Board**: Enter a text description of a scenario and receive a generated board.
- **Adjustable board size**: Configure minimum and maximum number of symbols.
- **Text-to-speech**: Click the "Play" button on any symbol to hear it spoken aloud.

## Project Structure

```
├── aac_board_generator.py    # Main application (combined single file)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    └── arasaac_pictograms_complete_20251106_130529.json  # ARASAAC dataset
```

## License

This project is for educational purposes as part of the 95-891 Introduction to Artificial Intelligence course at Carnegie Mellon University.

ARASAAC pictograms are used under the Creative Commons BY-NC-SA license.
