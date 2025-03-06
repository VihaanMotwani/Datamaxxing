# DeepContext

DeepContext is an advanced system for detecting misinformation in video content (especially from social media) and providing web-based context to help users verify claims. It combines video processing, text extraction, speech recognition, machine learning, and web search to create a comprehensive analysis pipeline.

## Overview

DeepContext works through a multi-stage process:

1. **Content Acquisition**: Downloads videos from Instagram Reels
2. **Content Extraction**: 
   - Extracts frames from the video at optimal intervals
   - Performs OCR (Optical Character Recognition) on text within the frames
   - Extracts and transcribes audio using OpenAI Whisper
3. **Misinformation Analysis**: 
   - Analyzes the extracted text and transcription using a Large Language Model (Llama via Groq)
   - Identifies potential misinformation based on 10 specific criteria
4. **Web Context Retrieval**:
   - If misinformation is detected, extracts the main claim
   - Searches the web for relevant context (using SerpAPI)
   - Evaluates sources for reliability
   - Synthesizes information to provide balanced perspective on the claim
5. **User Presentation**:
   - Presents findings through a Streamlit web interface or command-line outputs
   - Displays misinformation analysis, confidence scores, and web context with source ratings

## Features

- **Video Processing Pipeline**: Complete pipeline from video download to text/audio extraction
- **Multi-modal Analysis**: Combines text from both visual (OCR) and audio (transcription) sources
- **Comprehensive Misinformation Detection**: Uses 10 specific criteria to identify potential issues
- **Source Evaluation**: Rates web sources based on reliability and credibility
- **Balanced Perspective Presentation**: Presents multiple viewpoints on potentially controversial claims
- **Source Transparency**: Lists and rates all sources used for web context
- **Multiple Interfaces**: Command-line tool and Streamlit web application

## Requirements

- Python 3.8+
- Groq API key (for LLM access)
- SerpAPI key (for web search)
- Required Python packages:
  - `groq`
  - `yt-dlp`
  - `opencv-python`
  - `pillow`
  - `pytesseract`
  - `moviepy`
  - `whisper`
  - `streamlit`
  - `python-dotenv`
  - `requests`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/deepcontext.git
   cd deepcontext
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR (required for text extraction):
   - For Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - For macOS: `brew install tesseract`
   - For Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. EXPORT your API keys into the environment:
   ```
   EXPORT GROQ_API_KEY=your_groq_api_key_here
   EXPORT SERPAPI_KEY=your_serpapi_key_here
   ```

## Usage

### Web Interface

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Enter an Instagram Reel URL in the provided field and click "Process Reel"

3. View the results including misinformation analysis and web context

### Command Line Interface

Process a file with extracted text and transcription:
```
python main.py --json-file data.json
```

Analyze raw text:
```
python main.py --text "Scientists claim that lemon water cures diabetes."
```

Process multiple files in a directory:
```
python main.py --json-dir videos_folder --output-dir results
```

Additional options:
```
python main.py --help
```

## Components

- **reel_download.py**: Downloads Instagram Reels using yt-dlp
- **text_gen.py**: Extracts frames, performs OCR, and transcribes audio
- **misinformation_detector.py**: Analyzes content for potential misinformation
- **web_context_agent.py**: Searches and synthesizes web context for claims
- **integrated_system.py**: Combines all components into a unified system
- **main.py**: Command-line interface
- **app.py**: Streamlit web interface
- **Init_integrate.py**: Integrated extraction pipeline

## Misinformation Detection Criteria

The system analyzes content based on these criteria:

1. **Factual Accuracy**: Checking for demonstrably false claims
2. **Source Credibility**: Identifying unreliable or misrepresented sources
3. **Logical Consistency**: Detecting logical fallacies or contradictions
4. **Scientific Consensus**: Comparing claims against established scientific understanding
5. **Context Manipulation**: Identifying information presented in misleading ways
6. **Statistical Misrepresentation**: Detecting misused or cherry-picked statistics
7. **Emotional Manipulation**: Identifying emotional appeals over factual evidence
8. **Unverifiable Claims**: Flagging extraordinary claims without evidence
9. **Conspiracy Narratives**: Detecting unfounded conspiracy theories
10. **False Equivalence**: Identifying when unequal positions are presented as equal

## Future Improvements

- More advanced OCR for complex text layouts
- Configurable Whisper model sizes for transcription
- Local LLM options for misinformation detection
- Cached results to improve performance
- Support for additional languages

## License



## Acknowledgments

- [Groq](https://groq.com/) for LLM API access
- [SerpAPI](https://serpapi.com/) for web search capabilities
- [Whisper AI](https://github.com/openai/whisper) for audio transcription
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction
