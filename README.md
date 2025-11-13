# EDtech Chatbot - AI-Powered Student Performance Assistant

A comprehensive educational technology chatbot built with Streamlit and Google Gemini AI, designed to help students analyze their mathematics performance and solve problems interactively.

## Features

- 🤖 **Dual-Mode Intelligence**
  - **Tutor Mode**: Step-by-step problem solving with image support
  - **Analysis Mode**: Performance insights from homework and exam data

- 📊 **Smart Data Analysis**
  - Automatic homework and exam data processing
  - Intelligent filtering to focus on relevant information
  - Topic-based performance tracking
  - Strengths and weaknesses identification

- 🎤 **Voice Interaction** (v1 & v2)
  - Speech-to-text input using Faster-Whisper
  - Text-to-speech output with multi-language support (English, Hindi, Telugu)

- 🖼️ **Image Support**
  - Upload math problems as images
  - Visual problem-solving assistance

- 💬 **Conversational Memory**
  - Context-aware responses using LangChain
  - Natural follow-up conversations

- 📚 **Curriculum-Aligned**
  - Class 6-12 mathematics curriculum support
  - Unit priority recommendations based on exam weightage

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EDtech-Chatbot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   - Set your Google Gemini API key as an environment variable:
   ```bash
   # Windows
   set GEMINI_API_KEY=your_api_key_here

   # Linux/Mac
   export GEMINI_API_KEY=your_api_key_here
   ```
   - Alternatively, update the `Config.GEMINI_API_KEY` in the code (not recommended for production)

## Usage

### Running the Application

**Version 3** (Latest - Recommended):
```bash
streamlit run version3.py
```

**Version 2**:
```bash
streamlit run version2.py
```

**Version 1**:
```bash
streamlit run version1.py
```

### Getting Started

1. **Login**: Enter your student name and class
2. **Upload Data** (Optional): Paste homework or exam data in JSON format
3. **Start Chatting**: Ask questions about your performance or solve math problems

### Example Queries

**Performance Analysis**:
- "How did I do in my recent homework?"
- "What are my weak areas in algebra?"
- "Show me my exam results"
- "Which topics should I focus on?"

**Problem Solving**:
- "Solve this quadratic equation: x² + 5x + 6 = 0"
- "How do I find the area of a triangle?"
- Upload an image of a math problem and ask "Solve this"

**General**:
- "Explain the Pythagorean theorem"
- "What is the quadratic formula?"

## Version Comparison

| Feature | Version 1 | Version 2 | Version 3 |
|---------|-----------|-----------|-----------|
| **Audio I/O** | ✅ Input & Output | ✅ Input & Output | ✅ Input Only |
| **Tutor Mode** | ❌ | ✅ | ✅ |
| **Analysis Mode** | ❌ | ✅ | ✅ |
| **Smart Filtering** | ❌ | ✅ | ✅ |
| **AI Model** | Gemini 2.0 Flash | Gemini 2.0 Flash | Gemini 2.5 Flash |
| **Streaming** | ❌ | ❌ | ✅ |
| **Mode Detection** | ❌ | ✅ | ✅ Enhanced |
| **Data Normalization** | Basic | Standard | Advanced |

### Version Details

#### Version 1
- Basic chatbot with audio input/output
- Simple query-response system
- Takes username, homework, and exam data
- No specialized modes

#### Version 2
- Introduced **Tutor Mode** and **Analysis Mode**
- Smart data filtering based on query intent
- Query-specific context optimization
- Upgraded to Gemini 2.0 Flash

#### Version 3 (Current)
- Removed voice output for performance optimization
- Added streaming support for faster responses
- Enhanced data normalization for multiple formats
- Improved mode detection algorithm
- Upgraded to Gemini 2.5 Flash with optimized parameters
- Better handling of casual vs. analytical queries

## Data Format

### Homework Data
```json
{
  "data": [
    {
      "homework_id": "HW-001",
      "submission_date": "2024-11-01",
      "question": {
        "questions": [
          {
            "question_id": "Q1",
            "question_text": "Solve: x² - 5x + 6 = 0",
            "total_score": 8,
            "max_score": 10,
            "answer_category": "calculation_error",
            "concept_required": ["Quadratic Equations"]
          }
        ]
      }
    }
  ]
}
```

### Exam Data
```json
{
  "student_name": "John Doe",
  "roll_number": "10A23",
  "results": [
    {
      "exam_name": "Mid-Term Exam",
      "total_marks_obtained": 85,
      "total_max_marks": 100,
      "overall_percentage": 85.0,
      "grade": "A",
      "strengths": ["Algebra", "Geometry"],
      "areas_for_improvement": ["Trigonometry"]
    }
  ]
}
```

Sample data files are provided in `homework_data.txt` and `exam.txt`.

## Project Structure

```
EDtech-Chatbot/
├── version1.py              # Basic version with audio I/O
├── version2.py              # Added tutor/analysis modes
├── version3.py              # Latest with streaming (recommended)
├── requirements.txt         # Python dependencies
├── homework_data.txt        # Sample homework data
├── exam.txt                 # Sample exam data
├── README.md               # This file
├── CLAUDE.md               # Developer documentation
└── venv/                   # Virtual environment (not in git)
```

## Technology Stack

- **Frontend**: Streamlit
- **AI**: Google Gemini (2.5 Flash)
- **Memory**: LangChain
- **Voice Recognition**: Faster-Whisper
- **Audio Processing**: SoundFile, SciPy
- **Image Processing**: Pillow

## Contributing

When contributing to this project, please ensure:

1. Follow the existing code structure and patterns
2. Test with sample data before submitting
3. Update documentation for any new features
4. Maintain backward compatibility with existing data formats
5. Refer to `CLAUDE.md` for architectural guidance

## License

[Add your license here]

## Support

For issues or questions, please open an issue on the repository.

## Acknowledgments

- Google Gemini AI for natural language processing
- Streamlit for the web framework
- OpenAI Whisper for speech recognition
