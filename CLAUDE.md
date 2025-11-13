# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Educational Technology chatbot system designed to help students with mathematics performance analysis and problem-solving. The application uses Streamlit for the UI, Google Gemini AI for natural language processing, and supports voice input/output capabilities.

## Version History

The codebase has three main versions representing evolutionary improvements:

- **version1.py**: Basic chatbot with audio I/O, no tutor/analysis modes
- **version2.py**: Added tutor mode and analysis mode, filters data by query type, uses Gemini 2.0 Flash
- **version3.py**: Latest version - removed voice output, added streaming support, improved data normalization

Version 3 is the current production version. When making changes, focus on this file.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run version3.py

# Run specific version
streamlit run version1.py  # or version2.py
```

## Core Architecture

### Dual-Mode System

The application operates in two distinct modes, automatically detected based on user queries:

1. **TUTOR_MODE**: Problem-solving and mathematical explanations
   - Activated by keywords: "solve", "find", "calculate", "how to solve"
   - Always activated when images are uploaded with queries
   - Uses simplified prompts focused on step-by-step problem solving
   - Does NOT require student performance data

2. **ANALYSIS_MODE**: Student performance analysis
   - Activated by keywords: "my performance", "my homework", "my exam", "how did I do"
   - Requires homework_data and/or exam_data in session state
   - Uses data filtering to reduce token usage
   - Provides insights on strengths, weaknesses, and improvement areas

3. **CASUAL_MODE**: Simple greetings (no AI call)
   - Fast path for "hi", "hello", "thanks", "bye"
   - Returns template responses without AI

Mode detection is handled by `detect_query_mode()` in version3.py:975

### Data Architecture

#### Homework Data Flow
```
User JSON Input → parse_json_safely() → HomeworkDataNormalizer.normalize_homework_item()
→ HomeworkDataFilter.apply_smart_filter() → Filtered JSON → Gemini API
```

The system supports multiple homework data formats:
- **NEW format**: `homework_id + submission_date + question.questions` (nested structure)
- **OLD format**: `homework_id + creation_date + questions` (flat structure)

All formats are normalized through `HomeworkDataNormalizer` class (version2.py:34, version3.py:33).

#### Exam Data Flow
```
User JSON Input → parse_json_safely() → ExamDataFilter.normalize_exam_data()
→ ExamDataFilter.apply_smart_exam_filter() → Filtered JSON → Gemini API
```

Exam data formats:
- **Results-based format**: `results` array with `strengths` and `areas_for_improvement`
- **Question-based format**: `question_data` with detailed question-level analysis

### Data Filtering System

The application uses intelligent filtering to reduce token usage and provide relevant context:

**HomeworkDataFilter** (version2.py:197, version3.py:197):
- `filter_recent_n_homeworks()`: Get N most recent submissions
- `filter_by_topic()`: Filter by mathematical topic/concept
- `filter_errors_only()`: Show only questions with mistakes
- `create_summary_only()`: Aggregate statistics without question details
- `filter_specific_homework()`: Get specific homework by ID
- `apply_smart_filter()`: Main entry point that auto-selects appropriate filter

**ExamDataFilter** (version2.py:539, version3.py:539):
- `normalize_exam_data()`: Standardizes different exam data formats
- `apply_smart_exam_filter()`: Routes to summary/strengths/weaknesses based on query intent
- Supports both high-level results and detailed question-level data

### Query Classification

**QueryIntentClassifier** (version2.py:134, version3.py:134):
- Classifies queries as 'homework', 'exam', or 'general'
- Uses keyword matching and pattern detection
- Determines which data source to use for responses

## Configuration

### API Keys
The Gemini API key is configured in `Config.GEMINI_API_KEY` (hardcoded in code, should be moved to environment variables).

### AI Model Selection
- **version1.py, version2.py**: Uses `gemini-2.0-flash`
- **version3.py**: Uses `gemini-2.5-flash` with generation config (temperature=0.2, top_p=0.8, top_k=40)

### Curriculum Data
Class-specific curriculum weightages are defined in `Config.UNIT_WEIGHTAGE` (version2.py:826, version3.py:826):
- Supports Class 6-12 mathematics curriculum
- Defines high-priority (≥20%), medium-priority (15-19%), and low-priority (<15%) units
- Used to provide targeted recommendations based on exam weightage

## Session State Management

Streamlit session state variables (initialized in `initialize_session_state()`):
- `logged_in`: Boolean for authentication status
- `homework_data`: Normalized homework submissions
- `exam_data`: Normalized exam results
- `student_class`: Format "Class-X" (e.g., "Class-10")
- `student_name` (v3) / `student_id` (v1, v2): Student identifier
- `memory`: LangChain ConversationBufferMemory for chat history
- `messages`: List of chat messages for UI rendering
- `whisper_model`: Cached Faster-Whisper model for voice recognition
- `language`: Selected language for voice output ("en", "hi", "te")
- `processing`: Boolean to prevent concurrent query processing
- `last_audio_processed`: Boolean to prevent audio re-processing

## Voice Features

### Voice Input (All versions)
- Uses Faster-Whisper model (base, CPU, int8)
- Audio processing pipeline: Upload → Resample to 16kHz → Transcribe → Return text
- Handles mono/stereo conversion automatically
- Function: `process_voice_input()` (version1.py:928, version2.py:1283, version3.py:1283)

### Voice Output (version1.py, version2.py only)
- Uses Google Text-to-Speech (gTTS)
- Supports English, Hindi, Telugu
- Text cleaning removes markdown formatting and special characters
- Function: `text_to_speech()` (version1.py:1010, version2.py:1365)
- **Note**: Voice output removed in version3.py per README

## Image Support

All versions support image uploads:
- Converts to JPEG format
- Encodes as base64 for Gemini API
- Used primarily in TUTOR_MODE for problem-solving
- Function: `answer_query_with_gemini()` handles image processing

## Key Implementation Patterns

### Safe JSON Parsing
`parse_json_safely()` (version2.py:887, version3.py:887):
- Handles both JSON and Python literal formats
- Strips leading dots and whitespace
- Falls back to ast.literal_eval if json.loads fails

### Token Estimation
`estimate_tokens()` and `safe_json_data()` (version2.py:849, version3.py:849):
- Estimates tokens as `len(text) // 4`
- Truncates data to stay under 4000 token limit
- Automatically falls back to summary mode for large datasets

### Date Parsing
`HomeworkDataFilter.parse_date()` (version2.py:229, version3.py:229):
- Supports ISO format (with/without microseconds)
- Supports short format (DD-MM-YY)
- Supports standard format (YYYY-MM-DD)
- Returns `datetime.min` on failure

## Testing Data

Sample data files are provided:
- `homework_data.txt`: Sample homework submissions in JSON format
- `exam.txt`: Sample exam results in JSON format

These can be used for testing the data processing pipeline.

## Common Development Tasks

### Adding New Data Filters
1. Add method to `HomeworkDataFilter` or `ExamDataFilter` class
2. Update `apply_smart_filter()` to call new filter based on intent
3. Update `detect_query_intent()` if new keywords needed

### Modifying AI Prompts
- **TUTOR_MODE**: Edit `build_tutor_prompt()` in version3.py:1033
- **ANALYSIS_MODE**: Edit `build_analysis_prompt()` in version3.py:1058
- Consider token limits when adding context

### Changing AI Model
Update `initialize_gemini()` function:
- version2.py:1098
- version3.py:1098
- Adjust generation config parameters as needed

### Adding New Curriculum Classes
Add entry to `Config.UNIT_WEIGHTAGE` dictionary with unit weightages totaling 100.

## Important Notes

- The application automatically detects student class from student_id format (version1.py only)
- version3.py requires manual class input during login
- All data normalization should preserve backward compatibility with old formats
- Token usage is critical - always use filtering for large datasets
- The mode detection system prioritizes performance analysis keywords over solving keywords to prevent misclassification
