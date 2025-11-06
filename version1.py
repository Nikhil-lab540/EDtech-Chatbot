```import streamlit as st
import requests
import json
import tempfile
import os
import warnings
import logging
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime, timedelta
import re

warnings.filterwarnings('ignore')
logging.getLogger('asyncio').setLevel(logging.ERROR)

import sys
if 'torch' in sys.modules:
    import torch
    torch.set_warn_always(False)

import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from faster_whisper import WhisperModel
import soundfile as sf
from scipy import signal
from gtts import gTTS
from PIL import Image
import base64
import io

# ===========================
# QUERY INTENT CLASSIFIER
# ===========================
class QueryIntentClassifier:
    """Determines if query is about homework, exam, or general"""
    
    @staticmethod
    def classify_query_type(query: str) -> Literal['homework', 'exam', 'general']:
        """
        Classify query as homework-related, exam-related, or general
        Returns: 'homework', 'exam', or 'general'
        """
        query_lower = query.lower()
        
        # Strong exam indicators
        exam_keywords = [
            'exam', 'test', 'examination', 'exam result',
            'exam score', 'exam grade', 'exam performance',
            'exam analysis', 'test result', 'test score',
            'question paper', 'final exam', 'midterm'
        ]
        
        # Strong homework indicators
        homework_keywords = [
            'homework', 'hw-', 'hw0', 'hw1', 'hw2', 'hw3', 'hw4', 'hw5',
            'assignment', 'homework submission', 'hw ', 'daily work',
            'practice problem', 'homework score', 'homework performance'
        ]
        
        # Check for exam keywords first (more specific)
        if any(keyword in query_lower for keyword in exam_keywords):
            return 'exam'
        
        # Check for homework keywords
        if any(keyword in query_lower for keyword in homework_keywords):
            return 'homework'
        
        # Context-based classification
        if 'hw' in query_lower and any(char.isdigit() for char in query_lower):
            return 'homework'
        
        if any(word in query_lower for word in ['grade', 'marks', 'percentage', 'score']):
            return 'general'
        
        return 'general'

# ===========================
# HOMEWORK DATA FILTER
# ===========================
class HomeworkDataFilter:
    """Filter student homework data based on query context"""
    
    @staticmethod
    def detect_query_intent(query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
        
        if any(word in query_lower for word in ['recent', 'latest', 'last', 'current']):
            return 'recent_performance'
        
        if any(word in query_lower for word in ['homework', 'hw-', 'assignment']):
            return 'specific_homework'
        
        if any(word in query_lower for word in ['chapter', 'topic', 'unit', 'algebra', 'geometry', 
                                                  'calculus', 'trigonometry', 'statistics', 'rational',
                                                  'square', 'cube', 'polynomial', 'equation']):
            return 'topic_analysis'
        
        if any(word in query_lower for word in ['struggle', 'weak', 'improve', 'error', 'mistake']):
            return 'weakness_analysis'
        
        if any(word in query_lower for word in ['summary', 'report']):
            return 'overall_summary'
        
        return 'general'
    
    @staticmethod
    def parse_date(date_str: str) -> datetime:
        """Parse date from multiple formats"""
        if not date_str:
            return datetime.min
        
        try:
            if 'T' in date_str or '-' in date_str and len(date_str) > 10:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            pass
        
        try:
            return datetime.strptime(date_str, '%d-%m-%y')
        except:
            pass
        
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except:
            return datetime.min
    
    @staticmethod
    def filter_recent_n_homeworks(student_data: Dict, n: int = 3) -> Dict:
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        homework_list = student_data['homework_data']
        sorted_hw = sorted(
            homework_list, 
            key=lambda x: HomeworkDataFilter.parse_date(x.get('creation_date', '')),
            reverse=True
        )
        
        filtered_data = student_data.copy()
        filtered_data['homework_data'] = sorted_hw[:n]
        filtered_data['filter_applied'] = f'Showing {min(n, len(sorted_hw))} most recent homeworks'
        
        return filtered_data
    
    @staticmethod
    def filter_by_topic(student_data: Dict, topic_keyword: str) -> Dict:
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        filtered_homeworks = []
        
        for homework in student_data['homework_data']:
            filtered_questions = [
                q for q in homework.get('questions', [])
                if topic_keyword.lower() in q.get('topic', '').lower()
            ]
            
            if filtered_questions:
                hw_copy = homework.copy()
                hw_copy['questions'] = filtered_questions
                filtered_homeworks.append(hw_copy)
        
        filtered_data = student_data.copy()
        filtered_data['homework_data'] = filtered_homeworks
        filtered_data['filter_applied'] = f'Filtered by topic: {topic_keyword}'
        
        return filtered_data
    
    @staticmethod
    def filter_errors_only(student_data: Dict) -> Dict:
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        filtered_homeworks = []
        for homework in student_data['homework_data']:
            error_questions = [
                q for q in homework.get('questions', [])
                if q.get('answer_category', '').lower() not in ['no_error', 'correct']
                and q.get('total_score', 0) < q.get('max_score', 0)
            ]
            
            if error_questions:
                hw_copy = homework.copy()
                hw_copy['questions'] = error_questions
                filtered_homeworks.append(hw_copy)
        
        filtered_data = student_data.copy()
        filtered_data['homework_data'] = filtered_homeworks
        filtered_data['filter_applied'] = 'Showing only homework questions with errors'
        
        return filtered_data
    
    @staticmethod
    def create_summary_only(student_data: Dict) -> Dict:
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        summary = {
            'student_id': student_data.get('student_id'),
            'student_name': student_data.get('student_name'),
            'class': student_data.get('class'),
            'total_homeworks': len(student_data.get('homework_data', [])),
            'homework_scores': [],
            'topic_summary': {},
            'error_summary': {},
            'filter_applied': 'Homework summary only'
        }
        
        for hw in student_data.get('homework_data', []):
            hw_score = 0
            hw_max = 0
            
            for q in hw.get('questions', []):
                hw_score += q.get('total_score', 0)
                hw_max += q.get('max_score', 0)
            
            if hw_max > 0:
                summary['homework_scores'].append({
                    'date': hw.get('creation_date'),
                    'homework_id': hw.get('homework_id'),
                    'score': hw_score,
                    'max_score': hw_max,
                    'percentage': round((hw_score / hw_max * 100), 2)
                })
        
        for hw in student_data.get('homework_data', []):
            for q in hw.get('questions', []):
                topic = q.get('topic', 'Unknown')
                
                if topic not in summary['topic_summary']:
                    summary['topic_summary'][topic] = {
                        'total_questions': 0,
                        'score': 0,
                        'max_score': 0
                    }
                
                summary['topic_summary'][topic]['total_questions'] += 1
                summary['topic_summary'][topic]['score'] += q.get('total_score', 0)
                summary['topic_summary'][topic]['max_score'] += q.get('max_score', 0)
        
        for topic in summary['topic_summary']:
            data = summary['topic_summary'][topic]
            data['percentage'] = round((data['score'] / data['max_score'] * 100), 2) if data['max_score'] > 0 else 0
        
        for hw in student_data.get('homework_data', []):
            for q in hw.get('questions', []):
                error_type = q.get('answer_category', 'no_error')
                if error_type.lower() not in ['no_error', 'correct']:
                    summary['error_summary'][error_type] = summary['error_summary'].get(error_type, 0) + 1
        
        return summary
    
    @staticmethod
    def filter_specific_homework(student_data: Dict, homework_id: str) -> Dict:
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        specific_hw = [
            hw for hw in student_data['homework_data']
            if homework_id.upper() in hw.get('homework_id', '').upper()
        ]
        
        filtered_data = student_data.copy()
        filtered_data['homework_data'] = specific_hw
        filtered_data['filter_applied'] = f'Showing homework: {homework_id}'
        
        return filtered_data
    
    @staticmethod
    def apply_smart_filter(query: str, student_data: Dict) -> Dict:
        """Automatically apply the best filter based on query intent"""
        intent = HomeworkDataFilter.detect_query_intent(query)
        
        if intent in ['greeting']:
            return {
                'student_id': student_data.get('student_id'),
                'student_name': student_data.get('student_name'),
                'class': student_data.get('class'),
                'filter_applied': 'No homework data needed for greeting'
            }
        
        if intent == 'recent_performance':
            return HomeworkDataFilter.filter_recent_n_homeworks(student_data, n=3)
        
        elif intent == 'weakness_analysis':
            return HomeworkDataFilter.filter_errors_only(student_data)
        
        elif intent == 'overall_summary':
            return HomeworkDataFilter.create_summary_only(student_data)
        
        elif intent == 'specific_homework':
            hw_pattern = r'hw[_-]?(\d+)'
            match = re.search(hw_pattern, query.lower())
            if match:
                hw_id = f"HW{match.group(1).zfill(3)}"
                return HomeworkDataFilter.filter_specific_homework(student_data, hw_id)
            return HomeworkDataFilter.filter_recent_n_homeworks(student_data, n=5)
        
        elif intent == 'topic_analysis':
            topic_keywords = ['algebra', 'geometry', 'calculus', 'trigonometry', 
                            'statistics', 'rational', 'square', 'cube', 'polynomial',
                            'equation', 'mensuration', 'data handling', 'quadratic',
                            'derivative', 'integration', 'probability', 'coordinate']
            for keyword in topic_keywords:
                if keyword in query.lower():
                    return HomeworkDataFilter.filter_by_topic(student_data, keyword)
            return HomeworkDataFilter.create_summary_only(student_data)
        
        else:
            filtered = HomeworkDataFilter.filter_recent_n_homeworks(student_data, n=2)
            return filtered

# ===========================
# EXAM DATA FILTER
# ===========================
class ExamDataFilter:
    """Filter exam result data based on query context"""
    
    @staticmethod
    def normalize_exam_data(exam_data: Dict) -> Dict:
        """
        Normalize exam results data structure
        Supports multiple formats
        """
        if not exam_data:
            return exam_data
        
        # Handle NEW format: results array with exam metadata
        if isinstance(exam_data, dict) and 'results' in exam_data:
            results = exam_data.get('results', [])
            
            if not results:
                return exam_data
            
            exam_result = results[0]
            
            return {
                'student_name': exam_data.get('student_name', ''),
                'roll_number': exam_data.get('roll_number', ''),
                'exam_name': exam_result.get('exam_name', ''),
                'exam_type': exam_result.get('exam_type', ''),
                'class_section': exam_result.get('class_section', ''),
                'total_marks_obtained': exam_result.get('total_marks_obtained', 0),
                'total_max_marks': exam_result.get('total_max_marks', 0),
                'overall_percentage': exam_result.get('overall_percentage', 0),
                'grade': exam_result.get('grade', 'N/A'),
                'strengths': exam_result.get('strengths', []),
                'areas_for_improvement': exam_result.get('areas_for_improvement', []),
                'total_exams': len(results),
                'status': 'success'
            }
        
        # Handle OLD format: question_data with detailed questions
        if isinstance(exam_data, dict) and 'question_data' in exam_data:
            question_data_list = exam_data['question_data']
            
            all_questions = []
            total_score = 0
            total_max = 0
            
            for data_entry in question_data_list:
                questions = data_entry.get('questions_evaluation', [])
                for q in questions:
                    all_questions.append(q)
                    total_score += q.get('total_score', 0)
                    total_max += q.get('max_marks', 0)
            
            percentage = round((total_score / total_max * 100), 2) if total_max > 0 else 0
            
            if percentage >= 90:
                grade = 'A+'
            elif percentage >= 80:
                grade = 'A'
            elif percentage >= 70:
                grade = 'B+'
            elif percentage >= 60:
                grade = 'B'
            elif percentage >= 50:
                grade = 'C'
            elif percentage >= 40:
                grade = 'D'
            else:
                grade = 'F'
            
            return {
                'total_questions': len(all_questions),
                'total_marks_obtained': total_score,
                'total_max_marks': total_max,
                'overall_percentage': percentage,
                'grade': grade,
                'questions': all_questions,
                'status': 'success'
            }
        
        return exam_data
    
    @staticmethod
    def detect_exam_query_intent(query: str) -> str:
        """Detect exam query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['summary', 'grade', 'percentage']):
            return 'summary'
        
        if any(word in query_lower for word in ['strength', 'good', 'correct', 'right', 'what did i do well']):
            return 'strengths'
        
        if any(word in query_lower for word in ['weakness', 'improve', 'improvement', 'weak', 'mistake', 'error', 'wrong']):
            return 'weaknesses'
        
        if any(word in query_lower for word in ['detail', 'analysis', 'breakdown']):
            return 'detailed'
        
        return 'general'
    
    @staticmethod
    def apply_smart_exam_filter(query: str, exam_data: Dict) -> Dict:
        """Apply intelligent filtering based on query"""
        normalized = ExamDataFilter.normalize_exam_data(exam_data)
        
        if not normalized:
            return normalized
        
        intent = ExamDataFilter.detect_exam_query_intent(query)
        
        # Check if it's the NEW format (results-based)
        if 'strengths' in normalized and 'areas_for_improvement' in normalized:
            
            if intent == 'summary':
                return {
                    'student_name': normalized.get('student_name'),
                    'roll_number': normalized.get('roll_number'),
                    'exam_name': normalized.get('exam_name'),
                    'exam_type': normalized.get('exam_type'),
                    'total_marks_obtained': normalized.get('total_marks_obtained'),
                    'total_max_marks': normalized.get('total_max_marks'),
                    'overall_percentage': normalized.get('overall_percentage'),
                    'grade': normalized.get('grade'),
                    'filter_applied': 'Exam summary only',
                    'status': 'success'
                }
            
            elif intent == 'strengths':
                return {
                    'student_name': normalized.get('student_name'),
                    'exam_name': normalized.get('exam_name'),
                    'overall_percentage': normalized.get('overall_percentage'),
                    'grade': normalized.get('grade'),
                    'strengths': normalized.get('strengths', []),
                    'filter_applied': 'Showing only strengths',
                    'status': 'success'
                }
            
            elif intent == 'weaknesses':
                return {
                    'student_name': normalized.get('student_name'),
                    'exam_name': normalized.get('exam_name'),
                    'overall_percentage': normalized.get('overall_percentage'),
                    'grade': normalized.get('grade'),
                    'areas_for_improvement': normalized.get('areas_for_improvement', []),
                    'filter_applied': 'Showing only areas for improvement',
                    'status': 'success'
                }
            
            else:
                return normalized
        
        # OLD FORMAT with questions array
        if 'questions' in normalized:
            all_questions = normalized['questions']
            
            if intent == 'summary':
                return {
                    'total_questions': normalized['total_questions'],
                    'total_marks_obtained': normalized['total_marks_obtained'],
                    'total_max_marks': normalized['total_max_marks'],
                    'overall_percentage': normalized['overall_percentage'],
                    'grade': normalized['grade'],
                    'questions': [],
                    'summary_only': True,
                    'filter_applied': 'Exam summary only',
                    'status': 'success'
                }
            
            elif intent == 'strengths':
                correct_questions = [
                    q for q in all_questions
                    if q.get('error_type', '').lower() == 'no_error'
                ]
                return {
                    'total_questions': len(correct_questions),
                    'total_marks_obtained': normalized['total_marks_obtained'],
                    'total_max_marks': normalized['total_max_marks'],
                    'overall_percentage': normalized['overall_percentage'],
                    'grade': normalized['grade'],
                    'questions': correct_questions,
                    'filter_applied': f'Showing {len(correct_questions)} correct questions',
                    'status': 'success'
                }
            
            elif intent == 'weaknesses':
                error_questions = [
                    q for q in all_questions
                    if q.get('error_type', '').lower() not in ['no_error']
                ]
                return {
                    'total_questions': len(error_questions),
                    'total_marks_obtained': normalized['total_marks_obtained'],
                    'total_max_marks': normalized['total_max_marks'],
                    'overall_percentage': normalized['overall_percentage'],
                    'grade': normalized['grade'],
                    'questions': error_questions,
                    'filter_applied': f'Showing {len(error_questions)} questions with errors',
                    'status': 'success'
                }
            
            if len(all_questions) > 15:
                return {
                    'total_questions': normalized['total_questions'],
                    'total_marks_obtained': normalized['total_marks_obtained'],
                    'total_max_marks': normalized['total_max_marks'],
                    'overall_percentage': normalized['overall_percentage'],
                    'grade': normalized['grade'],
                    'questions': all_questions[:15],
                    'filter_applied': f'Showing first 15 of {len(all_questions)} questions',
                    'status': 'success'
                }
        
        return normalized

# ===========================
# CONFIG
# ===========================
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDp2W52XRwr68EVsLORgGLGiVtnmdgh4fQ")
    
    LANGUAGE_MAP = {
        "en": "en",
        "hi": "hi",
        "te": "te"
    }
    
    TLD_MAP = {
        "en": "com",
        "hi": "co.in",
        "te": "co.in"
    }
    
    UNIT_WEIGHTAGE = {
        "Class-6": {"Number Systems": 25, "Algebra": 20, "Geometry": 30, "Mensuration": 15, "Data Handling": 10},
        "Class-7": {"Integers": 20, "Fractions": 15, "Algebra": 25, "Geometry": 25, "Perimeter and Area": 15},
        "Class-8": {"Rational Numbers": 20, "Linear Equations": 25, "Quadrilaterals": 20, "Mensuration": 20, "Statistics": 15},
        "Class-9": {"Number Systems": 15, "Polynomials": 20, "Coordinate Geometry": 15, "Linear Equations": 20, "Geometry": 20, "Statistics": 10},
        "Class-10": {"Real Numbers": 20, "Polynomials": 15, "Linear Equations": 20, "Quadratic Equations": 20, "Trigonometry": 15, "Statistics": 10},
        "Class-11": {"Sets": 10, "Relations": 15, "Trigonometry": 25, "Limits": 20, "Mathematical Reasoning": 15, "Statistics": 15},
        "Class-12": {"Relations": 15, "Inverse Trigonometry": 15, "Matrices": 20, "Determinants": 15, "Calculus": 25, "Probability": 10}
    }
    
    FORMULAS_DESCRIPTION = """
    Mathematical formulas and concepts for analysis:
    - Quadratic formula: x = [-b ± √(b²-4ac)] / 2a
    - Distance formula: d = √[(x₂-x₁)² + (y₂-y₁)²]
    - Trigonometric identities: sin²θ + cos²θ = 1
    - Derivative rules: d/dx(xⁿ) = nxⁿ⁻¹
    - Integration: ∫xⁿdx = xⁿ⁺¹/(n+1) + C
    """

# ===========================
# UTILITY FUNCTIONS
# ===========================
def estimate_tokens(text: str) -> int:
    return len(text) // 4

def safe_json_data(filtered_data: Dict, max_tokens: int = 4000) -> str:
    json_str = json.dumps(filtered_data, indent=2, default=str, ensure_ascii=False)
    estimated = estimate_tokens(json_str)
    
    if estimated > max_tokens:
        if 'questions' in filtered_data:
            truncated = filtered_data.copy()
            truncated['questions'] = filtered_data['questions'][:10]
            truncated['_note'] = f"Showing first 10 of {len(filtered_data['questions'])} questions"
            return json.dumps(truncated, indent=2, default=str, ensure_ascii=False)
        
        if 'homework_data' in filtered_data:
            return json.dumps(
                HomeworkDataFilter.create_summary_only(filtered_data),
                indent=2,
                default=str,
                ensure_ascii=False
            )
    
    return json_str

def detect_class_from_student_id(student_id: str) -> str:
    try:
        if isinstance(student_id, str) and len(student_id) >= 2:
            first_char = student_id[0]
            second_char = student_id[1]
            
            if first_char == '1' and second_char in ['0', '1', '2']:
                return f"Class-1{second_char}"
            elif first_char in ['6', '7', '8', '9']:
                return f"Class-{first_char}"
        return "Class-8"
    except:
        return "Class-8"

def get_class_curriculum_info(class_name: str) -> tuple:
    chapters = Config.UNIT_WEIGHTAGE.get(class_name, {})
    return chapters, chapters

def get_priority_units(weightages: Dict, class_name: str) -> Dict:
    high = [unit for unit, weight in weightages.items() if weight >= 20]
    medium = [unit for unit, weight in weightages.items() if 15 <= weight < 20]
    low = [unit for unit, weight in weightages.items() if weight < 15]
    
    return {"high": high, "medium": medium, "low": low}

# ===========================
# DATA PROCESSING
# ===========================
def parse_json_safely(json_string: str) -> Optional[Any]:
    if not json_string or not json_string.strip():
        return None
    
    json_string = json_string.strip()
    if json_string.startswith('.'):
        json_string = json_string[1:].strip()
    
    if not json_string.startswith(('[', '{')):
        return None
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        try:
            import ast
            return ast.literal_eval(json_string)
        except:
            return None

def process_student_data(username: str, homework_json: str, exam_json: str) -> tuple:
    """Process both homework and exam data"""
    try:
        if not username or not username.strip():
            st.error("Please enter a valid username")
            return None, None, "", ""
        
        username = username.strip()
        detected_class = detect_class_from_student_id(username)
        
        # Process homework data
        homework_data = None
        if homework_json and homework_json.strip():
            hw_parsed = parse_json_safely(homework_json)
            if hw_parsed:
                if isinstance(hw_parsed, dict) and 'data' in hw_parsed:
                    homework_list = hw_parsed['data']
                elif isinstance(hw_parsed, list):
                    homework_list = hw_parsed
                else:
                    homework_list = []
                
                if homework_list:
                    homework_data = {
                        "student_id": username,
                        "student_name": f"Student {username}",
                        "class": detected_class.split('-')[1],
                        "homework_data": homework_list,
                        "status": "Homework data loaded successfully"
                    }
                    st.success(f"✅ Loaded {len(homework_list)} homework submissions!")
        
        # Process exam data
        exam_data = None
        if exam_json and exam_json.strip():
            exam_parsed = parse_json_safely(exam_json)
            if exam_parsed:
                exam_data = ExamDataFilter.normalize_exam_data(exam_parsed)
                
                if exam_data and 'exam_name' in exam_data:
                    st.success(f"✅ Loaded {exam_data.get('exam_name', 'exam')}! Grade: {exam_data.get('grade', 'N/A')} ({exam_data.get('overall_percentage', 0):.1f}%)")
                
                elif exam_data and 'questions' in exam_data:
                    st.success(f"✅ Loaded exam with {exam_data['total_questions']} questions! Grade: {exam_data['grade']}")
        
        # Create default if both are empty
        if not homework_data and not exam_data:
            homework_data = {
                "student_id": username,
                "student_name": f"Student {username}",
                "class": detected_class.split('-')[1],
                "homework_data": [],
                "status": "No data provided"
            }
            st.info(f"Welcome {username}! You're a {detected_class.split('-')[1]}th class student.")
        
        return homework_data, exam_data, username, detected_class
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, "", ""

# ===========================
# AI PROCESSING
# ===========================
def initialize_gemini():
    try:
        if not Config.GEMINI_API_KEY:
            st.error("GEMINI_API_KEY not configured")
            return None
        genai.configure(api_key=Config.GEMINI_API_KEY)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None

def is_casual_query(query: str) -> bool:
    casual_patterns = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'whats up', "what's up", 'sup', 'yo',
        'thanks', 'thank you', 'bye', 'goodbye', 'see you',
        'ok', 'okay', 'cool', 'nice', 'great', 'awesome'
    ]
    
    query_lower = query.lower().strip()
    
    if len(query_lower.split()) <= 3:
        for pattern in casual_patterns:
            if pattern in query_lower:
                return True
    
    return False

def get_casual_response(query: str, student_class: str) -> str:
    query_lower = query.lower().strip()
    class_number = student_class.split('-')[1] if '-' in student_class else student_class
    
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    if any(greet in query_lower for greet in greetings):
        return f"Hello! I'm your {class_number}th class mathematics assistant. How can I help you today?"
    
    thanks = ['thanks', 'thank you']
    if any(thank in query_lower for thank in thanks):
        return "You're welcome!"
    
    goodbye = ['bye', 'goodbye', 'see you']
    if any(bye in query_lower for bye in goodbye):
        return "Goodbye! Keep up the great work!"
    
    positive = ['ok', 'okay', 'cool', 'nice', 'great', 'awesome']
    if any(pos in query_lower for pos in positive):
        return "Great! What would you like to know?"
    
    return f"I'm here to help with your {class_number}th class mathematics. What would you like to know?"

def answer_query_with_gemini(query: str, homework_data: Any, exam_data: Any, 
                             memory: ConversationBufferMemory, 
                             image: Optional[Image.Image] = None) -> Optional[str]:
    """Intelligently route to homework or exam data based on query intent"""
    try:
        if not image and is_casual_query(query):
            return get_casual_response(query, st.session_state.student_class)
        
        gemini_model = initialize_gemini()
        if not gemini_model:
            return "Sorry, I'm having trouble connecting to the AI service."
        
        query_type = QueryIntentClassifier.classify_query_type(query)
        
        if query_type == 'exam' and exam_data:
            filtered_data = ExamDataFilter.apply_smart_exam_filter(query, exam_data)
            data_context = "EXAM DATA"
            
        elif query_type == 'homework' and homework_data:
            filtered_data = HomeworkDataFilter.apply_smart_filter(query, homework_data)
            data_context = "HOMEWORK DATA"
            
        elif query_type == 'general':
            if exam_data and homework_data:
                hw_filtered = HomeworkDataFilter.create_summary_only(homework_data)
                exam_filtered = ExamDataFilter.apply_smart_exam_filter(query, exam_data)
                exam_filtered['summary_only'] = True
                exam_filtered['questions'] = []
                
                filtered_data = {
                    'homework_summary': hw_filtered,
                    'exam_summary': exam_filtered,
                    'filter_applied': 'Combined homework and exam summary'
                }
                data_context = "COMBINED DATA"
            elif exam_data:
                filtered_data = ExamDataFilter.apply_smart_exam_filter(query, exam_data)
                data_context = "EXAM DATA"
            elif homework_data:
                filtered_data = HomeworkDataFilter.apply_smart_filter(query, homework_data)
                data_context = "HOMEWORK DATA"
            else:
                filtered_data = {}
                data_context = "NO DATA"
        else:
            filtered_data = homework_data or exam_data or {}
            data_context = "AVAILABLE DATA"
        
        json_data = safe_json_data(filtered_data, max_tokens=4000)
        
        student_class = st.session_state.get('student_class', 'Class-10')
        class_chapters, class_weightages = get_class_curriculum_info(student_class)
        priority_units = get_priority_units(class_weightages, student_class)
        class_number = student_class.split('-')[1]
        
        history_text = ""
        try:
            for msg in memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    history_text += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_text += f"Assistant: {msg.content}\n"
        except:
            history_text = ""
        
        curriculum_info = f"""
# CURRICULUM STRUCTURE FOR {student_class}

## Unit Weightages:
{json.dumps(class_weightages, indent=2)}

## Priority Analysis:
- High Priority (≥20%): {', '.join(priority_units['high']) if priority_units['high'] else 'None'}
- Medium Priority (15-19%): {', '.join(priority_units['medium']) if priority_units['medium'] else 'None'}
"""
        
        style_instruction = f"""
You are a friendly {class_number}th class mathematics performance assistant.

DATA CONTEXT: {data_context}
QUERY TYPE: {query_type}

Guidelines:
- If analyzing homework data, focus on homework performance, practice patterns, and homework-specific mistakes
- If analyzing exam data, focus on exam results, grades, test performance, and exam-specific errors
- If both available, provide holistic analysis comparing homework practice vs exam performance
- If no data, answer general {class_number}th class mathematics questions
- If image provided, analyze it and answer accordingly
- Focus on high-weightage units when giving recommendations
- Keep responses helpful and appropriate for {class_number}th class level

Filter Applied: {filtered_data.get('filter_applied', 'Standard data')}
"""
        
        prompt = f"""
{style_instruction}

Conversation History:
{history_text}

{curriculum_info}

Student Data ({data_context}):
{json_data}

{Config.FORMULAS_DESCRIPTION}

User Query: {query}
"""
        
        if image:
            try:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_part = {
                    "mime_type": "image/jpeg", 
                    "data": base64.b64encode(buffered.getvalue()).decode('utf-8')
                }
                response = gemini_model.generate_content([prompt, img_part])
            except:
                response = gemini_model.generate_content(prompt)
        else:
            response = gemini_model.generate_content(prompt)
        
        return response.text.strip()
        
    except Exception as e:
        st.error(f"AI processing failed: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

# ===========================
# AUDIO PROCESSING
# ===========================
def initialize_session_state():
    defaults = {
        "logged_in": False,
        "homework_data": None,
        "exam_data": None,
        "student_class": "Class-10",
        "student_id": None,
        "memory": ConversationBufferMemory(return_messages=True),
        "messages": [],
        "whisper_model": None,
        "language": "en",
        "processing": False,
        "last_audio_processed": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def initialize_whisper_model():
    if st.session_state.whisper_model is None:
        try:
            with st.spinner("Loading speech recognition model..."):
                st.session_state.whisper_model = WhisperModel(
                    "base", 
                    device="cpu", 
                    compute_type="int8"
                )
        except Exception as e:
            st.error(f"Failed to load Whisper model: {str(e)}")
            return False
    return True

def process_voice_input(audio_file) -> Optional[str]:
    if not initialize_whisper_model():
        return None
        
    try:
        if hasattr(audio_file, "read"):  
            audio_bytes = audio_file.read()
        else:
            audio_bytes = audio_file

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            audio_data, sample_rate = sf.read(temp_path)

            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            if sample_rate != 16000:
                num_samples = round(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                sample_rate = 16000

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as processed_audio:
                proc_path = processed_audio.name
                sf.write(proc_path, audio_data, sample_rate)

            segments, _ = st.session_state.whisper_model.transcribe(
                proc_path, 
                beam_size=5, 
                temperature=0.2
            )
            
            transcription = " ".join([seg.text for seg in segments])
            
            os.unlink(proc_path)
            return transcription.strip()
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        st.error(f"Voice processing failed: {str(e)}")
        return None

def clean_text_for_speech(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    text = re.sub(r'^[\s]*[•\-\*\+]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*[a-zA-Z]\.\s*', '', text, flags=re.MULTILINE)
    
    replacements = {
        '**': '', '*': '', '_': '', '•': '',
        '→': 'leads to', '✅': 'check', '❌': 'cross',
        '%': ' percent', '&': ' and ', '@': ' at ',
        '#': ' number ', '$': ' dollars ',
        '>': ' greater than ', '<': ' less than ',
        '=': ' equals ', '+': ' plus '
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if text and not text.endswith('.'):
        text += '.'
    
    return text

def text_to_speech(text: str, language: str = "en") -> Optional[bytes]:
    try:
        cleaned_text = clean_text_for_speech(text)
        
        if not cleaned_text:
            return None
        
        lang_code = Config.LANGUAGE_MAP.get(language, "en")
        tld = Config.TLD_MAP.get(language, "com")
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            tts = gTTS(text=cleaned_text, lang=lang_code, tld=tld, slow=False)
            tts.save(temp_path)
            
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            return audio_data
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception:
        return None

# ===========================
# UI COMPONENTS
# ===========================
def render_data_input_form():
    st.title("Student Performance Assistant")
    st.markdown("### Enter Username and Data")
    st.info("Your class will be automatically detected from your username")
    
    with st.form("data_input_form", clear_on_submit=False):
        username = st.text_input(
            "Username/Student ID", 
            help="Enter your student username or ID",
            placeholder="e.g., 9MME37"
        )
        
        st.markdown("#### Homework Data (Optional)")
        homework_json = st.text_area(
            "Homework Data (JSON format)", 
            help="Paste your homework data in JSON format.",
            placeholder='[{"creation_date": "27-09-25", "homework_id": "HW-27-9-25", "questions": [...]}]',
            height=150
        )
        
        st.markdown("#### Exam Results Data (Optional)")
        exam_json = st.text_area(
            "Exam Results (JSON format)",
            help="Paste exam results from API.",
            placeholder='{"student_name": "9MMB37", "results": [...]}',
            height=150
        )
        
        submitted = st.form_submit_button("Start Chat", use_container_width=True)
        
        if submitted and username:
            with st.spinner("Processing your data..."):
                hw_data, ex_data, student_id, detected_class = process_student_data(username, homework_json, exam_json)
                
                if hw_data is not None or ex_data is not None:
                    st.session_state.logged_in = True
                    st.session_state.homework_data = hw_data
                    st.session_state.exam_data = ex_data
                    st.session_state.student_id = student_id
                    st.session_state.student_class = detected_class
                    st.rerun()

def render_sidebar():
    student_class = st.session_state.get('student_class', 'Class-10')
    class_num = student_class.split('-')[1]
    
    st.sidebar.title(f"{class_num}th Class Assistant")
    st.sidebar.success(f"Student: {st.session_state.student_id}")
    st.sidebar.info(f"Class: {student_class}")
    
    # Show homework stats - DIRECT CALCULATION
    if st.session_state.homework_data and st.session_state.homework_data.get('homework_data'):
        homework_list = st.session_state.homework_data.get('homework_data', [])
        if homework_list:
            total_q = 0
            total_s = 0
            total_m = 0
            
            for hw in homework_list:
                questions = hw.get('questions', [])
                for q in questions:
                    total_q += 1
                    total_s += float(q.get('total_score', 0))
                    total_m += float(q.get('max_score', 0))
            
            st.sidebar.markdown("### Homework Stats")
            st.sidebar.metric("Total Homeworks", len(homework_list))
            st.sidebar.metric("Total Questions", total_q)
            st.sidebar.metric("Total Score", f"{total_s:.1f} / {total_m:.1f}")
            
            if total_m > 0:
                pct = (total_s / total_m) * 100
                st.sidebar.metric("Overall Score", f"{pct:.1f}%")
    
    # Show exam stats
    if st.session_state.exam_data:
        st.sidebar.markdown("### Exam Stats")
        
        if 'exam_name' in st.session_state.exam_data:
            st.sidebar.metric("Exam", st.session_state.exam_data.get('exam_name', 'N/A'))
            st.sidebar.metric("Marks", f"{st.session_state.exam_data.get('total_marks_obtained', 0)}/{st.session_state.exam_data.get('total_max_marks', 0)}")
            st.sidebar.metric("Percentage", f"{st.session_state.exam_data.get('overall_percentage', 0):.1f}%")
            st.sidebar.metric("Grade", st.session_state.exam_data.get('grade', 'N/A'))
        
        elif 'questions' in st.session_state.exam_data:
            st.sidebar.metric("Total Questions", st.session_state.exam_data.get('total_questions', 0))
            st.sidebar.metric("Score", f"{st.session_state.exam_data.get('overall_percentage', 0):.1f}%")
            st.sidebar.metric("Grade", st.session_state.exam_data.get('grade', 'N/A'))
        
        elif 'evaluation_results' in st.session_state.exam_data:
            num_exams = len(st.session_state.exam_data.get('evaluation_results', {}))
            st.sidebar.metric("Total Exams", num_exams)
    
    # Priority units
    if student_class in Config.UNIT_WEIGHTAGE:
        class_weightages = Config.UNIT_WEIGHTAGE[student_class]
        priority_units = get_priority_units(class_weightages, student_class)
        
        with st.sidebar.expander("📚 Unit Priorities", expanded=False):
            if priority_units['high']:
                st.write("**High Priority:**")
                for unit in priority_units['high']:
                    st.write(f"• {unit}")
            if priority_units['medium']:
                st.write("**Medium Priority:**")
                for unit in priority_units['medium']:
                    st.write(f"• {unit}")
    
    st.sidebar.markdown("---")
    st.sidebar.title("Language Settings")
    
    language_options = {
        "English": "en",
        "Hindi": "hi", 
        "Telugu": "te"
    }
    
    selected_lang = st.sidebar.selectbox(
        "Select language for voice responses:",
        options=list(language_options.keys()),
        index=0
    )
    
    st.session_state.language = language_options[selected_lang]
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Log Out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def render_chat_interface():
    st.title("Chat with Your AI Assistant")
    
    has_homework = st.session_state.homework_data and st.session_state.homework_data.get('homework_data')
    has_exam = st.session_state.exam_data and (
        st.session_state.exam_data.get('exam_name') or
        st.session_state.exam_data.get('questions')
    )
    
    if has_homework and has_exam:
        st.success("📚 Homework data and 📝 Exam data loaded! Ask me about either or both.")
    elif has_homework:
        st.success("📚 Homework data loaded! Ready to analyze your homework performance.")
    elif has_exam:
        st.success("📝 Exam data loaded! Ready to analyze your exam results.")
    else:
        class_num = st.session_state.student_class.split('-')[1]
        st.info(f"No data loaded. You can ask general {class_num}th class mathematics questions!")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "audio" in message:
                st.audio(message["audio"], format="audio/mp3")
    
    current_image = None
    uploaded_img = st.file_uploader(
        "Upload an image (optional)", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image to include in your question"
    )
    
    if uploaded_img:
        try:
            current_image = Image.open(uploaded_img).convert("RGB")
            st.image(current_image, caption="Uploaded image", width=200)
        except Exception as e:
            st.error(f"Failed to process image: {str(e)}")
    
    user_input = None
    audio_key = f"audio_input_{len(st.session_state.messages)}"
    audio_input_file = st.audio_input("Record your question", key=audio_key)
    
    if (audio_input_file and 
        not st.session_state.processing and 
        not st.session_state.get("last_audio_processed", False)):
        
        with st.spinner("Processing voice input..."):
            transcription = process_voice_input(audio_input_file)
            if transcription and transcription.strip():
                st.success(f"Voice input: {transcription}")
                user_input = transcription
                st.session_state.last_audio_processed = True
            else:
                st.warning("Could not transcribe audio. Please try again.")
    
    if not audio_input_file:
        st.session_state.last_audio_processed = False
    
    if not user_input:
        text_key = f"text_input_{len(st.session_state.messages)}"
        user_input = st.chat_input("Type your question here...", key=text_key)
    
    if user_input and not st.session_state.processing:
        st.session_state.processing = True
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.memory.chat_memory.add_user_message(user_input)
        
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = answer_query_with_gemini(
                    user_input,
                    st.session_state.homework_data,
                    st.session_state.exam_data,
                    st.session_state.memory,
                    current_image
                )
                
                if reply:
                    st.write(reply)
                    
                    try:
                        audio_reply = text_to_speech(reply, st.session_state.language)
                    except:
                        audio_reply = None
                    
                    message_data = {"role": "assistant", "content": reply}
                    if audio_reply:
                        st.audio(audio_reply, format="audio/mp3")
                        message_data["audio"] = audio_reply
                    
                    st.session_state.messages.append(message_data)
                    st.session_state.memory.chat_memory.add_ai_message(reply)
        
        st.session_state.processing = False
        st.session_state.last_audio_processed = False

# ===========================
# MAIN APPLICATION
# ===========================
def main():
    st.set_page_config(
        page_title="Student Performance Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    if not st.session_state.logged_in:
        render_data_input_form()
    else:
        render_sidebar()
        render_chat_interface()

if __name__ == "__main__":
    main()```
