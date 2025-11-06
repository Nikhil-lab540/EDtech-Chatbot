import streamlit as st
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
from PIL import Image
import base64
import io

# ===========================
# HOMEWORK DATA NORMALIZER
# ===========================
class HomeworkDataNormalizer:
    """Normalize different homework data formats to a standard structure"""
    
    @staticmethod
    def normalize_homework_item(homework_item: Dict) -> Dict:
        """
        Normalize a single homework submission to standard format
        Handles multiple input formats including:
        - NEW format: homework_id + submission_date + question.questions
        - OLD format: homework_id + creation_date + questions
        """
        normalized = {
            'homework_id': homework_item.get('homework_id', 'Unknown'),
            'creation_date': homework_item.get('submission_date', homework_item.get('creation_date', '')),
            'questions': []
        }
        
        # Handle NEW format: question -> questions (nested structure)
        if 'question' in homework_item and isinstance(homework_item['question'], dict):
            question_data = homework_item['question']
            
            # Extract grade-level info if present
            if 'grade' in question_data:
                normalized['grade'] = question_data['grade']
            if 'strengths' in question_data:
                normalized['strengths'] = question_data['strengths']
            if 'areas_for_improvement' in question_data:
                normalized['areas_for_improvement'] = question_data['areas_for_improvement']
            if 'overall_percentage' in question_data:
                normalized['overall_percentage'] = question_data['overall_percentage']
            
            # Extract questions array
            questions_array = question_data.get('questions', [])
            
        # Handle OLD format: questions directly at top level
        elif 'questions' in homework_item:
            questions_array = homework_item['questions']
        else:
            questions_array = []
        
        # Normalize each question
        for q in questions_array:
            # Handle concept_required which can be null, string, or list
            concepts = q.get('concept_required', q.get('concepts_required', []))
            if concepts is None:
                concepts = []
            elif isinstance(concepts, str):
                concepts = [concepts] if concepts.strip() else []
            elif not isinstance(concepts, list):
                concepts = []
            
            # Handle comment and correction_comment which can be null or "None" string
            comment = q.get('comment', q.get('gap_analysis', ''))
            if comment is None or comment == 'None':
                comment = ''
            
            correction_comment = q.get('correction_comment', q.get('mistakes_made', ''))
            if correction_comment is None or correction_comment == 'None':
                correction_comment = ''
            
            # Handle answer_category which can be null or "None" string
            answer_category = q.get('answer_category', q.get('error_type', 'no_error'))
            if answer_category is None or answer_category == 'None':
                answer_category = 'no_error'
            
            normalized_q = {
                'question_id': q.get('question_id', q.get('question_number', q.get('question', 'Unknown'))),
                'question_text': q.get('question_text', ''),
                'total_score': float(q.get('total_score', q.get('total_marks_obtained', 0))),
                'max_score': float(q.get('max_score', q.get('max_marks', 0))),
                'answer_category': answer_category,
                'comment': comment,
                'correction_comment': correction_comment,
                'topic': '',
                'concepts_required': concepts
            }
            
            # Derive topic from concepts_required
            if normalized_q['concepts_required'] and len(normalized_q['concepts_required']) > 0:
                primary_concept = normalized_q['concepts_required'][0]
                # Take first part before comma if exists
                normalized_q['topic'] = primary_concept.split(',')[0].strip()
            
            # If still no topic, use existing or default
            if not normalized_q['topic']:
                normalized_q['topic'] = q.get('topic', 'General Mathematics')
            
            normalized['questions'].append(normalized_q)
        
        return normalized
    
    @staticmethod
    def normalize_homework_list(homework_list: List[Dict]) -> List[Dict]:
        """Normalize a list of homework items"""
        return [HomeworkDataNormalizer.normalize_homework_item(hw) for hw in homework_list]

# ===========================
# QUERY INTENT CLASSIFIER
# ===========================

class QueryIntentClassifier:
    """Simplified query classifier with pattern matching"""
    
    @staticmethod
    def classify_query_type(query: str) -> Literal['homework', 'exam', 'general']:
        """
        Classify query as homework-related, exam-related, or general
        Returns: 'homework', 'exam', or 'general'
        """
        if not query or not query.strip():
            return 'general'
        
        query_lower = query.lower().strip()
        
        # Strong exam indicators (check first - more specific)
        exam_keywords = [
            'exam result', 'exam score', 'exam grade', 'test result',
            'exam analysis', 'examination', 'question paper', 
            'final exam', 'midterm', 'test score'
        ]
        
        # Strong homework indicators
        homework_keywords = [
            'homework', 'assignment', 'hw ', 'hw-', 'hw_',
            'daily work', 'practice problem', 'homework submission'
        ]
        
        # Pattern matching for hw followed by number
        hw_pattern = re.compile(r'\bhw[-_]?\d+\b', re.IGNORECASE)
        
        # Count matches for each category
        exam_matches = sum(1 for keyword in exam_keywords if keyword in query_lower)
        homework_matches = sum(1 for keyword in homework_keywords if keyword in query_lower)
        
        # Add bonus for hw pattern (hw1, hw-2, hw_3)
        if hw_pattern.search(query_lower):
            homework_matches += 2
        
        # Decide based on match count
        if exam_matches > homework_matches:
            return 'exam'
        
        if homework_matches > exam_matches:
            return 'homework'
        
        # If tied or no matches
        if exam_matches > 0 or homework_matches > 0:
            # Check which appears first in query
            exam_pos = min((query_lower.find(kw) for kw in exam_keywords if kw in query_lower), default=999)
            hw_pos = min((query_lower.find(kw) for kw in homework_keywords if kw in query_lower), default=999)
            
            if exam_pos < hw_pos:
                return 'exam'
            elif hw_pos < exam_pos:
                return 'homework'
        
        return 'general'


# ===========================
# HOMEWORK DATA FILTER
# ===========================

class HomeworkDataFilter:
    """Filter student homework data based on query context to reduce token usage"""
    
    @staticmethod
    def normalize_homework_data(student_data: Dict) -> Dict:
        """
        Normalize data structure to handle different formats
        - Handles top-level 'data' key wrapper
        - Supports both 'submission_date' and 'creation_date'
        - Normalizes question structure with HomeworkDataNormalizer
        """
        if not student_data:
            return student_data
        
        # Handle top-level 'data' key if present
        if 'data' in student_data and 'homework_data' not in student_data:
            homework_list = student_data['data']
        else:
            homework_list = student_data.get('homework_data', [])
        
        # Normalize each homework item using HomeworkDataNormalizer
        normalized_homeworks = HomeworkDataNormalizer.normalize_homework_list(homework_list)
        
        # Return normalized structure
        return {
            'student_name': student_data.get('student_name', ''),
            'class': student_data.get('class', ''),
            'homework_data': normalized_homeworks,
            'status': student_data.get('status', 'Data loaded')
        }
    
    @staticmethod
    def parse_date(date_str: str) -> datetime:
        """
        Parse date from multiple formats
        Returns datetime.min if parsing fails
        """
        if not date_str:
            return datetime.min
        
        try:
            # ISO format: 2025-06-23T06:31:07Z or 2025-08-29T11:52:19.615973Z
            if 'T' in date_str or ('-' in date_str and len(date_str) > 10):
                # Handle both with and without microseconds
                date_str_clean = date_str.replace('Z', '+00:00')
                return datetime.fromisoformat(date_str_clean)
        except:
            pass
        
        try:
            # Short format: 27-09-25
            return datetime.strptime(date_str, '%d-%m-%y')
        except:
            pass
        
        try:
            # Standard format: 2025-09-27
            return datetime.strptime(date_str, '%Y-%m-%d')
        except:
            return datetime.min
    
    @staticmethod
    def detect_query_intent(query: str) -> str:
        """Detect what type of information the user is asking for"""
        query_lower = query.lower()
        
        # Greeting detection
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
        
        # Recent performance queries
        if any(word in query_lower for word in ['recent', 'latest', 'last', 'current']):
            return 'recent_performance'
        
        # Specific homework queries
        if any(word in query_lower for word in ['homework', 'hw-', 'hw_', 'hw ', 'assignment']):
            return 'specific_homework'
        
        # Topic/chapter queries
        topic_keywords = [
            'chapter', 'topic', 'unit', 'algebra', 'geometry', 'calculus', 
            'trigonometry', 'statistics', 'rational', 'square', 'cube', 
            'polynomial', 'equation', 'quadratic', 'derivative', 'integration', 
            'probability', 'coordinate', 'mensuration', 'data handling',
            'factor', 'heron', 'identity', 'rationalize', 'expansion'
        ]
        if any(word in query_lower for word in topic_keywords):
            return 'topic_analysis'
        
        # Struggling/weak areas
        if any(word in query_lower for word in ['struggle', 'weak', 'improve', 'error', 'mistake']):
            return 'weakness_analysis'
        
        # Overall performance
        if any(word in query_lower for word in ['overall', 'total', 'summary', 'report']):
            return 'overall_summary'
        
        # Concept/explanation queries
        if any(word in query_lower for word in ['what is', 'explain', 'how to', 'solve', 'formula']):
            return 'concept_explanation'
        
        return 'general'
    
    @staticmethod
    def filter_recent_n_homeworks(student_data: Dict, n: int = 3) -> Dict:
        """Get only the N most recent homework submissions"""
        student_data = HomeworkDataFilter.normalize_homework_data(student_data)
        
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        homework_list = student_data['homework_data']
        
        # Sort by date (most recent first)
        sorted_hw = sorted(
            homework_list, 
            key=lambda x: HomeworkDataFilter.parse_date(x.get('creation_date', '')),
            reverse=True
        )
        
        filtered_data = student_data.copy()
        filtered_data['homework_data'] = sorted_hw[:n]
        filtered_data['filter_applied'] = f'Showing {min(n, len(sorted_hw))} most recent homeworks out of {len(sorted_hw)} total'
        
        return filtered_data
    
    @staticmethod
    def filter_by_topic(student_data: Dict, topic_keyword: str) -> Dict:
        """Filter homework data to include only specific topic/chapter"""
        student_data = HomeworkDataFilter.normalize_homework_data(student_data)
        
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        filtered_homeworks = []
        total_questions = 0
        
        for homework in student_data['homework_data']:
            filtered_questions = [
                q for q in homework.get('questions', [])
                if topic_keyword.lower() in q.get('topic', '').lower() or
                   any(topic_keyword.lower() in str(concept).lower() 
                       for concept in q.get('concepts_required', []))
            ]
            
            if filtered_questions:
                hw_copy = homework.copy()
                hw_copy['questions'] = filtered_questions
                filtered_homeworks.append(hw_copy)
                total_questions += len(filtered_questions)
        
        filtered_data = student_data.copy()
        filtered_data['homework_data'] = filtered_homeworks
        filtered_data['filter_applied'] = f'Filtered by topic: {topic_keyword} ({len(filtered_homeworks)} homeworks, {total_questions} questions)'
        
        return filtered_data
    
    @staticmethod
    def filter_errors_only(student_data: Dict) -> Dict:
        """Get only questions with errors (conceptual, logical, calculation)"""
        student_data = HomeworkDataFilter.normalize_homework_data(student_data)
        
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        filtered_homeworks = []
        total_errors = 0
        
        for homework in student_data['homework_data']:
            error_questions = [
                q for q in homework.get('questions', [])
                if q.get('answer_category', '').lower() not in ['no_error', 'correct', 'none']
                and q.get('total_score', 0) < q.get('max_score', 1)
            ]
            
            if error_questions:
                hw_copy = homework.copy()
                hw_copy['questions'] = error_questions
                filtered_homeworks.append(hw_copy)
                total_errors += len(error_questions)
        
        filtered_data = student_data.copy()
        filtered_data['homework_data'] = filtered_homeworks
        filtered_data['filter_applied'] = f'Showing only questions with errors ({total_errors} error questions across {len(filtered_homeworks)} homeworks)'
        
        return filtered_data
    
    @staticmethod
    def create_summary_only(student_data: Dict) -> Dict:
        """Create aggregated summary without question details"""
        student_data = HomeworkDataFilter.normalize_homework_data(student_data)
        
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        summary = {
            'student_name': student_data.get('student_name'),
            'class': student_data.get('class'),
            'total_homeworks': len(student_data.get('homework_data', [])),
            'homework_scores': [],
            'topic_summary': {},
            'error_summary': {}
        }
        
        # Aggregate by homework
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
        
        # Aggregate by topic
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
        
        # Calculate percentages
        for topic in summary['topic_summary']:
            data = summary['topic_summary'][topic]
            if data['max_score'] > 0:
                data['percentage'] = round((data['score'] / data['max_score'] * 100), 2)
            else:
                data['percentage'] = 0
        
        # Aggregate errors
        for hw in student_data.get('homework_data', []):
            for q in hw.get('questions', []):
                error_type = q.get('answer_category', 'no_error')
                if error_type.lower() not in ['no_error', 'correct', 'none']:
                    summary['error_summary'][error_type] = summary['error_summary'].get(error_type, 0) + 1
        
        summary['filter_applied'] = 'Summary view only (detailed questions omitted to save tokens)'
        
        return summary
    
    @staticmethod
    def filter_specific_homework(student_data: Dict, homework_id: str) -> Dict:
        """Get data for a specific homework assignment"""
        student_data = HomeworkDataFilter.normalize_homework_data(student_data)
        
        if not student_data or not student_data.get('homework_data'):
            return student_data
        
        specific_hw = [
            hw for hw in student_data['homework_data']
            if homework_id.upper() in hw.get('homework_id', '').upper()
        ]
        
        total_questions = sum(len(hw.get('questions', [])) for hw in specific_hw)
        
        filtered_data = student_data.copy()
        filtered_data['homework_data'] = specific_hw
        filtered_data['filter_applied'] = f'Showing homework: {homework_id} ({len(specific_hw)} found with {total_questions} questions)'
        
        return filtered_data
    
    @staticmethod
    def apply_smart_filter(query: str, student_data: Dict) -> Dict:
        """
        Automatically apply the best filter based on query intent
        Main function to call before sending data to LLM
        """
        intent = HomeworkDataFilter.detect_query_intent(query)
        
        # Normalize data first
        student_data = HomeworkDataFilter.normalize_homework_data(student_data)
        
        # No filtering needed for these intents
        if intent in ['greeting', 'concept_explanation']:
            return {
                'student_name': student_data.get('student_name'),
                'class': student_data.get('class'),
                'filter_applied': 'No homework data needed for this query'
            }
        
        # Apply appropriate filter based on intent
        if intent == 'recent_performance':
            return HomeworkDataFilter.filter_recent_n_homeworks(student_data, n=3)
        
        elif intent == 'weakness_analysis':
            return HomeworkDataFilter.filter_errors_only(student_data)
        
        elif intent == 'overall_summary':
            return HomeworkDataFilter.create_summary_only(student_data)
        
        elif intent == 'specific_homework':
            # Try to extract homework ID from query
            hw_pattern = r'hw[-_\s]?(\d+|[a-zA-Z0-9\-]+)'
            match = re.search(hw_pattern, query.lower())
            if match:
                hw_id = match.group(0).upper()
                return HomeworkDataFilter.filter_specific_homework(student_data, hw_id)
            # If no specific ID found, show recent
            return HomeworkDataFilter.filter_recent_n_homeworks(student_data, n=5)
        
        elif intent == 'topic_analysis':
            # Extract topic keyword
            topic_keywords = [
                'algebra', 'geometry', 'calculus', 'trigonometry', 
                'statistics', 'rational', 'square', 'cube', 'polynomial',
                'equation', 'mensuration', 'data handling', 'quadratic',
                'derivative', 'integration', 'probability', 'coordinate',
                'factor', 'heron', 'identity', 'rationalize', 'expansion'
            ]
            for keyword in topic_keywords:
                if keyword in query.lower():
                    return HomeworkDataFilter.filter_by_topic(student_data, keyword)
            # If no specific topic, show summary
            return HomeworkDataFilter.create_summary_only(student_data)
        
        else:  # general queries
            # For general queries, provide recent 3 homeworks
            return HomeworkDataFilter.filter_recent_n_homeworks(student_data, n=3)



# ===========================
# EXAM DATA FILTER
# ===========================

class ExamDataFilter:
    """Filter exam result data based on query context"""
    
    @staticmethod
    def normalize_exam_data(exam_data: Dict) -> Dict:
        """
        Normalize exam results data structure
        Supports multiple formats:
        1. NEW format: 'results' array with strengths/areas_for_improvement
        2. OLD format: 'question_data' with detailed question-level analysis
        """
        if not exam_data:
            return exam_data
        
        # NEW FORMAT: results array with exam metadata
        if isinstance(exam_data, dict) and 'results' in exam_data:
            results = exam_data.get('results', [])
            
            if not results:
                return {
                    'status': 'error',
                    'message': 'No exam results found'
                }
            
            # Get first exam result
            exam_result = results[0]
            
            return {
                'student_name': exam_data.get('student_name', ''),
                'roll_number': exam_data.get('roll_number', ''),
                'exam_name': exam_result.get('exam_name', 'Unknown Exam'),
                'exam_type': exam_result.get('exam_type', 'N/A'),
                'class_section': exam_result.get('class_section', ''),
                'total_marks_obtained': exam_result.get('total_marks_obtained', 0),
                'total_max_marks': exam_result.get('total_max_marks', 0),
                'overall_percentage': exam_result.get('overall_percentage', 0),
                'grade': exam_result.get('grade', 'N/A'),
                'strengths': exam_result.get('strengths', []),
                'areas_for_improvement': exam_result.get('areas_for_improvement', []),
                'total_exams': len(results),
                'data_format': 'results_based',
                'status': 'success'
            }
        
        # OLD FORMAT: question_data with detailed questions
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
            
            # Calculate grade
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
                'data_format': 'question_based',
                'status': 'success'
            }
        
        # Unknown format
        return {
            'status': 'error',
            'message': 'Unknown exam data format',
            'original_data': exam_data
        }
    
    @staticmethod
    def detect_exam_query_intent(query: str) -> Literal['summary', 'strengths', 'weaknesses', 'detailed', 'general']:
        """Detect exam query type"""
        query_lower = query.lower()
        
        # Summary queries
        if any(word in query_lower for word in ['summary', 'grade', 'percentage', 'overall', 'total']):
            return 'summary'
        
        # Strength queries
        if any(word in query_lower for word in ['strength', 'good', 'correct', 'right', 'what did i do well', 'where did i do well']):
            return 'strengths'
        
        # Weakness queries
        if any(word in query_lower for word in ['weakness', 'improve', 'improvement', 'weak', 'mistake', 'error', 'wrong', 'where did i go wrong']):
            return 'weaknesses'
        
        # Detailed analysis
        if any(word in query_lower for word in ['detail', 'analysis', 'breakdown', 'complete']):
            return 'detailed'
        
        return 'general'
    
    @staticmethod
    def apply_smart_exam_filter(query: str, exam_data: Dict) -> Dict:
        """
        Apply intelligent filtering based on query intent
        Main function to call before sending exam data to LLM
        """
        # Normalize first
        normalized = ExamDataFilter.normalize_exam_data(exam_data)
        
        if not normalized or normalized.get('status') == 'error':
            return normalized
        
        # Detect intent
        intent = ExamDataFilter.detect_exam_query_intent(query)
        
        # Check data format
        data_format = normalized.get('data_format', 'unknown')
        
        # RESULTS-BASED FORMAT (NEW)
        if data_format == 'results_based':
            
            if intent == 'summary':
                num_strengths = len(normalized.get('strengths', []))
                num_improvements = len(normalized.get('areas_for_improvement', []))
                
                return {
                    'student_name': normalized.get('student_name'),
                    'roll_number': normalized.get('roll_number'),
                    'exam_name': normalized.get('exam_name'),
                    'exam_type': normalized.get('exam_type'),
                    'total_marks_obtained': normalized.get('total_marks_obtained'),
                    'total_max_marks': normalized.get('total_max_marks'),
                    'overall_percentage': normalized.get('overall_percentage'),
                    'grade': normalized.get('grade'),
                    'filter_applied': f'Exam summary only ({num_strengths} strengths, {num_improvements} areas for improvement identified)',
                    'status': 'success'
                }
            
            elif intent == 'strengths':
                strengths = normalized.get('strengths', [])
                return {
                    'student_name': normalized.get('student_name'),
                    'exam_name': normalized.get('exam_name'),
                    'overall_percentage': normalized.get('overall_percentage'),
                    'grade': normalized.get('grade'),
                    'strengths': strengths,
                    'filter_applied': f'Showing only strengths ({len(strengths)} identified)',
                    'status': 'success'
                }
            
            elif intent == 'weaknesses':
                improvements = normalized.get('areas_for_improvement', [])
                return {
                    'student_name': normalized.get('student_name'),
                    'exam_name': normalized.get('exam_name'),
                    'overall_percentage': normalized.get('overall_percentage'),
                    'grade': normalized.get('grade'),
                    'areas_for_improvement': improvements,
                    'filter_applied': f'Showing only areas for improvement ({len(improvements)} identified)',
                    'status': 'success'
                }
            
            elif intent == 'detailed':
                # Return full data for detailed analysis
                return normalized
            
            else:  # general
                # Return balanced view
                return normalized
        
        # QUESTION-BASED FORMAT (OLD)
        elif data_format == 'question_based':
            all_questions = normalized.get('questions', [])
            
            if intent == 'summary':
                return {
                    'total_questions': normalized['total_questions'],
                    'total_marks_obtained': normalized['total_marks_obtained'],
                    'total_max_marks': normalized['total_max_marks'],
                    'overall_percentage': normalized['overall_percentage'],
                    'grade': normalized['grade'],
                    'questions': [],  # Empty for summary
                    'filter_applied': f'Exam summary only ({normalized["total_questions"]} questions total)',
                    'status': 'success'
                }
            
            elif intent == 'strengths':
                correct_questions = [
                    q for q in all_questions
                    if q.get('error_type', '').lower() == 'no_error'
                ]
                return {
                    'total_questions': normalized['total_questions'],
                    'total_marks_obtained': normalized['total_marks_obtained'],
                    'total_max_marks': normalized['total_max_marks'],
                    'overall_percentage': normalized['overall_percentage'],
                    'grade': normalized['grade'],
                    'questions': correct_questions,
                    'filter_applied': f'Showing only correct questions ({len(correct_questions)} out of {len(all_questions)})',
                    'status': 'success'
                }
            
            elif intent == 'weaknesses':
                error_questions = [
                    q for q in all_questions
                    if q.get('error_type', '').lower() not in ['no_error', '']
                ]
                return {
                    'total_questions': normalized['total_questions'],
                    'total_marks_obtained': normalized['total_marks_obtained'],
                    'total_max_marks': normalized['total_max_marks'],
                    'overall_percentage': normalized['overall_percentage'],
                    'grade': normalized['grade'],
                    'questions': error_questions,
                    'filter_applied': f'Showing only questions with errors ({len(error_questions)} out of {len(all_questions)})',
                    'status': 'success'
                }
            
            elif intent == 'detailed':
                # Limit to first 15 questions if too many
                if len(all_questions) > 15:
                    return {
                        'total_questions': normalized['total_questions'],
                        'total_marks_obtained': normalized['total_marks_obtained'],
                        'total_max_marks': normalized['total_max_marks'],
                        'overall_percentage': normalized['overall_percentage'],
                        'grade': normalized['grade'],
                        'questions': all_questions[:15],
                        'filter_applied': f'Showing first 15 of {len(all_questions)} questions (to reduce token usage)',
                        'status': 'success'
                    }
                return normalized
            
            else:  # general
                # Limit questions for general queries
                if len(all_questions) > 10:
                    return {
                        'total_questions': normalized['total_questions'],
                        'total_marks_obtained': normalized['total_marks_obtained'],
                        'total_max_marks': normalized['total_max_marks'],
                        'overall_percentage': normalized['overall_percentage'],
                        'grade': normalized['grade'],
                        'questions': all_questions[:10],
                        'filter_applied': f'Showing first 10 of {len(all_questions)} questions',
                        'status': 'success'
                    }
                return normalized
        
        # Unknown format - return as is
        return normalized

# ===========================
# CONFIG
# ===========================
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDp2W52XRwr68EVsLORgGLGiVtnmdgh4fQ")
    
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
Mathematical formulas relevant for problem solving:
- Distance formula: d = √[(x₂-x₁)² + (y₂-y₁)²]
- Area of circle: A = πr²
- Pythagorean theorem: a² + b² = c²
- Factor Theorem: If p(a)=0, then (x-a) is a factor
- Heron's Formula: Area = √[s(s-a)(s-b)(s-c)]
- Algebraic Identities: (a+b)³, (a-b)³, a³+b³+c³-3abc
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

def process_student_data(student_name: str, class_name: str, homework_json: str, exam_json: str) -> tuple:
    """Process both homework and exam data"""
    try:
        if not student_name or not student_name.strip():
            st.error("Please enter a valid student name")
            return None, None, ""
        
        student_name = student_name.strip()
        class_name = class_name.strip() if class_name else "10"
        
        # Normalize class name format
        if not class_name.startswith('Class-'):
            class_name = f"Class-{class_name}"
        
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
                        "student_name": student_name,
                        "class": class_name.split('-')[1] if '-' in class_name else class_name,
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
                "student_name": student_name,
                "class": class_name.split('-')[1] if '-' in class_name else class_name,
                "homework_data": [],
                "status": "No data provided"
            }
            st.info(f"Welcome {student_name}! You're a {class_name} student.")
        
        return homework_data, exam_data, class_name
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, ""

# ===========================
# MODE DETECTION & AI PROCESSING
# ===========================

def detect_query_mode(query: str, image: Optional[Image.Image]) -> Literal['TUTOR_MODE', 'ANALYSIS_MODE', 'CASUAL_MODE']:
    """
    Determine the interaction mode based on query and image
    
    TUTOR_MODE: Solving math problems (with or without image)
    ANALYSIS_MODE: Analyzing student performance data
    CASUAL_MODE: Simple greetings, no AI needed
    """
    query_lower = query.lower().strip()
    
    # Check casual first (fastest path)
    casual_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 
                      'good evening', 'thanks', 'thank you', 'bye', 'goodbye']
    
    if len(query_lower.split()) <= 3:
        if any(pattern in query_lower for pattern in casual_patterns):
            # But if image is provided, it's TUTOR_MODE
            if image:
                return 'TUTOR_MODE'
            return 'CASUAL_MODE'
    
    # CHECK PERFORMANCE KEYWORDS FIRST (higher priority than solve keywords)
    performance_keywords = [
        'my performance', 'my progress', 'how did i do', 'my mistakes',
        'my homework', 'my exam', 'my score', 'my grade', 'my result',
        'recent performance', 'recent progress', 'recent homework', 'recent exam',
        'where am i weak', 'what should i improve', 'my strengths', 'my weaknesses',
        'analyze my', 'show my results', 'how am i doing', 'my work'
    ]
    
    if any(keyword in query_lower for keyword in performance_keywords):
        return 'ANALYSIS_MODE'
    
    # Check for problem-solving intent (but exclude performance questions)
    solve_keywords = [
        'solve', 'find', 'calculate', 'compute', 'determine',
        'how to solve', 'show me how', 'help me solve', 'work out',
        'find the value', 'find r', 'find x', 'find the answer',
        'what is the value', 'what is the solution'
    ]
    
    if any(keyword in query_lower for keyword in solve_keywords):
        return 'TUTOR_MODE'
    
    # If image provided with short query, likely problem-solving
    if image and len(query_lower.split()) <= 5:
        return 'TUTOR_MODE'
    
    # Special case: "what is" + math term = TUTOR_MODE
    if 'what is' in query_lower:
        math_terms = ['equation', 'formula', 'theorem', 'proof', 'derivative', 
                     'integral', 'root', 'solution', 'area', 'volume', 'angle']
        if any(term in query_lower for term in math_terms):
            return 'TUTOR_MODE'
    
    # Default to analysis mode (safer for student data queries)
    return 'ANALYSIS_MODE'

def build_tutor_prompt(query: str, class_number: str) -> str:
    """Build prompt for problem-solving mode"""
    
    prompt = f"""You are a JEE ADVANCED mathematics tutor.

⚠️ WARNING: These problems have traps. Study these examples:



EXAMPLE 2 - Triangle:  
Trap: Assuming standard 30-60-90 triangle
Reality: Check given constraints → angles are different

EXAMPLE 3 - Parabola:
Trap: Assuming vertex at origin
Reality: Check tangency conditions → vertex elsewhere

PATTERN: The "obvious" geometric answer is wrong. Always verify with ALL constraints.

Student Question: {query}

Use 5-step approach:"""
    
    return prompt

def build_analysis_prompt(
    query: str,
    filtered_data: dict,
    data_context: str,
    query_type: str,
    student_class: str,
    curriculum_info: str,
    history_text: str
) -> str:
    """Build prompt for performance analysis mode - FULL CONTEXT"""
    
    class_number = student_class.split('-')[1] if '-' in student_class else student_class
    json_data = safe_json_data(filtered_data, max_tokens=4000)
    
    prompt = f"""You are a friendly {class_number}th class mathematics performance assistant.

DATA CONTEXT: {data_context}
QUERY TYPE: {query_type}

GUIDELINES:
- Analyze homework/exam data to provide insights on performance
- Focus on patterns, strengths, and areas needing improvement
- Give actionable recommendations
- Keep responses helpful and age-appropriate for {class_number}th class

Filter Applied: {filtered_data.get('filter_applied', 'Standard data')}

{f'Conversation History:\n{history_text}\n' if history_text else ''}

{curriculum_info}

Student Data ({data_context}):
{json_data}

User Query: {query}

Provide helpful analysis:"""
    
    return prompt

def initialize_gemini():
    try:
        if not Config.GEMINI_API_KEY:
            st.error("GEMINI_API_KEY not configured")
            return None
        genai.configure(api_key=Config.GEMINI_API_KEY)
        return genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                top_p=0.8,
                top_k=40
            )
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None

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
    
    return f"I'm here to help with your {class_number}th class mathematics. What would you like to know?"

def answer_query_with_gemini(query: str, homework_data: Any, exam_data: Any, 
                             memory: ConversationBufferMemory, 
                             image: Optional[Image.Image] = None) -> Any:
    """Route query to appropriate mode and generate response with streaming"""
    try:
        # Detect mode
        mode = detect_query_mode(query, image)
        
        # Handle casual mode
        if mode == 'CASUAL_MODE':
            return get_casual_response(query, st.session_state.student_class)
        
        # Initialize Gemini
        gemini_model = initialize_gemini()
        if not gemini_model:
            return "Sorry, I'm having trouble connecting to the AI service."
        
        student_class = st.session_state.get('student_class', 'Class-10')
        class_number = student_class.split('-')[1] if '-' in student_class else student_class
        
        # TUTOR MODE - Problem Solving
        if mode == 'TUTOR_MODE':
            prompt = build_tutor_prompt(query, class_number)
            
            # Add formulas for math problems
            prompt += f"\n\n{Config.FORMULAS_DESCRIPTION}"
        
        # ANALYSIS MODE - Performance Analysis
        else:  # ANALYSIS_MODE
            # Classify and filter data
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
                    if 'questions' in exam_filtered:
                        exam_filtered['questions'] = []
                    filtered_data = {
                        'homework_summary': hw_filtered,
                        'exam_summary': exam_filtered,
                        'filter_applied': 'Combined summary'
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
            
            # Get conversation history (limit to last 10 messages)
            history_text = ""
            try:
                messages = memory.chat_memory.messages[-10:]
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        history_text += f"User: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        history_text += f"Assistant: {msg.content[:200]}...\n"
            except:
                history_text = ""
            
            # Get curriculum info
            class_chapters, class_weightages = get_class_curriculum_info(student_class)
            priority_units = get_priority_units(class_weightages, student_class)
            
            curriculum_info = f"""
# CURRICULUM FOR {student_class}

High Priority Units (≥20%): {', '.join(priority_units['high']) if priority_units['high'] else 'None'}
Medium Priority (15-19%): {', '.join(priority_units['medium']) if priority_units['medium'] else 'None'}
"""
            
            prompt = build_analysis_prompt(
                query, filtered_data, data_context, query_type,
                student_class, curriculum_info, history_text
            )
        
        # Make API call with streaming
        try:
            if image:
                try:
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_part = {
                        "mime_type": "image/jpeg", 
                        "data": base64.b64encode(buffered.getvalue()).decode('utf-8')
                    }
                    response = gemini_model.generate_content([prompt, img_part], stream=True)
                except Exception as e:
                    st.warning(f"Image processing failed: {str(e)}")
                    response = gemini_model.generate_content(prompt, stream=True)
            else:
                response = gemini_model.generate_content(prompt, stream=True)
            
            # Return the streaming response object
            return response
        except Exception as e:
            # If streaming fails completely, try non-streaming as fallback
            st.warning("Streaming failed, using standard response mode...")
            try:
                if image:
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_part = {
                        "mime_type": "image/jpeg", 
                        "data": base64.b64encode(buffered.getvalue()).decode('utf-8')
                    }
                    response = gemini_model.generate_content([prompt, img_part], stream=False)
                else:
                    response = gemini_model.generate_content(prompt, stream=False)
                return response.text.strip()
            except Exception as e2:
                st.error(f"Both streaming and non-streaming failed: {str(e2)}")
                return None
        
    except Exception as e:
        st.error(f"AI processing failed: {str(e)}")
        return None

# ===========================
# AUDIO PROCESSING
# ===========================
def initialize_session_state():
    defaults = {
        "logged_in": False,
        "homework_data": None,
        "exam_data": None,
        "student_class": "Class-10",
        "student_name": None,
        "memory": ConversationBufferMemory(return_messages=True),
        "messages": [],
        "whisper_model": None,
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

# ===========================
# UI COMPONENTS
# ===========================
def render_data_input_form():
    st.title("Student Performance Assistant")
    st.markdown("### Login")
    
    with st.form("data_input_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            student_name = st.text_input(
                "Student Name *",
                help="Enter your full name",
                placeholder="e.g., Hafizshahed"
            )
        
        with col2:
            class_name = st.text_input(
                "Class Name *",
                help="Enter your class (e.g., 10, Class-10)",
                placeholder="e.g., 10 or Class-10"
            )
        
        st.markdown("#### Homework Data (Optional)")
        st.markdown("**Supports new format** with `question.questions` structure!")
        homework_json = st.text_area(
            "Homework Data (JSON format)", 
            help="Paste your homework data in JSON format. Supports both old and new formats!",
            placeholder='{"data": [{"homework_id": "HW-01", "question": {"questions": [...]}}]} or old format',
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
        
        if submitted:
            if not student_name or not class_name:
                st.error("Please fill in all required fields (Student Name and Class)")
            else:
                with st.spinner("Processing your data..."):
                    hw_data, ex_data, sclass = process_student_data(
                        student_name, class_name, homework_json, exam_json
                    )
                    
                    if hw_data is not None or ex_data is not None:
                        st.session_state.logged_in = True
                        st.session_state.homework_data = hw_data
                        st.session_state.exam_data = ex_data
                        st.session_state.student_name = student_name
                        st.session_state.student_class = sclass
                        st.rerun()

def render_sidebar():
    student_class = st.session_state.get('student_class', 'Class-10')
    class_num = student_class.split('-')[1] if '-' in student_class else student_class
    student_name = st.session_state.get('student_name', 'Student')
    
    st.sidebar.title(f"{class_num}th Class Assistant")
    st.sidebar.success(f"👤 {student_name}")
    st.sidebar.info(f"📚 {student_class}")
    
    # Show homework stats
    if st.session_state.homework_data and st.session_state.homework_data.get('homework_data'):
        # First normalize the data to ensure proper structure
        normalized_data = HomeworkDataFilter.normalize_homework_data(st.session_state.homework_data)
        homework_list = normalized_data.get('homework_data', [])
        
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
        class_num = st.session_state.student_class.split('-')[1] if '-' in st.session_state.student_class else st.session_state.student_class
        st.info(f"No data loaded. You can ask general {class_num}th class mathematics questions or solve problems!")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
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
            response_placeholder = st.empty()
            
            # Get response (either string for casual or streaming object for AI)
            response = answer_query_with_gemini(
                user_input,
                st.session_state.homework_data,
                st.session_state.exam_data,
                st.session_state.memory,
                current_image
            )
            
            if response:
                # Check if it's a string (casual response) or streaming object
                if isinstance(response, str):
                    # Casual response - just display
                    response_placeholder.write(response)
                    full_response = response
                else:
                    # Streaming response
                    full_response = ""
                    try:
                        for chunk in response:
                            # Check if chunk has text attribute
                            if hasattr(chunk, 'text') and chunk.text:
                                full_response += chunk.text
                                response_placeholder.write(full_response)
                            # Handle chunks with parts
                            elif hasattr(chunk, 'parts'):
                                for part in chunk.parts:
                                    if hasattr(part, 'text') and part.text:
                                        full_response += part.text
                                        response_placeholder.write(full_response)
                        
                        # If no response was generated, provide fallback
                        if not full_response.strip():
                            full_response = "I apologize, but I couldn't generate a response. This might be due to content filters. Could you please rephrase your question?"
                            response_placeholder.write(full_response)
                            
                    except StopIteration:
                        # Normal end of stream
                        if not full_response.strip():
                            full_response = "I apologize, but I couldn't generate a complete response. Please try rephrasing your question."
                            response_placeholder.write(full_response)
                    except Exception as e:
                        # Log the actual error for debugging
                        error_msg = str(e)
                        if "finish_reason" in error_msg or "response.text" in error_msg:
                            full_response = "I apologize, but I couldn't generate a response due to content safety filters. Let me try to help you differently.\n\nCould you please:\n1. Rephrase your question\n2. Be more specific about what data you'd like to see\n3. Try asking about a particular homework or topic"
                        else:
                            full_response = f"Sorry, I encountered an error: {error_msg}\n\nPlease try rephrasing your question or contact support if this persists."
                        response_placeholder.write(full_response)
                
                # Store in messages and memory
                message_data = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message_data)
                st.session_state.memory.chat_memory.add_ai_message(full_response)
        
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
    main()