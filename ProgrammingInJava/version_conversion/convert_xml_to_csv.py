#!/usr/bin/env python3
"""
Convert XML test bank questions to CSV format.
Processes multiple questions per file and generates individual CSV files.
"""

import xml.etree.ElementTree as ET
import csv
import os
from pathlib import Path
from typing import List, Dict, Optional
import re


def convert_xml_content_to_text(element) -> str:
    """
    Convert XML element content to text, preserving formatting.
    Handles HTML tags, LaTeX, code blocks, and images.
    """
    if element is None:
        return ""
    
    result = []
    
    # Handle direct text
    if element.text:
        result.append(element.text.strip())
    
    # Process children
    for child in element:
        if child.tag == 'p':
            # Paragraph
            p_content = convert_xml_content_to_text(child)
            if p_content:
                result.append(p_content)
        
        elif child.tag == 'line':
            # Line element
            line_content = convert_xml_content_to_text(child)
            if line_content:
                result.append(line_content)
        
        elif child.tag == 'code':
            # Inline code
            code_text = child.text if child.text else ""
            result.append(f'`{code_text}`')
        
        elif child.tag == 'codeBlock':
            # Code block
            language = child.get('language', '')
            code_text = child.text if child.text else ""
            code_text = code_text.strip()
            result.append(f'```{language}\\n{code_text}\\n```')
        
        elif child.tag == 'ilatex':
            # Inline LaTeX
            latex_text = child.text if child.text else ""
            result.append(f'${latex_text}$')
        
        elif child.tag == 'blatex':
            # Block LaTeX
            latex_text = child.text if child.text else ""
            latex_text = latex_text.strip()
            result.append(f'$${latex_text}$$')
        
        elif child.tag == 'image':
            # Image tag - convert to text description
            alt_text = child.get('alt', 'No description')
            result.append(f'[Image: {alt_text}]')
        
        elif child.tag == 'i':
            # Italics
            i_content = convert_xml_content_to_text(child)
            if i_content:
                result.append(f'<i>{i_content}</i>')
        
        else:
            # Other tags - recursively process
            child_content = convert_xml_content_to_text(child)
            if child_content:
                result.append(child_content)
        
        # Handle tail text (text after closing tag)
        if child.tail:
            tail_text = child.tail.strip()
            if tail_text:
                result.append(tail_text)
    
    # Join with newlines, then normalize
    text = '\\n'.join(result)
    
    # Clean up excessive newlines
    text = re.sub(r'\\n\\n+', '\\n\\n', text)
    text = text.strip()
    
    return text


def process_choice(choice_element) -> str:
    """Process a single choice element and return its text content."""
    return convert_xml_content_to_text(choice_element)


def parse_question(question_element) -> Optional[Dict[str, str]]:
    """
    Parse a single question element and return a dictionary with CSV row data.
    Returns None if the question structure is invalid.
    """
    try:
        # Extract instructions (question text)
        instructions = question_element.find('instructions')
        if instructions is None:
            return None
        
        question_text = convert_xml_content_to_text(instructions)
        
        # Extract choices
        choices_element = question_element.find('choices')
        if choices_element is None:
            return None
        
        choices = choices_element.findall('choice')
        if len(choices) != 4:
            return None
        
        # Process each choice and find the correct one
        choice_texts = []
        correct_answer = None
        
        for choice in choices:
            choice_text = process_choice(choice)
            choice_texts.append(choice_text)
            
            # Check if this is the correct answer
            if choice.get('correct') == 'true':
                correct_answer = choice_text
        
        if correct_answer is None:
            return None
        
        # Build CSV row data
        row_data = {
            'text': question_text,
            'answer': correct_answer,
            'a': choice_texts[0] if len(choice_texts) > 0 else '',
            'b': choice_texts[1] if len(choice_texts) > 1 else '',
            'c': choice_texts[2] if len(choice_texts) > 2 else '',
            'd': choice_texts[3] if len(choice_texts) > 3 else '',
            'e': '',
            'f': '',
            'g': '',
            'h': '',
            'hint': '',
            'feedback': ''
        }
        
        return row_data
    
    except Exception as e:
        print(f"  Error parsing question: {e}")
        return None


def process_xml_file(xml_file_path: Path) -> tuple[int, int]:
    """
    Process a single XML file and generate corresponding CSV.
    Returns (questions_processed, questions_failed).
    """
    try:
        # Read the file content
        with open(xml_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove namespace prefixes (xsi:type) to avoid parsing errors
        content = re.sub(r'\sxsi:type=["\'][^"\']*["\']', '', content)
        content = re.sub(r'\ssubjects=["\'][^"\']*["\']', '', content)
        content = re.sub(r'\sguid=["\'][^"\']*["\']', '', content)
        
        # Remove orphaned closing tags (like </sectionTestBank>)
        content = re.sub(r'</sectionTestBank\s*>', '', content)
        
        # Wrap content in a root element to make it valid XML
        wrapped_content = f'<root>{content}</root>'
        
        # Parse XML
        root = ET.fromstring(wrapped_content)
        
        # Find all question elements
        questions = root.findall('.//question')
        
        if not questions:
            print(f"  No questions found in {xml_file_path.name}")
            return 0, 0
        
        # Process each question
        rows = []
        failed = 0
        
        for question in questions:
            row_data = parse_question(question)
            if row_data:
                rows.append(row_data)
            else:
                failed += 1
        
        if not rows:
            print(f"  No valid questions found in {xml_file_path.name}")
            return 0, failed
        
        # Generate CSV file path (same directory, .csv extension)
        csv_file_path = xml_file_path.with_suffix('.csv')
        
        # Write CSV file
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['text', 'answer', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'hint', 'feedback']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write rows
            for row in rows:
                writer.writerow(row)
        
        print(f"✓ {xml_file_path.name}: {len(rows)} questions converted → {csv_file_path.name}")
        return len(rows), failed
    
    except ET.ParseError as e:
        print(f"✗ {xml_file_path.name}: XML parsing error - {e}")
        return 0, 0
    
    except Exception as e:
        print(f"✗ {xml_file_path.name}: Unexpected error - {e}")
        return 0, 0


def main():
    """Main function to process all XML files in the specified folders."""
    
    # Base path
    base_path = Path(__file__).parent / 'SME-questions'
    
    # Folders to process
    folders = [
        'DataSciencePython',
        'DiscreteMath',
        'ProgrammingInJava',
        'ProgrammingInPython'
    ]
    
    # Statistics
    stats = {
        'total_files': 0,
        'successful_files': 0,
        'total_questions': 0,
        'failed_questions': 0,
        'folder_stats': {}
    }
    
    print("=" * 60)
    print("XML to CSV Conversion - Test Bank Questions")
    print("=" * 60)
    print()
    
    # Process each folder
    for folder_name in folders:
        folder_path = base_path / folder_name
        
        if not folder_path.exists():
            print(f"Warning: Folder not found - {folder_path}")
            continue
        
        print(f"\nProcessing folder: {folder_name}")
        print("-" * 60)
        
        # Get all .txt files in the folder
        txt_files = sorted(folder_path.glob('*.txt'))
        
        if not txt_files:
            print(f"  No .txt files found in {folder_name}")
            continue
        
        folder_questions = 0
        folder_failed = 0
        folder_files_success = 0
        
        for txt_file in txt_files:
            stats['total_files'] += 1
            questions, failed = process_xml_file(txt_file)
            
            if questions > 0:
                folder_files_success += 1
                stats['successful_files'] += 1
            
            folder_questions += questions
            folder_failed += failed
        
        stats['total_questions'] += folder_questions
        stats['failed_questions'] += folder_failed
        stats['folder_stats'][folder_name] = {
            'files': len(txt_files),
            'successful_files': folder_files_success,
            'questions': folder_questions,
            'failed_questions': folder_failed
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"\nTotal files processed: {stats['total_files']}")
    print(f"Successful conversions: {stats['successful_files']}")
    print(f"Failed conversions: {stats['total_files'] - stats['successful_files']}")
    print(f"\nTotal questions converted: {stats['total_questions']}")
    print(f"Failed questions: {stats['failed_questions']}")
    
    print("\n" + "-" * 60)
    print("Per-folder breakdown:")
    print("-" * 60)
    
    for folder_name, folder_data in stats['folder_stats'].items():
        print(f"\n{folder_name}:")
        print(f"  Files: {folder_data['successful_files']}/{folder_data['files']}")
        print(f"  Questions: {folder_data['questions']}")
        if folder_data['failed_questions'] > 0:
            print(f"  Failed questions: {folder_data['failed_questions']}")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
