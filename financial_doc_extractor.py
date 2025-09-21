#!/usr/bin/env python3
"""
Financial Document Information Extractor for Compliance Review
Extracts specified information attributes from PDF documents using OCR and LLM processing.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import traceback
from datetime import datetime

# Third-party imports
import fitz  # PyMuPDF
import PyMuPDF
import easyocr
from openai import OpenAI
import cv2
import numpy as np
from PIL import Image
import io

class DocumentProcessor:
    """Handles document processing with OCR and LLM extraction capabilities."""
    
    def __init__(self, openai_api_key: str, enable_ocr: bool = True, enable_llm: bool = True):
        """
        Initialize the document processor.
        
        Args:
            openai_api_key: OpenAI API key for LLM processing
            enable_ocr: Enable OCR processing for images
            enable_llm: Enable LLM processing for text extraction
        """
        self.enable_ocr = enable_ocr
        self.enable_llm = enable_llm
        
        # Initialize OpenAI client
        if enable_llm:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize EasyOCR reader
        if enable_ocr:
            try:
                self.ocr_reader = easyocr.Reader(['en'])
                logging.info("EasyOCR initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize EasyOCR: {e}")
                self.enable_ocr = False
        
        # Track processing statistics
        self.stats = {
            'files_processed': 0,
            'pages_processed': 0,
            'ocr_pages': 0,
            'llm_calls': 0,
            'errors': 0
        }
    
    def load_target_attributes(self, attributes_file: str) -> List[str]:
        """Load target information attributes from text file."""
        try:
            with open(attributes_file, 'r', encoding='utf-8') as f:
                attributes = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(attributes)} target attributes from {attributes_file}")
            return attributes
        except Exception as e:
            logging.error(f"Failed to load attributes file {attributes_file}: {e}")
            raise
    
    def extract_text_from_pdf_page(self, page) -> str:
        """Extract text from a PDF page using PyMuPDF."""
        try:
            text = page.get_text()
            return text.strip()
        except Exception as e:
            logging.warning(f"Failed to extract text from PDF page: {e}")
            return ""
    
    def perform_ocr_on_page(self, page) -> str:
        """Perform OCR on a PDF page that appears to be an image."""
        if not self.enable_ocr:
            return ""
        
        try:
            # Convert PDF page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
            img_data = pix.tobytes("png")
            
            # Convert to numpy array for EasyOCR
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Perform OCR
            results = self.ocr_reader.readtext(img)
            
            # Extract text from results
            ocr_text = ""
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    ocr_text += text + " "
            
            self.stats['ocr_pages'] += 1
            logging.debug(f"OCR extracted {len(ocr_text)} characters")
            return ocr_text.strip()
            
        except Exception as e:
            logging.warning(f"OCR failed on page: {e}")
            return ""
    
    def create_extraction_prompt(self, text: str, target_attributes: List[str]) -> str:
        """Create a detailed prompt for LLM extraction."""
        attributes_xml = "\n".join([f"    <attribute>{attr}</attribute>" for attr in target_attributes])
        
        prompt = f"""
<task>
You are a Financial Services Compliance Officer assistant. Extract specific information attributes from the provided document text with absolute accuracy.

CRITICAL INSTRUCTIONS:
1. Extract information EXACTLY as it appears in the document - do not infer, guess, or modify values
2. If information is unclear, ambiguous, or partially visible, mark it as "UNCLEAR" in the value field
3. If information is completely missing, mark it as "NOT_FOUND"
4. Report any inconsistencies or data quality issues in the exception_notes field
5. Include the approximate location description where you found each piece of information
</task>

<target_attributes>
{attributes_xml}
</target_attributes>

<document_text>
{text}
</document_text>

<output_format>
Respond with a valid JSON object containing an array of extracted information:
{{
    "extracted_data": [
        {{
            "attribute": "attribute_name",
            "value": "exact_value_from_document_or_NOT_FOUND_or_UNCLEAR",
            "location_description": "description of where this information was found in the text",
            "confidence": "HIGH|MEDIUM|LOW",
            "exception_notes": "any issues, inconsistencies, or data quality concerns"
        }}
    ],
    "processing_notes": "overall observations about document quality and extraction challenges"
}}
</output_format>

Extract the requested information now:
"""
        return prompt
    
    def extract_with_llm(self, text: str, target_attributes: List[str]) -> Dict[str, Any]:
        """Extract information using LLM."""
        if not self.enable_llm:
            return {"extracted_data": [], "processing_notes": "LLM processing disabled"}
        
        try:
            prompt = self.create_extraction_prompt(text, target_attributes)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Use latest GPT-4 model
                messages=[
                    {"role": "system", "content": "You are a precise document extraction assistant. Return only valid JSON responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000
            )
            
            self.stats['llm_calls'] += 1
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Handle potential JSON formatting issues
            if response_text.startswith('```json'):
                response_text = response_text.strip('```json').strip('```').strip()
            
            result = json.loads(response_text)
            return result
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM JSON response: {e}")
            return {"extracted_data": [], "processing_notes": f"JSON parsing error: {e}"}
        except Exception as e:
            logging.error(f"LLM extraction failed: {e}")
            return {"extracted_data": [], "processing_notes": f"LLM error: {e}"}
    
    def process_pdf_file(self, file_path: str, target_attributes: List[str]) -> List[Dict[str, Any]]:
        """Process a single PDF file and extract information."""
        results = []
        
        try:
            doc = fitz.open(file_path)
            file_name = os.path.basename(file_path)
            
            logging.info(f"Processing {file_name} ({len(doc)} pages)")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_number = page_num + 1
                
                # Extract text using PyMuPDF
                text = self.extract_text_from_pdf_page(page)
                
                # If text is minimal, try OCR
                if len(text.strip()) < 50:  # Threshold for "text-poor" pages
                    ocr_text = self.perform_ocr_on_page(page)
                    if ocr_text:
                        text = ocr_text
                        logging.debug(f"Used OCR for page {page_number}")
                
                # Skip if no text found
                if len(text.strip()) < 10:
                    logging.warning(f"Minimal text found on page {page_number} of {file_name}")
                    continue
                
                # Process with LLM
                extraction_result = self.extract_with_llm(text, target_attributes)
                
                # Format results for this page
                for item in extraction_result.get('extracted_data', []):
                    result_item = {
                        'file_name': file_name,
                        'page_number': page_number,
                        'field_label': item.get('attribute', 'UNKNOWN'),
                        'field_value': item.get('value', 'NOT_FOUND'),
                        'location_description': item.get('location_description', ''),
                        'confidence': item.get('confidence', 'LOW'),
                        'exception_notes': item.get('exception_notes', ''),
                        'processing_notes': extraction_result.get('processing_notes', '')
                    }
                    results.append(result_item)
                
                self.stats['pages_processed'] += 1
            
            doc.close()
            self.stats['files_processed'] += 1
            
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")
            self.stats['errors'] += 1
            
            # Add error record
            results.append({
                'file_name': os.path.basename(file_path),
                'page_number': 0,
                'field_label': 'PROCESSING_ERROR',
                'field_value': str(e),
                'location_description': 'File level error',
                'confidence': 'LOW',
                'exception_notes': f'File processing failed: {traceback.format_exc()}',
                'processing_notes': 'Error during file processing'
            })
        
        return results
    
    def process_directory(self, input_dir: str, target_attributes: List[str]) -> List[Dict[str, Any]]:
        """Process all PDF files in the input directory."""
        all_results = []
        
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))
        
        logging.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            results = self.process_pdf_file(str(pdf_file), target_attributes)
            all_results.extend(results)
        
        return all_results
    
    def generate_exception_log(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate exception log from processing results."""
        exceptions = []
        
        for result in results:
            # Check for various exception conditions
            exception_entry = None
            
            if result.get('field_value') == 'UNCLEAR':
                exception_entry = {
                    'file_name': result['file_name'],
                    'page_number': result['page_number'],
                    'field_label': result['field_label'],
                    'issue_type': 'UNCLEAR_VALUE',
                    'description': 'Value found but unclear or ambiguous',
                    'exception_notes': result.get('exception_notes', ''),
                    'timestamp': datetime.now().isoformat()
                }
            
            elif result.get('field_value') == 'NOT_FOUND':
                exception_entry = {
                    'file_name': result['file_name'],
                    'page_number': result['page_number'],
                    'field_label': result['field_label'],
                    'issue_type': 'MISSING_VALUE',
                    'description': 'Required information not found in document',
                    'exception_notes': result.get('exception_notes', ''),
                    'timestamp': datetime.now().isoformat()
                }
            
            elif result.get('confidence') == 'LOW':
                exception_entry = {
                    'file_name': result['file_name'],
                    'page_number': result['page_number'],
                    'field_label': result['field_label'],
                    'issue_type': 'LOW_CONFIDENCE',
                    'description': 'Extracted value has low confidence',
                    'exception_notes': result.get('exception_notes', ''),
                    'timestamp': datetime.now().isoformat()
                }
            
            elif result.get('exception_notes'):
                exception_entry = {
                    'file_name': result['file_name'],
                    'page_number': result['page_number'],
                    'field_label': result['field_label'],
                    'issue_type': 'PROCESSING_ISSUE',
                    'description': 'Processing issue reported',
                    'exception_notes': result.get('exception_notes', ''),
                    'timestamp': datetime.now().isoformat()
                }
            
            if exception_entry:
                exceptions.append(exception_entry)
        
        return exceptions

def setup_logging(output_dir: str) -> None:
    """Set up logging configuration."""
    log_file = os.path.join(output_dir, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main function to run the document processor."""
    parser = argparse.ArgumentParser(description="Financial Document Information Extractor")
    
    parser.add_argument('--input-dir', required=True, help='Input directory containing PDF files')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--attributes-file', required=True, help='Text file containing target attributes')
    parser.add_argument('--openai-api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--enable-ocr', action='store_true', default=True, help='Enable OCR processing')
    parser.add_argument('--disable-ocr', action='store_true', help='Disable OCR processing')
    parser.add_argument('--enable-llm', action='store_true', default=True, help='Enable LLM processing')
    parser.add_argument('--disable-llm', action='store_true', help='Disable LLM processing')
    
    args = parser.parse_args()
    
    # Handle enable/disable flags
    enable_ocr = args.enable_ocr and not args.disable_ocr
    enable_llm = args.enable_llm and not args.disable_llm
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    
    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if enable_llm and not openai_api_key:
        logging.error("OpenAI API key required for LLM processing. Set --openai-api-key or OPENAI_API_KEY env var")
        sys.exit(1)
    
    try:
        # Initialize processor
        processor = DocumentProcessor(
            openai_api_key=openai_api_key,
            enable_ocr=enable_ocr,
            enable_llm=enable_llm
        )
        
        # Load target attributes
        target_attributes = processor.load_target_attributes(args.attributes_file)
        
        # Process documents
        logging.info(f"Starting processing with OCR={'enabled' if enable_ocr else 'disabled'}, LLM={'enabled' if enable_llm else 'disabled'}")
        results = processor.process_directory(args.input_dir, target_attributes)
        
        # Generate exception log
        exceptions = processor.generate_exception_log(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_file = os.path.join(args.output_dir, f"extraction_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        exceptions_file = os.path.join(args.output_dir, f"exception_log_{timestamp}.json")
        with open(exceptions_file, 'w', encoding='utf-8') as f:
            json.dump(exceptions, f, indent=2, ensure_ascii=False)
        
        # Print summary
        stats = processor.stats
        logging.info("=" * 50)
        logging.info("PROCESSING SUMMARY")
        logging.info("=" * 50)
        logging.info(f"Files processed: {stats['files_processed']}")
        logging.info(f"Pages processed: {stats['pages_processed']}")
        logging.info(f"OCR pages: {stats['ocr_pages']}")
        logging.info(f"LLM calls: {stats['llm_calls']}")
        logging.info(f"Errors: {stats['errors']}")
        logging.info(f"Total records extracted: {len(results)}")
        logging.info(f"Exception records: {len(exceptions)}")
        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Exceptions saved to: {exceptions_file}")
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
