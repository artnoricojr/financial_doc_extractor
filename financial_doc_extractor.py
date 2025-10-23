#!/usr/bin/env python3
"""
version 06 - Added Handwriting Detection
Financial Document Information Extractor for Compliance Review
Extracts specified information attributes from PDF documents using OCR and LLM processing.

NEW FEATURE: Handwriting Detection
- Automatically classifies extracted content as handwritten, typed, or uncertain
- Uses hybrid approach combining OCR confidence analysis, image processing, and LLM intelligence
- Adds "derived-from-handwriting" attribute with values: "yes", "maybe", "no"

HANDWRITING DETECTION APPROACH:
1. OCR-Based Analysis (for scanned/image pages):
   - Analyzes OCR confidence scores (handwriting typically has lower confidence)
   - Measures baseline consistency and spacing irregularity
   - Detects edge patterns and stroke width variations using Canny and Sobel filters
   - Evaluates text fragmentation and character length patterns

2. LLM-Based Analysis (for all pages):
   - GPT-4 analyzes extracted text for handwriting indicators
   - Looks for cursive patterns, irregular spacing, annotations, signatures
   - Identifies mixed printed/handwritten content
   - Provides contextual understanding beyond image analysis

3. Classification Rules:
   - "yes": Strong indicators of handwriting (cursive, signatures, irregular patterns)
   - "maybe": Mixed signals, uncertain origin, stylized fonts, low image quality
   - "no": Clearly typed, printed, or digitally generated text
"""

import argparse
import csv
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
        self.custom_prompt_text = None
        
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
    
    def load_custom_prompt(self, prompt_file: str) -> None:
        """Load custom prompt template from file."""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.custom_prompt_text = f.read()
            logging.info(f"Loaded custom prompt template from {prompt_file}")
        except Exception as e:
            logging.error(f"Failed to load custom prompt file {prompt_file}: {e}")
            raise
    
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
    
    def detect_handwriting_from_image(self, img: np.ndarray, ocr_results: List[Tuple]) -> str:
        """
        Analyze image and OCR results to detect if content is derived from handwriting.

        Classification logic:
        - "yes": Clearly handwritten (low OCR confidence, irregular patterns, high edge variation)
        - "maybe": Uncertain (mixed signals, stylized fonts, moderate confidence)
        - "no": Clearly typed/printed (high OCR confidence, uniform patterns)

        Args:
            img: The page image as numpy array
            ocr_results: EasyOCR results with bounding boxes, text, and confidence scores

        Returns:
            Classification as "yes", "maybe", or "no"
        """
        try:
            if len(ocr_results) == 0:
                return "maybe"  # No text detected - uncertain

            # Metric 1: Analyze OCR confidence scores
            # Handwriting typically has lower OCR confidence
            confidences = [conf for (bbox, text, conf) in ocr_results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            low_confidence_ratio = sum(1 for c in confidences if c < 0.7) / len(confidences)

            # Metric 2: Analyze bounding box regularity
            # Handwriting has more irregular spacing and alignment
            if len(ocr_results) > 1:
                # Calculate vertical position variance (baseline consistency)
                y_positions = [bbox[0][1] for (bbox, text, conf) in ocr_results]  # Top-left y-coordinate
                y_variance = np.var(y_positions) if len(y_positions) > 0 else 0

                # Calculate horizontal spacing variance
                boxes_sorted = sorted(ocr_results, key=lambda x: x[0][0][0])  # Sort by x-position
                spacings = []
                for i in range(len(boxes_sorted) - 1):
                    x_end = boxes_sorted[i][0][1][0]  # Right edge of current box
                    x_start = boxes_sorted[i+1][0][0][0]  # Left edge of next box
                    spacings.append(x_start - x_end)
                spacing_variance = np.var(spacings) if len(spacings) > 0 else 0
            else:
                y_variance = 0
                spacing_variance = 0

            # Metric 3: Analyze image edge characteristics
            # Handwriting has more varied edge patterns due to stroke variations
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Apply Sobel filter to detect stroke width variations
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_variance = np.var(gradient_magnitude)

            # Metric 4: Text characteristics from OCR
            # Handwriting often has shorter text segments and more single-word detections
            avg_text_length = sum(len(text) for (bbox, text, conf) in ocr_results) / len(ocr_results)
            short_text_ratio = sum(1 for (bbox, text, conf) in ocr_results if len(text) <= 3) / len(ocr_results)

            # Decision logic combining all metrics
            handwriting_score = 0

            # Confidence-based scoring (weight: 3)
            if avg_confidence < 0.6:
                handwriting_score += 3
            elif avg_confidence < 0.75:
                handwriting_score += 2
            elif avg_confidence < 0.85:
                handwriting_score += 1

            # Low confidence ratio (weight: 2)
            if low_confidence_ratio > 0.5:
                handwriting_score += 2
            elif low_confidence_ratio > 0.3:
                handwriting_score += 1

            # Baseline variance (weight: 2)
            if y_variance > 100:
                handwriting_score += 2
            elif y_variance > 50:
                handwriting_score += 1

            # Spacing irregularity (weight: 2)
            if spacing_variance > 1000:
                handwriting_score += 2
            elif spacing_variance > 500:
                handwriting_score += 1

            # Edge characteristics (weight: 1)
            if edge_density > 0.15 or gradient_variance > 5000:
                handwriting_score += 1

            # Text fragmentation (weight: 1)
            if short_text_ratio > 0.4 or avg_text_length < 4:
                handwriting_score += 1

            # Classification based on total score (max score: 11)
            if handwriting_score >= 7:
                classification = "yes"  # Strong indicators of handwriting
            elif handwriting_score >= 4:
                classification = "maybe"  # Mixed signals or uncertain
            else:
                classification = "no"  # Likely typed/printed text

            logging.debug(f"Handwriting detection - Score: {handwriting_score}/11, "
                         f"Avg confidence: {avg_confidence:.2f}, "
                         f"Low conf ratio: {low_confidence_ratio:.2f}, "
                         f"Y-variance: {y_variance:.2f}, "
                         f"Classification: {classification}")

            return classification

        except Exception as e:
            logging.warning(f"Handwriting detection failed: {e}")
            return "maybe"  # Default to uncertain on error

    def perform_ocr_on_page(self, page) -> Tuple[str, str]:
        """
        Perform OCR on a PDF page that appears to be an image.

        Returns:
            Tuple of (ocr_text, handwriting_classification)
            where handwriting_classification is "yes", "maybe", or "no"
        """
        if not self.enable_ocr:
            return "", "no"

        try:
            # Convert PDF page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
            img_data = pix.tobytes("png")

            # Convert to numpy array for EasyOCR
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Perform OCR
            results = self.ocr_reader.readtext(img)

            # Detect handwriting before filtering results
            handwriting_classification = self.detect_handwriting_from_image(img, results)

            # Extract text from results
            ocr_text = ""
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    ocr_text += text + " "

            self.stats['ocr_pages'] += 1
            logging.debug(f"OCR extracted {len(ocr_text)} characters, "
                         f"handwriting: {handwriting_classification}")
            return ocr_text.strip(), handwriting_classification

        except Exception as e:
            logging.warning(f"OCR failed on page: {e}")
            return "", "maybe"
    
    def create_extraction_prompt(self, text: str, target_attributes: List[str], custom_prompt_text: str = None) -> str:
        """Create a detailed prompt for LLM extraction."""
        attributes_xml = "\n".join([f"    <attribute>{attr}</attribute>" for attr in target_attributes])
        
        if custom_prompt_text:
            # Use custom prompt template, replacing placeholders
            prompt = custom_prompt_text.replace("{attributes_xml}", attributes_xml)
            prompt = prompt.replace("{text}", text)
            prompt = prompt.replace("{document_text}", text)  # Alternative placeholder
            return prompt
        
        # Default prompt template
        prompt = f"""
<task>
You are a Financial Services Compliance Officer assistant. Extract specific information attributes from the provided document text with absolute accuracy.

CRITICAL INSTRUCTIONS:
1. Extract information EXACTLY as it appears in the document - do not infer, guess, or modify values
2. If information is unclear, ambiguous, or partially visible, mark it as "UNCLEAR" in the value field
3. If information is completely missing, mark it as "NOT_FOUND"
4. Report any inconsistencies or data quality issues in the exception_notes field
5. Include the approximate location description where you found each piece of information
6. IMPORTANT: For each extracted attribute, determine if it was "derived-from-handwriting":
   - "yes" = Clearly handwritten (cursive, print handwriting, signatures, annotations, irregular text)
   - "maybe" = Uncertain origin (stylized fonts, unclear scans, mixed printed/handwritten content)
   - "no" = Clearly printed, typed, or digitally generated text
</task>

<handwriting_detection_guidelines>
Look for these indicators of handwriting:
- Irregular letter spacing and inconsistent baselines
- Variations in stroke width and pen pressure
- Non-standard letterforms compared to typical fonts
- Cursive or connected letters
- Signatures, initials, or annotations
- Checkmarks, circles, or other hand-drawn marks
- Uneven alignment or slanted text
- Mixed printed and handwritten content in the same field
</handwriting_detection_guidelines>

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
            "derived-from-handwriting": "yes|maybe|no",
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
            prompt = self.create_extraction_prompt(text, target_attributes, self.custom_prompt_text)
            
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

                # Track if OCR was used and handwriting classification
                used_ocr = False
                page_handwriting_classification = "no"  # Default for text-based PDFs

                # If text is minimal, try OCR
                if len(text.strip()) < 50:  # Threshold for "text-poor" pages
                    ocr_text, handwriting_classification = self.perform_ocr_on_page(page)
                    if ocr_text:
                        text = ocr_text
                        page_handwriting_classification = handwriting_classification
                        used_ocr = True
                        logging.debug(f"Used OCR for page {page_number}, "
                                    f"handwriting: {handwriting_classification}")

                # Skip if no text found
                if len(text.strip()) < 10:
                    logging.warning(f"Minimal text found on page {page_number} of {file_name}")
                    continue

                # Process with LLM
                extraction_result = self.extract_with_llm(text, target_attributes)

                # Format results for this page
                for item in extraction_result.get('extracted_data', []):
                    # Get handwriting classification from LLM or OCR
                    # Priority: LLM classification > OCR classification > default "no"
                    llm_handwriting = item.get('derived-from-handwriting', '').lower()

                    # Validate and use LLM classification if valid
                    if llm_handwriting in ['yes', 'maybe', 'no']:
                        handwriting_value = llm_handwriting
                    elif used_ocr:
                        # If LLM didn't provide valid classification, use OCR result
                        handwriting_value = page_handwriting_classification
                    else:
                        # Default for text-based PDFs (no OCR)
                        handwriting_value = "no"

                    result_item = {
                        'file_name': file_name,
                        'page_number': page_number,
                        'field_label': item.get('attribute', 'UNKNOWN'),
                        'field_value': item.get('value', 'NOT_FOUND'),
                        'location_description': item.get('location_description', ''),
                        'confidence': item.get('confidence', 'LOW'),
                        'derived-from-handwriting': handwriting_value,  # NEW ATTRIBUTE
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
                'derived-from-handwriting': 'no',  # Error records default to 'no'
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

    def save_results_to_csv(self, results: List[Dict[str, Any]], csv_file_path: str) -> None:
        """
        Save extraction results to CSV file.

        Args:
            results: List of extraction result dictionaries
            csv_file_path: Path to output CSV file
        """
        if not results:
            logging.warning("No results to save to CSV")
            return

        try:
            # Define CSV column headers
            fieldnames = [
                'file_name',
                'page_number',
                'field_label',
                'field_value',
                'location_description',
                'confidence',
                'derived-from-handwriting',
                'exception_notes',
                'processing_notes'
            ]

            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')

                # Write header
                writer.writeheader()

                # Write data rows
                for result in results:
                    writer.writerow(result)

            logging.info(f"Results saved to CSV: {csv_file_path}")

        except Exception as e:
            logging.error(f"Failed to save results to CSV: {e}")
            raise

    def save_exceptions_to_csv(self, exceptions: List[Dict[str, Any]], csv_file_path: str) -> None:
        """
        Save exception log to CSV file.

        Args:
            exceptions: List of exception dictionaries
            csv_file_path: Path to output CSV file
        """
        if not exceptions:
            logging.warning("No exceptions to save to CSV")
            # Create empty CSV file with headers
            try:
                fieldnames = [
                    'file_name',
                    'page_number',
                    'field_label',
                    'issue_type',
                    'description',
                    'exception_notes',
                    'timestamp'
                ]
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                logging.info(f"Empty exception CSV created: {csv_file_path}")
            except Exception as e:
                logging.error(f"Failed to create empty exception CSV: {e}")
            return

        try:
            # Define CSV column headers
            fieldnames = [
                'file_name',
                'page_number',
                'field_label',
                'issue_type',
                'description',
                'exception_notes',
                'timestamp'
            ]

            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')

                # Write header
                writer.writeheader()

                # Write data rows
                for exception in exceptions:
                    writer.writerow(exception)

            logging.info(f"Exceptions saved to CSV: {csv_file_path}")

        except Exception as e:
            logging.error(f"Failed to save exceptions to CSV: {e}")
            raise

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
    parser.add_argument('--prompt-file', help='Text file containing custom prompt template')
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
        
        # Load custom prompt if provided
        if args.prompt_file:
            processor.load_custom_prompt(args.prompt_file)
        
        # Load target attributes
        target_attributes = processor.load_target_attributes(args.attributes_file)
        
        # Process documents
        logging.info(f"Starting processing with OCR={'enabled' if enable_ocr else 'disabled'}, LLM={'enabled' if enable_llm else 'disabled'}")
        if args.prompt_file:
            logging.info(f"Using custom prompt template from {args.prompt_file}")
        results = processor.process_directory(args.input_dir, target_attributes)
        
        # Generate exception log
        exceptions = processor.generate_exception_log(results)

        # Save results in both JSON and CSV formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON files
        results_json_file = os.path.join(args.output_dir, f"extraction_results_{timestamp}.json")
        with open(results_json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        exceptions_json_file = os.path.join(args.output_dir, f"exception_log_{timestamp}.json")
        with open(exceptions_json_file, 'w', encoding='utf-8') as f:
            json.dump(exceptions, f, indent=2, ensure_ascii=False)

        # Save CSV files
        results_csv_file = os.path.join(args.output_dir, f"extraction_results_{timestamp}.csv")
        processor.save_results_to_csv(results, results_csv_file)

        exceptions_csv_file = os.path.join(args.output_dir, f"exception_log_{timestamp}.csv")
        processor.save_exceptions_to_csv(exceptions, exceptions_csv_file)

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
        logging.info("")
        logging.info("OUTPUT FILES:")
        logging.info(f"  Results (JSON): {results_json_file}")
        logging.info(f"  Results (CSV):  {results_csv_file}")
        logging.info(f"  Exceptions (JSON): {exceptions_json_file}")
        logging.info(f"  Exceptions (CSV):  {exceptions_csv_file}")
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
