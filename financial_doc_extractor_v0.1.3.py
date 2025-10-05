import json
import os
import logging
import easyocr
import fitz  # PyMuPDF
import openai
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import io
import base64
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExtractedInfo:
    """Data class for extracted information"""
    filename: str
    page_number: int
    field_label: str
    field_value: str
    confidence: float = 0.0
    extraction_method: str = ""

@dataclass
class ExceptionLog:
    """Data class for exception logging"""
    filename: str
    page_number: int
    field_label: str
    issue_type: str
    description: str
    timestamp: str

class FinancialDocumentExtractor:
    """
    Main class for extracting financial compliance information from PDF documents
    Supports text extraction, OCR, and vision-based extraction using OpenAI's GPT-4 Vision
    """
    
    def __init__(self, openai_api_key: str, attributes_file: str, use_vision: bool = True):
        """
        Initialize the extractor
        
        Args:
            openai_api_key: OpenAI API key
            attributes_file: Path to text file containing the 24 information attributes
            use_vision: Whether to use GPT-4 Vision for image-based extraction (default: True)
        """
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.ocr_reader = None  # Lazy load EasyOCR
        self.attributes = self._load_attributes(attributes_file)
        self.extracted_data: List[ExtractedInfo] = []
        self.exceptions: List[ExceptionLog] = []
        self.use_vision = use_vision
        
        # Context window settings
        self.max_tokens = 4000  # Conservative limit for context window
        self.chunk_overlap = 200  # Overlap between chunks
        
        # Vision API settings
        self.max_image_size = 20 * 1024 * 1024  # 20MB limit for OpenAI
        self.vision_model = "gpt-4o"  # GPT-4 with vision capabilities
        
    def _load_attributes(self, attributes_file: str) -> List[str]:
        """Load the 24 information attributes from text file"""
        try:
            with open(attributes_file, 'r', encoding='utf-8') as f:
                attributes = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(attributes)} attributes from {attributes_file}")
            return attributes
        except Exception as e:
            logger.error(f"Error loading attributes file: {e}")
            raise
    
    def _initialize_ocr(self):
        """Lazy initialization of EasyOCR"""
        if self.ocr_reader is None:
            logger.info("Initializing EasyOCR...")
            self.ocr_reader = easyocr.Reader(['en'])
    
    def _create_extraction_prompt(self, text_content: str, filename: str, page_num: int) -> str:
        """
        Create the LLM prompt for text-based extraction following Anthropic's XML tag guidelines
        """
        attributes_xml = "\n".join([f"<attribute>{attr}</attribute>" for attr in self.attributes])
        
        prompt = f"""
<instructions>
You are a Financial Services Compliance Officer assistant. Your task is to extract specific information attributes from financial documents for annuity policy suitability reviews.

CRITICAL REQUIREMENTS:
1. Extract content EXACTLY as reported in the document - do not guess, infer, or modify values
2. If information is unclear, ambiguous, or potentially inaccurate, report it in exceptions
3. Only extract information that is explicitly present in the text
4. Maintain exact formatting, spelling, and punctuation as found in source
5. If a field has multiple values or conflicting information, report all instances
</instructions>

<target_attributes>
{attributes_xml}
</target_attributes>

<document_context>
Filename: {filename}
Page Number: {page_num}
</document_context>

<document_text>
{text_content}
</document_text>

<output_format>
For each attribute found, respond with:
<extraction>
<field_label>[exact attribute name]</field_label>
<field_value>[exact value as found in document]</field_value>
<confidence>[high/medium/low]</confidence>
<location_context>[brief description of where found in text]</location_context>
</extraction>

For any issues or inconsistencies:
<exception>
<field_label>[attribute name if applicable]</field_label>
<issue_type>[unclear_text/conflicting_info/missing_data/poor_quality/other]</issue_type>
<description>[detailed description of the issue]</description>
</exception>

If no relevant information is found, respond with:
<no_extraction>No target attributes found in this text segment.</no_extraction>
</output_format>

Please analyze the document text and extract the required information following these guidelines exactly.
"""
        return prompt
    
    def _create_vision_extraction_prompt(self, filename: str, page_num: int) -> str:
        """
        Create the prompt for vision-based extraction
        """
        attributes_list = "\n".join([f"- {attr}" for attr in self.attributes])
        
        prompt = f"""You are a Financial Services Compliance Officer assistant analyzing a financial document image for annuity policy suitability reviews.

DOCUMENT INFORMATION:
Filename: {filename}
Page Number: {page_num}

TARGET INFORMATION ATTRIBUTES:
{attributes_list}

CRITICAL REQUIREMENTS:
1. Extract content EXACTLY as shown in the image - do not guess, infer, or modify values
2. Read all text carefully, including handwritten notes if present
3. If information is unclear, blurry, or ambiguous, report it as an exception
4. Only extract information that is explicitly visible in the image
5. Maintain exact formatting, spelling, and punctuation as found in the document
6. If a field has multiple values or conflicting information, report all instances

OUTPUT FORMAT:
For each attribute found, respond with:
<extraction>
<field_label>[exact attribute name from the list above]</field_label>
<field_value>[exact value as shown in the image]</field_value>
<confidence>[high/medium/low]</confidence>
<location_context>[brief description of where found on the page]</location_context>
</extraction>

For any issues or inconsistencies:
<exception>
<field_label>[attribute name if applicable]</field_label>
<issue_type>[unclear_text/conflicting_info/missing_data/poor_quality/handwriting_illegible/other]</issue_type>
<description>[detailed description of the issue]</description>
</exception>

If no relevant information is found, respond with:
<no_extraction>No target attributes found in this image.</no_extraction>

Please analyze the image carefully and extract the required information."""
        
        return prompt
    
    def _extract_text_from_pdf_page(self, pdf_page) -> str:
        """Extract text from a PDF page using PyMuPDF"""
        try:
            text = pdf_page.get_text()
            if text.strip():
                return text
            else:
                # If no text found, the page might be image-based
                return None
        except Exception as e:
            logger.warning(f"Error extracting text from PDF page: {e}")
            return None
    
    def _pdf_page_to_base64_image(self, pdf_page, dpi: int = 200) -> Tuple[str, int]:
        """
        Convert PDF page to base64-encoded PNG image
        
        Args:
            pdf_page: PyMuPDF page object
            dpi: Resolution for rendering (default: 200)
            
        Returns:
            Tuple of (base64_string, file_size_bytes)
        """
        try:
            # Render page to image with higher DPI for better quality
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = pdf_page.get_pixmap(matrix=mat)
            
            # Convert to PNG bytes
            img_data = pix.tobytes("png")
            
            # Check size and reduce DPI if necessary
            if len(img_data) > self.max_image_size and dpi > 100:
                logger.info(f"Image too large ({len(img_data)} bytes), reducing DPI")
                return self._pdf_page_to_base64_image(pdf_page, dpi=int(dpi * 0.7))
            
            # Encode to base64
            base64_image = base64.b64encode(img_data).decode("utf-8")
            
            return base64_image, len(img_data)
            
        except Exception as e:
            logger.error(f"Error converting PDF page to base64 image: {e}")
            raise
    
    def _ocr_pdf_page(self, pdf_page) -> str:
        """Perform OCR on a PDF page using EasyOCR"""
        try:
            self._initialize_ocr()
            
            # Convert PDF page to image
            mat = fitz.Matrix(2.0, 2.0)  # Increase resolution for better OCR
            pix = pdf_page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image for EasyOCR
            image = Image.open(io.BytesIO(img_data))
            img_array = np.array(image)
            
            # Perform OCR
            results = self.ocr_reader.readtext(img_array)
            
            # Combine OCR results into text
            ocr_text = "\n".join([result[1] for result in results if result[2] > 0.5])  # Filter by confidence
            
            return ocr_text if ocr_text.strip() else None
            
        except Exception as e:
            logger.error(f"Error performing OCR: {e}")
            return None
    
    def _chunk_text(self, text: str, max_chunk_size: int = 3000) -> List[str]:
        """Split text into manageable chunks with overlap"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find a good break point (end of sentence or paragraph)
            break_point = end
            for i in range(end - 100, end):
                if i > start and text[i] in '.!?\n':
                    break_point = i + 1
                    break
            
            chunks.append(text[start:break_point])
            start = break_point - self.chunk_overlap
            
        return chunks
    
    def _process_llm_response(self, response: str, filename: str, page_num: int, method: str = "LLM") -> None:
        """Parse LLM response and extract structured data"""
        try:
            # Extract extractions
            extraction_pattern = r'<extraction>(.*?)</extraction>'
            extractions = re.findall(extraction_pattern, response, re.DOTALL)
            
            for extraction in extractions:
                field_label = re.search(r'<field_label>(.*?)</field_label>', extraction, re.DOTALL)
                field_value = re.search(r'<field_value>(.*?)</field_value>', extraction, re.DOTALL)
                confidence = re.search(r'<confidence>(.*?)</confidence>', extraction, re.DOTALL)
                
                if field_label and field_value:
                    extracted_info = ExtractedInfo(
                        filename=filename,
                        page_number=page_num,
                        field_label=field_label.group(1).strip(),
                        field_value=field_value.group(1).strip(),
                        confidence=self._convert_confidence(confidence.group(1).strip() if confidence else "medium"),
                        extraction_method=method
                    )
                    self.extracted_data.append(extracted_info)
            
            # Extract exceptions
            exception_pattern = r'<exception>(.*?)</exception>'
            exceptions = re.findall(exception_pattern, response, re.DOTALL)
            
            for exception in exceptions:
                field_label = re.search(r'<field_label>(.*?)</field_label>', exception, re.DOTALL)
                issue_type = re.search(r'<issue_type>(.*?)</issue_type>', exception, re.DOTALL)
                description = re.search(r'<description>(.*?)</description>', exception, re.DOTALL)
                
                if issue_type and description:
                    exception_log = ExceptionLog(
                        filename=filename,
                        page_number=page_num,
                        field_label=field_label.group(1).strip() if field_label else "",
                        issue_type=issue_type.group(1).strip(),
                        description=description.group(1).strip(),
                        timestamp=self._get_timestamp()
                    )
                    self.exceptions.append(exception_log)
                    
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
    
    def _convert_confidence(self, confidence_str: str) -> float:
        """Convert confidence string to numeric value"""
        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
        return confidence_map.get(confidence_str.lower(), 0.7)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _call_openai_text_api(self, prompt: str) -> str:
        """Call OpenAI API with text-based extraction prompt"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better accuracy
                messages=[
                    {"role": "system", "content": "You are a precise financial document analysis assistant. Follow instructions exactly and extract information only as explicitly found in documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent, accurate extraction
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI text API: {e}")
            return f"<exception><issue_type>api_error</issue_type><description>Failed to process with LLM: {str(e)}</description></exception>"
    
    def _call_openai_vision_api(self, base64_image: str, prompt: str) -> str:
        """
        Call OpenAI Vision API with base64-encoded image
        
        Args:
            base64_image: Base64-encoded image data
            prompt: Extraction prompt
            
        Returns:
            API response text
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise financial document analysis assistant specialized in analyzing document images. Follow instructions exactly and extract information only as explicitly visible in images."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"  # High detail for better text recognition
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI Vision API: {e}")
            return f"<exception><issue_type>vision_api_error</issue_type><description>Failed to process with Vision API: {str(e)}</description></exception>"
    
    def _process_page_with_vision(self, pdf_page, filename: str, page_num: int) -> bool:
        """
        Process a PDF page using OpenAI Vision API
        
        Args:
            pdf_page: PyMuPDF page object
            filename: Name of the PDF file
            page_num: Page number (1-indexed)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing page {page_num} with Vision API")
            
            # Convert page to base64 image
            base64_image, img_size = self._pdf_page_to_base64_image(pdf_page)
            logger.info(f"Image size: {img_size / 1024:.2f} KB")
            
            # Create vision prompt
            prompt = self._create_vision_extraction_prompt(filename, page_num)
            
            # Call Vision API
            response = self._call_openai_vision_api(base64_image, prompt)
            
            # Process response
            self._process_llm_response(response, filename, page_num, method="Vision_API")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing page with Vision API: {e}")
            exception_log = ExceptionLog(
                filename=filename,
                page_number=page_num,
                field_label="",
                issue_type="vision_processing_error",
                description=f"Failed to process page with Vision API: {str(e)}",
                timestamp=self._get_timestamp()
            )
            self.exceptions.append(exception_log)
            return False
    
    def _process_page_with_text(self, text_content: str, filename: str, page_num: int) -> None:
        """Process extracted text using text-based LLM"""
        chunks = self._chunk_text(text_content)
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Processing text chunk {chunk_idx + 1}/{len(chunks)} for page {page_num}")
            
            # Create prompt and call LLM
            prompt = self._create_extraction_prompt(chunk, filename, page_num)
            response = self._call_openai_text_api(prompt)
            
            # Process response
            self._process_llm_response(response, filename, page_num, method="Text_LLM")
    
    def process_pdf_file(self, pdf_path: str, force_vision: bool = False) -> None:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            force_vision: If True, use Vision API for all pages regardless of text content
        """
        logger.info(f"Processing PDF file: {pdf_path}")
        filename = os.path.basename(pdf_path)
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                logger.info(f"Processing page {page_num + 1}/{len(pdf_document)} of {filename}")
                page = pdf_document[page_num]
                
                # Strategy selection
                if force_vision or self.use_vision:
                    # Try Vision API first for comprehensive extraction
                    vision_success = self._process_page_with_vision(page, filename, page_num + 1)
                    
                    if not vision_success and not force_vision:
                        # Fallback to text extraction if vision fails
                        logger.info(f"Vision API failed, falling back to text extraction for page {page_num + 1}")
                        text_content = self._extract_text_from_pdf_page(page)
                        
                        if text_content and len(text_content.strip()) >= 50:
                            self._process_page_with_text(text_content, filename, page_num + 1)
                        else:
                            # Last resort: OCR
                            logger.info(f"Using OCR for page {page_num + 1}")
                            text_content = self._ocr_pdf_page(page)
                            if text_content:
                                self._process_page_with_text(text_content, filename, page_num + 1)
                else:
                    # Traditional text extraction approach
                    text_content = self._extract_text_from_pdf_page(page)
                    
                    if text_content and len(text_content.strip()) >= 50:
                        self._process_page_with_text(text_content, filename, page_num + 1)
                    else:
                        # Use OCR for image-based pages
                        logger.info(f"Using OCR for page {page_num + 1}")
                        text_content = self._ocr_pdf_page(page)
                        if text_content:
                            self._process_page_with_text(text_content, filename, page_num + 1)
                        else:
                            logger.warning(f"No text extracted from page {page_num + 1} of {filename}")
            
            pdf_document.close()
            logger.info(f"Completed processing {filename}")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            exception_log = ExceptionLog(
                filename=filename,
                page_number=0,
                field_label="",
                issue_type="processing_error",
                description=f"Failed to process PDF: {str(e)}",
                timestamp=self._get_timestamp()
            )
            self.exceptions.append(exception_log)
    
    def process_directory(self, directory_path: str, force_vision: bool = False) -> None:
        """
        Process all PDF files in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            force_vision: If True, use Vision API for all pages
        """
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            self.process_pdf_file(str(pdf_file), force_vision=force_vision)
    
    def export_results(self, output_file: str = "extracted_financial_data.json", 
                      exceptions_file: str = "extraction_exceptions.json") -> None:
        """Export extracted data and exceptions to JSON files"""
        # Export extracted data
        extracted_dict = []
        for item in self.extracted_data:
            extracted_dict.append({
                "filename": item.filename,
                "page_number": item.page_number,
                "field_label": item.field_label,
                "field_value": item.field_value,
                "confidence": item.confidence,
                "extraction_method": item.extraction_method
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_dict, f, indent=2, ensure_ascii=False)
        
        # Export exceptions
        exceptions_dict = []
        for exception in self.exceptions:
            exceptions_dict.append({
                "filename": exception.filename,
                "page_number": exception.page_number,
                "field_label": exception.field_label,
                "issue_type": exception.issue_type,
                "description": exception.description,
                "timestamp": exception.timestamp
            })
        
        with open(exceptions_file, 'w', encoding='utf-8') as f:
            json.dump(exceptions_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(self.extracted_data)} extracted items to {output_file}")
        logger.info(f"Exported {len(self.exceptions)} exceptions to {exceptions_file}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the extraction process"""
        total_extractions = len(self.extracted_data)
        total_exceptions = len(self.exceptions)
        
        # Count by field
        field_counts = {}
        for item in self.extracted_data:
            field_counts[item.field_label] = field_counts.get(item.field_label, 0) + 1
        
        # Count by confidence
        confidence_counts = {"high": 0, "medium": 0, "low": 0}
        for item in self.extracted_data:
            if item.confidence >= 0.8:
                confidence_counts["high"] += 1
            elif item.confidence >= 0.6:
                confidence_counts["medium"] += 1
            else:
                confidence_counts["low"] += 1
        
        # Count by extraction method
        method_counts = {}
        for item in self.extracted_data:
            method_counts[item.extraction_method] = method_counts.get(item.extraction_method, 0) + 1
        
        return {
            "total_extractions": total_extractions,
            "total_exceptions": total_exceptions,
            "extractions_by_field": field_counts,
            "confidence_distribution": confidence_counts,
            "extraction_methods": method_counts,
            "unique_files_processed": len(set(item.filename for item in self.extracted_data))
        }


def main():
    """
    Example usage of the FinancialDocumentExtractor with Vision API support
    """
    # Configuration
    OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with actual key
    ATTRIBUTES_FILE = "financial_attributes.txt"  # File containing the 24 attributes
    PDF_DIRECTORY = "pdf_documents"  # Directory containing PDF files
    
    # Set to True to use Vision API, False for text-only extraction
    USE_VISION = True
    
    # Set to True to force Vision API on all pages (even those with extractable text)
    FORCE_VISION = False
    
    try:
        # Initialize extractor
        extractor = FinancialDocumentExtractor(
            OPENAI_API_KEY, 
            ATTRIBUTES_FILE,
            use_vision=USE_VISION
        )
        
        # Process all PDFs in directory
        extractor.process_directory(PDF_DIRECTORY, force_vision=FORCE_VISION)
        
        # Export results
        extractor.export_results()
        
        # Print summary statistics
        stats = extractor.get_summary_statistics()
        print("\n=== EXTRACTION SUMMARY ===")
        print(f"Total extractions: {stats['total_extractions']}")
        print(f"Total exceptions: {stats['total_exceptions']}")
        print(f"Files processed: {stats['unique_files_processed']}")
        print(f"Confidence distribution: {stats['confidence_distribution']}")
        print(f"Extraction methods used: {stats['extraction_methods']}")
        
        print("\nExtractions by field:")
        for field, count in stats['extractions_by_field'].items():
            print(f"  {field}: {count}")
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


# Example attributes file content (save as financial_attributes.txt)
"""
Customer Name
Customer Date of Birth
Customer Social Security Number
Customer Address
Customer Phone Number
Customer Email Address
Customer Annual Income
Customer Net Worth
Customer Investment Experience
Customer Risk Tolerance
Customer Investment Objectives
Beneficiary Information
Annuity Product Name
Annuity Product Type
Premium Amount
Payment Schedule
Surrender Period
Surrender Charges
Death Benefits
Guaranteed Minimum Income Benefits
Agent Name
Agent License Number
Sale Date
Customer Signature Date
"""

if __name__ == "__main__":
    main()