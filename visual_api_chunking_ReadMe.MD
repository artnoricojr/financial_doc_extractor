I've created an enhanced version of the code that adds **Vision API support** for processing PDF pages as base64-encoded images. Here are the key improvements:

## **New Features:**

### **1\. Vision API Processing (`_process_page_with_vision`)**

* Converts PDF pages to base64-encoded PNG images  
* Sends images directly to OpenAI's GPT-4 Vision API  
* Handles image size optimization (reduces DPI if image exceeds 20MB limit)  
* Particularly effective for:  
  * **Handwritten content**  
  * **Image-only pages**  
  * **Complex layouts** (forms, tables)  
  * **Poor quality scans**

### **2\. Base64 Image Conversion (`_pdf_page_to_base64_image`)**

* Renders PDF pages at configurable DPI (default: 200\)  
* Encodes as: `data:image/png;base64,{pdf_data}`  
* Automatically reduces resolution if image is too large  
* Returns both base64 string and file size for monitoring

### **3\. Vision-Specific Prompt (`_create_vision_extraction_prompt`)**

* Optimized instructions for image analysis  
* Handles handwriting recognition  
* Reports blurry/unclear content  
* Identifies visual elements and layouts

### **4\. Intelligent Processing Strategy**

The system now uses a cascading approach:

1\. Vision API (primary) → Best for all document types  
   ↓ (if fails or disabled)  
2\. Text Extraction → Fast for digital PDFs  
   ↓ (if no text found)  
3\. EasyOCR (fallback) → For image-only pages

## **Configuration Options:**

extractor \= FinancialDocumentExtractor(  
    openai\_api\_key=OPENAI\_API\_KEY,  
    attributes\_file=ATTRIBUTES\_FILE,  
    use\_vision=True  \# Enable Vision API  
)

\# Process with Vision API intelligently applied  
extractor.process\_directory(PDF\_DIRECTORY)

\# OR force Vision API on ALL pages  
extractor.process\_directory(PDF\_DIRECTORY, force\_vision=True)

## **Benefits of Vision API:**

1. **Superior Accuracy**: Understands visual context and layout  
2. **Handwriting Recognition**: Better than traditional OCR for handwritten notes  
3. **Complex Documents**: Handles forms, tables, and mixed content  
4. **Single-Pass Processing**: No need for separate OCR step

## **Output Tracking:**

The `extraction_method` field now shows:

* `"Vision_API"` \- Extracted using GPT-4 Vision  
* `"Text_LLM"` \- Extracted from digital text  
* `"LLM"` \- Extracted via OCR \+ text processing

## **Usage Example:**

\# Use Vision API for comprehensive extraction  
extractor \= FinancialDocumentExtractor(  
    openai\_api\_key=YOUR\_KEY,  
    attributes\_file="financial\_attributes.txt",  
    use\_vision=True  \# Enable Vision processing  
)

extractor.process\_directory("pdf\_documents/")  
extractor.export\_results()

\# Check which methods were used  
stats \= extractor.get\_summary\_statistics()  
print(stats\['extraction\_methods'\])  
\# Output: {'Vision\_API': 45, 'Text\_LLM': 12, 'OCR': 3}

This approach provides the most comprehensive extraction capability, especially valuable for the challenging documents you described (average to poor quality, handwriting, image-only pages).

