# CSV Output Format - Example

## Overview
The financial document extractor now outputs results in **both JSON and CSV formats** automatically. This provides flexibility for different analysis tools and workflows.

## Output Files Generated

For each processing run, the following 4 files are created:

1. **`extraction_results_YYYYMMDD_HHMMSS.json`** - Full extraction results in JSON format
2. **`extraction_results_YYYYMMDD_HHMMSS.csv`** - Full extraction results in CSV format
3. **`exception_log_YYYYMMDD_HHMMSS.json`** - Exception/error log in JSON format
4. **`exception_log_YYYYMMDD_HHMMSS.csv`** - Exception/error log in CSV format

---

## Extraction Results CSV Format

### Column Headers
```csv
file_name,page_number,field_label,field_value,location_description,confidence,derived-from-handwriting,exception_notes,processing_notes
```

### Sample Data

**extraction_results_20250123_143022.csv**

```csv
file_name,page_number,field_label,field_value,location_description,confidence,derived-from-handwriting,exception_notes,processing_notes
application_form_001.pdf,1,Account Holder Name,John A. Smith,Top section handwritten in blue ink,MEDIUM,yes,Handwritten name with slight variations in letter size,Clear handwriting detected with cursive signature style
application_form_001.pdf,1,Account Number,4502-8871-9234,Middle section printed in standard font,HIGH,no,,Computer-generated account number in standard typeface
application_form_001.pdf,1,Date of Birth,03/15/1985,Form field printed text,HIGH,no,,Standard printed date format
application_form_001.pdf,2,Signature,J.A. Smith,Bottom of page cursive signature,LOW,yes,Cursive signature with connected letters difficult to read,Handwritten signature with typical cursive characteristics
application_form_001.pdf,2,Date,03/15/2024,Next to signature handwritten,MEDIUM,maybe,Numbers appear handwritten but with some regularity,Uncertain - could be printed numbers or neat handwriting
statement_002.pdf,1,Transaction Amount,$1250.00,Transaction table column 3,HIGH,no,,Computer-generated statement with typed values
statement_002.pdf,1,Transaction Date,01/20/2024,Transaction table column 1,HIGH,no,,Typed date in standard format
check_003.pdf,1,Amount in Words,One Thousand Two Hundred Fifty,Center of check handwritten in cursive,LOW,yes,Irregular spacing and connected letters indicate handwriting,Classic handwritten check with cursive writing
check_003.pdf,1,Payee Name,ABC Services Inc.,Printed on check,HIGH,no,,Pre-printed check field
form_with_annotation.pdf,1,Notes,APPROVED - see manager notes,Margin annotation in pen,MEDIUM,yes,Hand-written annotation added to printed form,Mixed document - printed form with handwritten annotations
form_with_annotation.pdf,1,Application ID,APP-2024-00156,Header section printed,HIGH,no,,Computer-generated tracking number
```

---

## Exception Log CSV Format

### Column Headers
```csv
file_name,page_number,field_label,issue_type,description,exception_notes,timestamp
```

### Sample Data

**exception_log_20250123_143022.csv**

```csv
file_name,page_number,field_label,issue_type,description,exception_notes,timestamp
application_form_001.pdf,2,Signature,LOW_CONFIDENCE,Extracted value has low confidence,Cursive signature with connected letters difficult to read,2025-01-23T14:30:22.456789
application_form_001.pdf,2,Date,PROCESSING_ISSUE,Processing issue reported,Numbers appear handwritten but with some regularity,2025-01-23T14:30:22.567890
check_003.pdf,1,Amount in Words,LOW_CONFIDENCE,Extracted value has low confidence,Irregular spacing and connected letters indicate handwriting,2025-01-23T14:30:22.678901
damaged_document.pdf,3,Account Balance,UNCLEAR_VALUE,Value found but unclear or ambiguous,Text is faded and partially illegible,2025-01-23T14:30:22.789012
incomplete_form.pdf,1,Beneficiary Name,MISSING_VALUE,Required information not found in document,Field appears to be left blank,2025-01-23T14:30:22.890123
form_with_annotation.pdf,1,Notes,PROCESSING_ISSUE,Processing issue reported,Hand-written annotation added to printed form,2025-01-23T14:30:22.901234
```

---

## CSV Advantages

### 1. **Excel/Spreadsheet Compatible**
- Open directly in Microsoft Excel, Google Sheets, LibreOffice Calc
- No import/conversion required
- Immediate sorting, filtering, and analysis

### 2. **Database Import**
- Easy import into SQL databases (MySQL, PostgreSQL, SQL Server)
- Compatible with data warehouse tools
- Supports bulk loading operations

### 3. **Data Analysis Tools**
- Works with Python pandas: `pd.read_csv('extraction_results.csv')`
- Compatible with R, MATLAB, SAS, SPSS
- Supports business intelligence tools (Tableau, Power BI)

### 4. **Human Readable**
- Simple text format
- Can be viewed in any text editor
- Easy to share and collaborate

### 5. **Smaller File Size**
- More compact than JSON for large datasets
- Faster to load and process
- Better for network transfer

---

## Usage Examples

### Python - Reading CSV Results

```python
import pandas as pd

# Load extraction results
results_df = pd.read_csv('extraction_results_20250123_143022.csv')

# Filter handwritten entries
handwritten = results_df[results_df['derived-from-handwriting'] == 'yes']
print(f"Found {len(handwritten)} handwritten entries")

# Filter by confidence
high_confidence = results_df[results_df['confidence'] == 'HIGH']

# Filter by specific document
doc_results = results_df[results_df['file_name'] == 'application_form_001.pdf']

# Group by field label
grouped = results_df.groupby('field_label').size()
```

### Excel - Analysis

1. Open the CSV file in Excel
2. Use **Data > Filter** to enable column filters
3. Sort by confidence, handwriting status, or file name
4. Create pivot tables for summary analysis
5. Apply conditional formatting to highlight exceptions

### SQL - Import and Query

```sql
-- Import CSV into database table
LOAD DATA INFILE 'extraction_results_20250123_143022.csv'
INTO TABLE extraction_results
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Query handwritten entries
SELECT file_name, field_label, field_value
FROM extraction_results
WHERE derived_from_handwriting = 'yes';

-- Count by confidence level
SELECT confidence, COUNT(*) as count
FROM extraction_results
GROUP BY confidence;
```

---

## JSON vs CSV Comparison

| Feature | JSON | CSV |
|---------|------|-----|
| **Structure** | Hierarchical, nested | Flat, tabular |
| **Human Readable** | Moderate | High |
| **File Size** | Larger | Smaller |
| **Excel Compatible** | No (requires import) | Yes (direct open) |
| **Data Types** | Preserved | All strings |
| **Complex Data** | Excellent | Limited |
| **API Integration** | Excellent | Good |
| **Database Import** | Moderate | Excellent |
| **Multi-line Text** | Easy | Escaped/quoted |

---

## Special Character Handling

The CSV export properly handles:

- **Commas in values**: Enclosed in double quotes
  ```csv
  "Smith, John A."
  ```

- **Quotes in values**: Escaped with double quotes
  ```csv
  "He said ""approved"""
  ```

- **Line breaks in notes**: Preserved within quoted fields
  ```csv
  "Processing note:
  Multiple lines
  of text"
  ```

- **Unicode characters**: UTF-8 encoded
  ```csv
  "José García","€1,250.00"
  ```

---

## Processing Summary Output

When running the extractor, you'll see both formats confirmed:

```
==================================================
PROCESSING SUMMARY
==================================================
Files processed: 3
Pages processed: 5
OCR pages: 2
LLM calls: 5
Errors: 0
Total records extracted: 11
Exception records: 6

OUTPUT FILES:
  Results (JSON): ./output/extraction_results_20250123_143022.json
  Results (CSV):  ./output/extraction_results_20250123_143022.csv
  Exceptions (JSON): ./output/exception_log_20250123_143022.json
  Exceptions (CSV):  ./output/exception_log_20250123_143022.csv
```

---

## Notes

- **UTF-8 Encoding**: All CSV files use UTF-8 encoding to support international characters
- **No BOM**: Files are created without Byte Order Mark for maximum compatibility
- **Line Endings**: Windows CRLF (`\r\n`) line endings for Windows compatibility
- **Empty Values**: Empty fields are left blank (not "null" or "N/A")
- **Consistent Timestamps**: All files from same run share the same timestamp
- **Header Row**: Always included as the first row
- **Quote Character**: Double quotes (`"`) used for field enclosure
- **Delimiter**: Comma (`,`) used as field separator

---

## Best Practices

1. **Choose format based on use case:**
   - Use **CSV** for spreadsheet analysis, database import, or simple viewing
   - Use **JSON** for API integration, programmatic processing, or preserving data types

2. **Keep both formats:**
   - CSV for quick human review and filtering
   - JSON as the authoritative source with full fidelity

3. **Version control:**
   - Timestamp in filename prevents overwrites
   - JSON works better with git diff
   - CSV better for manual comparison

4. **Large datasets:**
   - CSV loads faster in Excel/pandas
   - JSON better for selective loading

5. **Automation:**
   - CSV easier for shell scripting (awk, grep, cut)
   - JSON better for Python/JavaScript automation
