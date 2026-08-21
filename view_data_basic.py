import csv
import os

def print_file_info(filepath, max_rows=5, max_cols=10):
    print(f"File: {os.path.basename(filepath)}")
    print(f"Size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Columns: {len(header)}")
            print(f"Header: {header[:max_cols]}" + ("..." if len(header) > max_cols else ""))
            
            rows = []
            for i, row in enumerate(reader):
                if i < max_rows:
                    rows.append(row)
                else:
                    break
            
            print(f"First {len(rows)} rows:")
            for row in rows:
                print(row[:max_cols], "..." if len(row) > max_cols else "")
    except Exception as e:
        print(f"Error reading file: {e}")

print("Analyzing data files...\n")

# Analyze all_ddi.csv
print_file_info("data/raw/all_ddi.csv")
print("\n" + "-"*50 + "\n")

# Analyze all_drug_embedding.csv
print_file_info("data/raw/all_drug_embedding.csv")
print("\n" + "-"*50 + "\n")

# Analyze gene_vector.csv
print_file_info("data/raw/gene_vector.csv")
