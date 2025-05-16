import csv
import pandas as pd

def line_by_line_reader(input_path):
    problematic_lines = []
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as infile:
            for i, line in enumerate(infile):
                # unmatched or misplaced quotes
                if line.count('"') % 2 != 0:
                    print(f"unmatched quote in line {i+1}")
                    problematic_lines.append((i+1, line.strip()))

                # commas and tabs within same line
                if '\t' in line and ',' in line:
                    print(f"mixed delimiters (tabs and commas) in line {i+1}")
                    problematic_lines.append((i+1, line.strip()))

                # prints first 5 problematic lines for inspection
                if len(problematic_lines) <= 5:
                    print(f"line {i+1}: {repr(line)}")
                
        print(f"total problematic lines detected: {len(problematic_lines)}")
    except Exception as e:
        print(f"error during plain text reading: {e}")

input_csv = "data/raw/papersum.csv"
line_by_line_reader(input_csv)

