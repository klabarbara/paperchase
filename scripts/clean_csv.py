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

def fix_quotes_and_multilines(input_path, output_path, log_path):
    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile, \
         open(log_path, 'w', encoding='utf-8') as logfile:
    

        buffer = []

        problem_count = 0

        for i, line in enumerate(infile):
            line = line.rstrip()
            
            # odd num of quote check
            if line.count('"') % 2 == 1:
                # if buffer is not empty, contains rest of problem line
                if buffer:
                    buffer.append(line)

                    combined = " ".join(buffer)
                    if combined.count('"') % 2 == 0:
                        outfile.write(combined + "\n")
                        buffer.clear()
                    else:
                        continue # keep adding to buffer if quotes not balanced

                else: 
                    # start new problem buffer
                    buffer = [line]

            else:
                if buffer: 
                    buffer.append(line)
                    combined = " ".join(buffer)
                    # if still unbalanced after combining, log it
                    if combined.count('"') % 2 ==1:
                        logfile.write(f"unfixable line {i+1}: {combined}\n")
                        problem_count += 1
                        buffer.clear()
                    else:
                        outfile.write(combined + "\n")
                        buffer.clear()
        if buffer:
            logfile.write(f"unfixable line at EOF: {' '.join(buffer)}\n")
            problem_count += 1

        print(f"fix completed. problem lines logged: {problem_count}")


input_csv = "data/raw/papersum.csv"
output_csv = "data/raw/papersum_fixed.csv"
log_csv = "data/raw/problematic_lines.log"

fix_quotes_and_multilines(input_csv, output_csv, log_csv)
