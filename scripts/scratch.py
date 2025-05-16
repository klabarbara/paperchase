def fix_quotes_and_multilines(input_path, output_path, log_path):
    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile, \
         open(log_path, 'w', encoding='utf-8') as logfile:

        buffer = []
        problematic_count = 0

        for i, line in enumerate(infile):
            # Strip any leading or trailing whitespace or newlines
            line = line.rstrip()

            # Check if the line has an odd number of quotes
            if line.count('"') % 2 == 1:
                # If buffer is not empty, it's a continuation of a problematic line
                if buffer:
                    buffer.append(line)
                    # Check if the combined lines now have balanced quotes
                    combined = " ".join(buffer)
                    if combined.count('"') % 2 == 0:
                        outfile.write(combined + "\n")
                        buffer.clear()
                    else:
                        continue  # Keep adding to the buffer
                else:
                    # Start a new problematic buffer
                    buffer = [line]
            else:
                # If the buffer is not empty, combine the lines
                if buffer:
                    buffer.append(line)
                    combined = " ".join(buffer)
                    # If still unbalanced after combining, log it
                    if combined.count('"') % 2 == 1:
                        logfile.write(f"Unfixable line {i+1}: {combined}\n")
                        problematic_count += 1
                        buffer.clear()
                    else:
                        outfile.write(combined + "\n")
                        buffer.clear()
                else:
                    # Regular, well-formed line
                    outfile.write(line + "\n")

        # Handle any leftover lines in the buffer
        if buffer:
            logfile.write(f"Unfixable line at EOF: {' '.join(buffer)}\n")
            problematic_count += 1

        print(f"Fix completed. Problematic lines logged: {problematic_count}")

# Paths
input_csv = "data/raw/papersum.csv"
output_csv = "data/raw/papersum_fixed.csv"
log_csv = "data/raw/problematic_lines.log"

fix_quotes_and_multilines(input_csv, output_csv, log_csv)
