import json
import gzip
from pathlib import Path
from tqdm import tqdm
from text_utils import simplify_nq_example

def process_nq_file(input_file, output_file):
    """Process a single NQ gzipped file and save simplified examples."""
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            example = json.loads(line)
            simplified = simplify_nq_example(example)
            f_out.write(json.dumps(simplified) + '\n')

def main():
    # Define paths
    dev_dir = Path('v1.0/dev')
    output_dir = Path('nq_open')
    output_dir.mkdir(exist_ok=True)
    
    # Get all dev files
    dev_files = list(dev_dir.glob('nq-dev-*.jsonl.gz'))
    
    # Process each file with progress bar
    output_file = output_dir / 'NQ-open.dev.jsonl'
    
    print(f"Processing {len(dev_files)} dev files...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for dev_file in tqdm(dev_files, desc="Processing dev files"):
            with gzip.open(dev_file, 'rt', encoding='utf-8') as f_in:
                for line in f_in:
                    example = json.loads(line)
                    simplified = simplify_nq_example(example)
                    f_out.write(json.dumps(simplified) + '\n')
    
    print(f"Simplified dev data saved to {output_file}")

if __name__ == '__main__':
    main()
