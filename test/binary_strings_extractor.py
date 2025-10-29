#!/usr/bin/env python3
import os
import sys
import re
import shutil
import subprocess
from pathlib import Path

# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import configuration from config module
from config.config import RAW_FILES_PATH

def is_binary(file_path):
    """Check if the file is a binary file"""
    try:
        # Use file command to determine file type
        result = subprocess.run(['file', file_path], 
                               capture_output=True, text=True, check=True)
        return 'ELF' in result.stdout or 'executable' in result.stdout.lower()
    except:
        return False

def process_string_output(stdout_content):
    """Process the output from strings command to extract and clean strings"""
    processed_lines = []
    for line in stdout_content.strip().split('\n'):
        # Trim whitespace and tabs from start and end
        trimmed = line.strip(' \t')
        # Replace non-alphabetic characters with spaces
        alpha_only = re.sub(r'[^a-zA-Z]', ' ', trimmed)
        
        # Replace sequences of 3 or more consecutive identical letters with spaces
        alpha_only = re.sub(r'(.)\1{2,}', ' ', alpha_only)
        
        # Split alpha_only by spaces and add each non-empty part
        for part in alpha_only.split(' '):
            part = part.strip()
            # Only add if part is not empty, has length > 2, and contains more than one unique character
            if part and len(part) > 1 and len(set(part)) > 1:
                processed_lines.append(part)
    return set(processed_lines)

def extract_strings_from_binary(binary_path):
    """Extract strings from binary file"""
    # First use strip to remove symbols
    try:
        # Create temporary file to save stripped binary
        temp_stripped = binary_path + '.stripped'
        subprocess.run(['strip', '-s', binary_path, '-o', temp_stripped], 
                       check=True, stderr=subprocess.DEVNULL)
        
        # Then use strings to extract strings
        result = subprocess.run(['strings', temp_stripped], 
                               capture_output=True, text=True, check=True)
        
        # Clean up temporary file
        try:
            os.remove(temp_stripped)
        except:
            pass
        
        # Process the output using the helper function
        return process_string_output(result.stdout)
    except:
        # If processing fails, try using strings directly on the original file
        try:
            result = subprocess.run(['strings', binary_path], 
                                   capture_output=True, text=True, check=True)
            # Process the output using the helper function
            return process_string_output(result.stdout)
        except:
            return set()

def binary_strings_extractor(target_path, output_filename):
    if not os.path.isdir(target_path):
        print(f"Error: {target_path} is not a valid directory path")
        sys.exit(1)

    all_strings = set()
    binary_count = 0
    
    # Traverse the directory and its subdirectories
    print(f"Starting to scan directory: {target_path}")
    for root, _, files in os.walk(target_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file is binary
            if is_binary(file_path):
                binary_count += 1
                print(f"Processing binary file ({binary_count}): {file_path}")
                # Extract strings and add to total set
                all_strings.update(extract_strings_from_binary(file_path))
    
    # Sort the strings
    sorted_strings = sorted(all_strings)
    
    # Save results to file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted_strings))
    
    print(f"Processing complete!")
    print(f"Total of {binary_count} binary files processed")
    print(f"Total of {len(all_strings)} unique strings extracted")
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    # Get the passed path parameter and optional output filename
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: {sys.argv[0]} <target_path> [output_filename]")
        sys.exit(1)
    
    target_path = sys.argv[1]
    # Default output filename if not provided
    output_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strings_all_binary.txt') if len(sys.argv) == 2 else sys.argv[2]
    binary_strings_extractor(target_path, output_filename)

    #如果output_filename存在拷贝到RAW_FILES_PATH
    if os.path.exists(output_filename):
        try:
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(RAW_FILES_PATH), exist_ok=True)
            shutil.copy(output_filename, RAW_FILES_PATH)
            print(f"Successfully copied {output_filename} to {RAW_FILES_PATH}")
            # 执行script\sens_finder.py
            try:
                # 在Windows上使用python而不是python3，并使用正确的路径分隔符
                subprocess.run(['python', os.path.join('script', 'sens_finder.py')], check=True)
                print("Successfully executed sens_finder.py")
            except Exception as e:
                print(f"Error executing sens_finder.py: {e}")
        except Exception as e:
            print(f"Error copying file: {e}")