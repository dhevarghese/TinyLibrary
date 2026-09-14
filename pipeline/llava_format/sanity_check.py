"""Check that LLaVA-format files contain paired human/GPT turns.

This file contains logic adapted from TinyLLaVA Factory and modified for
TinyLibrary. Licensed under the Apache License, Version 2.0; see
THIRD_PARTY_NOTICES.md and licenses/TinyLLaVA-Apache-2.0.txt.
"""

import argparse
import os
import json

# Adapted from _get_list_from_message() in:
# https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/74883f79083ec9b14ab8df1d47e342ce1a3aa029/tinyllava/data/template/base.py#L47
# Modified to report question/answer count mismatches for TinyLibrary files.
def get_mismatch_qa(messages, sample_id, report_path):
    """
    messages  ====>  [{from:human, value:message}, {from:gpt, value:message}]
    """
    question_list = []
    answer_list = []
    first_is_not_question = 0
    for i, message in enumerate(messages):
        if i == 0 and message['from'] != 'human':
            first_is_not_question = 1
            continue
        if i % 2 == first_is_not_question:
            question_list.append(message['value'])
        else:
            answer_list.append(message['value'])
    
    qa_mismatch = len(question_list) != len(answer_list)
    if qa_mismatch:
        with open(report_path, "a") as f:
            f.write(f"QA mismatch: questions={len(question_list)}, answers={len(answer_list)}, id={sample_id}\n")
            if len(answer_list) > len(question_list):
                f.write("More answers than questions\n")

    return qa_mismatch, len(question_list), len(answer_list)

def process_file(file_path, report_path):
    print("Checking file: ", file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)

    for datum in data:
        get_mismatch_qa(datum['conversations'], datum['id'], report_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_dir", required=True, help="Directory containing LLaVA JSON files.")
    parser.add_argument("--report", default="mismatched_qa.txt")
    args = parser.parse_args()

    files = sorted(name for name in os.listdir(args.check_dir) if name.endswith(".json"))
    for filename in files:
        process_file(os.path.join(args.check_dir, filename), args.report)


if __name__ == "__main__":
    main()
