from pathlib import Path
from typing import List

# run from the base directory


def read_list_from_file(file_path: Path) -> List[str]:
    with open(file_path) as f:
        return [line.strip() for line in f]


def write_list_to_file(file_path: Path, lines: List[str], append: bool = False):
    text = "\n".join(lines)
    write_text_to_file(file_path, text, append)


def write_text_to_file(file_path: Path, text: str, append: bool = False):
    mode = "a" if append else "w"
    with open(file_path, mode) as f:
        f.write(text)


IDs_FILE = "human_evaluation/selected_sample_ids.txt"
INPUTS_FILE = "human_evaluation/inputs.txt"
HUMAN_RESPONSES_FILE = "human_evaluation/human_responses.txt"
SYSTEM_RESPONSES_FILE = "human_evaluation/system_responses.txt"
OUTPUT_FILE = Path("human_evaluation/form_questions.txt")

ids = [int(id) - 1 for id in read_list_from_file(IDs_FILE)]

inputs = read_list_from_file(INPUTS_FILE)
human_responses = read_list_from_file(HUMAN_RESPONSES_FILE)
system_responses = read_list_from_file(SYSTEM_RESPONSES_FILE)

selected_inputs = [inputs[id] for id in ids]
selected_human_responses = [human_responses[id] for id in ids]
selected_system_responses = [system_responses[id] for id in ids]

lines = []
for i in range(len(selected_inputs)):
    system_line = f"𝗖𝗼𝗺𝗺𝗲𝗻𝘁: {selected_inputs[i]}_____𝗥𝗲𝘀𝗽𝗼𝗻𝘀𝗲: {selected_system_responses[i]}"
    human_line = f"𝗖𝗼𝗺𝗺𝗲𝗻𝘁: {selected_inputs[i]}_____𝗥𝗲𝘀𝗽𝗼𝗻𝘀𝗲: {selected_human_responses[i]}"
    lines.append(system_line)
    lines.append(human_line)

write_list_to_file(OUTPUT_FILE, lines)
