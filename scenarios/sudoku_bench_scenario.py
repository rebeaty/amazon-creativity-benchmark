"""
HELM Scenario: SUDOKU-BENCH

Paper: https://arxiv.org/abs/2505.16135
Code: https://github.com/SakanaAI/Sudoku-Bench
Dataset: https://huggingface.co/datasets/SakanaAI/Sudoku-Bench

Prompt format: Based on ONE_SHOT_VARIANT_PROMPT from official evaluation code
  (src/eval/prompts.py in GitHub repo)

Fields used: rules, initial_board, solution, rows, cols, visual_elements
Fields skipped: puzzle_id, sudokupad_url, author, title, encoded_puzzle (metadata/alternative formats)

Subsets:
  - challenge_100: 100 creative/variant Sudoku puzzles
  - nikoli_100: 100 handmade standard Sudoku puzzles by Nikoli
  - ctc: ~2,570 puzzles from Cracking the Cryptic

Evaluation: exact_match on 81-digit solution string
"""

import json
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Reference, Output,
    CORRECT_TAG, TEST_SPLIT
)


class SudokuBenchScenario(Scenario):
    name = "sudoku_bench"
    description = "SakanaAI/Sudoku-Bench"
    tags = ["creativity", "logical_reasoning", "constraint_satisfaction", "puzzle_solving"]

    def __init__(self, subset: str = "challenge_100"):
        """
        Args:
            subset: One of ['challenge_100', 'nikoli_100', 'ctc']
        """
        super().__init__()
        self.subset = subset

    def _format_board(self, board_string: str, rows: int = 9, cols: int = 9) -> str:
        """Convert 81-character board string to ASCII grid format."""
        board = []
        for i in range(rows):
            row = board_string[i * cols:(i + 1) * cols]
            # Replace dots with spaces for readability
            row = row.replace('.', ' ')
            # Add spacing every 3 characters for 3x3 boxes
            formatted_row = ' '.join([row[j:j+3] for j in range(0, len(row), 3)])
            board.append(formatted_row)
            # Add horizontal separator every 3 rows
            if i % 3 == 2 and i < rows - 1:
                board.append('-' * (cols + (cols // 3) - 1))
        return '\n'.join(board)

    def _format_visual_elements(self, visual_elements_str: str) -> str:
        """Format visual elements JSON into readable text."""
        if not visual_elements_str or visual_elements_str == '[]':
            return "None"

        try:
            elements = json.loads(visual_elements_str)
            if not elements:
                return "None"

            formatted = []
            for elem in elements:
                elem_type = elem.get('type', '')

                if elem_type == 'cage':
                    cells = ', '.join(elem.get('cells', []))
                    value = elem.get('value', '')
                    style = elem.get('style', 'killer')
                    formatted.append(f"{style} cage: cells {cells}, sum={value}")

                elif elem_type == 'lines':
                    color = elem.get('color_name', 'unknown')
                    coords = ' '.join(elem.get('coords', []))
                    formatted.append(f"line, color: {color}, coords: {coords}")

                elif elem_type == 'arrows':
                    color = elem.get('color_name', 'unknown')
                    coords = ' '.join(elem.get('coords', []))
                    formatted.append(f"arrow, color: {color}, coords (base to tip): {coords}")

                elif elem_type in ('overlays', 'underlays'):
                    coords = ', '.join(elem.get('coords', []))
                    shape = elem.get('shape', '')
                    text = elem.get('text', '')
                    color = elem.get('color_name', '')
                    parts = [f"{elem_type}:"]
                    if coords:
                        parts.append(f"at {coords}")
                    if shape:
                        parts.append(f"shape: {shape}")
                    if text:
                        parts.append(f"text: {text}")
                    if color:
                        parts.append(f"color: {color}")
                    formatted.append(' '.join(parts))

                else:
                    # Generic fallback
                    formatted.append(f"{elem_type}: {elem}")

            return '\n'.join(formatted)
        except (json.JSONDecodeError, Exception):
            return visual_elements_str

    def get_instances(self, output_path):
        dataset = load_dataset("SakanaAI/Sudoku-Bench", self.subset, split="test")

        instances = []
        for item in dataset:
            rows = item['rows']
            cols = item['cols']

            # Build prompt following official ONE_SHOT_VARIANT_PROMPT format
            prompt = """You are a professional Sudoku puzzle solver. Please solve the following Sudoku puzzle.

## Format Explanation ##
Coordinates:
- We use r{x}c{y} coordinates. For example, r1c1 is the top-left cell at row 1 column 1, r1c2 is the cell to the right, r2c1 is the cell below, and so on.

Visual Elements:
- Visual elements are described using rxcy coordinates
- If an element is "between" two cells, it appears on the edge between them
- Elements outside the grid use the same coordinate system (e.g., r0c1 is above r1c1)

## Size ##
{rows} x {cols}

## Rules ##
{rules}

## Visual Elements ##
{visual_elements}

## Initial Sudoku Board ##
{board}

## Answer Format ##
Please provide your answer at the end of your response. Put your answer within tags <ANSWER></ANSWER>. Your answer will be a sequence of {total} digits.

For example, the format should look like:
<ANSWER>
123456789123456789...
</ANSWER>
""".format(
                rows=rows,
                cols=cols,
                rules=item['rules'],
                visual_elements=self._format_visual_elements(item['visual_elements']),
                board=self._format_board(item['initial_board'], rows, cols),
                total=rows * cols
            )

            # Reference is the complete solution string
            references = [Reference(
                Output(text=item['solution']),
                tags=[CORRECT_TAG]
            )]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
