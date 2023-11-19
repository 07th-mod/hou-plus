from difflib import SequenceMatcher
import tokenize_higurashi
from typing import List

FORCE_MERGE_ON_OUTPUTLINEALL = True
DEBUG_PASSTHROUGH = False
PRINT_STATISTICS = False # This makes this script very slow, so disable if you don't need voice matching statistics

def strip_unquote(s: str):
    s = s.strip()

    if s[0] == '"':
        s = s[1:]

    if s[-1] == '"':
        s = s[:-1]

    return s

class OutputLine:
    def __init__(self, jp_name, jp_text, en_name, en_text, line_ending_type):
        self.jp_name = jp_name
        self.jp_text = jp_text
        self.en_name = en_name
        self.en_text = en_text
        self.line_ending_type = line_ending_type

    def as_outputline_call_str(self) -> str:
        return f"\tOutputLine({self.jp_name}, {self.jp_text},\n\t\t   {self.en_name}, {self.en_text}, {self.line_ending_type});\n"

    def is_wait_for_input_or_continue(self) -> bool:
        return self.line_ending_type == 'Line_WaitForInput' or self.line_ending_type == 'Line_ContinueAfterTyping'

    def is_line_normal(self) -> bool:
        return self.line_ending_type == 'Line_Normal'

    def preview(self) -> str:
        return self.en_text

    def merge(output_lines: List['OutputLine']) -> 'OutputLine':
        jp_text_noquote = ''.join([strip_unquote(o.jp_text) for o in output_lines])
        en_text_noquote = ''.join([strip_unquote(o.en_text) for o in output_lines])
        return OutputLine(
            output_lines[0].jp_name, # Use the name from the first item in list
            f'"{jp_text_noquote}"',
            output_lines[0].en_name, # Use the name from the first item in list
            f'"{en_text_noquote}"',
            output_lines[-1].line_ending_type # Use the line ending type from the last item in the list
        )

class Line:
    def __init__(self, line: str, obj: OutputLine, type: str):
        self.line = line
        self.obj = obj
        self.type = type
        self.action = 'INVALID' # To be set later
        self.action_payload = None # type: OutputLine

    def set_action(self, action: str, payload=None):
        self.action = action
        self.action_payload = payload



def parse_output_line(start, end):
    """
    Process a single OutputLine(...); function
    This takes two arguments, because it assumes that the OutputLine(...);
    function is split across two lines
    """
    combined = f'{start.rstrip()} {end.strip()}'.strip()
    args = combined.replace("OutputLine(", '').replace(');', '')

    tokens = tokenize_higurashi.tokenize_output_line(args)

    # Strip commas from tokens
    tokens = [t for t in tokens if t[1] != 'COMMA']

    # Strip leading/trailing whitespace from tokens
    tokens = [(t[0].strip(), t[1]) for t in tokens]

    if len(tokens) != 5:
        print(f"Line: {combined}")
        print(f"Tokens: {tokens}")
        raise Exception("Invalid number of arguments found for OutputLine")

    return OutputLine(tokens[0][0], tokens[1][0], tokens[2][0], tokens[3][0], tokens[4][0])


def parse_script(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()

    input_line_objs = []  # type: List[Line]

    # mark what each type of line is
    output_line_started = False
    last_start = None

    for line in lines:
        output_line_obj = None

        if output_line_started:
            line_type = 'INVALID_SPLIT_OUTPUTLINE'
        else:
            line_type = 'Copy'  # Just copy to output

        if output_line_started:
            if line.strip().endswith(');'):
                output_line_started = False
                line_type = 'OutputLineEnd'
                # Put the output line on the 'OutputLineEnd' line
                output_line_obj = parse_output_line(last_start, line)
                last_start = None
        else:
            if line.lstrip().startswith('OutputLine('):
                output_line_started = True
                line_type = 'OutputLineStart'
                last_start = line

        input_line_objs.append(Line(line, output_line_obj, line_type))

        # print(f"{line.rstrip()} | {line_type}\n", end='')

    return input_line_objs

class CheckSimilarity:
    def __init__(self, a, b) -> None:
        self.sm = SequenceMatcher(None, a, b)

    def ratio(self):
        return self.sm.ratio()

    def matching_large_sequences_count(self):
        MIN_BLOCK_SIZE = 2
        accum = 0

        for match in self.sm.get_matching_blocks():
            if match[2] > MIN_BLOCK_SIZE:
                accum += match[2]

        return accum

class ConsoleLine:
    def __init__(self, index, line):
        self.index = index
        self.line = line

    def from_list(console_lines: List[str]) -> List['ConsoleLine']:
        lines = []

        for (i, line) in enumerate(console_lines):
            lines.append(ConsoleLine(i, line))

        return lines

class OutputLineMatcher:
    def __init__(self, console_lines: List[str], supress_warnings=False):
        self.console_lines = ConsoleLine.from_list(console_lines)
        self.last_match_position = None # type: int
        self.last_match_console_text = None
        self.DEBUG_PRINT = False
        self.ENABLE_WARNINGS = not supress_warnings

    def check_merged_line_exists_in_console(self, lines_to_merge: List[Line]) -> OutputLine:
        # TODO: print error if match not sequential or distance too long
        # TODO: tune matching criteria
        # TODO: pick best match, as long as within a certain number of lines distance??
        # TODO: exclude already matched items?

        # merge the lines together before matching
        merged = OutputLine.merge([l.obj for l in lines_to_merge])

        return self.check_line_exists_in_console(merged)

    def check_line_exists_in_console(self, line: OutputLine) -> OutputLine:
        jp_text = strip_unquote(line.jp_text)

        if self.DEBUG_PRINT:
            print("Matching results for", jp_text.strip())

        result = None

        temp_last_match_position = self.last_match_position
        temp_last_match_console_text = self.last_match_console_text

        # First try searching near last match
        if self.last_match_position is not None:
            result = self.search(jp_text, self.last_match_position, self.last_match_position + 20)

        # If still not found, try searching entire area
        if result is None:
            # print("Quick match failed")
            result = self.search(jp_text)

        # Show warning if match went backwards or skipped forwards too fara
        if self.ENABLE_WARNINGS:
            if temp_last_match_position is not None and result is not None:
                need_print = False
                if self.last_match_position < temp_last_match_position:
                    print(f"WARNING: The current match is non sequential ({temp_last_match_position} -> {self.last_match_position})")
                    need_print = True
                elif self.last_match_position > temp_last_match_position + 100:
                    print(f"WARNING: The current match skipped forward many lines ({temp_last_match_position} -> {self.last_match_position})")
                    need_print = True

                if need_print:
                    print(f" - Previous console {temp_last_match_console_text.strip()}")
                    print(f" - Mangamer Merged {line.jp_text.strip()}")
                    print(f" - Console {result.strip()}")

        if result:
            return line
        else:
            return None

    def search(self, search_string: str, search_start=None, search_end=None):
        if search_start is None:
            search_start = 0

        if search_end is None:
            search_end = len(self.console_lines)

        best_match = None
        best_ratio = None

        for console_obj in self.console_lines[search_start:search_end]:
            sim = CheckSimilarity(search_string, console_obj.line)
            count = sim.matching_large_sequences_count()
            ratio = sim.ratio()
            if count > 5 and ratio > .80:
                if self.DEBUG_PRINT:
                    print(f"{console_obj.index} | {ratio} | {count} | {console_obj.line.strip()}")
                if best_match is None or ratio > best_ratio:
                    best_ratio = ratio
                    best_match = console_obj.line
                    self.last_match_position = console_obj.index
                    self.last_match_console_text = best_match

        return best_match

def try_merge(matcher: OutputLineMatcher, last_objs: List[Line], is_split_line):
    merged_outputline = None

    # Only attempt merge if there is more than 1 thing to merge, otherwise it is pointless
    if len(last_objs) > 1:
        merged_outputline = matcher.check_merged_line_exists_in_console(last_objs)

    # passthrough_split = merged_outputline and is_split_line
    if is_split_line:
        for line in last_objs:
            if line.obj:
                line.set_action('PASSTHROUGH')
            else:
                line.set_action('SKIP')

        if merged_outputline:
            last_objs[-1].set_action('PASSTHROUGH_SPLIT', payload=merged_outputline)
    elif merged_outputline is None:
        # no console match found, so do not merge the line - keep it the same as before
        for line in last_objs:
            if line.obj:
                line.set_action('PASSTHROUGH')
            else:
                line.set_action('SKIP')
    else:
        for line in last_objs:
            line.set_action('SKIP')

        last_objs[-1].set_action('USE_PAYLOAD', payload=merged_outputline)


def line_causes_split(input):
    # Whitespace
    if input.line.strip() == '':
        return False

    # Comment
    if input.line.strip().startswith('//'):
        return False

    # OutputLineAll
    if input.line.strip().startswith('OutputLineAll('):
        return False

    # # ShakeScreen
    # if input.line.strip().startswith('ShakeScreen('):
    #     return False

    # the first part of an outputLine
    if input.type == 'OutputLineStart':
        return False

    # the second part of an outputLine
    if input.type == 'OutputLineEnd':
        return False

    return True


def merge_output_lines(output_path, input_line_objs: List[Line], console_lines):
    last_was_wait_for_input = False
    last_objs = [] # type: List[Line]

    matcher = OutputLineMatcher(console_lines)

    is_split_line = False

    # Firstly, mark which lines should be merged
    for input in input_line_objs:
        if last_objs:
            if line_causes_split(input):
                is_split_line = True

        # Ugly hack to force merging every time you reach an OutputLineAll, which seems to improve matching significantly
        # in the outbreak scripts. I guess it's because thes OutputLineAll usually insert a \n, or has a Line_Normal, which implies that the
        # character has started a new phrase/sentence, which causes it to match better with the console scripts
        # For example, check line ".Well, I suppose you could summon up a ghost"
        if FORCE_MERGE_ON_OUTPUTLINEALL:
            if 'OutputLineAll' in input.line:
                if last_objs:
                    try_merge(matcher, last_objs, is_split_line)
                    is_split_line = False
                    last_objs = []

        if input.type == 'Copy':
            input.set_action('COPY') # copy the obj.line exactly
            # output_lines.append(input.line)
            ## Uncomment this if you want to avoid merging OutputLine() which have stuff inbetween them
            ## I checked and in alot of cases it does make sense to merge them despite having sprite calls etc. in between
            ## especially since it will be cross-checked against the console text
            ## TODO: need to output the skipped objects at their previous locations?
            # if last_objs:
            #     print(f"Warning: discarding objects {[o.preview() for o in last_objs]}")
            #     last_objs = []
        elif input.type == 'OutputLineStart':
            input.set_action('SKIP') # skip (will be handled in OutputLineEnd)
        elif input.type == 'OutputLineEnd':
            is_wait_for_input = False

            if input.obj:
                is_wait_for_input = input.obj.is_wait_for_input_or_continue()

            if is_wait_for_input:
                # If you get a 'Line_WaitForInput' or 'Line_ContinueAfterTyping', just continue to collect them
                last_was_wait_for_input = True
                # Postpone setting action to later when last_objs is processed
                last_objs.append(input)
            else:
                # When moving from wait for input to non-wait for input do some action
                if last_was_wait_for_input:
                    if input.obj.is_line_normal():
                        # If reached this point, last_objs contains a series of Line_WaitForInput followed by a Line_Normal
                        # Add the current obj which is a Line_Normal
                        last_objs.append(input)
                        # print([l.obj.preview() for l in last_objs])

                        try_merge(matcher, last_objs, is_split_line)
                        is_split_line = False
                    else:
                        # TODO:
                        # If reached this point, there was a series of Line_WaitForInput followed by something else. Not sure how to handle this
                        raise Exception("This situation not handled")
                        
                        # For now, mark current and previous lines as just passthrough (don't modify them)
                        for line in last_objs:
                            line.set_action('PASSTHROUGH')

                    # always clear the last_objs when finishing a group of Line_WaitForInput
                    last_objs = []
                else:
                    # Got two consecutive Line_Normal. In that case, just pass through the outputline
                    input.set_action('PASSTHROUGH')

                last_was_wait_for_input = False

        else:
            raise Exception(f"Got invalid/error line type {input.type}")

    # If the script does not end with a Line_Normal, any OutputLine() still in last_objs will not be handled
    # In that case, try to merge them
    if last_objs:
        try_merge(matcher, last_objs, is_split_line)
        is_split_line = False
        last_objs = []


    # Now handle the lines
    merge_count = 0
    split_count = 0
    output_lines = []
    for input in input_line_objs:
        # This test/debug code make sure we haven't mangled parsing the OutputLine arguments
        # When DEBUG_PASSTHROUGH is enabled, the generated output file should exactly match the input file
        # If there are any differences, there is a problem with either the output code, or the initial parsing.
        if DEBUG_PASSTHROUGH:
            if input.action == 'SKIP':
                pass
            elif input.obj:
                output_lines.append(input.obj.as_outputline_call_str())
            else:
                output_lines.append(input.line)

            continue

        if input.action == 'COPY':
            output_lines.append(input.line)
        elif input.action == 'SKIP':
            pass
        elif input.action == 'PASSTHROUGH':
            output_lines.append(input.obj.as_outputline_call_str())
        elif input.action == 'PASSTHROUGH_SPLIT':
            split_preview = input.action_payload.as_outputline_call_str().replace('\n', '').strip()
            output_lines.append(input.obj.as_outputline_call_str())
            output_lines.append(f'\t// 07th-mod Split Voice above - {split_preview}\n')
            split_count += 1
        elif input.action == 'USE_PAYLOAD':
            output_lines.append(input.action_payload.as_outputline_call_str())
            merge_count += 1
        else:
            raise Exception(f"Unknown action {input.action} when handling {input.line}")

        # print(f"{input.action} {input.line.strip()}")

    print(f"Made {merge_count} merges and {split_count} splits")

    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.writelines(output_lines)


def print_script_matching_statistics(input_line_objs, console_lines, name):
    # For debug purposes, print statistics of the unmodified file to see how well it matches
    matcher = OutputLineMatcher(console_lines, supress_warnings=True)
    total_console_voices = len(console_lines)
    match_count = 0
    for input_line in input_line_objs:
        if input_line.obj:
            if matcher.check_line_exists_in_console(input_line.obj):
                match_count += 1

    print(f"{name} Script: {match_count}/{total_console_voices} voices")

def process_file(file_path, console_script_path):
    """
    Process a single Higurashi script file
    """
    print(f"-------- Processing {file_path}... -------- ")
    output_path = file_path + '.out'

    # Load console as list of japanese lines (ignore voices for now)
    with open(console_script_path, encoding='utf-8') as f:
        console_lines = [l.split('\t', maxsplit=1)[1] for l in f.readlines()]

    ################### Parse Script ##################
    input_line_objs = parse_script(file_path)

    if PRINT_STATISTICS:
        print_script_matching_statistics(input_line_objs, console_lines, 'Unmodified')

    ################### Merge OutputLine ##################
    merge_output_lines(output_path, input_line_objs, console_lines)

    if PRINT_STATISTICS:
        output_line_objs = parse_script(output_path)
        print_script_matching_statistics(output_line_objs, console_lines, 'Merged')


process_file('Update/outbreak01_1.txt', 'console/s24.txt')
process_file('Update/outbreak02_1.txt', 'console/s25.txt')
process_file('Update/busstop01.txt', 'console/s26.txt')
