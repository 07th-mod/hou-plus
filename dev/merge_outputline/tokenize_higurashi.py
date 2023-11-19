import re

string_regex = re.compile(r'\s*(?<!\\)\".*?(?<!\\)\"\s*')
generic_arg_regex = re.compile(r'[^,]+')
comma = re.compile(r'\s*,')


def consume_regex(s, regex, type):
    """
    Given a string 's', try to match the regex to the start of the string
    Then on a match, split the line into the portion that matched the regex, and the portion which didn't
    Return a tuple of (match, rest_of_line, type) (type is just copied from
    argument)
    """
    match = regex.match(s)  # type: re.Match

    if match:
        mstr = match.group(0)
        return (s[:len(str(mstr))], s[len(str(mstr)):], type)
    else:
        return None


def take_token(s):
    """
    Take one token from a string, returning a tuple like (token,
    rest_of_line, type)
    """
    if s.strip() == '':
        return None

    res = consume_regex(s, comma, 'COMMA')
    if res:
        return res

    res = consume_regex(s, string_regex, 'STRING')
    if res:
        return res

    res = consume_regex(s, generic_arg_regex, 'ARG')
    if res:
        return res

    raise Exception(f"No match found for {s}")


def tokenize_output_line(line):
    """
    Given a string (argument list), tokenize into a list of tuples of
    (token_text, token_type)
    """
    tokens = []

    while True:
        retval = take_token(line)

        if retval is None:
            return tokens

        (arg, line, type) = retval

        tokens.append((arg, type))

        # print(f'{arg} | {type}')
