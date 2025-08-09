import sys
import re
import os

# Pure functional lambda calculus interpreter
# Lambda expressions are represented as tuples:
# ('var', name)                    - Variable
# ('lambda', param, body)          - Lambda abstraction
# ('app', func, arg)               - Function application

def is_var(expr):
    return isinstance(expr, tuple) and len(expr) == 2 and expr[0] == 'var'

def is_lambda(expr):
    return isinstance(expr, tuple) and len(expr) == 3 and expr[0] == 'lambda'

def is_app(expr):
    return isinstance(expr, tuple) and len(expr) == 3 and expr[0] == 'app'

def make_var(name):
    return ('var', name)

def make_lambda(param, body):
    return ('lambda', param, body)

def make_app(func, arg):
    return ('app', func, arg)

def expr_to_string(expr):
    if is_var(expr):
        return expr[1]
    elif is_lambda(expr):
        return f"λ{expr[1]}.{expr_to_string(expr[2])}"
    elif is_app(expr):
        func_str = f"({expr_to_string(expr[1])})" if is_lambda(expr[1]) else expr_to_string(expr[1])
        arg_str = f"({expr_to_string(expr[2])})" if is_lambda(expr[2]) or is_app(expr[2]) else expr_to_string(expr[2])
        return f"{func_str} {arg_str}"
    else:
        return str(expr)

def free_vars(expr):
    if is_var(expr):
        return {expr[1]}
    elif is_lambda(expr):
        return free_vars(expr[2]) - {expr[1]}
    elif is_app(expr):
        return free_vars(expr[1]) | free_vars(expr[2])
    else:
        return set()

def fresh_var(base, used_vars):
    counter = 1
    while f"{base}{counter}" in used_vars:
        counter += 1
    return f"{base}{counter}"

def substitute(expr, var, replacement):
    if is_var(expr):
        return replacement if expr[1] == var else expr
    elif is_lambda(expr):
        param, body = expr[1], expr[2]
        if var == param:
            return expr  # Variable is bound
        elif param in free_vars(replacement):
            # Alpha conversion needed
            new_param = fresh_var(param, free_vars(replacement) | free_vars(body) | {var})
            new_body = substitute(body, param, make_var(new_param))
            return make_lambda(new_param, substitute(new_body, var, replacement))
        else:
            return make_lambda(param, substitute(body, var, replacement))
    elif is_app(expr):
        return make_app(substitute(expr[1], var, replacement), substitute(expr[2], var, replacement))
    else:
        return expr

def beta_reduce_step(expr):
    if is_var(expr):
        return expr
    elif is_lambda(expr):
        return make_lambda(expr[1], beta_reduce_step(expr[2]))
    elif is_app(expr):
        func = beta_reduce_step(expr[1])
        arg = beta_reduce_step(expr[2])

        if is_lambda(func):
            # Beta reduction: apply lambda to argument
            return substitute(func[2], func[1], arg)
        else:
            return make_app(func, arg)
    else:
        return expr

def normalize(expr, max_steps=100000):
    for _ in range(max_steps):
        reduced = beta_reduce_step(expr)
        if expr_to_string(reduced) == expr_to_string(expr):
            break
        expr = reduced
    return expr

def parse_expression(text, pos=0):
    text = text.strip()
    pos = skip_whitespace(text, pos)
    return parse_application(text, pos)

def skip_whitespace(text, pos):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos

def parse_application(text, pos):
    expr, pos = parse_atom(text, pos)

    while True:
        pos = skip_whitespace(text, pos)
        if pos >= len(text) or text[pos] == ')':
            break

        arg, pos = parse_atom(text, pos)
        expr = make_app(expr, arg)

    return expr, pos

def parse_atom(text, pos):
    pos = skip_whitespace(text, pos)

    if pos >= len(text):
        raise ValueError("Unexpected end of input")

    if text[pos] == 'λ':
        return parse_lambda(text, pos)
    elif text[pos] == '(':
        return parse_parenthesized(text, pos)
    else:
        return parse_variable(text, pos)

def parse_lambda(text, pos):
    pos += 1  # consume 'λ'
    param, pos = parse_identifier(text, pos)

    if pos >= len(text) or text[pos] != '.':
        raise ValueError("Expected '.' after lambda parameter")

    pos += 1  # consume '.'
    body, pos = parse_expression(text, pos)
    return make_lambda(param, body), pos

def parse_parenthesized(text, pos):
    pos += 1  # consume '('
    expr, pos = parse_expression(text, pos)
    pos = skip_whitespace(text, pos)

    if pos >= len(text) or text[pos] != ')':
        raise ValueError("Expected ')' to close parentheses")

    pos += 1  # consume ')'
    return expr, pos

def parse_variable(text, pos):
    name, pos = parse_identifier(text, pos)
    return make_var(name), pos

def parse_identifier(text, pos):
    pos = skip_whitespace(text, pos)
    start = pos

    while pos < len(text) and text[pos].isalnum():
        pos += 1

    if start == pos:
        raise ValueError(f"Expected identifier at position {pos}")

    return text[start:pos], pos

headers = []

def convert_bracket_lists(line):
    """Convert bracket notation [x, y, z] to cons x (cons y (cons z nil))"""
    import re

    def bracket_to_cons(match):
        content = match.group(1).strip()
        if not content:
            return "nil"

        # Split by commas and trim whitespace
        elements = [elem.strip() for elem in content.split(',') if elem.strip()]

        if not elements:
            return "nil"

        # Build nested cons structure from right to left
        result = "nil"
        for element in reversed(elements):
            result = f"(cons {element} {result})"

        return result

    # Match bracket patterns [...]
    return re.sub(r'\[([^\]]*)\]', bracket_to_cons, line)

def preprocess(lines, base_dir="."):
    """Parse and apply headers, handle imports, return processed expression lines."""
    processed_lines = []

    for line in lines:
        line = line.strip()

        if not line or line.startswith('#'):
            continue

        # Handle imports
        if line.startswith('@'):
            target = line.removeprefix('@').strip()

            # Resolve relative path from base directory
            import_path = os.path.join(base_dir, target)

            try:
                # Check if it's a directory
                if os.path.isdir(import_path):
                    # Import all .lambda files in the directory
                    try:
                        lambda_files = [f for f in os.listdir(import_path) if f.endswith('.lambda')]
                        lambda_files.sort()  # Sort for consistent ordering

                        for lambda_file in lambda_files:
                            file_path = os.path.join(import_path, lambda_file)
                            try:
                                with open(file_path, 'rb') as f:
                                    import_lines = f.read().decode().split('\n')

                                # Recursively preprocess the imported file
                                import_base_dir = os.path.dirname(file_path)
                                imported_processed = preprocess(import_lines, import_base_dir)

                                # Add imported content
                                processed_lines.extend(imported_processed)

                            except Exception as e:
                                print(f"Error importing '{lambda_file}' from directory '{target}': {e}")

                    except Exception as e:
                        print(f"Error reading directory '{target}': {e}")

                else:
                    # Handle single file import
                    filename = target

                    # Automatically add .lambda extension if not present
                    if not filename.endswith('.lambda'):
                        filename += '.lambda'
                        import_path = os.path.join(base_dir, filename)

                    with open(import_path, 'rb') as f:
                        import_lines = f.read().decode().split('\n')

                    # Recursively preprocess the imported file
                    import_base_dir = os.path.dirname(import_path)
                    imported_processed = preprocess(import_lines, import_base_dir)

                    # Add imported content
                    processed_lines.extend(imported_processed)

            except FileNotFoundError:
                print(f"Warning: Could not import '{target}' from '{import_path}'")
            except Exception as e:
                print(f"Error importing '{target}': {e}")

            continue

        # Defining headers for syntactic sugar
        if line.startswith('$'):
            header_def = line.removeprefix('$').strip()
            parts = header_def.split('->', 1)
            if len(parts) == 2:
                pattern_part, replacement = parts
                pattern_part = pattern_part.strip()
                replacement = replacement.strip()

                # Check if this is a repetition regex pattern like [0-9]+# ([0-9+])|x
                if '#' in pattern_part:
                    repeat_pattern, actual_pattern = pattern_part.split('#', 1)
                    repeat_pattern = repeat_pattern.strip()
                    actual_pattern = actual_pattern.strip()
                    headers.append(('repeat_regex', repeat_pattern, actual_pattern, replacement))
                else:
                    # Regular pattern (treat as regex)
                    headers.append(('regex', pattern_part, replacement))
            continue

        # Apply headers to transform syntax (with recursive expansion)
        changed = True
        max_expansions = 1000  # Prevent infinite loops
        expansions = 0

        # First, convert bracket notation to cons/nil
        line = convert_bracket_lists(line)

        while changed and expansions < max_expansions:
            changed = False
            expansions += 1

            for header in headers:
                if header[0] == 'repeat_regex':
                    # Handle repetition regex: $3# ([0-9]+) -> lfx. (f ${1})
                    _, repeat_pattern, actual_pattern, replacement = header

                    # Find matches for the actual pattern
                    matches = list(re.finditer(actual_pattern, line))
                    if matches:
                        for match in matches:
                            matched_text = match.group(0)

                            # Check if this match determines repetition count
                            if re.match(repeat_pattern, matched_text):
                                try:
                                    repeat_count = int(matched_text)

                                    if repeat_count <= 100:  # Reasonable limit
                                        # Build Church numeral structure: λf.λx.f (f (f ... x))
                                        # For count=3: λf.λx.f (f (f x))
                                        if repeat_count == 0:
                                            # Church numeral 0: λf.λx.x
                                            final_result = "λf.λx.x"
                                        else:
                                            # Build the application chain: f applied repeat_count times
                                            inner = "x"
                                            for i in range(repeat_count):
                                                inner = f"f ({inner})" if i > 0 else f"f {inner}"
                                            final_result = f"λf.λx.{inner}"

                                        new_line = line.replace(matched_text, final_result, 1)
                                        if new_line != line:
                                            line = new_line
                                            changed = True
                                            break
                                except ValueError:
                                    pass  # Not a valid number

                elif header[0] == 'regex':
                    # Handle regular regex patterns: $l -> λ
                    _, pattern, replacement = header

                    def replace_func(match):
                        result = replacement
                        # Replace ${1}, ${2}, etc. with captured groups
                        for i, group in enumerate(match.groups()):
                            result = result.replace(f'${{{i+1}}}', group if group else '')
                        result = result.replace('${0}', match.group(0))

                        # If the replacement contains lambda and we're in a context where it might be applied,
                        # we need to parenthesize it properly
                        if 'λ' in result:
                            # Check if the match is followed by whitespace and another token (indicating application)
                            match_end = match.end()
                            if (match_end < len(line) and
                                line[match_end:].lstrip() and
                                not line[match_end:].lstrip()[0] in ')'):
                                result = f"({result})"

                        return result

                    new_line = re.sub(pattern, replace_func, line)
                    if new_line != line:
                        line = new_line
                        changed = True
                else:
                    # Legacy string replacement (fallback)
                    pattern, replacement = header
                    new_line = line.replace(pattern, replacement)
                    if new_line != line:
                        line = new_line
                        changed = True

        processed_lines.append(line)

    return processed_lines

def decode_church_number(expr):
    """Try to decode a Church numeral. Returns (True, number) if successful, (False, None) otherwise."""
    if not is_lambda(expr):
        return False, None

    # Church numeral should be λf.λx.BODY where BODY applies f to x some number of times
    if not is_lambda(expr[2]):
        return False, None

    f_param = expr[1]
    inner_lambda = expr[2]
    x_param = inner_lambda[1]
    body = inner_lambda[2]

    # Count how many times f is applied to x
    def count_f_applications(expr, f_var, x_var):
        """Count consecutive applications of f_var starting from x_var"""
        if is_var(expr) and expr[1] == x_var:
            return 0
        elif is_app(expr):
            func, arg = expr[1], expr[2]
            if is_var(func) and func[1] == f_var:
                # This is f applied to something
                inner_count = count_f_applications(arg, f_var, x_var)
                if inner_count is not None:
                    return inner_count + 1
        return None

    count = count_f_applications(body, f_param, x_param)
    if count is not None:
        return True, count
    return False, None

def decode_church_boolean(expr):
    """Try to decode a Church boolean. Returns (True, bool) if successful, (False, None) otherwise."""
    if not is_lambda(expr):
        return False, None

    # Church boolean should be λx.λy.x (True) or λx.λy.y (False)
    if not is_lambda(expr[2]):
        return False, None

    x_param = expr[1]
    inner_lambda = expr[2]
    y_param = inner_lambda[1]
    body = inner_lambda[2]

    if is_var(body):
        if body[1] == x_param:
            return True, True
        elif body[1] == y_param:
            return True, False

    return False, None

def decode_church_list(expr, max_depth=10):
    """Try to decode a Church list. Returns (True, list) if successful, (False, None) otherwise."""
    if max_depth <= 0:
        return False, None

    # Try to decode as NIL (empty list): λc.λn.n
    if is_lambda(expr) and is_lambda(expr[2]):
        c_param = expr[1]
        inner_lambda = expr[2]
        n_param = inner_lambda[1]
        body = inner_lambda[2]

        if is_var(body) and body[1] == n_param:
            return True, []

    # Try to decode as CONS: λc.λn.c head (tail c n)
    if is_lambda(expr) and is_lambda(expr[2]):
        c_param = expr[1]
        inner_lambda = expr[2]
        n_param = inner_lambda[1]
        body = inner_lambda[2]

        if is_app(body) and is_app(body[2]):
            # Check if it's c head (tail c n)
            if (is_var(body[1]) and body[1][1] == c_param):
                head_expr = body[2][1]  # The head element
                tail_app = body[2][2]   # The (tail c n) part

                # Try to extract the tail by applying it to a continuation
                # This is a simplified approach - in practice, Church lists are complex to decode
                if is_app(tail_app) and is_app(tail_app[2]):
                    # Recursively try to decode the tail
                    tail_success, tail_list = decode_church_list(tail_app[1], max_depth - 1)
                    if tail_success:
                        # Try to decode the head element
                        head_readable = decode_to_readable(head_expr)
                        return True, [head_readable] + tail_list

    return False, None

def decode_to_readable(expr):
    """Convert Church encodings to readable form."""
    # Try Church boolean first (since False = 0 structurally)
    is_bool, bool_val = decode_church_boolean(expr)
    if is_bool:
        return "True" if bool_val else "False"

    # Try Church number
    is_num, num = decode_church_number(expr)
    if is_num:
        return str(num)

    # Try Church list (basic attempt)
    is_list, list_val = decode_church_list(expr)
    if is_list:
        return f"[{', '.join(list_val)}]"

    # If no Church encoding detected, return the lambda expression
    return expr_to_string(expr)

def split_into_statements(processed_lines):
    """Split processed lines into separate statements based on parentheses balance and logical breaks."""
    statements = []
    current_statement = []
    paren_depth = 0
    brace_depth = 0

    i = 0
    while i < len(processed_lines):
        line = processed_lines[i].strip()

        if not line:
            # Empty line might indicate statement boundary if we're at depth 0
            if paren_depth == 0 and brace_depth == 0 and current_statement:
                statement_text = ' '.join(current_statement).strip()
                if statement_text:
                    statements.append(current_statement.copy())
                current_statement = []
            i += 1
            continue

        # Count parentheses and braces to track nesting depth
        for char in line:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1

        current_statement.append(line)

        # Check if this completes a balanced expression
        if paren_depth == 0 and brace_depth == 0:
            # Look ahead to see if the next non-empty line suggests a new statement
            next_non_empty = None
            j = i + 1
            while j < len(processed_lines):
                next_line = processed_lines[j].strip()
                if next_line:
                    next_non_empty = next_line
                    break
                j += 1

            # If there's a next line and it doesn't start with an operator or continuation,
            # this is likely a statement boundary
            if (next_non_empty is None or
                not next_non_empty.startswith((')', '}', '+', '-', '*', '/', '&&', '||', '.', ','))):
                statement_text = ' '.join(current_statement).strip()
                if statement_text:
                    statements.append(current_statement.copy())
                current_statement = []

        i += 1

    # Handle any remaining statement
    if current_statement:
        statement_text = ' '.join(current_statement).strip()
        if statement_text:
            statements.append(current_statement)

    return statements

def evaluate_expressions(processed_lines):
    """Parse and evaluate expressions, handling multiple statements."""
    statements = split_into_statements(processed_lines)

    if not statements:
        return

    results = []

    for i, statement_lines in enumerate(statements):
        # Join statement lines into a single expression
        full_text = ' '.join(statement_lines).strip()

        if not full_text:
            continue

        try:
            expr, _ = parse_expression(full_text, 0)
            normalized = normalize(expr)

            # Try to decode to readable form
            readable = decode_to_readable(normalized)

            # If it's just the lambda expression (no Church encoding), return the original
            if readable == expr_to_string(normalized):
                result = expr_to_string(normalized)
            else:
                # Return readable form
                result = readable

            results.append(result)
        except Exception as e:
            results.append(f"Error in statement {i+1}: {e}")

    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <file.lambda>")
        return

    file = sys.argv[1]

    with open(file, 'rb') as f:
        lines = f.read().decode().split('\n')

    # Get the base directory for relative imports
    base_dir = os.path.dirname(os.path.abspath(file))
    processed_lines = preprocess(lines, base_dir)

    result = evaluate_expressions(processed_lines)

    if result:
        if isinstance(result, list):
            for i, res in enumerate(result):
                print(f"Statement {i+1}: {res}")
        else:
            print(result)

if __name__ == "__main__":
    main()
