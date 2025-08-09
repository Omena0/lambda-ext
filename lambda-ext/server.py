from lsprotocol.types import Hover, MarkupContent, Diagnostic, DiagnosticSeverity, Position, Range, TextDocumentSyncKind, DocumentSymbol, SymbolKind, DocumentSymbolParams, Location, LocationLink, DefinitionParams, ReferenceParams, MarkupKind, CompletionItem, CompletionItemKind, CompletionList, CompletionParams
from pygls.server import LanguageServer
import logging
import sys
import re
import os

import importlib.util
import importlib
spec = importlib.util.spec_from_file_location("main", "../lambda/main.py")
if not spec or not spec.loader:
    exit("Failed to load main.py")

main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)

def log(msg):
    logging.info(msg)
    print(msg, file=sys.stderr, flush=True)

logging.basicConfig(filename='lambda-ls.log', level=logging.DEBUG)

class LambdaLanguageServer(LanguageServer):
    def __init__(self):
        super().__init__("lambda-extension", "1.0.0")
        self.symbols_by_uri = {}

lambda_server = LambdaLanguageServer()

HEADER_PATTERN = re.compile(r'^(?P<indent>\s*)\$(?P<name>.+?)\s*->\s*(?P<replacement>.+)$')
LAMBDA_PATTERN = re.compile(r'λ(?P<var>[a-zA-Z_][a-zA-Z0-9_]*)')

@lambda_server.feature('initialize')
def on_initialize(params):
    log('Language server initialized')
    return {'capabilities': {
        'textDocumentSync': TextDocumentSyncKind.Full,
        'hoverProvider': True,
        'documentSymbolProvider': True,
        'definitionProvider': True,
        'referencesProvider': True
    }}

def extract_symbols(doc):
    lines = doc.source.splitlines()
    symbols = []
    for i, line in enumerate(lines):
        header_match = HEADER_PATTERN.match(line)
        if header_match:
            name = header_match.group('name').strip()
            symbols.append({
                'name': f'${name}',
                'kind': 'header',
                'line': i,
                'char': line.find('$'),
                'range': (i, line.find('$'), i, len(line)),
            })
        for m in LAMBDA_PATTERN.finditer(line):
            var = m.group('var')
            symbols.append({
                'name': f'λ{var}',
                'kind': 'lambda',
                'line': i,
                'char': m.start(),
                'range': (i, m.start(), i, m.end()),
            })
    return symbols

def parse_headers_from_lines(lines):
    headers = {}
    for line in lines:
        m = HEADER_PATTERN.match(line)
        if m:
            name = m.group('name').strip()
            headers[name] = line.strip()
    return headers

def get_imported_headers(doc, base_dir):
    lines = doc.source.splitlines()
    headers = {}
    for line in lines:
        if line.strip().startswith('@'):
            target = line.strip()[1:].strip()
            import_path = os.path.join(base_dir, target)
            if os.path.isdir(import_path):
                for fname in os.listdir(import_path):
                    if fname.endswith('.lambda'):
                        fpath = os.path.join(import_path, fname)
                        try:
                            with open(fpath, 'r', encoding='utf-8') as f:
                                headers.update(parse_headers_from_lines(f.readlines()))
                        except Exception as e:
                            log(f'Error reading {fpath}: {e}')
            else:
                if not import_path.endswith('.lambda'):
                    import_path += '.lambda'
                try:
                    with open(import_path, 'r', encoding='utf-8') as f:
                        headers.update(parse_headers_from_lines(f.readlines()))
                except Exception as e:
                    log(f'Error reading {import_path}: {e}')
    return headers

def get_header_docs(lines):
    docs = {}
    last_comment = []
    for line in lines:
        if line.strip().startswith('#'):
            last_comment.append(line.strip().lstrip('#').strip())
        else:
            m = HEADER_PATTERN.match(line)
            if m:
                name = m.group('name').strip()
                docs[name] = '\n'.join(last_comment) if last_comment else ''
            last_comment = []
    return docs

def extract_lambda_expr(line, char_pos):
    # Try to extract the full lambda expression under the cursor, including application
    # Find the nearest lambda abstraction to the left of the cursor
    matches = list(re.finditer(r'λ([a-zA-Z_][a-zA-Z0-9_]*)\.', line))
    for m in reversed(matches):
        if m.start() <= char_pos:
            # Try to include the application if present (e.g., λx.x x)
            expr = line[m.start():].strip()
            # If there is an application after, include it
            after = line[m.end():].strip()
            if after:
                # Only include the next word (argument) for simple cases
                arg_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*|\([^\)]+\))', after)
                if arg_match:
                    expr = expr + ' ' + arg_match.group(1)
            return expr
    return None

def beta_reduce(expr, steps=5):
    # Very basic beta reduction for λ-calculus strings
    # Only for demo: does not handle all edge cases
    import re
    def step(e):
        # λx.E V => E[x:=V] (very naive, only for λx.x y)
        m = re.match(r'λ([a-zA-Z_][a-zA-Z0-9_]*)\.(.+) ([^\s\)]+)$', e)
        if m:
            var, body, val = m.group(1), m.group(2), m.group(3)
            # Replace all free occurrences of var in body with val
            # (not safe for variable capture, demo only)
            return body.replace(var, val)
        return e
    history = [expr]
    for _ in range(steps):
        nex = step(history[-1])
        if nex == history[-1]:
            break
        history.append(nex)
    return history

@lambda_server.feature('textDocument/documentSymbol')
def on_document_symbol(ls, params: DocumentSymbolParams):
    uri = params.text_document.uri
    doc = ls.workspace.get_text_document(uri)
    symbols = []
    for sym in extract_symbols(doc):
        symbols.append(DocumentSymbol(
            name=sym['name'],
            kind=SymbolKind.Function if sym['kind'] == 'header' else SymbolKind.Variable,
            range=Range(
                start=Position(line=sym['range'][0], character=sym['range'][1]),
                end=Position(line=sym['range'][2], character=sym['range'][3])
            ),
            selection_range=Range(
                start=Position(line=sym['range'][0], character=sym['range'][1]),
                end=Position(line=sym['range'][2], character=sym['range'][3])
            ),
            children=[]
        ))
    # Store for later use
    ls.symbols_by_uri[uri] = extract_symbols(doc)
    log(f'Extracted {len(symbols)} symbols from {uri}')
    return symbols

@lambda_server.feature('textDocument/definition')
def on_definition(ls, params: DefinitionParams):
    uri = params.text_document.uri
    pos = params.position
    doc = ls.workspace.get_text_document(uri)
    symbols = ls.symbols_by_uri.get(uri) or extract_symbols(doc)
    lines = doc.source.splitlines()
    if pos.line < len(lines):
        line = lines[pos.line]
        for sym in symbols:
            if sym['line'] == pos.line and sym['char'] <= pos.character < sym['char'] + len(sym['name']):
                log(f'Go to definition for {sym["name"]} at {sym["range"]}')
                return [Location(
                    uri=uri,
                    range=Range(
                        start=Position(line=sym['range'][0], character=sym['range'][1]),
                        end=Position(line=sym['range'][2], character=sym['range'][3])
                    )
                )]
    return None

@lambda_server.feature('textDocument/references')
def on_references(ls, params: ReferenceParams):
    uri = params.text_document.uri
    pos = params.position
    doc = ls.workspace.get_text_document(uri)
    symbols = ls.symbols_by_uri.get(uri) or extract_symbols(doc)
    lines = doc.source.splitlines()
    target_name = None
    if pos.line < len(lines):
        line = lines[pos.line]
        for sym in symbols:
            if sym['line'] == pos.line and sym['char'] <= pos.character < sym['char'] + len(sym['name']):
                target_name = sym['name']
                break
    if not target_name:
        return []
    refs = []
    for sym in symbols:
        if sym['name'] == target_name:
            refs.append(Location(
                uri=uri,
                range=Range(
                    start=Position(line=sym['range'][0], character=sym['range'][1]),
                    end=Position(line=sym['range'][2], character=sym['range'][3])
                )
            ))
    log(f'Found {len(refs)} references for {target_name}')
    return refs

def apply_headers_to_line(line, headers, applied=None, depth=0, max_depth=10):
    # Apply headers to a line, preventing runaway/recursive expansion
    if applied is None:
        applied = set()
    if depth > max_depth:
        return line
    changed = True
    while changed:
        changed = False
        for pattern, replacement in headers.items():
            if pattern in applied:
                continue  # Prevent re-applying the same header in this chain
            try:
                new_line = re.sub(pattern, replacement, line)
                if new_line != line:
                    # Mark this header as applied for this expansion chain
                    applied.add(pattern)
                    # Recursively apply headers to the new line, increasing depth
                    return apply_headers_to_line(new_line, headers, applied, depth+1, max_depth)
            except Exception:
                pass
    return line

def parse_multiline_statement(lines, line_idx):
    # Find the full statement (possibly multiline) containing line_idx
    # Simple heuristic: join lines until next blank or end
    start = line_idx
    while start > 0 and lines[start-1].strip() != '':
        start -= 1
    end = line_idx
    while end+1 < len(lines) and lines[end+1].strip() != '':
        end += 1
    return '\n'.join(lines[start:end+1])

@lambda_server.feature('textDocument/hover')
def on_hover(ls, params):
    try:
        doc = ls.workspace.get_text_document(params.text_document.uri)
        pos = params.position
        lines = doc.source.splitlines()
        base_dir = os.path.dirname(doc.path)

        # Find the full statement for hover (multiline aware)
        if pos.line < len(lines):
            statement = parse_multiline_statement(lines, pos.line)
            # Skip hover for comments and headers
            if statement.strip().startswith('#') or statement.strip().startswith('$'):
                log('Hover on comment or header, skipping beta reduction.')
                return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=''))
            log(f'[HOVER] Statement: {statement}')
            # Use main.preprocess for header expansion
            processed = main.preprocess([statement], base_dir)
            expanded = processed[0] if processed else statement
            log(f'[HOVER] Expanded: {expanded}')

            # Check for folder import hover (line starts with @ and is a directory)
            if statement.strip().startswith('@'):
                target = statement.strip().removeprefix('@').strip()
                import_path = os.path.join(base_dir, target)
                doc_md_path = os.path.join(import_path, 'doc.md')
                if os.path.isdir(import_path) and os.path.isfile(doc_md_path):
                    try:
                        with open(doc_md_path, 'r', encoding='utf-8') as f:
                            doc_md_content = f.read()
                        return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=doc_md_content))
                    except Exception as e:
                        log(f'[HOVER] Error reading doc.md: {e}')
                        return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=f'Error reading doc.md: {e}'))

            hover_md = ''
            # Only show expansion if it changes the statement
            if expanded != statement:
                hover_md += f'**Header Expansion:**\n\nOriginal:\n```lambda\n{statement}\n```\n\nExpanded:\n```lambda\n{expanded}\n```'
            # Beta reduction steps using main.py logic
            steps = []
            try:
                expr, _ = main.parse_expression(expanded, 0)
                steps = [main.expr_to_string(expr)]
                max_steps = 10
                for _ in range(max_steps):
                    next_expr = main.beta_reduce_step(expr)
                    next_str = main.expr_to_string(next_expr)
                    if next_str == steps[-1]:
                        break
                    steps.append(next_str)
                    expr = next_expr
                log(f'[HOVER] Beta steps: {steps}')
            except Exception as e:
                log(f'[HOVER] Beta reduction error: {e}')
            # Only show beta reduction if there is more than one step
            if len(steps) > 1:
                hover_md += '\n\n**Beta Reduction Steps:**'
                for i, s in enumerate(steps):
                    hover_md += f"\nStep {i}:\n```lambda\n{s}\n```"
            if hover_md:
                return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=hover_md))
        log('Hover on unknown')
        return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value='`Lambda expression`'))
    except Exception as e:
        log(f'Hover error: {e}')
        return None

@lambda_server.feature('textDocument/didOpen')
def did_open(ls, params):
    log(f'Document opened: {params.text_document.uri}')
    _validate(ls, params.text_document.uri)

@lambda_server.feature('textDocument/didChange')
def did_change(ls, params):
    log(f'Document changed: {params.text_document.uri}')
    _validate(ls, params.text_document.uri)

def _validate(ls, uri):
    try:
        doc = ls.workspace.get_text_document(uri)
        lines = doc.source.splitlines()
        base_dir = os.path.dirname(doc.path)
        # Preprocess the entire document to apply all headers
        processed_lines = main.preprocess(lines, base_dir)
        diagnostics = []
        for i, (orig, expanded) in enumerate(zip(lines, processed_lines)):
            if orig.strip().startswith('$') and not HEADER_PATTERN.match(orig):
                diag = Diagnostic(
                    Range(
                        start=Position(line=i, character=0),
                        end=Position(line=i, character=len(orig))
                    ),
                    'Malformed header. Expected: $pattern -> replacement'
                )
                diag.severity = DiagnosticSeverity.Warning
                diagnostics.append(diag)
            # Only check for lambda errors if not a header or comment line
            if not orig.strip().startswith('$') and not orig.strip().startswith('#'):
                for m in re.finditer(r'λ(?![a-zA-Z_])|λ$', expanded):
                    diag = Diagnostic(
                        Range(
                            start=Position(line=i, character=m.start()),
                            end=Position(line=i, character=m.end())
                        ),
                        'Lambda missing variable name'
                    )
                    diag.severity = DiagnosticSeverity.Warning
                    diagnostics.append(diag)
        log(f'Publishing {len(diagnostics)} diagnostics for {uri}')
        ls.publish_diagnostics(uri, diagnostics)
    except Exception as e:
        log(f'Validate error: {e}')

@lambda_server.feature('textDocument/completion')
def on_completion(ls, params: CompletionParams):
    doc = ls.workspace.get_text_document(params.text_document.uri)
    pos = params.position
    lines = doc.source.splitlines()
    # Use original lines for position-sensitive completion
    line = lines[pos.line][:pos.character] if pos.line < len(lines) else ''
    items = []
    if line.strip().endswith('$'):
        for l in lines:
            if l.strip().startswith('$'):
                header = l.strip().split('->')[0].strip()
                item = CompletionItem(label=header)
                item.kind = CompletionItemKind.Snippet
                item.detail = 'Header'
                items.append(item)

    elif line.strip().endswith('λ'):
        item = CompletionItem(label='λx.')
        item.kind = CompletionItemKind.Snippet
        item.detail = 'Lambda abstraction'
        items.append(item)

    else:
        for label, kind, detail in [
            ('λ', CompletionItemKind.Keyword, 'Lambda abstraction'),
            ('->', CompletionItemKind.Operator, 'Arrow operator'),
            ('$header', CompletionItemKind.Snippet, 'Header definition'),
        ]:
            item = CompletionItem(label=label)
            item.kind = kind
            item.detail = detail
            items.append(item)
    return CompletionList(is_incomplete=False, items=items)

if __name__ == '__main__':
    log('Starting lambda language server...')
    lambda_server.start_io()
