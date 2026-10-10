"""Validate the repository without installing scientific dependencies."""

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 50_000_000
PRIVATE_NAMES = {'CDSE_USER', 'CDSE_PASS', 'CDSE_S3_KEY', 'CDSE_S3_SECRET'}


def main():
    errors = []
    count = 0
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if any(part in {'.git', '.venv', '__pycache__'} for part in rel.parts) or not path.is_file():
            continue
        count += 1
        if path.stat().st_size > LIMIT:
            errors.append(f'{rel}: exceeds 50 MB')
        if path.suffix == '.py':
            try:
                text = path.read_text(encoding='utf-8-sig')
                tree = ast.parse(text, filename=str(rel))
                compile(tree, str(rel), 'exec')
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value:
                        if any(isinstance(t, ast.Name) and t.id in PRIVATE_NAMES for t in node.targets):
                            errors.append(f'{rel}: literal credential assignment')
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f'{rel}: {exc}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'Validated {count} files: Python syntax, credential assignments and 50 MB limit.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
