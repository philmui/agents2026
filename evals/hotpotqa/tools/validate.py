"""Check the delivered notebook, local teaching assets and data integrity."""
from pathlib import Path
from urllib.parse import unquote, urlparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import nbformat

ROOT = Path(__file__).resolve().parents[1]
nb = nbformat.read(ROOT / 'hotpotqa_rag_evaluation.ipynb', as_version=4)
nbformat.validate(nb)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code_cells), 'Notebook must be executed'
assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
assert 'RUN_LIVE = False' in '\n'.join(c.source for c in code_cells)
for cell in nb.cells:
    if cell.cell_type == 'markdown':
        for target in re.findall(r'\]\(([^)]+)\)', cell.source):
            parsed = urlparse(target)
            if not parsed.scheme and parsed.path:
                assert (ROOT / unquote(parsed.path)).exists(), f'Missing notebook link {target}'
for path in (ROOT / 'assets').glob('*.svg'):
    tree = ET.parse(path)
    assert tree.getroot().attrib.get('viewBox')
    assert tree.find('{http://www.w3.org/2000/svg}title') is not None
provenance = json.loads((ROOT / 'data/provenance.json').read_text())
for name, expected in provenance['file_sha256'].items():
    assert hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest() == expected, name
records = json.loads((ROOT / 'data/dev_sample.json').read_text())
for record in records:
    assert len(record['context']) == 10
    context = dict(record['context'])
    for title, index in record['supporting_facts']:
        assert 0 <= index < len(context[title])
print(f'Validated {len(nb.cells)} cells, {len(code_cells)} executed code cells, local links, SVGs, and {len(records)} real records.')
