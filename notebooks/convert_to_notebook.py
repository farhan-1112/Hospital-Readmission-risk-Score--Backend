"""Convert percent-format .py scripts to .ipynb notebooks."""
import nbformat
import sys
from pathlib import Path

def py_to_notebook(py_path):
    """Convert a percent-style Python script to a Jupyter notebook."""
    py_path = Path(py_path)
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }
    
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split on cell markers
    blocks = content.split('\n# %%')
    
    for i, block in enumerate(blocks):
        if i == 0 and not block.strip():
            continue
        
        # Check if it's a markdown cell
        lines = block.strip().split('\n')
        if lines and lines[0].strip().startswith('[markdown]'):
            # Remove the [markdown] marker
            md_lines = lines[1:]
            # Remove leading '# ' from markdown lines
            clean_lines = []
            for line in md_lines:
                if line.startswith('# '):
                    clean_lines.append(line[2:])
                elif line == '#':
                    clean_lines.append('')
                else:
                    clean_lines.append(line)
            md_source = '\n'.join(clean_lines)
            nb.cells.append(nbformat.v4.new_markdown_cell(md_source))
        else:
            code_source = block.strip()
            if code_source:
                nb.cells.append(nbformat.v4.new_code_cell(code_source))
    
    ipynb_path = py_path.with_suffix('.ipynb')
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Created: {ipynb_path}")
    return ipynb_path

if __name__ == '__main__':
    for path in sys.argv[1:]:
        py_to_notebook(path)
