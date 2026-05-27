import json
import os

def update_ipynb():
    notebook_path = 'fake_news_colab_standalone.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    def read_file_content(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    # Read the updated python files
    gnn_encoders_code = read_file_content('src/models/gnn_encoders.py')
    hgfnd_code = read_file_content('src/models/hgfnd.py')
    
    # Update the cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if source.startswith('!mkdir -p src/models\n%%writefile src/models/gnn_encoders.py'):
                new_source = f"!mkdir -p src/models\n%%writefile src/models/gnn_encoders.py\n{gnn_encoders_code}"
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                # remove trailing newline on last item
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
            
            elif source.startswith('!mkdir -p src/models\n%%writefile src/models/hgfnd.py'):
                new_source = f"!mkdir -p src/models\n%%writefile src/models/hgfnd.py\n{hgfnd_code}"
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_ipynb()
