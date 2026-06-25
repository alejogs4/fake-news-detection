import json

def update_ipynb():
    notebook_path = 'fake_news_colab_standalone.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    def read_file(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    files_to_update = {
        '%%writefile src/trainer.py': read_file('src/trainer.py'),
        '%%writefile main.py': read_file('main.py'),
        '%%writefile src/models/hgfnd.py': read_file('src/models/hgfnd.py')
    }
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            for header, content in files_to_update.items():
                if header in source:
                    if header == '%%writefile src/models/hgfnd.py':
                        new_source = f"!mkdir -p src/models\n{header}\n{content}"
                    elif header == '%%writefile src/trainer.py':
                        new_source = f"!mkdir -p src\n{header}\n{content}"
                    else:
                        new_source = f"{header}\n{content}"
                    
                    cell['source'] = [line + '\n' for line in new_source.split('\n')]
                    if cell['source']:
                        cell['source'][-1] = cell['source'][-1].rstrip('\n')

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_ipynb()
    print("Notebook completely updated.")
