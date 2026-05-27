import json
import os

def update_ipynb_settings():
    notebook_path = 'fake_news_colab_standalone.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open('colab_experiments.py', 'r', encoding='utf-8') as f:
        new_colab_exp = f.read()
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Update colab_experiments.py cell
            if source.startswith('%%writefile colab_experiments.py'):
                new_source = f"%%writefile colab_experiments.py\n{new_colab_exp}"
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_ipynb_settings()
    print("Notebook updated successfully.")
