"""
Author: Cyril Mauclair
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
This work is licensed under the Creative Commons Attribution 4.0 International License.
You are free to:
- Share: Copy and redistribute the material in any medium or format.
- Adapt: Remix, transform, and build upon the material for any purpose, even commercially.
Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, 
  and indicate if changes were made. You may do so in any reasonable manner, 
  but not in any way that suggests the licensor endorses you or your use.
"""

import os
import tkinter as tk
from tkinter import scrolledtext
import subprocess

def list_python_scripts():
    files = [f for f in os.listdir('.') if f.startswith("Scenario") and f.endswith(".py")]
    return files

def show_docstring(event):
    selected_file = script_listbox.get(script_listbox.curselection())
    docstring = extract_docstring(selected_file)
    doc_text.config(state=tk.NORMAL)
    doc_text.delete(1.0, tk.END)
    doc_text.insert(tk.END, docstring)
    doc_text.config(state=tk.DISABLED)

def extract_docstring(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        # Vérifier si le fichier commence par une docstring
        if lines and (lines[0].strip().startswith('"""') or lines[0].strip().startswith("''")):
            docstring_lines = []
            delimiter = lines[0].strip()[:3]  # """ ou '''
            for line in lines[1:]:
                if line.strip().startswith(delimiter):
                    return '\n'.join(docstring_lines)
                docstring_lines.append(line.strip())
        
        # Si aucune docstring en début de fichier, chercher celle de la première fonction
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                if i + 1 < len(lines) and ('"""' in lines[i + 1] or "'''" in lines[i + 1]):
                    return lines[i + 1].strip().strip('"""').strip("'''")
    
    return "Aucune docstring trouvée."

def run_selected_script():
    selected_file = script_listbox.get(script_listbox.curselection())
    subprocess.run(["python3", selected_file])

# GUI Setup
root = tk.Tk()
root.title("Choose the scenario you want to run")

script_listbox = tk.Listbox(root, width=50, height=10)
script_listbox.pack()
script_listbox.bind("<<ListboxSelect>>", show_docstring)

scripts = list_python_scripts()
for script in scripts:
    script_listbox.insert(tk.END, script)

doc_text = scrolledtext.ScrolledText(root, width=60, height=10, state=tk.DISABLED)
doc_text.pack()

run_button = tk.Button(root, text="Exécuter", command=run_selected_script)
run_button.pack()


root.mainloop()

