"""
PyLUMINA : Laser Unveiling Modelling for Interactive Numerical Applications

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
from tkinter import scrolledtext, messagebox
import subprocess
from PIL import Image, ImageTk, ImageOps

def list_python_scripts():
    """Lists all Python scripts starting with 'Scenario' in the current directory, sorted alphabetically."""
    return sorted([f for f in os.listdir('.') if f.startswith("Scenario") and f.endswith(".py")])

def show_docstring(event):
    """Displays the docstring of the selected Python script in the text box."""
    selected_file = script_listbox.get(script_listbox.curselection())
    docstring = extract_docstring(selected_file)
    doc_text.config(state=tk.NORMAL)
    doc_text.delete(1.0, tk.END)
    doc_text.insert(tk.END, docstring)
    doc_text.config(state=tk.DISABLED)

def extract_docstring(file_path):
    """Extracts the first docstring found in a Python script."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        if lines and (lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''")):
            docstring_lines = []
            delimiter = lines[0].strip()[:3]
            for line in lines[1:]:
                if line.strip().startswith(delimiter):
                    return '\n'.join(docstring_lines)
                docstring_lines.append(line.strip())

        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                if i + 1 < len(lines) and ('"""' in lines[i + 1] or "'''" in lines[i + 1]):
                    return lines[i + 1].strip().strip('"""').strip("'''")
    
    return "No description found."

def run_selected_script():
    """Executes the selected Python script."""
    selected_file = script_listbox.get(script_listbox.curselection())
    subprocess.run(["python3", selected_file])

def open_readme():
    """Opens README.md in a new window."""
    if not os.path.exists("README.md"):
        messagebox.showerror("Error", "README.md not found in the current directory.")
        return

    with open("README.md", "r", encoding="utf-8") as file:
        readme_content = file.read()

    readme_window = tk.Toplevel(root)
    readme_window.title("README.md")
    readme_window.geometry("600x400")

    readme_text = scrolledtext.ScrolledText(readme_window, wrap=tk.WORD, width=70, height=20)
    readme_text.insert(tk.END, readme_content)
    readme_text.config(state=tk.DISABLED)
    readme_text.pack(padx=10, pady=10)

def load_logo(invert=False):
    """Loads the logo and inverts colors if needed."""
    img = Image.open("logo.png")
    if invert:
        img = ImageOps.invert(img.convert("RGB"))  # Convert to RGB before inverting
    img = ImageTk.PhotoImage(img)
    return img

def toggle_theme():
    """Switches between Light and Dark theme and updates the logo accordingly."""
    current_theme = root.tk.call("ttk::style", "theme", "use")
    if current_theme == "azure-dark":
        root.tk.call("set_theme", "light")
        logo_label.config(image=original_logo)
        logo_label.image = original_logo
    else:
        root.tk.call("set_theme", "dark")
        logo_label.config(image=inverted_logo)
        logo_label.image = inverted_logo

# GUI Setup
root = tk.Tk()
root.title("Welcome to pyLUMINA")

# Load images
original_logo = load_logo(invert=False)
inverted_logo = load_logo(invert=True)

# Define icon
root.iconphoto(True, original_logo)

# GUI Theme
theme_path = os.path.join(os.path.dirname(__file__), "Azure-ttk-theme-main", "azure.tcl")
root.tk.call("source", theme_path)
root.tk.call("set_theme", "light")  # Default to light theme

# Logo placement at the top
logo_label = tk.Label(root, image=original_logo)
logo_label.pack(pady=10)

# GUI Title
title_label = tk.Label(root, text="Welcome to pyLUMINA", font=("Arial", 16, "bold"))
title_label.pack(pady=5)

# Listbox for scenario selection
script_listbox = tk.Listbox(root, width=50, height=10)
script_listbox.pack(pady=5)
script_listbox.bind("<<ListboxSelect>>", show_docstring)

# Populate the listbox with available scenarios
scripts = list_python_scripts()
for script in scripts:
    script_listbox.insert(tk.END, script)

# Description box with default message
doc_text = scrolledtext.ScrolledText(root, width=60, height=10, state=tk.NORMAL)
doc_text.insert(tk.END, "Click on a Scenario to get its description")
doc_text.config(state=tk.DISABLED)
doc_text.pack(pady=5)

# Frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10, fill=tk.X)

# Theme switch button (Left)
theme_button = tk.Button(button_frame, text="Switch Theme", command=toggle_theme)
theme_button.pack(side=tk.LEFT, padx=10)

# Run button (Centered)
run_button = tk.Button(button_frame, text="Run Selected Scenario", command=run_selected_script)
run_button.pack(side=tk.LEFT, expand=True)

# Readme button (Right)
readme_button = tk.Button(button_frame, text="Open README.md", command=open_readme)
readme_button.pack(side=tk.RIGHT, padx=10)

# Run the GUI loop
root.mainloop()

