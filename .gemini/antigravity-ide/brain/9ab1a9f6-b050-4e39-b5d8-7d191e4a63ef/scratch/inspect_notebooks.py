import json
import sys

def inspect_notebook(filepath, output_path):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(f"=== NOTEBOOK: {filepath} ===\n\n")
        for i, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type')
            source = "".join(cell.get('source', []))
            if cell_type == 'code':
                out.write(f"--- Code Cell {i} ---\n{source}\n\n")
            elif cell_type == 'markdown':
                out.write(f"--- Markdown Cell {i} ---\n{source}\n\n")

inspect_notebook(
    r"c:\Users\Mahir sakhuja\New folder (4)\Day_2_Course_Material\Project_2_outliers_percentile\1_outliers_percentile_exercise.ipynb",
    r"c:\Users\Mahir sakhuja\New folder (4)\.gemini\antigravity-ide\brain\9ab1a9f6-b050-4e39-b5d8-7d191e4a63ef\scratch\notebook_project2.txt"
)
inspect_notebook(
    r"c:\Users\Mahir sakhuja\New folder (4)\Day_4_Course_Material\Project_6_Kmeans_Algorithm\kmeans_exercise.ipynb",
    r"c:\Users\Mahir sakhuja\New folder (4)\.gemini\antigravity-ide\brain\9ab1a9f6-b050-4e39-b5d8-7d191e4a63ef\scratch\notebook_project6.txt"
)
inspect_notebook(
    r"c:\Users\Mahir sakhuja\New folder (4)\Day_4_Course_Material\KMeans\kmeans\kmeans.ipynb",
    r"c:\Users\Mahir sakhuja\New folder (4)\.gemini\antigravity-ide\brain\9ab1a9f6-b050-4e39-b5d8-7d191e4a63ef\scratch\notebook_kmeans_iris.txt"
)
print("Done extracting notebooks!")
