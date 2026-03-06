import os
import csv
from pathlib import Path
import pandas as pd
import shutil
import ast

current_dir = Path(os.getcwd())
PROJECT_ROOT = current_dir.parent


all_files = PROJECT_ROOT / "transfer_learning/output"
stored_position = PROJECT_ROOT / "transfer_learning/model_path/best"
all_best_final = []
entries = os.listdir(all_files)

for i in entries:

    temp_files = os.listdir(os.path.join(all_files, i))
    csvs = []
    for j in temp_files:
        if ".csv" in j:
            csvs.append(j)
    all_best = []
    all_best_row = []
    for n in csvs:
        temp_path = os.path.join(all_files, i, n)
        df = pd.read_csv(temp_path)
        min_index = df.iloc[:, 4].idxmin()
        min_row = list(df.loc[min_index])
        all_best_row.append(min_row)
        all_best.append(min_row[4])
    min_value = min(all_best)
    min_indices = [index for index, value in enumerate(all_best) if value == min_value]
    best_model_name = "fold_"+str(min_indices[0])+"_best_model_weight.pth"
    temp_file_name =os.path.join(all_files, i, best_model_name)
    new_file_position = os.path.join(stored_position, i)
    best_meric = all_best_row[min_indices[0]]
    all_best_final.append(best_meric)
    print(i+"best")
    print(csvs[min_indices[0]]+"  "+str(best_meric[0]))
    if not Path(new_file_position).exists():
        Path(new_file_position).mkdir(parents=True, exist_ok=True)
    print(new_file_position)
    print(best_model_name)
    shutil.copy(temp_file_name, os.path.join(new_file_position,best_model_name))

