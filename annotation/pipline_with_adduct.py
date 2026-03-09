#!/usr/bin/env python
# coding=utf-8


import pandas as pd
import csv
from Metabolite_annotation_with_adduct import searching
from pathlib import Path
import os
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR

data = PROJECT_ROOT/"6mix_data.csv"
with open(data, "rt", encoding="utf-8", errors='ignore') as csvfile:
    reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=",")
    output = [row for row in reader]

data_path =PROJECT_ROOT/"result_temp.csv"
result_path = PROJECT_ROOT/"result_no_adduct"
if not os.path.exists(result_path):
    os.makedirs(result_path)

with open(data_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for h in output:
            writer.writerow(h[:-1])

searching(data_path,result_path)
print("finish")