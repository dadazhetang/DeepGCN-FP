import json
import csv
from collections import Counter
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR

weight_path = PROJECT_ROOT/"filtered_weights.json"
group_path = PROJECT_ROOT/"Functional group results.json"
all_data = PROJECT_ROOT/"MminH_train.txt"
FUNCTIONAL_GROUP_SMARTS = {
    "Alcohol": "[CX4][OX2H]",
    "Phenol": "[cX3][OX2H]",
    "Ether": "[CX4][OX2][CX4]",
    "Aldehyde": "[CX3H1](=[OX1])[#6]",
    "Ketone": "[#6][CX3](=[OX1])[#6]",
    "Carboxylic_Acid": "[CX3](=[OX1])[OX2H]",
    "Ester": "[CX3](=[OX1])[OX2][#6]",
    "Amide": "[N,n][CX3]=[OX1]",

    "Amine_Primary": "[NX3H2][#6]",
    "Amine_Secondary": "[NX3H1]([#6])[#6]",
    "Amine_Tertiary": "[NX3]([#6])([#6])[#6]",
    "Nitrile": "[CX2]#[NX1]",
    "Nitro": "[NX3+](=[OX1])[OX1-]",
    "Nitroso": "[NX2]=[OX1]",

    "Thiol": "[CX4][SX2H]",
    "Sulfide": "[CX4][SX2][CX4]",
    "Disulfide": "[SX2][SX2]",

    "Alkene": "[CX3]=[CX3]",
    "Alkyne": "[CX2]#[CX2]",
    "Conjugated_Diene": "[CX3]=[CX3][CX3]=[CX3]",

    "Halogen": "[F,Cl,Br,I]",
    "Phenyl": "c1ccccc1",
}
with open(weight_path , 'r', encoding='utf-8') as f:
    weight = json.load(f)

with open(group_path, 'r', encoding='utf-8') as f:
    group = json.load(f)

data_all = []
with open(all_data, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        data_all.append(row)
    data_all = data_all[1:]
def is_nonzero_half_or_more(float_list):
    if not float_list:
        return False
    total = len(float_list)
    non_zero = sum(1 for x in float_list if x != 0)
    return non_zero >= total / 2

def count_functional_groups(group_list):
    counter = Counter(group_list)
    ordered_results = []
    ordered_dict = {}
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        count = counter.get(smarts, 0)
        ordered_results.append((name, smarts, count))
        ordered_dict[name] = count
    return ordered_results, ordered_dict

all_count = []
for i in group:
    for k in i[0]:
        all_count.append(k["smarts"])
ordered_results, ordered_dict = count_functional_groups(all_count)

with open(PROJECT_ROOT/"Frequency of occurrence across all standard compounds.csv", 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Functional_Group', 'SMARTS', 'Count'])
    for name, smarts, count in ordered_results:
        writer.writerow([name, smarts, count])
    writer.writerow([])
    writer.writerow(['Summary', 'Total_Types', '', len(ordered_results)])
    writer.writerow(['Summary', 'Total_Instances', '', sum(r[2] for r in ordered_results)])

all_count_MHyes = []
all_count_MHno = []
for i in range(len(group)):
    if data_all[i][1] == "1":
        for k in group[i][0]:
            all_count_MHyes.append(k["smarts"])
    if data_all[i][1] == "0":
        for k in group[i][0]:
            all_count_MHno.append(k["smarts"])
ordered_results_MHyes, ordered_dict_MHyes = count_functional_groups(all_count_MHyes)
ordered_results_MHno, ordered_dict_MHno = count_functional_groups(all_count_MHno)

with open(PROJECT_ROOT/"Frequency of occurrence in standards forming [M-H]⁻.csv", 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Functional_Group', 'SMARTS', 'Count'])
    for name, smarts, count in ordered_results_MHyes:
        writer.writerow([name, smarts, count])
    writer.writerow([])
    writer.writerow(['Summary', 'Total_Types', '', len(ordered_results_MHyes)])
    writer.writerow(['Summary', 'Total_Instances', '', sum(r[2] for r in ordered_results_MHyes)])

with open(PROJECT_ROOT/"Frequency of occurrence in standards not forming [M-H]⁻", 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Functional_Group', 'SMARTS', 'Count'])
    for name, smarts, count in ordered_results_MHno:
        writer.writerow([name, smarts, count])
    writer.writerow([])
    writer.writerow(['Summary', 'Total_Types', '', len(ordered_results_MHno)])
    writer.writerow(['Summary', 'Total_Instances', '', sum(r[2] for r in ordered_results_MHno)])

all_need = []
for i in range(len(weight)):
    temp_group_all = []
    temp_group_all.append(group[i][1])
    temp_group_all.append(data_all[i][1])
    temp_weight = weight[i]
    temp_group = group[i][0]
    ttemp = []
    if temp_group == []:
        continue
    else:
        for k in temp_group:
            count = 0
            temp = k
            positon_atoms = temp['atoms']
            sum_count = []
            for j in positon_atoms:
                sum_count.append(temp_weight[j])
                count+=temp_weight[j]
            if is_nonzero_half_or_more(sum_count):
                ttemp.append([temp,count])
    if ttemp ==[]:
        continue
    else:
        temp_group_all.append(ttemp)
        all_need.append(temp_group_all)
all_count_MHyes_imp = []
all_count_MHno_imp = []
temp_num = 0
for i in all_need:
    temp_num+=1
    print(temp_num)
    if i[1] == "1":
        for k in i[2]:
            if k[1] >0:
                all_count_MHyes_imp.append(k[0]["smarts"])
            else:
                all_count_MHno_imp.append(k[0]["smarts"])
    if i[1] == "0":
        for k in i[2]:
            if k[1] <0:
                all_count_MHno_imp.append(k[0]["smarts"])
            else:
                all_count_MHyes_imp.append(k[0]["smarts"])
ordered_results_MHyes_imp, ordered_dict_MHyes_imp = count_functional_groups(all_count_MHyes_imp)
ordered_results_MHno_imp, ordered_dict_MHno_imp = count_functional_groups(all_count_MHno_imp)

with open(PROJECT_ROOT/"Number of functional groups significantly promoting [M-H]⁻ formation.csv", 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Functional_Group', 'SMARTS', 'Count'])
    for name, smarts, count in ordered_results_MHyes_imp:
        writer.writerow([name, smarts, count])
    writer.writerow([])
    writer.writerow(['Summary', 'Total_Types', '', len(ordered_results_MHyes_imp)])
    writer.writerow(['Summary', 'Total_Instances', '', sum(r[2] for r in ordered_results_MHyes_imp)])

with open(PROJECT_ROOT/"Number of functional groups significantly suppressing [M-H]⁻ formation.csv", 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Functional_Group', 'SMARTS', 'Count'])
    for name, smarts, count in ordered_results_MHno_imp:
        writer.writerow([name, smarts, count])
    writer.writerow([])
    writer.writerow(['Summary', 'Total_Types', '', len(ordered_results_MHno_imp)])
    writer.writerow(['Summary', 'Total_Instances', '', sum(r[2] for r in ordered_results_MHno_imp)])
