#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
import csv
import time
import math
import re
from tqdm import tqdm
import os
import ast
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR

database =PROJECT_ROOT/'HMDB_20260204.csv'

adduct_all= PROJECT_ROOT/"new_adduct(20260204).csv"
exdatapos = ""

H = 1.00727646677
Na = 22.989218
K = 38.963158
H2O = 18.011114
NH3 = 17.02654653323  # NH4-H
NH4=18.033823   #NH3+H
ACN=41.02654653323
CH3OH=32.026213
IsoProp=60.05806353323
Cl=34.969402
FA=46.00547746677
Hac=60.02112746677
TFA=113.99286246677
HCOOH = 46.005479308
COOH = HCOOH - H
CH3COO = Hac -H
NaCOOH = 44.997654276 +Na
CF3COO = 113.992863891 -H  #TFA-H
Li = 7.016004049
DAN = 158.08439833
Br = 78.918885



def swap_numbers(expression):

    match = re.search(r'(\d*\.?\d*)\*mz1.*?/(\d*\.?\d*)', expression)

    if match:

        num_before_mz1 = match.group(1)
        num_after_slash = match.group(2)


        new_expression = re.sub(
            r'(\d*\.?\d*)\*mz1(.*?)/(\d*\.?\d*)',
            f'{num_after_slash}*mz1\\2/{num_before_mz1}',
            expression
        )

        return new_expression
    else:

        return expression

def find_duplicate_positions(lst):
    df = pd.DataFrame(lst, columns=["item"])
    duplicate_positions = {}
    df['is_duplicate'] = df.duplicated('item', keep=False)
    for item, group in df.groupby('item'):
        if group['is_duplicate'].any():
            duplicate_positions[item] = group.index.tolist()
    return duplicate_positions

class Identify_Point:
    def __init__(self):
        self.fliter = Filter()

    def get_data(self,data):
        with open(data, "rt", encoding="utf-8", errors='ignore') as csvfile:
            reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=",")
            output = [row for row in reader]
            output = np.array(output)
            return output

    def getdataformlist(self,number, lst):
        temp = []
        k = 1
        if len(lst) == 0:
            k = 0
        elif len(lst) < number:
            for n in range(len(lst)):
                temp.append(lst[n][0])
            lst = []
        else:
            for i in range(number):
                temp.append(lst[i][0])
            lst = lst[number:]  # 切片
        return temp, lst, k

    def point1(self,database, exdata, adduct):
        start_time = time.time()
        print("point1")
        number1 = 2000
        number2 = 10
        temp1 = self.get_data(exdata)
        temp2 = self.get_data(database)
        mz = temp1[:, 0].astype(np.float64)
        temp2[:, 2] = temp2[:, 2].astype(np.float64)
        with open(adduct, 'r', encoding='utf-8') as f:
            reader = csv.reader((line.replace('\0', '') for line in f), delimiter=",")
            adduct = [row for row in reader]
            adduct111 = adduct
            for i in range(len(adduct111)):
                adduct111[i][0]=adduct111[i][0].replace("--", "&").replace("-", "+").replace("&", "-")
            total_iterations = math.ceil(len(temp2) / number1) * math.ceil(len(adduct) / number2)
            with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
                while True:
                        adduct = adduct111
                        if np.size(temp2) == 0:
                            break
                        temp_hmdb = temp2[:2000, :]
                        temp2 = temp2[2000:, :]
                        mz1 = [float(row[2]) for row in temp_hmdb]
                        mz1 = np.array(mz1).reshape(-1, 1)
                        while True:
                            missadduct = []
                            temp_adduct, adduct, k2 = self.getdataformlist(number2,adduct)
                            changed_adduct = []
                            for hh in range(len(temp_adduct)):
                                changed_adduct.append(swap_numbers(temp_adduct[hh]))
                            if k2 == 0:
                                break

                            else:
                                for i in range(len(changed_adduct)):
                                    missadduct.append(eval(changed_adduct[i]))
                            missadduct = np.array(missadduct)
                            missadduct = missadduct.astype(np.float64)
                            temp2_expanded = mz[np.newaxis, :]
                            temp2_expanded = temp2_expanded.astype(np.float64)
                            result = np.absolute((missadduct - temp2_expanded) /(missadduct.astype(np.float64))) * 1000000

                            position = np.where(result < 100)
                            templist = []
                            for i in range(len(position[0])):
                                a = position[0][i]
                                b = position[1][i]
                                c = position[2][i]
                                temp = [mz[c],temp_adduct[a],temp_hmdb[b][0],temp_hmdb[b][1],temp_hmdb[b][2],temp_hmdb[b][3],result[a][b][c],temp_hmdb[b][5]]
                                templist.append(temp)
                            del result
                            with open(os.path.join(path1, "pt1.csv"), "a", newline="",
                                      encoding="utf-8") as csvfile:
                                writer = csv.writer(csvfile)
                                for h in templist:
                                    if isinstance(h[6],float):
                                        writer.writerow(h)
                            del templist
                            pbar.update(1)
        return self.get_data(database)


    def point2(self):
        print("point2")
        with open(os.path.join(path1, "pt1.csv"), "rt", encoding="utf-8", errors='ignore') as csvfile:
            reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=",")
            output = [row for row in reader]
        pt1 = output
       #pt1 = self.fliter.Molecular_formula(output)
        m = []
        for i in pt1:
            m.extend([i])
        my_dict = {}
        for sublist in m:
            key = sublist[0]
            value = sublist

            if key not in my_dict:
                my_dict[key] = []

            my_dict[key].append(value)
        dict_meta = {}
        for sublist in m:
            key = sublist[3]
            value = 0
            if key not in my_dict:
                dict_meta[key] = value

        all_list = []
        for key in my_dict:
            temp = my_dict[key]
            all_list.append(temp)


        adduct_pool = ["(1*mz1-1*H)/2",'(1*mz1-1*H-1*H2O)/2','(1*mz1+1*Cl+1*H2O)/2','(1*mz1+1*K-2*H)/2',"(1*mz1+1*Na-2*H)/2",
                       '(1*mz1+1*Cl+1*H2O)/3','(1*mz1+1*K-2*H)/3',"(1*mz1-1*H)/1",'(1*mz1-1*H-1*H2O)/1','(1*mz1-1*H+1*DAN)/1']
        for tempn1 in all_list:
            for tempn2 in tempn1:
                if tempn2[1] in adduct_pool:
                    position = adduct_pool.index(tempn2[1])
                    all_adduct = ast.literal_eval(tempn2[7])
                    if all_adduct[position]==1:
                        tempn2.append(1)
                        if dict_meta[tempn2[3]]<sum(all_adduct):
                            dict_meta[tempn2[3]] +=1
                    else:
                        tempn2.append(0)
                else:
                    tempn2.append(0)
        for i in dict_meta.keys():
            if dict_meta[i]>len(adduct_pool):
                dict_meta[i] = len(adduct_pool)

        for kk1 in all_list:
            for kk2 in kk1:
                kk2.append(dict_meta[kk2[3]])
        final_pt2 = []
        for i in  all_list:
            for k in i:
                final_pt2.append(k)

        return final_pt2





    def final_score(self):
        pt2 = self.point2()

        new_pt3 = []
        for i in pt2:
            temp_ppm = float(i[6])
            score = -(1 * temp_ppm)
            i.append(score)
            new_pt3.append(i)
        print()
        return new_pt3
class Filter:
    def get_data(self, data):
        with open(data, "rt", encoding="utf-8", errors='ignore') as csvfile:
            reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=",")
            output = [row for row in reader]
            output = np.array(output)
            return output



    def extract_elements(self,string):
        elements = re.findall(r'([A-Z][a-z]*)(\d*)', string)
        dictionary = {}

        for element in elements:
            letter = element[0]
            number = element[1]

            if number:
                dictionary[letter] = int(number)
            else:
                dictionary[letter] = 1
        return dictionary

    def merge_dicts(self,dict1, dict2):
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value

        return result

    def multiply_values(self,dictionary, multiplier):
        for key in dictionary:
            dictionary[key] *= multiplier
        return dictionary

    def extract_content(self,string):
        matches = re.findall(r'-([^-*][a-zA-Z0-9*]+)', string)
        alldic = []
        fanal = {}
        for i in range(len(matches)):
            temp1 = matches[i].split("*")
            num = temp1[0]
            group = temp1[1]
            dic = self.extract_elements(group)
            dic = self.multiply_values(dic, int(num))
            alldic.append(dic)
        for i in alldic:
            fanal = self.merge_dicts(fanal, i)
        return fanal

    def Molecular_formula(self,list):
        for element in tqdm(list, desc="Processing elements"):
            adduct = element[1]
            formula = element[5]
            if re.search(r"-", adduct):
                temp1 = self.extract_elements(formula)
                result1 = self.extract_content(adduct)
                for key in result1.keys():
                    if key not in temp1.keys():
                        list.remove(element)
                        break
                    else:
                        value1 = temp1[key]
                        value2 = result1[key]
                        if value1 < value2:
                            list.remove(element)
                            break
            else:
                continue

        return list
class Format_Output:

    def changeadduct(self,adduct):

        mul_before = re.findall(r'(\d+)\*', adduct)
        a = [s.rstrip('*') for s in mul_before]
        a = [elem.replace('1', '') for elem in a]

        b = re.findall(r'\*([^()\-*\/+]*)', adduct)

        c = re.findall(r'\/([^()\-*\/]+)', adduct)
        c = [elem.replace('1', '') for elem in c]
        matches = re.findall(r'[+-]', adduct)
        temp = "[" + c[0] + "M"
        for i in range(len(a)):
            if i + 1 < len(a):
                temp = temp +matches[i]+ a[i + 1] + b[i + 1]
            else:
                if exdatapos == "":
                    temp = temp + "]" + a[0] + "-"
                    break
                if exdataneg == "":
                    temp = temp + "]" + a[0] + "+"
                    break
        return temp

    def output(self,results):
        aggregated_dict = {}

        for sublist in results:
            key = sublist[0]
            if key in aggregated_dict:
                aggregated_dict[key].append(sublist)
            else:
                aggregated_dict[key] = [sublist]

        for key in aggregated_dict:
            value_list = aggregated_dict[key]
            for value in value_list:
                value[1] = self.changeadduct(value[1])
        for key in aggregated_dict:
            aggregated_dict[key] = sorted(aggregated_dict[key], key=lambda x: (x[10],x[3]), reverse=True)



        with open(exdataneg, "rt", encoding="utf-8", errors='ignore') as csvfile:
            reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=",")
            output = [row[0] for row in reader]
            print()
        temp = aggregated_dict.keys()
        for i in output:
            if i in temp:
                continue
            else:
                aggregated_dict[i] = i

        sorted_keys = sorted(aggregated_dict.keys(), key=float)
        with open(os.path.join(path1, "outputnew(11).csv"), "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            for h in sorted_keys:
                if aggregated_dict[h] == h:
                    writer.writerow([h])
                    continue
                value = aggregated_dict[h]
                value_lengths = len(value)
                row = []
                for i in range(value_lengths):
                    row.extend(value[i])
                    row.extend(" ")
                writer.writerow(row)
        with open(os.path.join(path1, "outputnew(22).csv"), "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for h in sorted_keys:
                value = aggregated_dict[h]
                value_lengths = len(value)
                for i in range(value_lengths):
                    if isinstance(value, str):
                        writer.writerow([value])
                        break
                    else:
                        writer.writerow(value[i])

def searching(mz,save_path):
    global exdataneg
    exdataneg = mz
    global path1
    path1 = save_path
    format = Format_Output()
    pt =Identify_Point()
    pt.point1(database, exdataneg, adduct_all)
    results = pt.final_score()
    unique_list = list(set(map(tuple,results)))
    unique_list = [list(item) for item in unique_list]

    format.output(unique_list)