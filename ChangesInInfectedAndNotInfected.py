import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Step 1: MRI data
mri_data = [
    {'Mouse': '3X', 'Infected': False, 'Week': 7, 'Uterus_Volume': 92600, 'Cyst_Volume': 4175, 'Cysts_to_Uterus': 4.51, 'LN_Volume': 6.90},
    {'Mouse': '3R', 'Infected': False, 'Week': 7, 'Uterus_Volume': 185800, 'Cyst_Volume': 36490, 'Cysts_to_Uterus': 19.64, 'LN_Volume': 3.54},
    {'Mouse': '3L', 'Infected': False, 'Week': 7, 'Uterus_Volume': 337200, 'Cyst_Volume': 103700, 'Cysts_to_Uterus': 30.75, 'LN_Volume': 3.05},
    {'Mouse': '3RR', 'Infected': False, 'Week': 7, 'Uterus_Volume': 128300, 'Cyst_Volume': 5762, 'Cysts_to_Uterus': 4.49, 'LN_Volume': 4.58},
    {'Mouse': '4R', 'Infected': True, 'Week': 7, 'Uterus_Volume': 91940, 'Cyst_Volume': 3735, 'Cysts_to_Uterus': 4.06, 'LN_Volume': 13.82},
    {'Mouse': '4L', 'Infected': True, 'Week': 7, 'Uterus_Volume': 95500, 'Cyst_Volume': 5212, 'Cysts_to_Uterus': 5.46, 'LN_Volume': 11.89},
    {'Mouse': '4RR', 'Infected': True, 'Week': 7, 'Uterus_Volume': 137200, 'Cyst_Volume': 31300, 'Cysts_to_Uterus': 22.81, 'LN_Volume': 9.25},
    {'Mouse': '4RL', 'Infected': True, 'Week': 7, 'Uterus_Volume': 75040, 'Cyst_Volume': 500, 'Cysts_to_Uterus': 0.67, 'LN_Volume': 10.08},
    {'Mouse': '3X', 'Infected': False, 'Week': 9, 'Uterus_Volume': 0, 'Cyst_Volume': 0, 'Cysts_to_Uterus': 0, 'LN_Volume': 5.66},
    {'Mouse': '3R', 'Infected': False, 'Week': 9, 'Uterus_Volume': 151400, 'Cyst_Volume': 24450, 'Cysts_to_Uterus': 16.15, 'LN_Volume': 3.60},
    {'Mouse': '3L', 'Infected': False, 'Week': 9, 'Uterus_Volume': 209300, 'Cyst_Volume': 75150, 'Cysts_to_Uterus': 35.91, 'LN_Volume': 4.19},
    {'Mouse': '3RR', 'Infected': False, 'Week': 9, 'Uterus_Volume': 146200, 'Cyst_Volume': 7581, 'Cysts_to_Uterus': 5.19, 'LN_Volume': 2.99},
    {'Mouse': '4R', 'Infected': True, 'Week': 9, 'Uterus_Volume': 118100, 'Cyst_Volume': 11150, 'Cysts_to_Uterus': 9.44, 'LN_Volume': 11.43},
    {'Mouse': '4L', 'Infected': True, 'Week': 9, 'Uterus_Volume': 152900, 'Cyst_Volume': 15700, 'Cysts_to_Uterus': 10.27, 'LN_Volume': 5.98},
    {'Mouse': '4RR', 'Infected': True, 'Week': 9, 'Uterus_Volume': 190000, 'Cyst_Volume': 32450, 'Cysts_to_Uterus': 17.08, 'LN_Volume': 7.46},
    {'Mouse': '4RL', 'Infected': True, 'Week': 9, 'Uterus_Volume': 84230, 'Cyst_Volume': 3735, 'Cysts_to_Uterus': 4.43, 'LN_Volume': 6.64},
    {"Mouse": "3X", 'Infected': False, 'Week': 11, "Uterus_Volume": 151900, "Cyst_Volume": 44480, "Cysts_to_Uterus": 29.28, "LN_Volume": 2.89},
    {"Mouse": "3R", 'Infected': False, 'Week': 11, "Uterus_Volume": 135900, "Cyst_Volume": 4236, "Cysts_to_Uterus": 3.12, "LN_Volume": 3.97},
    {"Mouse": "3L",'Infected': False, 'Week': 11, "Uterus_Volume": 306300, "Cyst_Volume": 92110, "Cysts_to_Uterus": 30.07, "LN_Volume": 2.99},
    {"Mouse": "3RR",'Infected': False, 'Week': 11, "Uterus_Volume": 244600, "Cyst_Volume": 20570, "Cysts_to_Uterus": 8.41, "LN_Volume": 2.36},
    {"Mouse": "4R",'Infected': True, 'Week': 11, "Uterus_Volume": 126200, "Cyst_Volume": 24340, "Cysts_to_Uterus": 19.29, "LN_Volume": 11.83},
    {"Mouse": "4L", 'Infected': True, 'Week': 11,"Uterus_Volume": 207600, "Cyst_Volume": 24790, "Cysts_to_Uterus": 11.94, "LN_Volume": 6.29},
    {"Mouse": "4RR", 'Infected': True, 'Week': 11,"Uterus_Volume": 124900, "Cyst_Volume": 26450, "Cysts_to_Uterus": 21.18, "LN_Volume": 6.78},
    {"Mouse": "4RL",'Infected': True, 'Week': 11, "Uterus_Volume": 91490, "Cyst_Volume": 1917, "Cysts_to_Uterus": 2.10, "LN_Volume": 4.48},
    {"Mouse": "3X", "Week": 13, "Infected": False, "Uterus_Volume": 109000, "Cyst_Volume": 18350, "Cysts_to_Uterus": 16.83, "LN_Volume": 5.3710},
    {"Mouse": "3R", "Week": 13, "Infected": False, "Uterus_Volume": 92130, "Cyst_Volume": 2087, "Cysts_to_Uterus": 2.27, "LN_Volume": 4.2110},
    {"Mouse": "3L", "Week": 13, "Infected": False, "Uterus_Volume": 223400, "Cyst_Volume": 30960, "Cysts_to_Uterus": 13.86, "LN_Volume": 2.9790},
    {"Mouse": "3RR", "Week": 13, "Infected": False, "Uterus_Volume": 111500, "Cyst_Volume": 6433, "Cysts_to_Uterus": 5.77, "LN_Volume": 2.2460},
    {"Mouse": "4R", "Week": 13, "Infected": True, "Uterus_Volume": 91690, "Cyst_Volume": 14270, "Cysts_to_Uterus": 15.56, "LN_Volume": 11.44},
    {"Mouse": "4L", "Week": 13, "Infected": True, "Uterus_Volume": 107900, "Cyst_Volume": 8032, "Cysts_to_Uterus": 7.44, "LN_Volume": 6.86},
    {"Mouse": "4RR", "Week": 13, "Infected": True, "Uterus_Volume": 107700, "Cyst_Volume": 21290, "Cysts_to_Uterus": 19.77, "LN_Volume": 8.62},
     {"Mouse": "3X", "Week": 15, "Infected": False, "Uterus_Volume": 188400, "Cyst_Volume": 27500, "Cysts_to_Uterus": 14.60, "LN_Volume": 5.4200},
    {"Mouse": "3R", "Week": 15, "Infected": False, "Uterus_Volume": 97610, "Cyst_Volume": 13380, "Cysts_to_Uterus": 13.71, "LN_Volume": 4.0040},
    {"Mouse": "3L", "Week": 15, "Infected": False, "Uterus_Volume": 340100, "Cyst_Volume": 116600, "Cysts_to_Uterus": 34.28, "LN_Volume": 3.2590},
    {"Mouse": "3RR", "Week": 15, "Infected": False, "Uterus_Volume": 153300, "Cyst_Volume": 29850, "Cysts_to_Uterus": 19.47, "LN_Volume": 2.3190}, 
    {"Mouse": "4R", "Week": 15, "Infected": True, "Uterus_Volume": 109400, "Cyst_Volume": 29200, "Cysts_to_Uterus": 26.69, "LN_Volume": 13.99},
    {"Mouse": "4L", "Week": 15, "Infected": True, "Uterus_Volume": 66600, "Cyst_Volume": 1941, "Cysts_to_Uterus": 2.91, "LN_Volume": 6.19},
    {"Mouse": "4RR", "Week": 15, "Infected": True, "Uterus_Volume": 138600, "Cyst_Volume": 18540, "Cysts_to_Uterus": 13.38, "LN_Volume": 7.05}
]

variables = [
    ('Uterus_Volume', 'Uterus Volume (mm³)'),
    ('Cyst_Volume', 'Cyst Volume (mm³)'),
    ('Cysts_to_Uterus', 'Cysts/Uterus Volume (%)'),
    ('LN_Volume', 'Para-aortic LN Volume (mm³)')
]

# Unique mice
mice = sorted(set(entry['Mouse'] for entry in mri_data))

# Plot each variable over time
for var_key, var_label in variables:
    plt.figure(figsize=(8, 5))
    
    for mouse in mice:
        # Filter data for this mouse and sort by week
        mouse_data = sorted([entry for entry in mri_data if entry['Mouse'] == mouse], key=lambda x: x['Week'])
        weeks = [entry['Week'] for entry in mouse_data]
        values = [entry[var_key] for entry in mouse_data]

        plt.plot(weeks, values, marker='o', label=mouse)

    plt.title(f'{var_label} Over Time')
    plt.xlabel('Week')
    plt.ylabel(var_label)
    plt.grid(True)
    plt.legend(title='Mouse', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


