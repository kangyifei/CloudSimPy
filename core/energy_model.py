import csv
import numpy as n
input_file_path = "C:\\Users\\kangyifei\\Desktop\\atop-icx-20120315.csv"
output_file_path = "C:\\Users\\kangyifei\\Desktop\\atop-icx-20120315_cpu+power.csv"
csvsheet=[]
cpu_usage = []
power = []
with open(input_file_path, "r") as f:
    r = csv.reader(f)
    for row in r:
        csvsheet.append(row)
csvsheet.pop(0)

for i in range(190):
    for j in range(len(csvsheet)):
        cpu_usage.append(csvsheet[j][1 + i])
        power.append(csvsheet[j][191 + i])

with open(output_file_path, "w",newline='') as f:
    w = csv.writer(f)
    for i in range(len(cpu_usage)):
        w.writerow([cpu_usage[i],power[i]])
