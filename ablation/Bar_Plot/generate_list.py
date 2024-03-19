import numpy as np

radar = '93.47±0.45,94.32±0.94,95.11±0.82,95.31±0.75,94.48±0.58,94.64±0.91'
hdm05 = '60.76±0.80,62.36±0.98,70.22±0.81,63.06±0.76,70.20±0.91,49.10±1.94'
inter_session = '53.83±9.77,55.27±8.68,54.48±9.21,55.26±8.93,55.54±7.45,56.43±8.79'
inter_subject = '49.68±7.88,51.15±7.83,51.38±5.77,52.52±6.83,51.67±8.73,54.14±8.36'

results=[radar,hdm05,inter_session,inter_subject]

means=[];stds=[];
for ith in range(len(results)):
# Split the string by commas and remove any leading or trailing spaces
    tmp = results[ith]
    values = tmp.split(',')

    # Extract the mean and std values using list comprehension
    tmp_means = [float(val.split('±')[0]) for val in values]
    tmp_stds = [float(val.split('±')[1]) for val in values]
    means.append(tmp_means);stds.append(tmp_stds)


print("Mean:", means)
print("Standard Deviation:", stds)
