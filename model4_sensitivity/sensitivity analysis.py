import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analysis(target_column):
    ROA_mean = []
    static_mean = []

    for P in np.arange(0.3, 0.8, 0.1):
        print(f"P: {P}")
        file = f"simulation_result_model4_Queue2_fleet20_redeploylimit5_P{P:.1f}.xlsx"
        static = pd.read_excel(file, sheet_name='static')
        ROA = pd.read_excel(file, sheet_name='ROA')
        
        for column in static.columns:
            if column == target_column:
                print(column)
                static_mean.append(static[column].mean())
                ROA_mean.append(ROA[column].mean())


    # Plotting the data
    # plt.plot(np.arange(0.3, 0.8, 0.1), static_mean, label=f'Static {target_column}')
    plt.plot(np.arange(0.3, 0.8, 0.1), ROA_mean, label=f'ROA {target_column}', color = 'orange')

    # Adding title and labels to the axes
    plt.title(f'Acceptance threshold Sensitibity analysis: {target_column}')
    plt.xlabel(f'Acceptance threshold')
    plt.ylabel(f'{target_column} Mean Value')

    # Adding a legend to differentiate the two lines
    plt.legend()

    # Display the plot
    plt.savefig(f"./P_0.3to0.7/{target_column}.png")
    plt.clf()

file = f"simulation_result_model4_Queue2_fleet20_redeploylimit5_P0.5.xlsx"
static = pd.read_excel(file, sheet_name='static')
for column in static.columns:
    analysis(column)

