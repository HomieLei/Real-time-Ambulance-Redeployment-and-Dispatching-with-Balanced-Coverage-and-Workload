import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def analysis(shift):
    ROA_workload_mean = []
    # ROA_to_scene_mean = []
    # ROA_service_mean = []
    # ROA_to_hospital_mean = []
    # ROA_redeploy_mean = []
    # static_mean = []

    for limit in np.arange(5,11):
        print(f"limit: {limit}")
        file = f"simulation_result_model4_Queue2_fleet20_redeploylimit{limit}.xlsx"
        # static = pd.read_excel(file, sheet_name='static')
        ROA = pd.read_excel(file, sheet_name='ROA')
        ROA_total_workload = 0
        for column in ROA.columns:
            if column == shift+"average number in queue":
                print(column)
                # static_mean.append(static[column].mean())
                ROA_total_workload = ROA[column].mean()
                ROA_workload_mean.append(ROA[column].mean())
                
            # elif column == shift+"to scene average":
            #     ROA_to_scene_mean.append(ROA[column].mean())
            # elif column == shift+"service on scene average":
            #     ROA_service_mean.append(ROA[column].mean())
            # elif column == shift+"to hospital average":
            #     ROA_to_hospital_mean.append(ROA[column].mean())
            # elif column == shift+"redeployment average":
            #     ROA_redeploy_mean.append(ROA[column].mean())

    # Plotting the data
    # plt.plot(np.arange(0.3, 0.8, 0.1), static_mean, label=f'Static {target_column}')
    plt.plot(np.arange(5,11), ROA_workload_mean, label=f'average number in queue')
    # plt.plot(np.arange(5,11), ROA_to_scene_mean, label=f'ROA to scene average')
    # plt.plot(np.arange(5,11), ROA_service_mean, label=f'ROA service on scene average')
    # plt.plot(np.arange(5,11), ROA_to_hospital_mean, label=f'ROA to hospital average')
    # plt.plot(np.arange(5,11), ROA_redeploy_mean, label=f'ROA redeployment average')
    # Adding title and labels to the axes
    plt.title(f'Redeploy limit Sensitibity analysis: {shift}Average number in queue')
    plt.xlabel(f'Rdeploy limit')
    plt.ylabel(f'Average number in queue')

    # Adding a legend to differentiate the two lines
    plt.legend() #loc='center right', bbox_to_anchor=(1, 0.7)
    plt.tight_layout()
    # Display the plot
    plt.savefig(f"{shift}Average number in queue.png")
    plt.clf()

# file = f"simulation_result_model4_Queue2_fleet20_redeploylimit5_P0.5.xlsx"
# static = pd.read_excel(file, sheet_name='static')
# for column in static.columns:
analysis("")

