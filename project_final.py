# initial 要用的import
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra
import scipy.stats as stats
from tqdm import tqdm
# DES 要用到的import
import os, sys
# sys.path.append(r'C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/python/3.10/x64_win64/')
import cplex
# PythonSim imports
import SimClasses
import SimFunctions
import SimRNG

def simulation_model(coverage_threshold, redeployment_limit):
    print(f"coverage_threshold: {coverage_threshold}")
    zone_num = 168
    cell_size = 2
    # coverage_threshold = 6 # miles
    acceptance_threshold = 0.5 # 50%
    fleet_num = 20
    maximum_workload = 8*60  #minutes
    vehicle_velocity = 6/10  #miles/ minutes
    # redeployment_limit = 5  #minutes
    shift_length = 12*60 # minutes
    arrival_rate_1 = 6.86/60 #minutes
    arrival_rate_2 = 9.84/60 #minutes
    service_rate_1 = 1.32/60 #minutes
    service_rate_2 = 1.39/60 #minutes
    resecue_time = [15, 10, 5] #minutes
    # 用 initialize_setting 這個function去初始化下面的字典
    Policy = ['static','ROA']

    spatial_xy = {} # zone_num : (x,y)
    spatial_volume = {} # zone_num : demand volume
    vehicle_position = {} # vehicle_num : zone_num
    vehicle_status = {} # vehicle_num : idle(busy)
    vehicle_workload = {} # vehicle_num : accumulated_workload
    coverage_position = [[] for _ in range(zone_num)] # 每個zone i 可以cover到的zone_num (zone i 可以被哪些zone cover)
    redeployment_position = [[] for _ in range(zone_num)] # 每個zone i 在redeployment limit之內可以cover到的zone_num (論文中的 Big lambda)
    zone_index = np.arange(0, zone_num) 
    travel_time_matrix = np.zeros((zone_num, zone_num)) # shortest_time
    distance_matrix = np.full((zone_num, zone_num), float('inf')) # 紀錄connect及距離，用來算dijkstra //1126
    hospital_position = [21, 120, 136] # hospital at which demand zone

    # 輸出excel用
    SC = {}
    AWA = {}
    df_call_all = {}
    df_call_tp1 = {}
    df_call_tp2 = {}
    df_call_tp3 = {}
    df_call_lost = {}
    df_call_hospital ={}
    df_call_scene = {}
    df_call_within_15 = {}
    df_call_within_10 = {}
    df_call_within_5 = {}
    df_call_without_15 = {}
    df_call_without_10 = {}
    df_call_without_5 = {}
    df_shift1_avg_workload = {}
    df_shift1_std_workload = {}
    df_shift1_avg_to_scene = {}
    df_shift1_std_to_scene = {}
    df_shift1_avg_service_on_scene = {}
    df_shift1_std_service_on_scene = {}
    df_shift1_avg_to_hospital = {}
    df_shift1_std_to_hospital = {}
    df_shift1_avg_redeployment = {}
    df_shift1_std_redeployment = {}
    df_shift2_avg_workload = {}
    df_shift2_std_workload = {}
    df_shift2_avg_to_scene = {}
    df_shift2_std_to_scene = {}
    df_shift2_avg_service_on_scene = {}
    df_shift2_std_service_on_scene = {}
    df_shift2_avg_to_hospital = {}
    df_shift2_std_to_hospital = {}
    df_shift2_avg_redeployment = {}
    df_shift2_std_redeployment = {}
    df_avg_wait_time_all = {}
    df_avg_wait_time_tp1 = {}
    df_avg_wait_time_tp2 = {}
    df_avg_wait_time_tp3 = {}
    df_avg_number_in_queue = {}

    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def read_data(filename):
        df = pd.read_excel(filename,header = None)
        zone_count = 0
        for row in range(len(df.index)):
            for col in range(15):
                if not pd.isna(df.iloc[row,col]):
                    center_x, center_y = col * cell_size + cell_size / 2, row * cell_size + cell_size / 2
                    spatial_xy[zone_count] = (center_x, center_y)
                    spatial_volume[zone_count] = df.iloc[row,col]
                    zone_count += 1
                    
    def initial():
        for v in range(fleet_num):
            vehicle_status[v] = 'idle'
            vehicle_workload[v] = 0

        for cur_zone in zone_index:
            cur_x = spatial_xy[cur_zone][0]
            cur_y = spatial_xy[cur_zone][1]
            for check_zone in zone_index:
                if cur_zone != check_zone:
                    check_x = spatial_xy[check_zone][0]
                    check_y = spatial_xy[check_zone][1]
                    distance = euclidean_distance(cur_x, cur_y, check_x, check_y)
                    if distance == cell_size:
                        distance_matrix[cur_zone, check_zone] = distance
                else:
                    distance_matrix[cur_zone, check_zone] = 0
        
        # dijkstra (one-to-all nodes shortest distance)
        distances, predecessors = dijkstra(csgraph=distance_matrix, directed=False, return_predecessors=True)
        for cur_zone in zone_index:
            for check_zone in zone_index:
                if cur_zone != check_zone:
                    travel_time = distances[cur_zone][check_zone] / vehicle_velocity
                    travel_time_matrix[cur_zone, check_zone] = travel_time
                    if distances[cur_zone][check_zone] <= coverage_threshold:
                        coverage_position[cur_zone].append(check_zone)
                    if travel_time <= redeployment_limit:
                        redeployment_position[cur_zone].append(check_zone)
                else:
                    travel_time_matrix[cur_zone, check_zone] = 0

        # add empty list to excel dictionary
        for policy in Policy:
            SC[policy] = []
            AWA[policy] = []
            df_call_all[policy] = []
            df_call_tp1[policy] = []
            df_call_tp2[policy] = []
            df_call_tp3[policy] = []
            df_call_lost[policy] = []
            df_call_hospital[policy] =[]
            df_call_scene[policy] = []
            df_call_within_15[policy] = []
            df_call_within_10[policy] = []
            df_call_within_5[policy] = []
            df_call_without_15[policy] = []
            df_call_without_10[policy] = []
            df_call_without_5[policy] = []
            df_shift1_avg_workload[policy] = []
            df_shift1_std_workload[policy] = []
            df_shift1_avg_to_scene[policy] = []
            df_shift1_std_to_scene[policy] = []
            df_shift1_avg_service_on_scene[policy] = []
            df_shift1_std_service_on_scene[policy] = []
            df_shift1_avg_to_hospital[policy] = []
            df_shift1_std_to_hospital[policy] = []
            df_shift1_avg_redeployment[policy] = []
            df_shift1_std_redeployment[policy] = []
            df_shift2_avg_workload[policy] = []
            df_shift2_std_workload[policy] = []
            df_shift2_avg_to_scene[policy] = []
            df_shift2_std_to_scene[policy] = []
            df_shift2_avg_service_on_scene[policy] = []
            df_shift2_std_service_on_scene[policy] = []
            df_shift2_avg_to_hospital[policy] = []
            df_shift2_std_to_hospital[policy] = []
            df_shift2_avg_redeployment[policy] = []
            df_shift2_std_redeployment[policy] = []
            df_avg_wait_time_all[policy] = []
            df_avg_wait_time_tp1[policy] = []
            df_avg_wait_time_tp2[policy] = []
            df_avg_wait_time_tp3[policy] = []
            df_avg_number_in_queue[policy] = []
        
        ################## MCLP模型 ######################
        # 定義目標函數中y的係數
        y_obj = [spatial_volume[idx] for idx in zone_index]
        # 定義y變數的類型（'B'代表二元變數）
        y_types = ['B' for _ in range(zone_num)]
        # 定義y變數的名稱
        y_names = [f'y_{idx}' for idx in zone_index]
        
        # 定義目標函數中x的係數
        x_obj = [0 for _ in range(zone_num)]
        # 定義x變數的類型（'B'代表二元變數）
        x_types = ['B' for _ in range(zone_num)]
        # 定義x變數的名稱
        x_names = [f'x_{idx}' for idx in zone_index]
        
        # 創建Cplex模型
        m = cplex.Cplex()
        # 最大化目標函數
        m.objective.set_sense(m.objective.sense.maximize)
        # 添加y變數
        m.variables.add(obj = y_obj, types = y_types, names = y_names)
        # 添加x變數
        m.variables.add(obj = x_obj, types = x_types, names = x_names)
        
        # 定義限制式1: sum(x_j for j in N_i) >= y_i  for all demand zone i
        constr1 = []
        for i in zone_index:
            constr1.append([[f'x_{j}' for j in coverage_position[i]] + [y_names[i]], [1 for j in coverage_position[i]] + [-1]])
            
        senses1 = "G" * zone_num  # 這表示每個限制式的形式是大於等於
        rhs1 = [0 for _ in range(zone_num)]  # 限制式的右邊值，均為0

        # 定義限制式2：保證選擇的救護車站數量為N
        constr2 = [[[f'x_{j}' for j in zone_index], [1 for _ in range(zone_num)]]]
        rhs2 = [fleet_num]
        # 添加所有線性約束到模型中
        m.linear_constraints.add(lin_expr = constr1, senses = senses1, rhs = rhs1)
        m.linear_constraints.add(lin_expr = constr2, senses = 'E', rhs = rhs2)
        # m.write('test.lp')
        # 把求解的訊息關掉
        m.set_log_stream(None)
        m.set_results_stream(None)
        # 求解模型
        m.solve()
        # 印出目標函數的值
        optimal_value = m.solution.get_objective_value()
        print(f"Optimal objective value: {optimal_value}")
        
        # 獲取每個變數的值
        variable_values = m.solution.get_values()

        # 獲取變數名稱
        variable_names = m.variables.get_names()

        initial_ambulance_location = []
        covered_demand_zone = set()
        # 印出每個變數的值，並找出最佳解中選擇的救護車位置
        for var_name, var_value in zip(variable_names, variable_values):
            if var_value >= 1e-6 and 'x' in var_name:
                # print(f"{var_name}: {var_value}")
                initial_ambulance_location.append(int(var_name.split('_')[-1]))
            elif var_value == 1 and 'y' in var_name:
                covered_demand_zone.add(int(var_name.split('_')[-1]))
        # 將model求出的救護車位置存到vehicle_position
        ambulance_num = 0
        for l in initial_ambulance_location:
            vehicle_position[ambulance_num] = l
            ambulance_num += 1

        return optimal_value, covered_demand_zone

    def plot():
        # 創建繪圖
        fig, (ax1, ax2) = plt.subplots(figsize=(10, 36), ncols=2)

        for policy in Policy:
            ax = ax1 if policy == 'static' else ax2
            ambulance_position = list(Ambulance[policy].position.values())
            call_position = list(all_call[policy])
            alpha_t = SimClasses.Clock % shift_length / shift_length * maximum_workload
            invalid_vehicle_num = [vehicle_num for vehicle_num in Ambulance[policy].status if Ambulance[policy].status[vehicle_num] == 'busy' or Ambulance[policy].workload[vehicle_num] > alpha_t]
            redeployment_position = list(all_redeployment[policy])
        
            for idx in zone_index:
                # 計算每個格子的中心點座標
                center_x, center_y = spatial_xy[idx][0], spatial_xy[idx][1]
                # 如果編號是醫院位置，則給它塗上黑色
                if idx in hospital_position:
                    color = 'black'
                # 如果當前編號是模型所選擇的，則給它塗上紅色
                elif idx in ambulance_position:
                    ambulance_index = ambulance_position.index(idx)  # 找到該格子在 list 中的 index
                    color = 'yellow' if Ambulance[policy].status[ambulance_index] == 'idle' else 'red'
                    ax.text(center_x-0.5, center_y-0.5, str(ambulance_index), color='black', ha='center', va='center', fontsize=10) # 把救護車編號標上去
                # 否則設為淺藍色
                else:
                    color = 'lightblue'
                
                # 添加矩形
                rect = plt.Rectangle((center_x - cell_size/2, center_y - cell_size/2), cell_size, cell_size, linewidth=1, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
                ax.text(center_x+0.5, center_y+0.5, str(idx), color='white', ha='center', va='center', fontsize=10) # 把demand zone編號標上去

                # 如果編號是call的位置，則給它塗上紫色
                if idx in call_position:
                    color = 'purple'
                    radius = 0.5
                    circle = plt.Circle((center_x, center_y), radius, linewidth=1, edgecolor='black', facecolor=color)
                    ax.text(center_x, center_y, str(all_call[policy][idx]), color='white', ha='center', va='center', fontsize=10) # 把demand zone編號標上去
                    ax.add_patch(circle)

                if idx in redeployment_position:
                    color = 'gray'
                    radius = 0.5
                    circle = plt.Circle((center_x, center_y), radius, linewidth=1, edgecolor='black', facecolor=color)
                    ax.text(center_x, center_y, str(all_redeployment[policy][idx]), color='white', ha='center', va='center', fontsize=10) # 把demand zone編號標上去
                    ax.add_patch(circle)

            
            for idx in zone_index:
                center_x, center_y = spatial_xy[idx][0], spatial_xy[idx][1]
                if idx in ambulance_position and ambulance_position.index(idx) not in invalid_vehicle_num:
                    coverage_range = plt.Circle((center_x, center_y), coverage_threshold, color='green', alpha=0.15, linewidth=1, fill=True)
                    ax.add_patch(coverage_range)
            
            ax.set_xlim(0, 15 * cell_size)
            ax.set_ylim(18 * cell_size, 0)

        plt.title("Location after running MCLP model")
        plt.tight_layout()  # 自動調整子圖參數，避免圖例遮擋
        plt.show()

    file_name = './demand_volume.xlsx'
    read_data(file_name)
    initial()
    # Initialization
    SimClasses.Clock = 0 # 初始化 clock = 0
    ZSimRNG = SimRNG.InitializeRNSeed() 
    Calendar = SimClasses.EventCalendar() # 建立一個Calendar物件

    Queue = {} # policy: Queue()
    Ambulance = {} # policy: SimClasses.Resource
    all_call = {}  # policy: {call_position: vehicle_num}
    all_redeployment = {} # policy: {redeployment_position: vehicle_num}
    total_call = {} # policy: total num of call
    total_call_within_limit = {} # policy: total num of call within limit
    total_call_without_limit = {} # policy: total num of call without limit
    total_call_hospital = {} # policy: total num of call to hospital
    total_call_scene = {} # policy: total num of call to scene
    total_call_lost = {} # policy: total num of call to lost
    total_workload = {} # policy: [workload in a shift]
    vehicle_workload_record = {} #policy:[each vehicle workload in a shift]
    vehicle_to_scene_record = {} #policy:[each vehicle workload in a shift]
    vehicle_service_on_scene_record = {} #policy:[each vehicle workload in a shift]
    vehicle_to_hospital_record = {} #policy:[each vehicle workload in a shift]
    vehicle_redeployment_record = {} #policy:[each vehicle workload in a shift]
    total_waiting_time = {}

    clock = []
    SC_every_t = {}
    avg_waiting_time_every_t = {}
    avg_number_in_queue_ecery_t = {}
    cur_number_in_queue_every_t = {}

    EndSimulationTime = 8.5*24*60 # minutes
    warm_up_period = 1.5*24*60 # minutes
    # 計算每個 demand zone 出現 call 的機率
    arrival_position_prob = list(spatial_volume.values())
    arrival_position_prob = [x / sum(arrival_position_prob) for x in arrival_position_prob] 
    NumReps = 10 # Number of simulation experiment replications

    def initialize_setting(reset_ambulance = True):
        # 把需要用到的變數全部初始化
        for policy in Policy:
            Queue[policy] = SimClasses.FIFOQueue()
            if reset_ambulance:
                Ambulance[policy] = SimClasses.Resource(vehicle_position.copy(), vehicle_status.copy(), vehicle_workload.copy(), vehicle_workload.copy(), vehicle_workload.copy(), vehicle_workload.copy(), vehicle_workload.copy())
                Ambulance[policy].SetUnits(fleet_num)
                SC_every_t[policy] = []
                avg_waiting_time_every_t[policy] = []
                avg_number_in_queue_ecery_t[policy] = []
                cur_number_in_queue_every_t[policy] = []
                clock.clear()

            all_call[policy] = {}
            all_redeployment[policy] = {}
            total_call[policy] = [0, 0, 0]
            total_call_within_limit[policy] = [0, 0, 0]
            total_call_without_limit[policy] = [0, 0, 0]
            total_call_hospital[policy] = 0
            total_call_scene[policy] = 0
            total_call_lost[policy] = 0
            total_workload[policy] = []
            vehicle_workload_record[policy] = []
            vehicle_to_scene_record[policy] = []
            vehicle_service_on_scene_record[policy] = [] 
            vehicle_to_hospital_record[policy] = []
            vehicle_redeployment_record[policy] = []
            total_waiting_time[policy] = [0, 0, 0]
            
    def find_cloest_vehicle(Ambulance, call_position, patient_type):
        # 找到現在救護車分布在哪些位置
        vehicle_now_position = list(Ambulance.position.values()) # [demand zone]
        # print(vehicle_now_position)
        # 篩選出不合的救護車編號 (busy 或是 目前workload + 過去的travel time > maximum_workload )
        invalid_vehicle_num = [vehicle_num for vehicle_num in Ambulance.status if Ambulance.status[vehicle_num] == 'busy' 
                                or Ambulance.workload[vehicle_num] + travel_time_matrix[Ambulance.position[vehicle_num]][call_position] > maximum_workload]
        # print(f"Clock: {SimClasses.Clock}, new call arrive at demand zone {call_position} and now invalid vehicle: {invalid_vehicle_num}")
        
        cloest_vehicle_num = -1 # 預設最近的救護車編號為 -1，代表沒有一台救護車可以用
        cloest_vehicle_position = -1 # 預設最近的救護車位置為 -1，代表沒有一台救護車可以用
        if len(invalid_vehicle_num) < fleet_num: # 如果有閒置的救護車
            # 篩選出所有救護車位置到 call_position 的 travel time 矩陣
            vehicle_travel_time_matrix = travel_time_matrix[vehicle_now_position] # 20*168的二維list
            # 把不合的救護車到 call_position 的 travel time 都改成無限大
            for vehicle_num in invalid_vehicle_num:
                vehicle_travel_time_matrix[vehicle_num, call_position] = math.inf
            cloest_vehicle_num = np.argmin(vehicle_travel_time_matrix[:, call_position]) # 找出traval time最短的vehicle_num
            cloest_vehicle_position = vehicle_now_position[cloest_vehicle_num] # 找到那一台救護車的位置
        return cloest_vehicle_num, cloest_vehicle_position

    def dispatch_vehicle(policy, Next_patient):
        # print(f"{policy}:")
        if Next_patient.ClosestVehicleNum != -1: # no idle

            travel_time_vehicle_call = travel_time_matrix[Next_patient.ClosestVehicleLocation][Next_patient.CallLocation] # 計算過去接病人的時間
            pred_position = Ambulance[policy].position[Next_patient.ClosestVehicleNum]
            travel_time_call_hospital = 0

            if Next_patient.HospitalLocation != -1: # 要送去醫院
                travel_time_call_hospital = travel_time_matrix[Next_patient.CallLocation][Next_patient.HospitalLocation] # 計算把病人送到醫院的時間
                # 計算總服務時間 (不包含結束後送回去的時間，在redeployment的時候才計算)
                total_service_time = travel_time_vehicle_call + travel_time_call_hospital + Next_patient.ServiceTime # 總服務時間 = 救護車位置到現場 + 現場到醫院 + 單純服務時間

                # Schedule depature，還要多紀錄派送哪一台救護車，和醫院的位置 
                SimFunctions.SchedulePlus(Calendar, f"Departure_{policy}", total_service_time, Next_patient)
                # print(f"    Clock: {SimClasses.Clock}, dispatch cloest vehicle {Next_patient.ClosestVehicleNum} from demand zone {pred_position} to call position {Next_patient.HospitalLocation} (Expected time to go back: {SimClasses.Clock+total_service_time})")

                total_call_hospital[policy] += 1
                Ambulance[policy].position[Next_patient.ClosestVehicleNum] = Next_patient.HospitalLocation
            else: # 直接現場處理
                # 計算總服務時間 (不包含結束後送回去的時間，在redeployment的時候才計算)
                total_service_time = travel_time_vehicle_call + Next_patient.ServiceTime # 總旅行時間 = 救護車位置到現場 + 單純服務時間

                # Schedule depature，還要多紀錄派送哪一台救護車，和call的位置
                SimFunctions.SchedulePlus(Calendar, f"Departure_{policy}", total_service_time, Next_patient)
                # print(f"    Clock: {SimClasses.Clock}, dispatch cloest vehicle {Next_patient.ClosestVehicleNum} from demand zone {pred_position} to call position {Next_patient.CallLocation} (Expected time to go back: {SimClasses.Clock+total_service_time})")

                total_call_scene[policy] += 1
                Ambulance[policy].position[Next_patient.ClosestVehicleNum] = Next_patient.CallLocation

            # 將最近救護車編號的status設為busy，並將該救護車的 workload 加上 total_service_time (Q:改位子?)
            Ambulance[policy].Seize(1, Next_patient.ClosestVehicleNum, travel_time_vehicle_call, Next_patient.ServiceTime, travel_time_call_hospital, 0)
            
            # 如果是 ROA policy 的話，還要跑模型看是否超過 acceptance_threshold
            if policy == "ROA":
                m, ROA1_optimal_value = ROA(Ambulance[policy]) # run ROA model
                # 計算原始的覆蓋範圍
                original_coverage = 0
                covered_demand_zone = set() # {zone_num} (因為是set，所以不會有重複的編號)
                for vehicle_num in Ambulance[policy].status:
                    # 把目前 idle 的救護車覆蓋到的 zone_num 都加入到 set 中 (有重複的會自動刪除)
                    if Ambulance[policy].status[vehicle_num] == 'idle':
                        covered_demand_zone.update(coverage_position[Ambulance[policy].position[vehicle_num]])
                # 計算 set 中所有 zone_num 的 spatial_volumn 總和
                for idx in covered_demand_zone:
                    original_coverage += spatial_volume[idx]
                # 計算 ROA 相對 原本覆蓋範圍 的改進幅度 (如果分母是0，代表沒有idle的救護車，跑ROA也不可能更好，所以直接讓改善幅度等於0)
                improve_proportion = (ROA1_optimal_value - original_coverage)/original_coverage if original_coverage != 0 else 0 
                # print(f'before:{original_coverage}, after:{ROA1_optimal_value}')
                if improve_proportion >= acceptance_threshold: # 如果有超過 acceptance_threshold
                    # print(f"    Pass acceptance_threshold (Original: {original_coverage}, ROA1_OBJ: {ROA1_optimal_value}, imporve {improve_proportion})")
                    # 從ROA模型得到哪些救護車要重新佈署
                    redeployment_vehicle = get_redeployment_vehicle(m)  # redeploy_vehicle_num: {redeployment time, redeployment position}
                    for redeploy_vehicle_num in redeployment_vehicle:
                        # 取出重新佈署的時間和位置
                        redeployment_time = redeployment_vehicle[redeploy_vehicle_num]['redeployment_time']
                        k = redeployment_vehicle[redeploy_vehicle_num]['redeployment_position']
                        # Schedule redeployment，還要多紀錄重新佈署哪一台救護車，和重新佈署後的位置
                        SimFunctions.SchedulePlus(Calendar, f"Redeployment_{policy}", redeployment_time, redeploy_vehicle_num)
                        # 將要重新佈署的救護護車編號的 status 設為 busy，並將該救護車的 workload 加上 redeployment_time
                        Ambulance[policy].Seize(1, redeploy_vehicle_num, 0, 0, 0, redeployment_time)
                        previous_position = Ambulance[policy].position[redeploy_vehicle_num]
                        Ambulance[policy].position[redeploy_vehicle_num] = k
                        
                        # print(f"    Clock: {SimClasses.Clock}, redeploy vehicle {redeploy_vehicle_num} from demand zone {previous_position} to demand zone {k} (Expected time to go back: {SimClasses.Clock+redeployment_time})")
                        all_redeployment[policy][k] = redeploy_vehicle_num

            waiting_time = travel_time_vehicle_call + (SimClasses.Clock - Next_patient.CreateTime)
            if waiting_time <= resecue_time[int(Next_patient.PatientType) - 1]:
                total_call_within_limit[policy][int(Next_patient.PatientType) - 1] += 1
            else:
                total_call_without_limit[policy][int(Next_patient.PatientType) - 1] += 1
                total_call_lost[policy] += 1
            total_waiting_time[policy][int(Next_patient.PatientType) - 1] += waiting_time
            all_call[policy][Next_patient.CallLocation] = Next_patient.ClosestVehicleNum
        else:
            # print(f"Clock: {SimClasses.Clock}, No idle ambulance")
            Next_patient.PatientType += 0.001 # 加一個很小的值，讓這個patient在同一個type中會被優先取出來
            Queue[policy].Add(Next_patient)

    # Called when an arrival event occurs
    def Arrival():
        # 生成patient
        service_time = SimRNG.Expon(1/service_rate, 2) # 計算單純服務的時間
        call_position = np.random.choice(zone_num, 1, replace=False, p=arrival_position_prob)[0]
        while(call_position in all_call['static']): 
            call_position = np.random.choice(zone_num, 1, replace=False, p=arrival_position_prob)[0]
        hospital = -1 # 預設醫院位置為 -1 ，代表沒有要送去醫院
        # 符合特定條件才會送到醫院
        tp = 1 if service_time <= 29 else (2 if np.random.rand() <= 0.23 else 3) # 數字 3 最嚴重，需要送到醫院，type 1 2現場處理
        if tp == 3:
            hospital = np.random.choice(hospital_position)

        closest_vehicle_num, closest_vehicle_location = -1, -1
            
        for i,policy in enumerate(Policy):
            Patient = SimClasses.Entity(call_position, service_time, tp, hospital, closest_vehicle_num, closest_vehicle_location)
            total_call[policy][Patient.PatientType - 1] += 1

            Queue[policy].Add(Patient)
            Next = Queue[policy].Remove()
            Next.ClosestVehicleNum, Next.ClosestVehicleLocation = find_cloest_vehicle(Ambulance[policy], Next.CallLocation, Next.PatientType) # 找到最近的救護車是哪一個
            dispatch_vehicle(policy, Next) # 派遣最近的救護車過去

        SimFunctions.Schedule(Calendar,"Arrival", SimRNG.Expon(1/arrival_rate,1))
        
    # Called when a departure event occurs
    def Departure(policy, Next_patient):
        # print(f"{policy}:")
        vehicle_num = Next_patient.ClosestVehicleNum

        if Next_patient.HospitalLocation == -1:
            call_position = Next_patient.CallLocation
        else:
            call_position = Next_patient.HospitalLocation
        
        if policy == "static":
            # original_position = Ambulance[policy].position[vehicle_num] # 先記錄原本救護車的位置
            Ambulance[policy].Free(1, vehicle_num) # 將救護車的狀態改為 idle
            # print(f"    Clock: {SimClasses.Clock}, vehicle {vehicle_num} is idle now and current position at demand zone {Ambulance[policy].position[vehicle_num]} ")
            
            if Queue[policy].NumQueue() > 0:
                Next = Queue[policy].Remove()
                Next.ClosestVehicleNum, Next.ClosestVehicleLocation = find_cloest_vehicle(Ambulance[policy], Next.CallLocation, Next.PatientType) # 找到最近的救護車是哪一個
                dispatch_vehicle(policy, Next) # 派遣最近的救護車過去

            else:
                # 計算重新佈署的時間，static 就是直接計算從 call position 回到 原本救護車位置的時間
                redeployment_time = travel_time_matrix[call_position][Next_patient.ClosestVehicleLocation] 
                # Schedule redeployment，還要多紀錄哪一台救護車，和救護車要回去的位置 (原本救護車的位置)
                SimFunctions.SchedulePlus(Calendar, f"Redeployment_{policy}", redeployment_time, vehicle_num)
                # 將救護車編號的status設為busy，並將該救護車的 workload 加上 redeployment_time
                Ambulance[policy].Seize(1, vehicle_num, 0, 0, 0, redeployment_time)
                Ambulance[policy].position[vehicle_num] = Next_patient.ClosestVehicleLocation
                
                # print(f"    Clock: {SimClasses.Clock}, redeploy vehicle {vehicle_num} from demand zone {call_position} to demand zone {Next_patient.ClosestVehicleLocation} (Expected time to go back: {SimClasses.Clock+redeployment_time})")
                all_redeployment[policy][Next_patient.ClosestVehicleLocation] = vehicle_num

        else:
            Ambulance[policy].Free(1, vehicle_num) # 將救護車的狀態改為 idle

            if Queue[policy].NumQueue() > 0:
                Next = Queue[policy].Remove()
                Next.ClosestVehicleNum, Next.ClosestVehicleLocation = find_cloest_vehicle(Ambulance[policy], Next.CallLocation, Next.PatientType) # 找到最近的救護車是哪一個
                dispatch_vehicle(policy, Next) # 派遣最近的救護車過去
            else:
                
                # print(f"    Clock: {SimClasses.Clock}, vehicle {vehicle_num} is idle now and current position at demand zone {Ambulance[policy].position[vehicle_num]} ")
                m, ROA1_optimal_value = ROA(Ambulance[policy]) # run ROA model
                # 不用檢查改善幅度是否超過 acceptance_threshold
                # 從ROA模型得到哪些救護車要重新佈署
                redeployment_vehicle = get_redeployment_vehicle(m) # redeploy_vehicle_num: {redeployment time, redeployment position}
                for redeploy_vehicle_num in redeployment_vehicle:
                    # 取出重新佈署的時間和位置
                    redeployment_time = redeployment_vehicle[redeploy_vehicle_num]['redeployment_time']
                    k = redeployment_vehicle[redeploy_vehicle_num]['redeployment_position']
                    # Schedule redeployment，還要多紀錄重新佈署哪一台救護車，和重新佈署後的位置
                    SimFunctions.SchedulePlus(Calendar, f"Redeployment_{policy}", redeployment_time, redeploy_vehicle_num)
                    # 將要重新佈署的救護護車編號的 status 設為 busy，並將該救護車的 workload 加上 redeployment_time
                    Ambulance[policy].Seize(1, redeploy_vehicle_num, 0, 0, 0, redeployment_time)
                    previous_position = Ambulance[policy].position[redeploy_vehicle_num]
                    Ambulance[policy].position[redeploy_vehicle_num] = k
                    
                    # print(f"    Clock: {SimClasses.Clock}, redeploy vehicle {redeploy_vehicle_num} from demand zone {previous_position} to demand zone {k} (Expected time to go back: {SimClasses.Clock+redeployment_time})")
                    all_redeployment[policy][k] = redeploy_vehicle_num
        
        # 服務結束後，就刪除 all_call 中的 call position 和 服務它的 vehicle_num
        for position in all_call[policy]:
            if all_call[policy][position] == vehicle_num:
                del all_call[policy][position]
                break
            
    def Redeployment(policy, redeployment_vehicle_num):
        # print(f"{policy}:")
        Ambulance[policy].Free(1, redeployment_vehicle_num) # 將要重新佈署的救護車 status 設為 idle
        if Queue[policy].NumQueue() > 0:
            Next = Queue[policy].Remove()
            Next.ClosestVehicleNum, Next.ClosestVehicleLocation = find_cloest_vehicle(Ambulance[policy], Next.CallLocation, Next.PatientType) # 找到最近的救護車是哪一個
            dispatch_vehicle(policy, Next) # 派遣最近的救護車過去

        # print(f"    Clock: {SimClasses.Clock}, vehicle {redeployment_vehicle_num} is back to demand zone {Ambulance[policy].position[redeployment_vehicle_num]} and is idle now")
        
        # 重新佈署結束後，就刪除 all_redeployment 中的 redeployment position 和 移動過去的 redeploy_vehicle_num
        for position in all_redeployment[policy]:
            if all_redeployment[policy][position] == redeployment_vehicle_num:
                del all_redeployment[policy][position]
                break

    def ROA(Ambulance):

        ######################## ROA step 1 ##############################
        # 先計算 alpha_t，並篩選出 'busy' 或是 workload 超過 alpha_t 的救護車編號
        alpha_t = SimClasses.Clock % shift_length / shift_length * maximum_workload
        invalid_vehicle_num = [vehicle_num for vehicle_num in Ambulance.status if Ambulance.status[vehicle_num] == 'busy' or Ambulance.workload[vehicle_num] >= alpha_t]

        # 創建Cplex模型
        m = cplex.Cplex()

        # 定義目標函數中Y的係數 (d_i)
        Y_obj = [spatial_volume[idx] for idx in zone_index] 
        # 定義Y變數的類型（'B'代表二元變數）
        Y_types = ['B' for _ in range(zone_num)]
        # 定義Y變數的名稱
        Y_names = [f'Y_{idx}' for idx in zone_index]
        
        # 定義目標函數中X的係數，都是0
        X_obj = [0 for _ in range(zone_num)]
        # 定義X變數的類型（'B'代表二元變數）
        X_types = ['B' for _ in range(zone_num)]
        # 定義X變數的名稱
        X_names = [f'X_{idx}' for idx in zone_index]

        # 定義目標函數中X的係數，都是0
        R_obj = [0 for j in range(zone_num) for k in range(zone_num) if j !=  k]
        # 定義X變數的類型（'B'代表二元變數）
        R_types = ['B' for j in range(zone_num) for k in range(zone_num) if j !=  k]
        # 定義X變數的名稱
        R_names = [f'R_{j}_{k}' for j in range(zone_num) for k in range(zone_num) if j !=  k ]
        
        # 目標函數1：最大化覆蓋範圍
        m.objective.set_sense(m.objective.sense.maximize)
        # 添加Y變數
        m.variables.add(obj = Y_obj, types = Y_types, names = Y_names)
        # 添加X變數
        m.variables.add(obj = X_obj, types = X_types, names = X_names)
        # 添加R變數
        m.variables.add(obj = R_obj, types = R_types, names = R_names)

        # 定義限制式2：更新救護車重新佈署後的位置
        ambulance_position = list(Ambulance.position.values())
        x = [1 if idx in ambulance_position and ambulance_position.index(idx) not in invalid_vehicle_num else 0 for idx in zone_index]
        # x = [1 if idx in ambulance_position else 0 for idx in zone_index]
        constr2 = [] #1127
        for j in zone_index: 
            constr2.append([[f'X_{j}'] + [f'R_{j}_{k}' for k in redeployment_position[j] if k not in ambulance_position] + [f'R_{k}_{j}' for k in redeployment_position[j]], 
                            [1] + [1 for k in redeployment_position[j] if k not in ambulance_position] + [-1 for _ in redeployment_position[j]]])
        senses2 = "E" * zone_num  # 限制式的關係是等於
        rhs2 = [x[j] for j in zone_index]  # 限制式的右邊值，x_jt
        names2 = [f"C2_{i}" for i in range(len(rhs2))]

        # 定義限制式3: demand zone i ∈ D is covered if ambulance/s are located in the covering locations of the zone.
        constr3 = []
        for i in zone_index:
            constr3.append([[f'X_{j}' for j in coverage_position[i]] + [f'Y_{i}'], [1 for j in coverage_position[i]] + [-1]])
            
        senses3 = "G" * zone_num  # 限制式的關係是大於等於
        rhs3 = [0 for _ in zone_index]  # 限制式的右邊值，均為0
        names3 = [f"C3_{i}" for i in range(len(rhs3))]

        # 定義限制式4: at most one ambulance can move to each location
        constr4 = []
        for k in zone_index:
            constr4.append([[f'R_{j}_{k}' for j in redeployment_position[k]], [1 for j in redeployment_position[k]]])
        senses4 = "L" * zone_num
        rhs4 = [0 if idx in ambulance_position else 1 for idx in zone_index]
        names4 = [f"C4_{i}" for i in range(len(rhs4))]

        # 定義限制式5: from each location, at most one ambulance can be redeployed to another location only if there is an ambulance already located there
        constr5 = [] #1127
        for j in zone_index:
            constr5.append([[f'R_{j}_{k}' for k in zone_index if k != j], [1 for k in zone_index if k != j]])
        senses5 = "L" * zone_num
        rhs5 = [x[j] for j in zone_index]
        names5 = [f"C5_{i}" for i in range(len(rhs5))]

        # 定義限制式6: Restriction workload of each ambulance at time t is considered dynamically
        ambulance_workload = Ambulance.workload # beta, {vehicle_num: workload at time t}
        # l: 20*168 的二維 list，l[v][j] = 1 代表編號 v 的救護車位於 demand zone j 上，否則為0
        l = [[1 if ambulance_position[v] == j and v not in invalid_vehicle_num else 0 for j in zone_index] for v in range(fleet_num)]
        # l = [[1 if ambulance_position[v] == j else 0 for j in zone_index] for v in range(fleet_num)] 
        constr6 = [] #1127
        for v in range(fleet_num):
            if v not in invalid_vehicle_num: # ROA 模型不考慮 invalid 的救護車
                constr6.append([[f'R_{j}_{k}' for j in zone_index for k in redeployment_position[j] if k not in ambulance_position],
                                [travel_time_matrix[j][k] * l[v][j] for j in zone_index for k in redeployment_position[j] if k not in ambulance_position]])
        senses6 = "L" * (fleet_num - len(invalid_vehicle_num))
        rhs6 = [alpha_t - ambulance_workload[v] for v in range(fleet_num) if v not in invalid_vehicle_num]
        names6 = [f"C6_{i}" for i in range(len(rhs6))]

        # 添加所有限制式到模型中
        m.linear_constraints.add(lin_expr = constr2, senses = senses2, rhs = rhs2, names = names2)
        m.linear_constraints.add(lin_expr = constr3, senses = senses3, rhs = rhs3, names = names3)
        m.linear_constraints.add(lin_expr = constr4, senses = senses4, rhs = rhs4, names = names4)
        m.linear_constraints.add(lin_expr = constr5, senses = senses5, rhs = rhs5, names =names5)
        m.linear_constraints.add(lin_expr = constr6, senses = senses6, rhs = rhs6, names = names6)
        # 把求解的訊息關掉
        m.set_log_stream(None)
        m.set_results_stream(None)
        # m.write('test.lp')
        # 求解模型
        m.solve()
        # 紀錄目標函數的值
        ROA1_optimal_value = m.solution.get_objective_value()
        # print(f"First step of ROA, optimal objective value: {ROA1_optimal_value}")

        ######################## ROA step 2 ##############################
        
        # 更新目標函數: 最小化總移動時間
        m.objective.set_sense(m.objective.sense.minimize)

        Y_obj2 = [0 for _ in zone_index]
        X_obj2 = [0 for _ in zone_index]
        R_obj2 = [travel_time_matrix[j][k] for j in zone_index for k in zone_index if j != k]

        m.objective.set_linear(list(zip(Y_names, Y_obj2)))  # 更新 Y 變數的目標函數係數
        m.objective.set_linear(list(zip(X_names, X_obj2)))  # 更新 X 變數的目標函數係數
        m.objective.set_linear(list(zip(R_names, R_obj2)))  # 更新 Y 變數的目標函數係數

        # 定義限制式7: 確保覆蓋範要大於等於 ROA step 1 計算出來的目標函數值
        constr7 = [[[f'Y_{i}' for i in zone_index], [spatial_volume[i] for i in zone_index]]]
        senses7 = "G"
        rhs7 = [ROA1_optimal_value]
        names7 = [f"C7_{i}" for i in range(len(rhs7))]

        m.linear_constraints.add(lin_expr = constr7, senses = senses7, rhs = rhs7, names = names7)

        m.set_log_stream(None)
        m.set_results_stream(None)
        m.solve()

        ROA2_optimal_value = m.solution.get_objective_value()
        # print(f"Second step of ROA, optimal objective value: {ROA2_optimal_value}\n")
        return m, ROA1_optimal_value

    def get_redeployment_vehicle(m):
        redeployment_vehicle = {} # vehicle_num: {redeployment position, redeployment time}
        # 獲取每個變數的值
        variable_values = m.solution.get_values()
        # 獲取每個變數的名稱
        variable_names = m.variables.get_names()
        # 找出最佳解中選擇的救護車位置
        # print("")
        ambulance_position = list(Ambulance['ROA'].position.values()) # [zone_num]

        for var_name, var_value in zip(variable_names, variable_values):
            if 'R' in var_name and var_value == 1:
                # print(f"{var_name}: {var_value}")
                var_name = var_name.split('_')
                j = int(var_name[1])
                k = int(var_name[2])
                redeployment_time = travel_time_matrix[j][k] # 計算 zone_num j 到 k 的 travel_time
                redeployment_vehicle_num = ambulance_position.index(j) # 找到 zone_num j 上的救護車的編號
                redeployment_vehicle[redeployment_vehicle_num] = {"redeployment_position": k, "redeployment_time": redeployment_time}
        return redeployment_vehicle

    for Reps in range(NumReps):
        # Initialization
        SimClasses.Clock = 0 # 初始化 clock = 0
        ZSimRNG = SimRNG.InitializeRNSeed() 
        Calendar = SimClasses.EventCalendar() # 建立一個Calendar物件
        pbar = tqdm(total=EndSimulationTime, desc=f"Replication {Reps+1}", position=0)
        initialize_setting()
        not_clear = True
        
        # Initialize simulation objects
        SimFunctions.SimFunctionsInit(Calendar)
        # Schedule first event
        SimFunctions.Schedule(Calendar,"Arrival", SimRNG.Expon(1/arrival_rate_1,1))
        # Schedule end-of-simulation event
        SimFunctions.Schedule(Calendar, "EndSimulation", EndSimulationTime)
        (arrival_rate, service_rate) = (arrival_rate_1, service_rate_1)
        # Main simulation loop
        while Calendar.N() > 0:
            # Pop event from queue
            NextEvent = Calendar.Remove()
            # Update the simulation clock
            SimClasses.Clock = NextEvent.EventTime

            # 更新不同shift的arrival、service rate並將上一個shift的workload記錄起來
            if SimClasses.Clock % (24*60) >= shift_length and arrival_rate == arrival_rate_1:
                (arrival_rate, service_rate) = (arrival_rate_2, service_rate_2)
                for policy in Policy:
                    total_workload[policy].append(sum(Ambulance[policy].workload.values()))
                    vehicle_workload_record[policy].append(list(Ambulance[policy].workload.values()))
                    vehicle_to_scene_record[policy].append(list(Ambulance[policy].to_scene.values()))
                    vehicle_service_on_scene_record[policy].append(list(Ambulance[policy].service_on_scene.values()))
                    vehicle_to_hospital_record[policy].append(list(Ambulance[policy].to_hospital.values()))
                    vehicle_redeployment_record[policy].append(list(Ambulance[policy].redeployment.values()))
                    Ambulance[policy].NextShift()
                    # print(sum(vehicle_workload_record[policy][-1]), (sum(vehicle_to_scene_record[policy][-1]) + sum(vehicle_service_on_scene_record[policy][-1]) +
                    #                                                 sum(vehicle_to_hospital_record[policy][-1]) + sum(vehicle_redeployment_record[policy][-1])))
                # print("Shift 2 start------------------------------")
            elif SimClasses.Clock % (24*60) < shift_length and arrival_rate == arrival_rate_2:
                (arrival_rate, service_rate) = (arrival_rate_1, service_rate_1)
                for policy in Policy:
                    total_workload[policy].append(sum(Ambulance[policy].workload.values()))
                    vehicle_workload_record[policy].append(list(Ambulance[policy].workload.values()))
                    vehicle_to_scene_record[policy].append(list(Ambulance[policy].to_scene.values()))
                    vehicle_service_on_scene_record[policy].append(list(Ambulance[policy].service_on_scene.values()))
                    vehicle_to_hospital_record[policy].append(list(Ambulance[policy].to_hospital.values()))
                    vehicle_redeployment_record[policy].append(list(Ambulance[policy].redeployment.values()))
                    Ambulance[policy].NextShift()
                # print("Shift 1 start")
                
            if SimClasses.Clock >= warm_up_period and not_clear:
                initialize_setting(reset_ambulance=False)
                not_clear = False

            if NextEvent.EventType == "Arrival":
                Arrival()
            elif "Departure" in NextEvent.EventType:
                # 取出是哪個policy的depature，和哪台救護車去服務哪個位置的call
                policy = NextEvent.EventType.split('_')[1]
                Departure(policy, NextEvent.WhichObject)
            elif "Redeployment" in NextEvent.EventType:
                # 取出是哪個policy的redeployment，和哪台救護車去重新佈署到哪個位置
                policy = NextEvent.EventType.split('_')[1]
                redeployment_vehicle_num = NextEvent.WhichObject
                Redeployment(policy, redeployment_vehicle_num)
            elif NextEvent.EventType == "EndSimulation":
                break
            # plot()

            clock.append(SimClasses.Clock)
            for policy in Policy:
                SC_every_t[policy].append(round(sum(total_call_within_limit[policy])/sum(total_call[policy]), 3) if sum(total_call[policy]) != 0 else 1)
                avg_waiting_time_every_t[policy].append(round(sum(total_waiting_time[policy])/(total_call_hospital[policy] + total_call_scene[policy]), 3) if sum(total_call[policy]) != 0 else 1)
                avg_number_in_queue_ecery_t[policy].append(round(Queue[policy].Mean(), 3))
                cur_number_in_queue_every_t[policy].append(round(Queue[policy].NumQueue(), 3))
            pbar.set_postfix({"Clock": SimClasses.Clock})  # 顯示當前值
            pbar.update(SimClasses.Clock-pbar.n)  # 更新進度條

        # 創建一個2行1列的子圖，這樣可以將兩張圖放在一起
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))  # 2行1列的子圖

        for i,policy in enumerate(Policy):
            print(f"{policy}:")
            print(f"    ---------------------   overall performance   -------------------")
            print(f"    SC: {round(sum(total_call_within_limit[policy])/sum(total_call[policy]), 3)}")
            print(f"    AWA: {round(sum(total_workload[policy])/len(total_workload[policy])/fleet_num/maximum_workload, 3)}")
            print(f"    ---------------------   call statistic   -------------------")
            print(f"    total_call (all_type): {sum(total_call[policy])}")
            print(f"    total_call (type1): {total_call[policy][0]}, proportion: {round(total_call[policy][0] / sum(total_call[policy]), 3)}")
            print(f"    total_call (type2): {total_call[policy][1]}, proportion: {round(total_call[policy][1] / sum(total_call[policy]), 3)}")
            print(f"    total_call (type3): {total_call[policy][2]}, proportion: {round(total_call[policy][2] / sum(total_call[policy]), 3)}")
            print(f"    ---------------------   call performance     -------------------")
            print(f"    total_call_hospital: {total_call_hospital[policy]}, total_call_scene: {total_call_scene[policy]}")
            print(f"    total_whithin_15: {total_call_within_limit[policy][0]}, proportion:{round(total_call_within_limit[policy][0] / total_call[policy][0], 3)}")
            print(f"    total_whithin_10: {total_call_within_limit[policy][1]}, proportion:{round(total_call_within_limit[policy][1] / total_call[policy][1], 3)}")
            print(f"    total_whithin_5: {total_call_within_limit[policy][2]}, proportion:{round(total_call_within_limit[policy][2] / total_call[policy][2], 3)}")
            print(f"    total_whithout_15: {total_call_without_limit[policy][0]}, proportion:{round(total_call_without_limit[policy][0] / total_call[policy][0], 3)}")
            print(f"    total_whithout_10: {total_call_without_limit[policy][1]}, proportion:{round(total_call_without_limit[policy][1] / total_call[policy][1], 3)}")
            print(f"    total_whithout_5: {total_call_without_limit[policy][2]}, proportion:{round(total_call_without_limit[policy][2] / total_call[policy][2], 3)}")
            print(f"    total lost call: {total_call_lost[policy]}, proportion: {round(total_call_lost[policy]/ sum(total_call[policy]), 3)}")
            print(f"    ---------------------   ambulance workload   -------------------")
            shift1_workload = [item for sublist in vehicle_workload_record[policy][0::2] for item in sublist]
            shift1_to_scene = [item for sublist in vehicle_to_scene_record[policy][0::2] for item in sublist]
            shift1_service_on_scene = [item for sublist in vehicle_service_on_scene_record[policy][0::2] for item in sublist]
            shift1_to_hospital = [item for sublist in vehicle_to_hospital_record[policy][0::2] for item in sublist]
            shift1_redeployment = [item for sublist in vehicle_redeployment_record[policy][0::2] for item in sublist]
            shift2_workload = [item for sublist in vehicle_workload_record[policy][1::2] for item in sublist]
            shift2_to_scene = [item for sublist in vehicle_to_scene_record[policy][1::2] for item in sublist]
            shift2_service_on_scene = [item for sublist in vehicle_service_on_scene_record[policy][1::2] for item in sublist]
            shift2_to_hospital = [item for sublist in vehicle_to_hospital_record[policy][1::2] for item in sublist]
            shift2_redeployment = [item for sublist in vehicle_redeployment_record[policy][1::2] for item in sublist]
            print(f"    shift 1 average workload: {round(np.mean(shift1_workload), 3)}")
            print(f"    shift 1 workload std: {round(np.std(shift1_workload), 3)}")
            print(f"    shift 1 average time to scene: {round(np.mean(shift1_to_scene), 3)}")
            print(f"    shift 1 time to scene std: {round(np.std(shift1_to_scene), 3)}")
            print(f"    shift 1 average time service on scene: {round(np.mean(shift1_service_on_scene), 3)}")
            print(f"    shift 1 time service on scene std: {round(np.std(shift1_service_on_scene), 3)}")
            print(f"    shift 1 average time to hospital: {round(np.mean(shift1_to_hospital), 3)}")
            print(f"    shift 1 time to hospital std: {round(np.std(shift1_to_hospital), 3)}")
            print(f"    shift 1 average time to redeployment: {round(np.mean(shift1_redeployment), 3)}")
            print(f"    shift 1 time to redeployment std: {round(np.std(shift1_redeployment), 3)}")
            print(f"    shift 2 average workload: {round(np.mean(shift2_workload), 3)}")
            print(f"    shift 2 workload std: {round(np.std(shift2_workload), 3)}")
            print(f"    shift 2 average time to scene: {round(np.mean(shift2_to_scene), 3)}")
            print(f"    shift 2 time to scene std: {round(np.std(shift2_to_scene), 3)}")
            print(f"    shift 2 average time service on scene: {round(np.mean(shift2_service_on_scene), 3)}")
            print(f"    shift 2 time service on scene std: {round(np.std(shift2_service_on_scene), 3)}")
            print(f"    shift 2 average time to hospital: {round(np.mean(shift2_to_hospital), 3)}")
            print(f"    shift 2 time to hospital std: {round(np.std(shift2_to_hospital), 3)}")
            print(f"    shift 2 average time to redeployment: {round(np.mean(shift2_redeployment), 3)}")
            print(f"    shift 2 time to redeployment std: {round(np.std(shift2_redeployment), 3)}")

            print(f"    ---------------------   patient performance   -------------------")
            print(f"    average waiting time (all_type): {round(sum(total_waiting_time[policy])/(total_call_hospital[policy] + total_call_scene[policy]), 3)}")
            print(f"    average waiting time (type1): {round(total_waiting_time[policy][0]/(total_call_within_limit[policy][0] + total_call_without_limit[policy][0]), 3)}")
            print(f"    average waiting time (type2): {round(total_waiting_time[policy][1]/(total_call_within_limit[policy][1] + total_call_without_limit[policy][1]), 3)}")
            print(f"    average waiting time (type3): {round(total_waiting_time[policy][2]/(total_call_within_limit[policy][2] + total_call_without_limit[policy][2]), 3)}")
            print(f"    average number in queue: {round(Queue[policy].Mean(), 3)}")

            SC[policy].append(round(sum(total_call_within_limit[policy])/sum(total_call[policy]), 3))
            AWA[policy].append(round(sum(total_workload[policy])/len(total_workload[policy])/fleet_num/maximum_workload, 3)) 
            df_call_all[policy].append(sum(total_call[policy]))
            df_call_tp1[policy].append(total_call[policy][0])
            df_call_tp2[policy].append(total_call[policy][1])
            df_call_tp3[policy].append(total_call[policy][2])
            df_call_lost[policy].append(total_call_lost[policy])
            df_call_hospital[policy].append(total_call_hospital[policy])
            df_call_scene[policy].append(total_call_scene[policy])
            df_call_within_15[policy].append(total_call_within_limit[policy][0])
            df_call_within_10[policy].append(total_call_within_limit[policy][1])
            df_call_within_5[policy].append(total_call_within_limit[policy][2])
            df_call_without_15[policy].append(total_call_without_limit[policy][0])
            df_call_without_10[policy].append(total_call_without_limit[policy][1])
            df_call_without_5[policy].append(total_call_without_limit[policy][2])
            df_shift1_avg_workload[policy].append(round(np.mean(shift1_workload), 3))
            df_shift1_std_workload[policy].append(round(np.std(shift1_workload), 3))
            df_shift1_avg_to_scene[policy].append(round(np.mean(shift1_to_scene), 3))
            df_shift1_std_to_scene[policy].append(round(np.std(shift1_to_scene), 3))
            df_shift1_avg_service_on_scene[policy].append(round(np.mean(shift1_service_on_scene), 3))
            df_shift1_std_service_on_scene[policy].append(round(np.std(shift1_service_on_scene), 3))
            df_shift1_avg_to_hospital[policy].append(round(np.mean(shift1_to_hospital), 3))
            df_shift1_std_to_hospital[policy].append(round(np.std(shift1_to_hospital), 3))
            df_shift1_avg_redeployment[policy].append(round(np.mean(shift1_redeployment), 3))
            df_shift1_std_redeployment[policy].append(round(np.std(shift1_redeployment), 3))
            df_shift2_avg_workload[policy].append(round(np.mean(shift2_workload), 3))
            df_shift2_std_workload[policy].append(round(np.std(shift2_workload), 3))
            df_shift2_avg_to_scene[policy].append(round(np.mean(shift2_to_scene), 3))
            df_shift2_std_to_scene[policy].append(round(np.std(shift2_to_scene), 3))
            df_shift2_avg_service_on_scene[policy].append(round(np.mean(shift2_service_on_scene), 3))
            df_shift2_std_service_on_scene[policy].append(round(np.std(shift2_service_on_scene), 3))
            df_shift2_avg_to_hospital[policy].append(round(np.mean(shift2_to_hospital), 3))
            df_shift2_std_to_hospital[policy].append(round(np.std(shift2_to_hospital), 3))
            df_shift2_avg_redeployment[policy].append(round(np.mean(shift2_redeployment), 3))
            df_shift2_std_redeployment[policy].append(round(np.std(shift2_redeployment), 3))
            df_avg_wait_time_all[policy].append(round(sum(total_waiting_time[policy])/(total_call_hospital[policy] + total_call_scene[policy]), 3))
            df_avg_wait_time_tp1[policy].append(round(total_waiting_time[policy][0]/(total_call_within_limit[policy][0] + total_call_without_limit[policy][0]), 3))
            df_avg_wait_time_tp2[policy].append(round(total_waiting_time[policy][1]/(total_call_within_limit[policy][1] + total_call_without_limit[policy][1]), 3))
            df_avg_wait_time_tp3[policy].append(round(total_waiting_time[policy][2]/(total_call_within_limit[policy][2] + total_call_without_limit[policy][2]), 3))
            df_avg_number_in_queue[policy].append(round(Queue[policy].Mean(), 3))

            axs[i].plot(clock, SC_every_t[policy], label="SC")
            axs[i].plot(clock, avg_waiting_time_every_t[policy], label="avg_waiting_time")
            axs[i].plot(clock, avg_number_in_queue_ecery_t[policy], label="avg_number_in_queue")
            axs[i].plot(clock, cur_number_in_queue_every_t[policy], label="cur_number_in_queue")
            # 設定每個子圖的標題
            axs[i].set_title(f"{policy} - performance through time T")
            
            # 設定x軸和y軸的標籤
            axs[i].set_xlabel("Clock")
            axs[i].set_ylabel("Accumulated Workload")
            
            # 顯示每個圖的圖例
            axs[i].legend()

        # 調整布局，避免標籤重疊
        plt.tight_layout()

        # 顯示圖表
        plt.savefig(f"./figure/Queue_every_t_four_rep{Reps+1}_redeploylimit{redeployment_limit}_fleet{fleet_num}_P{acceptance_threshold}.png")
        
        # 創建一個2行1列的子圖，這樣可以將兩張圖放在一起
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))  # 2行1列的子圖
        for i,policy in enumerate(Policy):
            axs[i].plot(clock, SC_every_t[policy], label="SC")
            # 設定每個子圖的標題
            axs[i].set_title(f"{policy} - performance through time T")
            
            # 設定x軸和y軸的標籤
            axs[i].set_xlabel("Clock")
            axs[i].set_ylabel("Accumulated Workload")
            
            # 顯示每個圖的圖例
            axs[i].legend()

        # 調整布局，避免標籤重疊
        plt.tight_layout()

        # 顯示圖表
        plt.savefig(f"./figure/Queue_every_t_SC_rep{Reps+1}_redeploylimit{redeployment_limit}_fleet{fleet_num}_P{acceptance_threshold}.png")

        # 創建一個2行1列的子圖，這樣可以將兩張圖放在一起
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))  # 2行1列的子圖
        for i,policy in enumerate(Policy):
            axs[i].plot(clock, avg_number_in_queue_ecery_t[policy], label="avg_number_in_queue")
            # 設定每個子圖的標題
            axs[i].set_title(f"{policy} - performance through time T")
            
            # 設定x軸和y軸的標籤
            axs[i].set_xlabel("Clock")
            axs[i].set_ylabel("Accumulated Workload")
            
            # 顯示每個圖的圖例
            axs[i].legend()

        # 調整布局，避免標籤重疊
        plt.tight_layout()

        # 顯示圖表
        plt.savefig(f"./figure/Queue_every_t_avg_number_in_queue_rep{Reps+1}_redeploylimit{redeployment_limit}_fleet{fleet_num}_P{acceptance_threshold}.png")

        # 創建一個2行1列的子圖，這樣可以將兩張圖放在一起
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))  # 2行1列的子圖
        for i,policy in enumerate(Policy):
            axs[i].plot(clock, cur_number_in_queue_every_t[policy], label="cur_number_in_queue")
            # 設定每個子圖的標題
            axs[i].set_title(f"{policy} - performance through time T")
            
            # 設定x軸和y軸的標籤
            axs[i].set_xlabel("Clock")
            axs[i].set_ylabel("Accumulated Workload")
            
            # 顯示每個圖的圖例
            axs[i].legend()

        # 調整布局，避免標籤重疊
        plt.tight_layout()

        # 顯示圖表
        plt.savefig(f"./figure/Queue_every_t_cur_number_in_queue_rep{Reps+1}_redeploylimit{redeployment_limit}_fleet{fleet_num}_P{acceptance_threshold}.png")

    df_all = []
    for policy in Policy:
        temp_df = pd.DataFrame({
            "SC": SC[policy],
            "AWA": AWA[policy],
            "total_call (all_type)": df_call_all[policy],
            "total_call (type1)": df_call_tp1[policy],
            "total_call (type2)": df_call_tp2[policy],
            "total_call (type3)": df_call_tp3[policy],
            "total_lost_call": df_call_lost[policy],
            "total_call_hospital": df_call_hospital[policy],
            "total_call_scene": df_call_scene[policy],
            "whthin 15 min arrival": df_call_within_15[policy],
            "whthin 10 min arrival": df_call_within_10[policy],
            "whthin 5 min arrival": df_call_within_5[policy],
            "whthout 15 min arrival": df_call_without_15[policy],
            "whthout 10 min arrival": df_call_without_10[policy],
            "whthout 5 min arrival": df_call_without_5[policy],
            "shift 1 workload average": df_shift1_avg_workload[policy],
            "shift 1 workload std": df_shift1_std_workload[policy],
            "shift 1 to scene average": df_shift1_avg_to_scene[policy],
            "shift 1 to scene std": df_shift1_std_to_scene[policy],
            "shift 1 service on scene average": df_shift1_avg_service_on_scene[policy],
            "shift 1 service on scene std": df_shift1_std_service_on_scene[policy],
            "shift 1 to hospital average": df_shift1_avg_to_hospital[policy],
            "shift 1 to hospital std": df_shift1_std_to_hospital[policy],
            "shift 1 redeployment average": df_shift1_avg_redeployment[policy],
            "shift 1 redeployment std": df_shift1_std_redeployment[policy],
            "shift 2 workload average": df_shift2_avg_workload[policy],
            "shift 2 workload std": df_shift2_std_workload[policy],
            "shift 2 to scene average": df_shift2_avg_to_scene[policy],
            "shift 2 to scene std": df_shift2_std_to_scene[policy],
            "shift 2 service on scene average": df_shift2_avg_service_on_scene[policy],
            "shift 2 service on scene std": df_shift2_std_service_on_scene[policy],
            "shift 2 to hospital average": df_shift2_avg_to_hospital[policy],
            "shift 2 to hospital std": df_shift2_std_to_hospital[policy],
            "shift 2 redeployment average": df_shift2_avg_redeployment[policy],
            "shift 2 redeployment std": df_shift2_std_redeployment[policy],
            "AWT (all_type)": df_avg_wait_time_all[policy],
            "AWT (type1)": df_avg_wait_time_tp1[policy],
            "AWT (type2)": df_avg_wait_time_tp2[policy],
            "AWT (type3)": df_avg_wait_time_tp3[policy],
            "average number in queue": df_avg_number_in_queue[policy],
        })
        df_all.append(temp_df)

    with pd.ExcelWriter(f"simulation_result_model4_Queue2_fleet{fleet_num}_redeploylimit{redeployment_limit}_P{acceptance_threshold}.xlsx") as writer:
        for i, policy in enumerate(Policy):
            df_all[i].to_excel(writer, sheet_name=policy, index=False)


for coverage_threshold in [6/10*7, 6/10*6, 6/10*5]:
    simulation_model(coverage_threshold = coverage_threshold, redeployment_limit= 5)
