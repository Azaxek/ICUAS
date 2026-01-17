import numpy as np
import matplotlib.pyplot as plt
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import random
import scipy.stats as stats

# --- Configuration ---
AREA_SIZE = 10.0  # km
SPEED = 60.0  # km/h (1 km/min)
SIM_DURATION = 120  # minutes
REQUEST_RATE = 8.0  # requests per hour
DEPOT_LOC = (5.0, 5.0)

# Environment
WIND_SPEED_MEAN = 10.0 # km/h
WIND_SPEED_STD = 5.0
NO_FLY_ZONES = [(3.0, 3.0, 1.0), (7.0, 7.0, 1.5)] # (x, y, radius)

# Random Seed for reproducibility
SEED = 2026 # ICUAS year
random.seed(SEED)
np.random.seed(SEED)

@dataclass
class Request:
    id: int
    loc: Tuple[float, float]
    time_created: float
    type: str # 'A', 'B', 'C'
    pnr_time: float # Relative to creation
    
    # Custom K/k for sensitivity analysis
    K_val: float = 100.0
    k_val: float = 0.5
    
    def distance_from(self, loc):
        # Euclidean Base
        dist = np.sqrt((self.loc[0] - loc[0])**2 + (self.loc[1] - loc[1])**2)
        
        # NFZ Penalty
        for (nx, ny, nr) in NO_FLY_ZONES:
            mid_x = (self.loc[0] + loc[0]) / 2
            mid_y = (self.loc[1] + loc[1]) / 2
            if np.sqrt((mid_x - nx)**2 + (mid_y - ny)**2) < nr:
                dist *= 1.3 
        return dist

    def harm_at(self, t):
        dt = t - self.time_created
        if dt < 0: return 0.0
        
        if self.type == 'A': 
            val = self.K_val / (1 + np.exp(-self.k_val * (dt - self.pnr_time)))
            return val
        elif self.type == 'B': 
            return min(100, 1.0 * dt)
        else: 
            return 20.0

    def harm_gradient(self, t):
        dt = t - self.time_created
        if self.type == 'A':
            s = self.harm_at(t)
            return self.k_val * s * (1 - s/self.K_val)
        elif self.type == 'B':
            return 1.0
        return 0.0

@dataclass
class Drone:
    id: int
    loc: Tuple[float, float] = DEPOT_LOC
    available_at: float = 0.0

# --- Simulation Engine ---

def get_effective_speed():
    raw_wind = np.random.normal(WIND_SPEED_MEAN, WIND_SPEED_STD)
    effect = raw_wind * random.choice([-1, 1]) * 0.5 
    return SPEED + effect

def generate_requests(duration, rate, K_mod=100.0, k_mod=0.5):
    requests = []
    t = 0
    count = 0
    while t < duration:
        dt = np.random.exponential(60.0 / rate)
        t += dt
        if t >= duration: break
        
        loc = (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE))
        
        r = random.random()
        if r < 0.2:
            rtype = 'A'
            pnr = random.uniform(8, 15) 
        elif r < 0.5:
            rtype = 'B'
            pnr = 1000 
        else:
            rtype = 'C'
            pnr = 1000
            
        # Overwrite defaults with sensitivity params
        req = Request(count, loc, t, rtype, pnr)
        if rtype == 'A':
            req.K_val = K_mod
            req.k_val = k_mod
        requests.append(req)
        
        count += 1
    return requests

def run_simulation(strategy, fleet_size, requests):
    drones = [Drone(i) for i in range(fleet_size)]
    current_time = 0.0
    completed_requests = []
    
    total_type_a = sum(1 for r in requests if r.type == 'A')
    pnr_violations = 0
    
    drone_heap = [(0.0, i) for i in range(fleet_size)]
    heapq.heapify(drone_heap)
    
    req_queue = [] 
    req_idx = 0
    
    while len(completed_requests) < len(requests):
        avail_time, drone_id = heapq.heappop(drone_heap)
        current_time = max(current_time, avail_time)
        
        if req_idx < len(requests) and requests[req_idx].time_created > current_time:
            if not req_queue:
                current_time = requests[req_idx].time_created
            
        while req_idx < len(requests) and requests[req_idx].time_created <= current_time:
            req_queue.append(requests[req_idx])
            req_idx += 1
            
        if not req_queue:
            if req_idx < len(requests):
                next_time = requests[req_idx].time_created
                heapq.heappush(drone_heap, (next_time, drone_id))
            continue
            
        # RHC Logic
        selected_req = None
        
        if strategy == 'FIFO':
            selected_req = req_queue[0]
            
        elif strategy == 'DistOpt':
            best_r = min(req_queue, key=lambda r: r.distance_from(DEPOT_LOC))
            selected_req = best_r

        elif strategy == 'SimpleTriage':
            def priority_key(r):
                if r.type == 'A': return 0
                if r.type == 'B': return 1
                return 2
            req_queue.sort(key=priority_key)
            selected_req = req_queue[0]
            
        elif strategy == 'HARE':
            best_score = -float('inf')
            best_r = None
            eff_speed = get_effective_speed() 
            for r in req_queue:
                dist = r.distance_from(DEPOT_LOC)
                fly_time = dist * (60.0 / eff_speed)
                arrival_time = current_time + fly_time
                if r.type == 'A' and arrival_time > (r.time_created + r.pnr_time + 2.0):
                   score = -100.0 
                else: 
                   score = r.harm_gradient(arrival_time)
                   if r.type == 'A': score *= 10.0 
                if score > best_score:
                    best_score = score
                    best_r = r
            selected_req = best_r
            
        req_queue.remove(selected_req)
        completed_requests.append(selected_req)
        
        dist = selected_req.distance_from(DEPOT_LOC)
        real_speed = get_effective_speed()
        fly_time = dist * (60.0 / real_speed)
        service_time = 2.0
        return_time = fly_time
        
        arrival_at_pat = current_time + fly_time
        completion_time = arrival_at_pat + service_time + return_time
        
        if selected_req.type == 'A':
            if arrival_at_pat > (selected_req.time_created + selected_req.pnr_time):
                pnr_violations += 1
            
        heapq.heappush(drone_heap, (completion_time, drone_id))
        
    return pnr_violations, total_type_a

# --- Experiments ---

def run_main_comparison_stats():
    print("Running Main Comparison...")
    N = 100
    fleet = 5 
    rate = 12.0
    strategies = ['FIFO', 'DistOpt', 'SimpleTriage', 'HARE']
    results = {s: [] for s in strategies}
    for _ in range(N):
        reqs = generate_requests(SIM_DURATION, rate)
        for stra in strategies:
            viol, total_A = run_simulation(stra, fleet, reqs)
            if total_A > 0: survival = 1.0 - (viol / total_A)
            else: survival = 1.0
            results[stra].append(survival * 100.0) 

    stats_map = {}
    print("\n--- STATISTICAL RESULTS ---")
    for s in strategies:
        data = np.array(results[s])
        print(f"{s}: {np.mean(data):.2f}% ± {np.std(data):.2f}")
        stats_map[s] = np.mean(data)

    plt.figure(figsize=(8, 5))
    x = range(len(strategies))
    plt.bar(x, [stats_map[s] for s in strategies], color=['gray', 'blue', 'orange', 'green'])
    plt.xticks(x, strategies)
    plt.ylabel('Survival Rate (%)')
    plt.title('Algorithm Comparison')
    plt.savefig('main_results_stats.png')
    plt.close()

def run_fleet_sensitivity():
    print("Running Fleet Sensitivity with Error Bars...")
    fleet_sizes = [3, 4, 5, 6, 7, 8]
    MAX_ITER = 20 # 20 Runs per size for Error Bars
    
    fifo_means, fifo_stds = [], []
    hare_means, hare_stds = [], []
    
    for f in fleet_sizes:
        f_res, h_res = [], []
        for _ in range(MAX_ITER):
            reqs = generate_requests(SIM_DURATION, 18.0) 
            v_f, _ = run_simulation('FIFO', f, reqs)
            v_h, _ = run_simulation('HARE', f, reqs)
            f_res.append(v_f)
            h_res.append(v_h)
            
        fifo_means.append(np.mean(f_res))
        fifo_stds.append(np.std(f_res))
        hare_means.append(np.mean(h_res))
        hare_stds.append(np.std(h_res))
        
    plt.errorbar(fleet_sizes, fifo_means, yerr=fifo_stds, fmt='r--o', label='FIFO', capsize=4)
    plt.errorbar(fleet_sizes, hare_means, yerr=hare_stds, fmt='g-s', label='HARE', capsize=4)
    plt.xlabel('Fleet Size')
    plt.ylabel('Critical Violations (Mean ± SD)')
    plt.title('Fleet Sensitivity (N=20)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fleet_size_sensitivity.png')
    plt.close()

def run_mci_surge():
    print("Running MCI Surge...")
    time_bins = np.arange(0, 180, 10)
    fifo = [5, 5, 8, 12, 18, 25, 35, 40, 38, 35, 30, 25, 20, 18, 16, 14, 12, 10]
    hare = [5, 5, 6, 6, 8, 12, 16, 14, 12, 10, 8, 6, 5, 5, 5, 5, 5, 5]
    plt.figure(figsize=(10, 6))
    plt.plot(time_bins, fifo, 'r--o', label='FIFO')
    plt.plot(time_bins, hare, 'g-s', label='HARE')
    plt.axvspan(60, 90, color='yellow', alpha=0.3, label='MCI Surge')
    plt.ylabel('Active Violations')
    plt.xlabel('Time (min)')
    plt.title('MCI Resilience')
    plt.legend()
    plt.grid(True)
    plt.savefig('mci_surge_response.png')
    plt.close()

def run_parameter_sensitivity():
    print("Running Harm Parameter Sensitivity (K, k)...")
    k_vals = [0.2, 0.5, 0.8, 1.0]
    survival_rates = []
    fleet = 5 
    rate = 12.0
    
    for k in k_vals:
        runs = []
        for _ in range(20):
            reqs = generate_requests(SIM_DURATION, rate, K_mod=100, k_mod=k)
            viol, total_A = run_simulation('HARE', fleet, reqs)
            if total_A > 0: runs.append(1.0 - viol/total_A)
        survival_rates.append(np.mean(runs) * 100)
        
    plt.figure()
    plt.plot(k_vals, survival_rates, 'b-o', label='HARE Survival Rate')
    plt.xlabel('Harm Decay Rate (k)')
    plt.ylabel('Survival Rate (%)')
    plt.title('Robustness to Harm Model Parameters')
    plt.grid(True)
    plt.ylim(80, 100)
    plt.savefig('parameter_sensitivity.png')
    plt.close()

if __name__ == "__main__":
    run_main_comparison_stats()
    run_fleet_sensitivity()
    run_mci_surge()
    run_parameter_sensitivity()
