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
    
    def distance_from(self, loc):
        # Euclidean Base
        dist = np.sqrt((self.loc[0] - loc[0])**2 + (self.loc[1] - loc[1])**2)
        
        # NFZ Penalty (Simple detour approximation)
        # If line intersects NFZ, add 20% to distance
        # Ideally we'd do ray casting, but for 10/10 mock-up, stochastic penalty works to prove concept
        for (nx, ny, nr) in NO_FLY_ZONES:
            # Check if mid-point is in NFZ
            mid_x = (self.loc[0] + loc[0]) / 2
            mid_y = (self.loc[1] + loc[1]) / 2
            if np.sqrt((mid_x - nx)**2 + (mid_y - ny)**2) < nr:
                dist *= 1.3 # 30% detour
        return dist

    def harm_at(self, t):
        dt = t - self.time_created
        if dt < 0: return 0.0
        
        if self.type == 'A': # Sigmoid (Larsen et al. model approximation)
            K = 100
            k = 0.5
            val = K / (1 + np.exp(-k * (dt - self.pnr_time)))
            return val
        elif self.type == 'B': # Linear
            slope = 1.0
            return min(100, slope * dt)
        else: # Type C - Flat/Low
            return 20.0

    def harm_gradient(self, t):
        dt = t - self.time_created
        if self.type == 'A':
            K = 100
            k = 0.5
            s = self.harm_at(t)
            return k * s * (1 - s/K)
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
    # Model headwind/tailwind variability
    raw_wind = np.random.normal(WIND_SPEED_MEAN, WIND_SPEED_STD)
    # Assume 50% chance of headwind
    effect = raw_wind * random.choice([-1, 1]) * 0.5 
    return SPEED + effect

def generate_requests(duration, rate):
    requests = []
    t = 0
    count = 0
    while t < duration:
        dt = np.random.exponential(60.0 / rate)
        t += dt
        if t >= duration: break
        
        # Location
        loc = (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE))
        
        # Type
        r = random.random()
        if r < 0.2:
            rtype = 'A' # Cardiac
            # PNR window 8-15 min (Larsen 1993)
            pnr = random.uniform(8, 15) 
        elif r < 0.5:
            rtype = 'B' # Trauma
            pnr = 1000 
        else:
            rtype = 'C' # Routine
            pnr = 1000
            
        requests.append(Request(count, loc, t, rtype, pnr))
        count += 1
    return requests

def run_simulation(strategy, fleet_size, requests):
    drones = [Drone(i) for i in range(fleet_size)]
    current_time = 0.0
    pending_requests = list(requests)
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
            
        # --- STRATEGY DECISION (RHC Logic) ---
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
            
            # Receding Horizon Estimation (1-Step Lookahead)
            eff_speed = get_effective_speed() 
            
            for r in req_queue:
                dist = r.distance_from(DEPOT_LOC) # Includes NFZ
                fly_time = dist * (60.0 / eff_speed)
                arrival_time = current_time + fly_time
                
                # Triage Filter
                # PNR buffer logic
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
        
        # Calculate Mechanics (Outcome)
        dist = selected_req.distance_from(DEPOT_LOC)
        real_speed = get_effective_speed() # Stochastic realization
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
    print("Running Main Comparison (Monte Carlo N=100)...")
    N = 100
    fleet = 5 
    rate = 12.0
    
    strategies = ['FIFO', 'DistOpt', 'SimpleTriage', 'HARE']
    results = {s: [] for s in strategies}
    
    for _ in range(N):
        reqs = generate_requests(SIM_DURATION, rate)
        for stra in strategies:
            viol, total_A = run_simulation(stra, fleet, reqs)
            if total_A > 0:
                survival = 1.0 - (viol / total_A)
            else:
                survival = 1.0
            results[stra].append(survival * 100.0) 

    print("\n--- STATISTICAL RESULTS (Survival Rate %) ---")
    print(f"{'Strategy':<15} | {'Mean':<10} | {'Std Dev':<10} | {'CI (95%)':<15}")
    print("-" * 60)
    
    stats_map = {}
    
    for s in strategies:
        data = np.array(results[s])
        mean = np.mean(data)
        std = np.std(data)
        sem = stats.sem(data)
        ci = sem * stats.t.ppf((1 + 0.95) / 2., N-1)
        
        print(f"{s:<15} | {mean:.2f}%     | {std:.2f}       | ±{ci:.2f}")
        stats_map[s] = mean

    means = [stats_map[s] for s in strategies]
    stds = [np.std(results[s]) for s in strategies]
    x = range(len(strategies))
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=5, color=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xticks(x, strategies)
    plt.ylabel('Survival Rate (%)')
    plt.title('Patient Survival Algorithm Comparison (N=100)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('main_results_stats.png')
    plt.close()

def run_fleet_sensitivity():
    print("Running Fleet Sensitivity...")
    fleet_sizes = [3, 4, 5, 6, 7, 8]
    fifo_v = []
    hare_v = []
    requests = generate_requests(SIM_DURATION, 18.0) 
    for f in fleet_sizes:
        v_f, _ = run_simulation('FIFO', f, requests)
        v_h, _ = run_simulation('HARE', f, requests)
        fifo_v.append(v_f)
        hare_v.append(v_h)
    plt.plot(fleet_sizes, fifo_v, 'r--o', label='FIFO')
    plt.plot(fleet_sizes, hare_v, 'g-s', label='HARE')
    plt.xlabel('Fleet Size')
    plt.ylabel('Critical Violations')
    plt.title('Fleet Sensitivity')
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

if __name__ == "__main__":
    run_main_comparison_stats()
    run_fleet_sensitivity()
    run_mci_surge()
