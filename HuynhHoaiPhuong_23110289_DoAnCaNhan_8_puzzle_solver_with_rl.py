import heapq
import time
import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext, filedialog
from collections import deque
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import pandas as pd
from tkinter import filedialog

class Puzzle8:
    def __init__(self, initial, goal):
        self.initial = tuple(initial)
        self.goal = tuple(goal)
        self.n = 3
        self.goal_positions = {self.goal[i]: (i // self.n, i % self.n) for i in range(self.n * self.n)}
        self.stats = {"nodes_expanded": 0, "max_fringe_size": 0, "states_visited": 0}
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        self.action_names = ["Up", "Down", "Left", "Right"]
        self.action_colors = ["#FF4040", "#40FF40", "#4040FF", "#FFD700"]
        self.max_runtime = 60
        self.visited_states = set()
        self.neighbor_cache = {}
        self.manhattan_cache = {}
        self.linear_conflict_cache = {}
        self.current_belief_states = []
        self.parent_map = {}

    def is_solvable(self, state):
        state_list = [x for x in state if x != 0]
        inversions = sum(
            1 for i in range(len(state_list)) for j in range(i + 1, len(state_list))
            if state_list[i] > state_list[j]
        )
        return inversions % 2 == 0


    def reset_stats(self):
        self.stats = {"nodes_expanded": 0, "max_fringe_size": 0, "states_visited": 0}
        self.neighbor_cache.clear()
        self.manhattan_cache.clear()
        self.linear_conflict_cache.clear()
        self.visited_states.clear()
        self.parent_map.clear()

    def get_neighbors(self, state):
        if state in self.neighbor_cache:
            self.stats["nodes_expanded"] += 1
            return self.neighbor_cache[state]
        state_list = list(state)
        zero_index = state_list.index(0)
        row, col = divmod(zero_index, self.n)
        moves = []
        actions = []
        for idx, (dr, dc) in enumerate(self.actions):
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.n and 0 <= new_col < self.n:
                new_index = new_row * self.n + new_col
                state_list[zero_index], state_list[new_index] = state_list[new_index], state_list[zero_index]
                new_state = tuple(state_list)
                moves.append(new_state)
                actions.append(idx)
                state_list[zero_index], state_list[new_index] = state_list[new_index], state_list[zero_index]
        self.stats["nodes_expanded"] += 1
        self.neighbor_cache[state] = moves
        for move, action_idx in zip(moves, actions):
            if move not in self.parent_map:
                self.parent_map[move] = (state, action_idx)
        return moves

    def manhattan_distance(self, state):
        if state in self.manhattan_cache:
            return self.manhattan_cache[state]
        distance = sum(
            abs(i // self.n - self.goal_positions[value][0]) +
            abs(i % self.n - self.goal_positions[value][1])
            for i, value in enumerate(state) if value != 0
        )
        self.manhattan_cache[state] = distance
        return distance

    def linear_conflict(self, state):
        if state in self.linear_conflict_cache:
            return self.linear_conflict_cache[state]
        distance = self.manhattan_distance(state)
        for dim, size in [(0, self.n), (1, self.n)]:
            for i in range(size):
                tiles = [(state[r * self.n + c], (r, c))
                         for r in range(self.n)
                         for c in range(self.n)
                         if (r if dim == 0 else c) == i and state[r * self.n + c] != 0]
                conflicts = sum(1 for j in range(len(tiles))
                               for k in range(j + 1, len(tiles))
                               if tiles[j][0] > tiles[k][0] and
                               self.goal_positions[tiles[j][0]][dim] < self.goal_positions[tiles[k][0]][dim])
                distance += 2 * conflicts
        self.linear_conflict_cache[state] = distance
        return distance

    def get_action(self, state, action):
        state_list = list(state)
        zero_index = state_list.index(0)
        row, col = divmod(zero_index, self.n)
        dr, dc = self.actions[action]
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < self.n and 0 <= new_col < self.n:
            new_index = new_row * self.n + new_col
            state_list[zero_index], state_list[new_index] = state_list[new_index], state_list[zero_index]
            return tuple(state_list), True
        return state, False

    def q_learning(self, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        q_table = {}
        max_steps = 200
        for _ in range(episodes):
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            state = self.initial
            self.visited_states.add(state)
            path = [state]
            for _ in range(max_steps):
                self.stats["states_visited"] += 1
                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    state_key = state
                    if state_key not in q_table:
                        q_table[state_key] = [0.0] * 4
                    action = np.argmax(q_table[state_key])
                next_state, valid = self.get_action(state, action)
                self.visited_states.add(next_state)
                if valid:
                    if len(path) > 1:
                        self.parent_map[next_state] = (state, action)
                    path.append(next_state)
                reward = 100 if next_state == self.goal else -1 if valid else -10
                self.stats["nodes_expanded"] += 1
                next_state_key = next_state
                if next_state_key not in q_table:
                    q_table[next_state_key] = [0.0] * 4
                q_table[state_key][action] += alpha * (
                    reward + gamma * max(q_table[next_state_key]) - q_table[state_key][action]
                )
                state = next_state
                if state == self.goal:
                    break
        state = self.initial
        path = []
        visited = set()
        steps = 0
        while state != self.goal and steps < max_steps:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            if state in visited:
                return None, time.time() - start_time, 0, self.stats, []
            visited.add(state)
            state_key = state
            if state_key not in q_table:
                return None, time.time() - start_time, 0, self.stats, []
            action = np.argmax(q_table[state_key])
            next_state, valid = self.get_action(state, action)
            if not valid:
                return None, time.time() - start_time, 0, self.stats, []
            path.append(next_state)
            state = next_state
            steps += 1
            self.stats["states_visited"] += 1
            self.visited_states.add(state)
        if state == self.goal:
            return path, time.time() - start_time, len(path), self.stats, []
        return None, time.time() - start_time, 0, self.stats, []

    def sarsa(self, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        q_table = {}
        max_steps = 200
        for _ in range(episodes):
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            state = self.initial
            self.visited_states.add(state)
            path = [state]
            state_key = state
            if state_key not in q_table:
                q_table[state_key] = [0.0] * 4
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[state_key])
            for _ in range(max_steps):
                self.stats["states_visited"] += 1
                next_state, valid = self.get_action(state, action)
                self.visited_states.add(next_state)
                if valid:
                    if len(path) > 1:
                        self.parent_map[next_state] = (state, action)
                    path.append(next_state)
                reward = 100 if next_state == self.goal else -1 if valid else -10
                self.stats["nodes_expanded"] += 1
                next_state_key = next_state
                if next_state_key not in q_table:
                    q_table[next_state_key] = [0.0] * 4
                if random.random() < epsilon:
                    next_action = random.randint(0, 3)
                else:
                    next_action = np.argmax(q_table[next_state_key])
                q_table[state_key][action] += alpha * (
                    reward + gamma * q_table[next_state_key][next_action] - q_table[state_key][action]
                )
                state = next_state
                state_key = next_state_key
                action = next_action
                if state == self.goal:
                    break
        state = self.initial
        path = []
        visited = set()
        steps = 0
        while state != self.goal and steps < max_steps:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            if state in visited:
                return None, time.time() - start_time, 0, self.stats, []
            visited.add(state)
            state_key = state
            if state_key not in q_table:
                return None, time.time() - start_time, 0, self.stats, []
            action = np.argmax(q_table[state_key])
            next_state, valid = self.get_action(state, action)
            if not valid:
                return None, time.time() - start_time, 0, self.stats, []
            path.append(next_state)
            state = next_state
            steps += 1
            self.stats["states_visited"] += 1
            self.visited_states.add(state)
        if state == self.goal:
            return path, time.time() - start_time, len(path), self.stats, []
        return None, time.time() - start_time, 0, self.stats, []

    def bfs(self):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        queue = deque([(self.initial, [])])
        visited = {self.initial}
        self.visited_states.add(self.initial)
        while queue:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(queue))
            state, path = queue.popleft()
            self.stats["states_visited"] += 1
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    self.visited_states.add(neighbor)
                    new_path = path + [neighbor]
                    if neighbor == self.goal:
                        return new_path, time.time() - start_time, len(new_path), self.stats, []
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return None, time.time() - start_time, 0, self.stats, []

    def dfs(self, max_depth=100):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        stack = [(self.initial, [self.initial], 0)]
        visited = set()
        self.visited_states.add(self.initial)
        while stack:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(stack))
            state, path, depth = stack.pop()
            self.stats["states_visited"] += 1
            if state == self.goal:
                return path[1:], time.time() - start_time, len(path) - 1, self.stats, []
            if depth > max_depth or state in visited:
                continue
            visited.add(state)
            neighbors = self.get_neighbors(state)
            for idx, neighbor in enumerate(neighbors):
                if neighbor not in visited:
                    self.visited_states.add(neighbor)
                    new_path = path + [neighbor]
                    self.parent_map[neighbor] = (state, idx)
                    stack.append((neighbor, new_path, depth + 1))
        return None, time.time() - start_time, 0, self.stats, []

    def ucs(self):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        open_set = [(0, 0, self.initial, [])]
        closed_set = set()
        g_scores = {self.initial: 0}
        self.visited_states.add(self.initial)
        counter = 1
        while open_set:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(open_set))
            cost, _, state, path = heapq.heappop(open_set)
            if state == self.goal:
                return path, time.time() - start_time, len(path), self.stats, []
            if state in closed_set:
                continue
            closed_set.add(state)
            self.stats["states_visited"] += 1
            for idx, neighbor in enumerate(self.get_neighbors(state)):
                if neighbor in closed_set:
                    continue
                self.visited_states.add(neighbor)
                tentative_g = g_scores[state] + 1
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    new_path = path + [neighbor]
                    self.parent_map[neighbor] = (state, idx)
                    heapq.heappush(open_set, (tentative_g, counter, neighbor, new_path))
                    counter += 1
        return None, time.time() - start_time, 0, self.stats, []

    def iterative_deepening(self, max_depth=50):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        for depth_limit in range(max_depth):
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            result, path_length = self._depth_limited_search(self.initial, [], depth_limit, set())
            if result is not None:
                return result, time.time() - start_time, path_length, self.stats, []
        return None, time.time() - start_time, 0, self.stats, []

    def _depth_limited_search(self, state, path, depth_limit, visited):
        self.stats["states_visited"] += 1
        self.visited_states.add(state)
        if state == self.goal:
            return path, len(path)
        if depth_limit <= 0:
            return None, 0
        visited.add(state)
        for neighbor in self.get_neighbors(state):
            if neighbor not in visited:
                self.visited_states.add(neighbor)
                new_path = path + [neighbor]
                result, path_length = self._depth_limited_search(neighbor, new_path, depth_limit - 1, visited.copy())
                if result is not None:
                    return result, path_length
        return None, 0

    def ida_star(self):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        h = self.manhattan_distance(self.initial)
        threshold = h
        path = [self.initial]
        self.visited_states.add(self.initial)
        while threshold < float("inf"):# Lặp lại quá trình nếu còn hy vọng (ngưỡng threshold chưa quá vô hạn).
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            found, new_threshold = self._ida_star_search(path, 0, threshold, set())
            if found:
                return path[1:], time.time() - start_time, len(path) - 1, self.stats, []
            if new_threshold == float("inf"):
                break
            threshold = new_threshold
        return None, time.time() - start_time, 0, self.stats, []

    def _ida_star_search(self, path, g, threshold, visited):
        state = path[-1]
        self.stats["states_visited"] += 1
        self.visited_states.add(state)
        f = g + self.manhattan_distance(state)
        if f > threshold:
            return False, f
        if state == self.goal:
            return True, 0
        visited.add(state)
        min_threshold = float("inf")
        for neighbor in self.get_neighbors(state):
            if neighbor not in visited and neighbor not in path:
                self.visited_states.add(neighbor)
                path.append(neighbor)
                found, new_threshold = self._ida_star_search(path, g + 1, threshold, visited.copy())
                if found:
                    return True, 0
                path.pop()
                min_threshold = min(min_threshold, new_threshold)
        return False, min_threshold

    def a_star(self, use_linear_conflict=False):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        h_func = self.linear_conflict if use_linear_conflict else self.manhattan_distance
        open_set = [(h_func(self.initial), 0, 0, self.initial, [])]
        closed_set = set()
        g_scores = {self.initial: 0}
        self.visited_states.add(self.initial)
        counter = 1
        while open_set:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(open_set))
            f, g, _, state, path = heapq.heappop(open_set) #Lấy trạng thái có f thấp nhất
            if state == self.goal:
                return path, time.time() - start_time, len(path), self.stats, []
            if state in closed_set:
                continue
            closed_set.add(state)
            self.stats["states_visited"] += 1
            for idx, neighbor in enumerate(self.get_neighbors(state)):#self.get_neighbors(state) trả về các trạng thái có thể đạt được từ trạng thái hiện tại.
                if neighbor in closed_set:
                    continue
                self.visited_states.add(neighbor)
                tentative_g = g + 1
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + h_func(neighbor)
                    new_path = path + [neighbor]
                    self.parent_map[neighbor] = (state, idx)
                    heapq.heappush(open_set, (f_score, tentative_g, counter, neighbor, new_path))
                    counter += 1
        return None, time.time() - start_time, 0, self.stats, []

    def simulated_annealing(self):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []

        current_state = list(self.initial)
        current_cost = self.manhattan_distance(tuple(current_state))
        best_state = current_state[:]
        best_cost = current_cost
        temperature = 10000.0
        cooling_rate = 0.001  # Giảm cooling_rate để khám phá lâu hơn
        max_iterations = 100000
        max_memory = 100000
        max_runtime = 30.0  # Tăng thời gian chạy tối đa

        self.visited_states.add(tuple(current_state))
        self.parent_map[tuple(current_state)] = (None, None)  # Đánh dấu trạng thái ban đầu

        iteration = 0
        while temperature > 0.1 and time.time() - start_time < max_runtime and iteration < max_iterations:
            if len(self.visited_states) > max_memory or len(self.parent_map) > max_memory:
                print("Memory limit exceeded")
                break

            neighbor = self.get_random_neighbor(current_state)
            if not neighbor:
                print("No neighbors available")
                break
        
            neighbor_state, action_idx = neighbor
            neighbor_cost = self.manhattan_distance(neighbor_state)

            # Lưu mọi trạng thái lân cận vào parent_map
            self.visited_states.add(tuple(neighbor_state))
            self.parent_map[tuple(neighbor_state)] = (tuple(current_state), action_idx)

            cost_diff = neighbor_cost - current_cost
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_state = list(neighbor_state)
                current_cost = neighbor_cost

            # Cập nhật best_state nếu tốt hơn hoặc ngẫu nhiên
            if current_cost < best_cost or random.random() < 0.05:
                best_state = current_state[:]
                best_cost = current_cost

            temperature *= (1 - cooling_rate)
            self.stats["states_visited"] += 1
            iteration += 1

            if best_cost == 0:
                print("Goal reached!")
                break

            if iteration % 1000 == 0:
                print(f"Iteration {iteration}, temperature: {temperature}, best_cost: {best_cost}, best_state: {tuple(best_state)}")

            # Tái khởi động nếu kẹt quá lâu
            if iteration % 10000 == 0 and best_cost > 0:
                print("Restarting to escape local minimum")
                current_state = list(self.initial)
                current_cost = self.manhattan_distance(tuple(current_state))
                temperature = 5000.0  # Đặt lại nhiệt độ

        print("parent_map size:", len(self.parent_map))
        path = self.reconstruct_path(tuple(best_state), self.parent_map)
        print(f"Final path length: {len(path)}, best_cost: {best_cost}")
        return path, time.time() - start_time, len(path) - 1, self.stats, [tuple(best_state)]

    def get_random_neighbor(self, state):
        neighbors = self.get_neighbors(tuple(state))
        if not neighbors:
            return None, None
        idx = random.randint(0, len(neighbors) - 1)
        return neighbors[idx], idx
    
    def reconstruct_path(self, state, parent_map=None):
        parent_map = parent_map if parent_map is not None else self.parent_map
        state = tuple(state) if isinstance(state, list) else state
        path = []
        seen = set()
    
        try:
            while state is not None and state not in seen:
                path.append(state)
                seen.add(state)
                parent_info = parent_map.get(state)
                if parent_info is None:
                    print(f"No parent for state {state}")
                    break
                print(f"State: {state}, parent_info: {parent_info}, type: {type(parent_info)}")
                if isinstance(parent_info, tuple) and len(parent_info) > 0:
                    parent = parent_info[0]
                elif isinstance(parent_info, list):
                    parent = tuple(parent_info)
                else:
                    print(f"Error: Invalid parent_info type {type(parent_info)} for state {state}: {parent_info}")
                    break
                state = tuple(parent) if isinstance(parent, list) else parent
        except Exception as e:
            print(f"Exception in reconstruct_path: {str(e)}")
    
        return path[::-1]

    def greedy(self):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        open_set = [(self.manhattan_distance(self.initial), 0, self.initial, [])]
        visited = set()
        self.visited_states.add(self.initial)
        counter = 1
        while open_set:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(open_set))
            _, _, state, path = heapq.heappop(open_set)
            if state == self.goal:
                return path, time.time() - start_time, len(path), self.stats, []
            if state not in visited:
                visited.add(state)
                self.stats["states_visited"] += 1
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        self.visited_states.add(neighbor)
                        new_path = path + [neighbor]
                        heapq.heappush(open_set, (self.manhattan_distance(neighbor), counter, neighbor, new_path))
                        counter += 1
        return None, time.time() - start_time, 0, self.stats, []

    def hill_climbing_simple(self):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        current_state = self.initial
        path = [current_state]
        visited = {current_state}
        self.stats["states_visited"] = 1
        self.visited_states.add(current_state)
        while True:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            neighbors = self.get_neighbors(current_state)
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(neighbors))
            best_neighbor = None
            best_h = self.manhattan_distance(current_state)
            for idx, neighbor in enumerate(neighbors):
                if neighbor not in visited:
                    self.visited_states.add(neighbor)
                    h = self.manhattan_distance(neighbor)
                    self.stats["states_visited"] += 1
                    if h < best_h:
                        best_neighbor = neighbor
                        best_h = h
                        self.parent_map[neighbor] = (current_state, idx)
                        break
            if best_neighbor is None:
                if current_state == self.goal:
                    return path, time.time() - start_time, len(path) - 1, self.stats, []
                return None, time.time() - start_time, 0, self.stats, []
            current_state = best_neighbor
            path.append(current_state)
            visited.add(current_state)
            self.visited_states.add(current_state)

    def hill_climbing_steepest(self):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []  # Đã sửa lỗi
        current_state = self.initial
        path = [current_state]
        visited = {current_state}
        self.stats["states_visited"] = 1
        self.visited_states.add(current_state)
        while True:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            neighbors = self.get_neighbors(current_state)
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(neighbors))
            best_neighbor = None
            best_h = self.manhattan_distance(current_state)
            best_action_idx = None
            for idx, neighbor in enumerate(neighbors):
                if neighbor not in visited:
                    self.visited_states.add(neighbor)
                    h = self.manhattan_distance(neighbor)
                    self.stats["states_visited"] += 1
                    if h < best_h:
                        best_neighbor = neighbor
                        best_h = h
                        best_action_idx = idx
            if best_neighbor is None:
                if current_state == self.goal:
                    return path, time.time() - start_time, len(path) - 1, self.stats, []
                return None, time.time() - start_time, 0, self.stats, []
            self.parent_map[best_neighbor] = (current_state, best_action_idx)
            current_state = best_neighbor
            path.append(current_state)
            visited.add(current_state)
            self.visited_states.add(current_state)

    def stochastic_hill_climbing(self):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            print("Initial state is not solvable")
            return None, time.time() - start_time, 0, self.stats, []

        current_state = tuple(self.initial)
        current_cost = self.manhattan_distance(current_state)
        max_iterations = 10000  # Tăng số lần lặp
        max_runtime = 30.0  # Tăng thời gian chạy

        self.visited_states.add(current_state)
        self.parent_map[current_state] = (None, None)  # Đánh dấu trạng thái ban đầu

        for iteration in range(max_iterations):
            if time.time() - start_time > max_runtime:
                print("Time limit exceeded")
                break
        
            neighbor, action_idx = self.get_random_neighbor(current_state)
            if not neighbor:
                print("No neighbors available")
                break
        
            neighbor_state = neighbor
            neighbor_cost = self.manhattan_distance(neighbor_state)

            # Lưu mọi trạng thái lân cận
            self.visited_states.add(neighbor_state)
            self.parent_map[neighbor_state] = (current_state, action_idx)

            if neighbor_cost <= current_cost or random.random() < 0.5:  # Tăng xác suất chấp nhận
                current_state = neighbor_state
                current_cost = neighbor_cost

            self.stats["states_visited"] += 1
            if current_cost == 0:
                print("Goal reached!")
                break

            if iteration % 1000 == 0:
                print(f"Iteration {iteration}, current_cost: {current_cost}, current_state: {current_state}")

        print("parent_map size:", len(self.parent_map))
        path = self.reconstruct_path(current_state, self.parent_map)
        print(f"Final path length: {len(path)}, final_cost: {current_cost}")
        return path, time.time() - start_time, len(path) - 1, self.stats, [current_state]     
    
    def hill_climbing_random(self, max_restarts=10):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        def random_solvable_state():
            state = list(self.goal)
            while True:
                random.shuffle(state)
                state_tuple = tuple(state)
                if self.is_solvable(state_tuple) and state_tuple != self.goal:
                    return state_tuple
        for restart in range(max_restarts):
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            current_state = self.initial if restart == 0 else random_solvable_state()
            path = [current_state]
            visited = {current_state}
            self.stats["states_visited"] += 1
            self.visited_states.add(current_state)
            while True:
                if time.time() - start_time > self.max_runtime:
                    return None, time.time() - start_time, 0, self.stats, []
                neighbors = self.get_neighbors(current_state)
                self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(neighbors))
                better_neighbors = []
                better_indices = []
                current_h = self.manhattan_distance(current_state)
                for idx, neighbor in enumerate(neighbors):
                    if neighbor not in visited:
                        self.visited_states.add(neighbor)
                        h = self.manhattan_distance(neighbor)
                        self.stats["states_visited"] += 1
                        if h < current_h:
                            better_neighbors.append(neighbor)
                            better_indices.append(idx)
                if not better_neighbors:
                    if current_state == self.goal:
                        return path[1:], time.time() - start_time, len(path) - 1, self.stats, []
                    break
                choice_idx = random.randint(0, len(better_neighbors) - 1)
                current_state = better_neighbors[choice_idx]
                self.parent_map[current_state] = (path[-1], better_indices[choice_idx])
                path.append(current_state)
                visited.add(current_state)
                self.visited_states.add(current_state)
        return None, time.time() - start_time, 0, self.stats, []

    def alchemy(self, max_stagnation=10):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        open_set = [(self.manhattan_distance(self.initial), 0, 0, self.initial, [])]
        closed_set = set()
        g_scores = {self.initial: 0}
        self.visited_states.add(self.initial)
        counter = 1
        stagnation_count = 0
        best_h = self.manhattan_distance(self.initial)
        while open_set:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(open_set))
            f, g, _, state, path = heapq.heappop(open_set)
            if state == self.goal:
                return path, time.time() - start_time, len(path), self.stats, []
            if state in closed_set:
                continue
            closed_set.add(state)
            self.stats["states_visited"] += 1
            self.visited_states.add(state)
            neighbors = self.get_neighbors(state)
            found_better = False
            for idx, neighbor in enumerate(neighbors):
                if neighbor in closed_set:
                    continue
                self.visited_states.add(neighbor)
                tentative_g = g + 1
                h = self.manhattan_distance(neighbor)
                if h < best_h:
                    best_h = h
                    found_better = True
                    stagnation_count = 0
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + h
                    new_path = path + [neighbor]
                    self.parent_map[neighbor] = (state, idx)
                    heapq.heappush(open_set, (f_score, tentative_g, counter, neighbor, new_path))
                    counter += 1
            if not found_better:
                stagnation_count += 1
            if stagnation_count >= max_stagnation:
                if neighbors:
                    idx = random.randint(0, len(neighbors) - 1)
                    random_neighbor = neighbors[idx]
                    if random_neighbor not in closed_set:
                        tentative_g = g + 1
                        h = self.manhattan_distance(random_neighbor)
                        g_scores[random_neighbor] = tentative_g
                        f_score = tentative_g + h
                        new_path = path + [random_neighbor]
                        self.parent_map[random_neighbor] = (state, idx)
                        heapq.heappush(open_set, (f_score, tentative_g, counter, random_neighbor, new_path))
                        counter += 1
                        stagnation_count = 0
        return None, time.time() - start_time, 0, self.stats, []

    def beam_search(self, beam_width=5):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        current_beam = [(self.manhattan_distance(self.initial), 0, self.initial, [])]
        visited = {self.initial}
        self.visited_states.add(self.initial)
        counter = 1
        while current_beam:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(current_beam))
            next_beam = []
            for h, _, state, path in current_beam:
                self.stats["states_visited"] += 1
                self.visited_states.add(state)
                if state == self.goal:
                    return path, time.time() - start_time, len(path), self.stats, []
                for idx, neighbor in enumerate(self.get_neighbors(state)):
                    if neighbor not in visited:
                        self.visited_states.add(neighbor)
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        h = self.manhattan_distance(neighbor)
                        self.parent_map[neighbor] = (state, idx)
                        next_beam.append((h, counter, neighbor, new_path))
                        counter += 1
            next_beam.sort(key=lambda x: x[0])
            current_beam = next_beam[:beam_width]
            if not current_beam:
                break
        return None, time.time() - start_time, 0, self.stats, []

    def genetic_algorithm(self, population_size=200, generations=1000, mutation_rate=0.1):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        def generate_individual(length):
            individual = []
            current_state = list(self.initial)
            for _ in range(length):
                neighbors = self.get_neighbors(tuple(current_state))
                if not neighbors:
                    break
                idx = random.randint(0, len(neighbors) - 1)
                next_state = neighbors[idx]
                self.parent_map[next_state] = (tuple(current_state), idx)
                individual.append(next_state)
                current_state = list(next_state)
                self.stats["states_visited"] += 1
                self.visited_states.add(next_state)
            return individual
        def fitness(individual):
            final_state = individual[-1] if individual else self.initial
            return -self.manhattan_distance(final_state)
        def selection(population, fitnesses, num_parents):
            sorted_pairs = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
            return [pair[1] for pair in sorted_pairs[:num_parents]]
        def crossover(parent1, parent2):
            if not parent1 or not parent2:
                return parent1[:]
            crossover_point = random.randint(0, min(len(parent1), len(parent2)))
            child = parent1[:crossover_point] + parent2[crossover_point:]
            return child
        def mutation(individual):
            if not individual or random.random() > mutation_rate:
                return individual[:]
            mutation_point = random.randint(0, len(individual) - 1)
            current_state = list(self.initial if mutation_point == 0 else individual[mutation_point - 1])
            neighbors = self.get_neighbors(tuple(current_state))
            if not neighbors:
                return individual[:]
            idx = random.randint(0, len(neighbors) - 1)
            individual[mutation_point] = neighbors[idx]
            self.parent_map[individual[mutation_point]] = (tuple(current_state), idx)
            self.visited_states.add(individual[mutation_point])
            for i in range(mutation_point + 1, len(individual)):
                neighbors = self.get_neighbors(individual[i - 1])
                if not neighbors:
                    individual = individual[:i]
                    break
                idx = random.randint(0, len(neighbors) - 1)
                individual[i] = neighbors[idx]
                self.parent_map[individual[i]] = (individual[i - 1], idx)
                self.stats["states_visited"] += 1
                self.visited_states.add(individual[i])
            return individual
        max_path_length = 30
        population = [generate_individual(random.randint(5, max_path_length)) for _ in range(population_size)]
        self.stats["max_fringe_size"] = population_size
        for _ in range(generations):
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            fitnesses = [fitness(ind) for ind in population]
            if max(fitnesses) == 0:
                best_individual = population[fitnesses.index(0)]
                return best_individual, time.time() - start_time, len(best_individual), self.stats, []
            parents = selection(population, fitnesses, population_size // 2)
            next_population = parents[:]
            while len(next_population) < population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2)
                child = mutation(child)
                next_population.append(child)
                self.stats["nodes_expanded"] += 1
            population = next_population[:population_size]
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(population))
        fitnesses = [fitness(ind) for ind in population]
        best_individual = population[fitnesses.index(max(fitnesses))]
        if best_individual and best_individual[-1] == self.goal:
            return best_individual, time.time() - start_time, len(best_individual), self.stats, []
        return None, time.time() - start_time, 0, self.stats, []

    def belief_state_search(self, belief_states=None):
        """
        A* search with belief states, using linear_conflict heuristic to find a common
        action sequence for all belief states to reach the goal.
        """
        self.reset_stats()
        start_time = time.time()

        # Sử dụng belief_states cố định nếu không được cung cấp
        if belief_states is None or len(belief_states) != 3:
            belief_states = [
                (1, 2, 3, 4, 5, 6, 7, 0, 8),
                (1, 2, 3, 4, 5, 0, 7, 8, 6),
                (1, 2, 3, 4, 0, 5, 7, 8, 6)
            ]
            print(f"Using default fixed belief states: {belief_states}")
        else:
            belief_states = list(belief_states)

        # Kiểm tra tính hợp lệ
        if len(belief_states) != 3:
            print("Invalid belief states: Must have exactly 3 belief states")
            return None, time.time() - start_time, 0, self.stats, belief_states

        for state in belief_states:
            if not self.is_solvable(state):
                print(f"Belief state {state} is not solvable")
                return None, time.time() - start_time, 0, self.stats, belief_states

        print(f"Initial belief states: {belief_states}")

        # Khởi tạo hàng đợi ưu tiên
        counter = 0
        initial_heuristic = sum(self.linear_conflict(state) for state in belief_states)
        open_set = [(initial_heuristic, 0, counter, belief_states, [])]
        g_scores = {frozenset(belief_states): 0}
        self.visited_states.update(belief_states)
        self.parent_map[belief_states[0]] = (None, None)
        best_heuristic = initial_heuristic
        iterations = 0

        while open_set and time.time() - start_time < self.max_runtime:
            f, g, _, current_belief_states, actions = heapq.heappop(open_set)
            belief_states_tuple = frozenset(current_belief_states)
            iterations += 1

            self.stats["states_visited"] += len(current_belief_states)
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(open_set) + 1)

            # Debug mỗi 1000 lần lặp
            if iterations % 1000 == 0:
                current_heuristic = sum(self.linear_conflict(state) for state in current_belief_states)
                print(f"Iteration {iterations}: states_visited={self.stats['states_visited']}, "
                      f"open_set_size={len(open_set)}, best_heuristic={current_heuristic}")

            # Kiểm tra mục tiêu
            if all(state == self.goal for state in current_belief_states):
                print(f"Goal reached for all belief states after {iterations} iterations!")
                paths = []
                for belief_state in belief_states:
                    path = [belief_state]
                    current = list(belief_state)
                    for action in actions:
                        next_state, valid = self.get_action(current, action)
                        if valid:
                            next_state_tuple = tuple(next_state)
                            path.append(next_state_tuple)
                            current = next_state
                            if next_state_tuple not in self.parent_map:
                                self.parent_map[next_state_tuple] = (tuple(current), action)
                    paths.append(path)
                print(f"Solution path length: {len(actions)}")
                return paths, time.time() - start_time, len(actions), self.stats, belief_states

            # Tạo hành động tiếp theo
            action_scores = []
            for action_idx in range(len(self.actions)):
                new_belief_states = []
                valid_action = False
                for state in current_belief_states:
                    next_state, valid = self.get_action(list(state), action_idx)
                    if valid:
                        new_belief_states.append(tuple(next_state))
                        valid_action = True
                    else:
                        new_belief_states.append(state)

                if not valid_action:
                    continue

                total_heuristic = sum(self.linear_conflict(state) for state in new_belief_states)
                action_scores.append((action_idx, new_belief_states, total_heuristic))

            # Sắp xếp theo heuristic
            action_scores.sort(key=lambda x: x[2])

            for action_idx, new_belief_states, total_heuristic in action_scores:
                new_belief_states_tuple = frozenset(new_belief_states)
                new_g = g + 1

                # Cho phép xem xét lại nếu g_score tốt hơn
                if new_belief_states_tuple not in g_scores or new_g < g_scores[new_belief_states_tuple]:
                    g_scores[new_belief_states_tuple] = new_g
                    f_score = new_g + total_heuristic
                    new_actions = actions + [action_idx]

                    # Cập nhật parent_map
                    for new_state, old_state in zip(new_belief_states, current_belief_states):
                        if new_state != old_state and new_state not in self.parent_map:
                            self.parent_map[new_state] = (old_state, action_idx)

                    counter += 1
                    heapq.heappush(open_set, (f_score, new_g, counter, new_belief_states, new_actions))
                    self.visited_states.update(new_belief_states)
                    if total_heuristic < best_heuristic:
                        best_heuristic = total_heuristic
                        print(f"Improved heuristic: {best_heuristic} with action {action_idx}")

        print(f"Failed to find a solution after {iterations} iterations. "
              f"Best heuristic: {best_heuristic}, states_visited: {self.stats['states_visited']}")
        return None, time.time() - start_time, 0, self.stats, belief_states
    def generate_belief_states(self):
        """
        Generate 3 solvable belief states if self.belief_states is not set or invalid.
        States are derived from neighbors of the initial state to ensure relevance.
        """
        if hasattr(self, 'belief_states') and self.belief_states and len(self.belief_states) == 3:
            valid = True
            for state in self.belief_states:
                if not self.is_solvable(state) or state == self.goal:
                    valid = False
                    break
            if valid:
                print(f"Using existing belief states: {self.belief_states}")
                return list(self.belief_states)
        
        print("Generating new belief states...")
        belief_states = []
        candidates = set()
        
        # Lấy các trạng thái lân cận từ initial state
        current_states = [self.initial]
        visited = {self.initial}
        max_depth = 2  # Giới hạn độ sâu để tránh tạo quá nhiều trạng thái
        
        for _ in range(max_depth):
            next_states = []
            for state in current_states:
                neighbors = self.get_neighbors(state)
                for neighbor in neighbors:
                    if neighbor not in visited and self.is_solvable(neighbor) and neighbor != self.goal:
                        candidates.add(neighbor)
                        next_states.append(neighbor)
                        visited.add(neighbor)
            current_states = next_states
        
        # Chọn ngẫu nhiên 3 trạng thái từ candidates
        candidates = list(candidates)
        if len(candidates) < 3:
            print("Not enough solvable neighbors; generating random solvable states")
            while len(candidates) < 3:
                state = self.generate_solvable_initial_state()
                if state not in candidates and state != self.initial and state != self.goal:
                    candidates.append(state)
        
        belief_states = random.sample(candidates, min(3, len(candidates)))
        
        # Lưu vào self.belief_states
        self.belief_states = belief_states
        print(f"Generated belief states: {belief_states}")
        return belief_states
    
    def partial_observation_search(self):
        """
        A* search with partial observation, using linear_conflict heuristic to find a common
        action sequence for all fixed belief states to reach the goal.
        """
        self.reset_stats()
        start_time = time.time()

        # Gán trạng thái niềm tin cố định
        fixed_belief_states = [
            (1, 2, 3, 4, 5, 6, 7, 8, 0),
            (1, 2, 3, 4, 5, 6, 7, 0, 8),
            (1, 2, 3, 4, 5, 6, 0, 7, 8)
        ]
        belief_states = fixed_belief_states

        # Kiểm tra tính hợp lệ
        if len(belief_states) != 3:
            print("Invalid belief states: Must have exactly 3 belief states")
            return None, time.time() - start_time, 0, self.stats, belief_states

        for state in belief_states:
            if not self.is_solvable(state):
                print(f"Belief state {state} is not solvable")
                return None, time.time() - start_time, 0, self.stats, belief_states
            if state == self.goal:
                print(f"Belief state {state} is already the goal state")
                return None, time.time() - start_time, 0, self.stats, belief_states

        print(f"Using fixed belief states: {belief_states}")

        # Khởi tạo hàng đợi ưu tiên: (f_score, g_score, counter, belief_states, actions)
        counter = 0
        initial_heuristic = sum(self.linear_conflict(state) for state in belief_states)
        open_set = [(initial_heuristic, 0, counter, tuple(belief_states), [])]
        closed_set = set()
        g_scores = {frozenset(belief_states): 0}
        self.visited_states.update(belief_states)
        for state in belief_states:
            self.parent_map[state] = (None, None)

        iterations = 0
        best_heuristic = initial_heuristic
        max_open_set_size = 100000

        while open_set and time.time() - start_time < self.max_runtime:
            if len(open_set) > max_open_set_size:
                print("Open set size limit exceeded")
                break

            f, g, _, current_belief_states, actions = heapq.heappop(open_set)
            current_belief_states = list(current_belief_states)
            belief_states_tuple = frozenset(current_belief_states)
            iterations += 1

            if belief_states_tuple in closed_set:
                continue

            closed_set.add(belief_states_tuple)
            self.stats["states_visited"] += len(current_belief_states)
            self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(open_set) + 1)

            if iterations % 1000 == 0:
                current_heuristic = sum(self.linear_conflict(state) for state in current_belief_states)
                print(f"Iteration {iterations}: states_visited={self.stats['states_visited']}, "
                      f"open_set_size={len(open_set)}, current_heuristic={current_heuristic}")

            if all(state == self.goal for state in current_belief_states):
                print(f"Goal reached after {iterations} iterations!")
                paths = []
                for belief_state in belief_states:
                    path = [belief_state]
                    current = list(belief_state)
                    for action in actions:
                        next_state, valid = self.get_action(tuple(current), action)
                        if valid:
                            next_state_tuple = tuple(next_state)
                            path.append(next_state_tuple)
                            current = next_state
                            if next_state_tuple not in self.parent_map:
                                self.parent_map[next_state_tuple] = (tuple(current), action)
                    paths.append(path)
                print(f"Solution path length: {len(actions)}")
                return paths, time.time() - start_time, len(actions), self.stats, belief_states

            action_scores = []
            for action_idx in range(len(self.actions)):
                new_belief_states = []
                valid_action = True
                for state in current_belief_states:
                    next_state, valid = self.get_action(state, action_idx)
                    if valid:
                        new_belief_states.append(tuple(next_state))
                    else:
                        valid_action = False
                        break
                if valid_action:
                    total_heuristic = sum(self.linear_conflict(state) for state in new_belief_states)
                    action_scores.append((action_idx, new_belief_states, total_heuristic))
                else:
                    print(f"Action {self.action_names[action_idx]} is invalid for all states")

            if not action_scores:
                print("No valid actions found for current belief states")

            action_scores.sort(key=lambda x: x[2])
            for action_idx, new_belief_states, total_heuristic in action_scores:
                new_belief_states_tuple = frozenset(new_belief_states)
                new_g = g + 1
                if new_belief_states_tuple not in g_scores or new_g < g_scores[new_belief_states_tuple]:
                    g_scores[new_belief_states_tuple] = new_g
                    f_score = new_g + total_heuristic
                    new_actions = actions + [action_idx]
                    for old_state, new_state in zip(current_belief_states, new_belief_states):
                        if new_state != old_state and new_state not in self.parent_map:
                            self.parent_map[new_state] = (old_state, action_idx)
                    counter += 1
                    heapq.heappush(open_set, (f_score, new_g, counter, tuple(new_belief_states), new_actions))
                    self.visited_states.update(new_belief_states)
                    if total_heuristic < best_heuristic:
                        best_heuristic = total_heuristic
                        print(f"Improved heuristic: {best_heuristic} with action {self.action_names[action_idx]}")

        print(f"Failed to find a solution after {iterations} iterations. "
              f"Best heuristic: {best_heuristic}, states_visited: {self.stats['states_visited']}, "
              f"open_set_size: {len(open_set)}")
        return None, time.time() - start_time, 0, self.stats, belief_states
    
    def and_or_tree_search(self, initial_state=None, depth_limit=20):
        """
        AND-OR Tree Search for the 8-puzzle, treating states as OR nodes (choose one action)
        and actions as AND nodes (explore outcomes of each action). Uses a recursive approach
        to reflect the true spirit of AND-OR Tree Search without heuristic guidance.
        Defaults to self.initial if initial_state is None.
        """
        self.reset_stats()
        start_time = time.time()
        state_count = [1]  # Sử dụng list để thay đổi giá trị trong hàm đệ quy
        
        # Sử dụng self.initial nếu initial_state không được truyền
        initial_state = initial_state if initial_state is not None else self.initial
        
        # Trường hợp đặc biệt: trạng thái ban đầu đã là mục tiêu
        if initial_state == self.goal:
            self.stats["states_visited"] = 1
            return [initial_state], time.time() - start_time, 0, self.stats, []
        
        # Kiểm tra tính khả thi
        if not self.is_solvable(initial_state):
            return None, time.time() - start_time, 0, self.stats, []
        
        # Tập hợp các trạng thái đã thăm
        visited = set()
        self.visited_states.add(initial_state)
        
        def and_or_search(state, depth, path):
            """
            Recursive AND-OR Tree Search.
            - OR node: State, where we try each action to find one that leads to the goal.
            - AND node: Outcomes of an action, where all outcomes must lead to the goal (in 8-puzzle, one outcome per action).
            """
            # Kiểm tra giới hạn độ sâu
            if depth > depth_limit:
                return None
            
            # Kiểm tra mục tiêu
            if state == self.goal:
                return path + [state]
            
            # Đánh dấu trạng thái đã thăm
            if state in visited:
                return None
            visited.add(state)
            self.stats["states_visited"] += 1
            
            # Nút OR: Thử tất cả các hành động từ trạng thái này
            neighbors = self.get_neighbors(state)
            self.stats["nodes_expanded"] += 1
            
            for idx, neighbor in enumerate(neighbors):
                if neighbor in visited:
                    continue
                
                # Nút AND: Trong 8-puzzle, mỗi hành động chỉ có một kết quả (deterministic)
                self.visited_states.add(neighbor)
                state_count[0] += 1
                
                # Lưu thông tin cha để tái tạo đường đi
                self.parent_map[neighbor] = (state, idx)
                
                # Đệ quy để khám phá trạng thái tiếp theo (nút OR)
                result = and_or_search(neighbor, depth + 1, path + [state])
                
                # Nếu tìm thấy đường đi đến mục tiêu qua hành động này, trả về
                if result is not None:
                    return result
            
            # Không tìm thấy đường đi qua trạng thái này
            return None
        
        # Gọi hàm đệ quy từ trạng thái ban đầu
        result = and_or_search(initial_state, 0, [])
        elapsed = time.time() - start_time
        
        # Cập nhật state_count vào stats
        self.stats["state_count"] = state_count[0]
        
        # Trả về với stats là dictionary
        if result is None:
            return None, elapsed, 0, self.stats, []
        return result, elapsed, len(result) - 1, self.stats, []

    def backtracking(self):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        depth_limit = 1
        while True:
            if time.time() - start_time > self.max_runtime:
                return None, time.time() - start_time, 0, self.stats, []
            visited = set()
            result = self._backtracking_search(self.initial, [], 0, depth_limit, visited)
            if result is not None:
                return result, time.time() - start_time, len(result), self.stats, []
            depth_limit += 1
    

    def Min_conflict_search(self):
        """
        Min-conflict search for the 8-puzzle as a CSP, assigning values (0-8) to each tile to minimize
        conflicts based on the linear conflict heuristic, ensuring unique values and solvability.
        Runs silently as a simulation until the goal state is reached, with no iteration or runtime limits.
        Uses simulated annealing-inspired exploration and robust perturbation to guarantee solution.
        """
        self.reset_stats()
        start_time = time.time()
        
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            print("Initial state is not solvable")
            return None, time.time() - start_time, 0, self.stats, []

        # Initialize current and best states
        current_state = list(self.initial)
        best_state = current_state[:]
        best_conflicts = self.linear_conflict(tuple(current_state))
        self.visited_states.add(tuple(current_state))
        self.parent_map[tuple(current_state)] = (None, None)
        
        # Domain: numbers 0 to 8
        domain = list(range(9))
        
        # Track path
        path = [tuple(current_state)]
        
        # Simulated annealing parameters
        initial_temperature = 20.0
        cooling_rate = 0.99995
        temperature = initial_temperature
        stagnation_count = 0
        max_stagnation = 5000
        iteration = 0
        
        while True:
            # Compute current conflicts
            current_conflicts = self.linear_conflict(tuple(current_state))
            self.stats["states_visited"] += 1
            
            # Check if goal state is reached
            if tuple(current_state) == self.goal:
                return path, time.time() - start_time, len(path) - 1, self.stats, []

            # Update best state
            if current_conflicts < best_conflicts:
                best_state = current_state[:]
                best_conflicts = current_conflicts
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Select a tile to reassign
            tile_idx = random.randint(0, 8)
            current_value = current_state[tile_idx]
            
            # Get available values
            used_values = set(current_state[:tile_idx] + current_state[tile_idx + 1:])
            available_values = [v for v in domain if v not in used_values]
            
            if not available_values:
                continue
            
            # Prioritize values based on goal positions
            goal_positions = {self.goal[i]: i for i in range(9)}
            available_values.sort(key=lambda v: abs(goal_positions[v] - tile_idx))
            
            # Evaluate values
            best_value = current_value
            best_conflicts_new = current_conflicts
            best_state_new = current_state[:]
            
            for value in available_values:
                new_state = current_state[:]
                new_state[tile_idx] = value
                new_state_tuple = tuple(new_state)
                
                if not self.is_solvable(new_state_tuple):
                    continue
                
                new_conflicts = self.linear_conflict(new_state_tuple)
                conflict_diff = new_conflicts - current_conflicts
                if conflict_diff <= 0 or random.random() < math.exp(-conflict_diff / max(temperature, 0.01)):
                    best_value = value
                    best_conflicts_new = new_conflicts
                    best_state_new = new_state
            
            # Update state
            if best_value != current_value:
                new_state_tuple = tuple(best_state_new)
                self.visited_states.add(new_state_tuple)
                self.parent_map[new_state_tuple] = (tuple(current_state), None)
                current_state = best_state_new
                path.append(new_state_tuple)
                self.stats["nodes_expanded"] += 1
            
            # Perturb if stagnant
            if stagnation_count >= max_stagnation:
                current_state = best_state[:]
                for _ in range(random.randint(2, 4)):
                    tile1, tile2 = random.sample(range(9), 2)
                    current_state[tile1], current_state[tile2] = current_state[tile2], current_state[tile1]
                new_state_tuple = tuple(current_state)
                
                attempts = 0
                max_attempts = 50
                while not self.is_solvable(new_state_tuple) and attempts < max_attempts:
                    current_state = best_state[:]
                    for _ in range(random.randint(2, 4)):
                        tile1, tile2 = random.sample(range(9), 2)
                        current_state[tile1], current_state[tile2] = current_state[tile2], current_state[tile1]
                    new_state_tuple = tuple(current_state)
                    attempts += 1
                
                if attempts >= max_attempts:
                    current_state = best_state[:]
                    tile1, tile2 = random.sample(range(9), 2)
                    current_state[tile1], current_state[tile2] = current_state[tile2], current_state[tile1]
                    new_state_tuple = tuple(current_state)
                    if not self.is_solvable(new_state_tuple):
                        current_state = best_state[:]
                        new_state_tuple = tuple(current_state)
                
                self.visited_states.add(new_state_tuple)
                self.parent_map[new_state_tuple] = (tuple(best_state), None)
                path.append(new_state_tuple)
                self.stats["nodes_expanded"] += 1
                stagnation_count = 0
                temperature = initial_temperature * 0.5
            
            # Update temperature
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01
            
            # Full restart if no progress
            if iteration % 50000 == 0 and best_conflicts >= self.linear_conflict(tuple(self.initial)) * 0.9:
                current_state = list(self.initial)
                best_state = current_state[:]
                best_conflicts = self.linear_conflict(tuple(current_state))
                path = [tuple(current_state)]
                self.visited_states = {tuple(current_state)}
                self.parent_map = {tuple(current_state): (None, None)}
                stagnation_count = 0
                temperature = initial_temperature
            
            iteration += 1

    def _backtracking_search(self, state, path, depth, depth_limit, visited):
        self.stats["states_visited"] += 1
        self.visited_states.add(state)
        if state == self.goal:
            return path
        if depth >= depth_limit:
            return None
        visited.add(state)
        neighbors = self.get_neighbors(state)
        neighbors_with_indices = [(neighbor, idx) for idx, neighbor in enumerate(neighbors)]
        neighbors_with_indices.sort(key=lambda x: self.manhattan_distance(x[0]))
        self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(neighbors))
        for neighbor, idx in neighbors_with_indices:
            if neighbor not in visited:
                self.visited_states.add(neighbor)
                self.parent_map[neighbor] = (state, idx)
                new_path = path + [neighbor]
                result = self._backtracking_search(neighbor, new_path, depth + 1, depth_limit, visited)
                if result is not None:
                    return result
        visited.remove(state)
        return None

    def forward_checking(self, depth_limit=50):
        self.reset_stats()
        start_time = time.time()
        if self.initial == self.goal:
            return [], time.time() - start_time, 0, self.stats, []
        if not self.is_solvable(self.initial):
            return None, time.time() - start_time, 0, self.stats, []
        result = self._forward_checking_search(self.initial, [], 0, depth_limit, set())
        if result is not None:
            return result, time.time() - start_time, len(result), self.stats, []
        return None, time.time() - start_time, 0, self.stats, []

    def _forward_checking_search(self, state, path, depth, depth_limit, visited):
        self.stats["states_visited"] += 1
        self.visited_states.add(state)
        if state == self.goal:
            return path
        if depth >= depth_limit:
            return None
        visited.add(state)
        neighbors = self.get_neighbors(state)
        self.stats["max_fringe_size"] = max(self.stats["max_fringe_size"], len(neighbors))
        neighbors = sorted(neighbors, key=lambda x: self.manhattan_distance(x))
        for idx, neighbor in enumerate(neighbors):
            if neighbor not in visited:
                self.visited_states.add(neighbor)
                self.parent_map[neighbor] = (state, idx)
                if self.manhattan_distance(neighbor) <= self.manhattan_distance(state) + 1:
                    new_path = path + [neighbor]
                    result = self._forward_checking_search(neighbor, new_path, depth + 1, depth_limit, visited.copy())
                    if result is not None:
                        return result
        return None

    def solve(self, algo):
        algorithms = {
        "BFS": self.bfs,
        "DFS": self.dfs,
        "UCS": self.ucs,
        "A*": self.a_star,
        "A* (Linear Conflict)": lambda: self.a_star(use_linear_conflict=True),
        "Greedy": self.greedy,
        "Iterative Deepening": self.iterative_deepening,
        "IDA*": self.ida_star,
        "Hill Climbing (Simple)": self.hill_climbing_simple,
        "Hill Climbing (Steepest)": self.hill_climbing_steepest,
        "Hill Climbing (Random)": self.hill_climbing_random,
        "Simulated Annealing": self.simulated_annealing,  # Thêm mới
        "Genetic Algorithm": self.genetic_algorithm,      # Thêm mới (đã có trong mã, nhưng để rõ ràng)
        "Min-Conflict Search": self.Min_conflict_search,  # Thêm mới
        "Stochastic Hill Climbing": self.stochastic_hill_climbing,  # Thêm mới
        "Search with Partial Observation": self.partial_observation_search,  # Thêm mới
        "Alchemy": self.alchemy,
        "Beam Search": self.beam_search,
        "Belief State Search": lambda: self.belief_state_search(getattr(self, 'belief_states', None)),
        "AND-OR Tree Search": self.and_or_tree_search,
        "Backtracking": self.backtracking,
        "Forward Checking": self.forward_checking,
        "Q-learning": self.q_learning,
        "SARSA": self.sarsa
        }
        return algorithms.get(algo, lambda: (None, 0, 0, self.stats, []))()

class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.geometry("1400x900")
        self.goal_state = tuple([1, 2, 3, 4, 5, 6, 7, 8, 0])
        self.fixed_initial_state = tuple([2, 1, 3, 8, 6, 4, 7, 0, 5])
        self.other_initial_state = tuple([1, 2, 3,5, 0, 6,4, 7, 8])
        self.initial_state = self.fixed_initial_state
        self.puzzle = Puzzle8(self.initial_state, self.goal_state)
        self.solving = False
        self.tree_window = None
        self.tile_size = 30
        self.current_job_id = None
        self.animation_jobs = []
        self.evaluation_data = []
        self.current_belief_states = []
        self.setup_ui()

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame cho toàn bộ giao diện chính
        self.content_frame = tk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Frame cho Controls, 8-Puzzle Board, Belief States và Statistics
        self.top_frame = tk.Frame(self.content_frame)
        self.top_frame.pack(fill=tk.X, pady=10)

        # Sử dụng grid để sắp xếp các frame ngang
        self.main_content_frame = tk.Frame(self.content_frame)
        self.main_content_frame.pack(fill=tk.BOTH, expand=True)

        # Frame cho Solution Path
        self.solution_frame = tk.LabelFrame(self.main_content_frame, text="Solution Path", font=("Arial", 12))
        self.solution_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Frame cho Algorithm Evaluation (bên phải)
        self.evaluation_frame = tk.LabelFrame(self.main_content_frame, text="Algorithm Evaluation", font=("Arial", 12, "bold"))
        self.evaluation_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Cân bằng các cột để mở rộng đều
        self.main_content_frame.grid_columnconfigure(0, weight=2)  # Solution Path
        self.main_content_frame.grid_columnconfigure(1, weight=2)  # Algorithm Evaluation

        self.create_top_ui()
        self.create_right_ui()
        self.create_evaluation_ui()

    def generate_solvable_initial_state(self):
        state = list(self.goal_state)
        while True:
            random.shuffle(state)
            state_tuple = tuple(state)
            puzzle = Puzzle8(state_tuple, self.goal_state)
            if puzzle.is_solvable(state_tuple) and state_tuple != self.goal_state:
                return state_tuple

    def create_top_ui(self):
        # Sử dụng grid để sắp xếp Controls, Puzzle Board, Belief States và Statistics
        # Controls ở trên cùng
        control_frame = tk.LabelFrame(self.top_frame, text="Controls", font=("Arial", 12))
        control_frame.pack(fill=tk.X, pady=5)
        tk.Label(control_frame, text="Algorithm:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
        self.combo = ttk.Combobox(control_frame, values=[
            "BFS", "DFS", "UCS","Iterative Deepening", "A*", "A* (Linear Conflict)", "Greedy",  "IDA*",
            "Hill Climbing (Simple)", "Hill Climbing (Steepest)","Stochastic Hill Climbing", "Hill Climbing (Random)","Beam Search",
            "Simulated Annealing", "Genetic Algorithm", "Min-Conflict Search","Backtracking", "Forward Checking",  "Search with Partial Observation",
            "Belief State Search", "AND-OR Tree Search",  "Q-learning", "SARSA"
        ], state="readonly", width=25)
        self.combo.grid(row=0, column=1, padx=5, pady=5)
        self.combo.current(0)
        self.combo.bind("<<ComboboxSelected>>", self.on_algorithm_select)
        self.start_button = tk.Button(control_frame, text="Solve", command=self.run_algorithm, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=10)
        self.start_button.grid(row=0, column=2, padx=10, pady=5)
        self.reset_button = tk.Button(control_frame, text="Reset", command=self.reset_puzzle, bg="#FF5733", fg="white", font=("Arial", 12, "bold"), padx=10)
        self.reset_button.grid(row=0, column=3, padx=10, pady=5)
        self.tree_button = tk.Button(control_frame, text="Draw Tree", command=self.draw_state_space_tree, bg="#4682B4", fg="white", font=("Arial", 12, "bold"), padx=10)
        self.tree_button.grid(row=0, column=4, padx=10, pady=5)

        # Frame con để chứa Puzzle Board, Belief States và Statistics
        self.board_and_stats_frame = tk.Frame(self.top_frame)
        self.board_and_stats_frame.pack(fill=tk.X, pady=5)

        # 8-Puzzle Board (bên trái)
        puzzle_frame = tk.LabelFrame(self.board_and_stats_frame, text="8-Puzzle Board", font=("Arial", 12))
        puzzle_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.tiles_frame = tk.Frame(puzzle_frame)
        self.tiles_frame.pack(pady=20)
        self.tiles = []
        for i in range(3):
            for j in range(3):
                tile = tk.Label(self.tiles_frame, text="", width=5, height=2, relief="raised", font=("Arial", 16, "bold"))
                tile.grid(row=i, column=j, padx=2, pady=2)
                self.tiles.append(tile)
        self.update_puzzle_display(self.initial_state)

        # Frame chứa Belief States và Statistics (bên phải)
        self.right_side_frame = tk.Frame(self.board_and_stats_frame)
        self.right_side_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Belief States
        self.belief_frame = tk.LabelFrame(self.right_side_frame, text="Belief States", font=("Arial", 12))
        self.belief_frame.pack(fill=tk.X, pady=5)

        # Statistics
        self.stats_frame = tk.LabelFrame(self.right_side_frame, text="Statistics", font=("Arial", 12))
        self.stats_frame.pack(fill=tk.X, pady=5)

        # Cân bằng các cột
        self.board_and_stats_frame.grid_columnconfigure(0, weight=1)  # Puzzle Board
        self.board_and_stats_frame.grid_columnconfigure(1, weight=1)  # Belief States và Statistics

    def create_right_ui(self):
        # Belief States (hiển thị ngang trong right_side_frame)
        self.belief_canvases = []
        for i in range(3):
            canvas = tk.Canvas(self.belief_frame, width=100, height=100, bg="#FFFFFF")
            canvas.grid(row=0, column=i, padx=5, pady=5)
            tk.Label(self.belief_frame, text=f"State {i+1}", font=("Arial", 10)).grid(row=1, column=i, padx=5, pady=5)
            self.belief_canvases.append(canvas)

        # Statistics (trong right_side_frame)
        self.stats_labels = {}
        stats_items = [("Running time:", "time"), ("Steps:", "steps"), ("Expanded nodes:", "nodes_expanded"),
                       ("Max fringe size:", "max_fringe_size"), ("Visited states:", "states_visited")]
        for i, (label_text, key) in enumerate(stats_items):
            tk.Label(self.stats_frame, text=label_text, font=("Arial", 14)).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            value_label = tk.Label(self.stats_frame, text="0", font=("Arial", 14), fg="#000000")
            value_label.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            self.stats_labels[key] = value_label

        # Solution Path
        self.result_text = scrolledtext.ScrolledText(self.solution_frame, width=50, height=15, font=("Arial", 14), fg="#000000")
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_evaluation_ui(self):
        # Tạo frame con với 2 phần
        eval_subframe = tk.Frame(self.evaluation_frame)
        eval_subframe.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Phần 1: Bảng đánh giá thuật toán
        eval_table_frame = tk.LabelFrame(eval_subframe, text="Performance Metrics", font=("Arial", 11))
        eval_table_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Tạo bảng để hiển thị các thông số đánh giá thuật toán
        self.eval_table = ttk.Treeview(eval_table_frame, columns=("algorithm", "time", "states_visited", "nodes_expanded", "path_length", "space_complexity"), show="headings", height=5)
        
        # Định nghĩa các cột
        self.eval_table.heading("algorithm", text="Algorithm")
        self.eval_table.heading("time", text="Time (s)")
        self.eval_table.heading("states_visited", text="States Visited")
        self.eval_table.heading("nodes_expanded", text="Nodes Expanded")
        self.eval_table.heading("path_length", text="Path Length")
        self.eval_table.heading("space_complexity", text="Space Complexity")
        
        # Đặt chiều rộng cột
        self.eval_table.column("algorithm", width=150)
        self.eval_table.column("time", width=80)
        self.eval_table.column("states_visited", width=100)
        self.eval_table.column("nodes_expanded", width=120)
        self.eval_table.column("path_length", width=80)
        self.eval_table.column("space_complexity", width=120)
        
        # Thêm thanh cuộn
        table_scrollbar = ttk.Scrollbar(eval_table_frame, orient="vertical", command=self.eval_table.yview)
        self.eval_table.configure(yscrollcommand=table_scrollbar.set)
        
        # Định vị bảng và thanh cuộn
        self.eval_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        table_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Phần 2: Các nút điều khiển và biểu đồ
        eval_controls_frame = tk.LabelFrame(eval_subframe, text="Comparison Tools", font=("Arial", 11))
        eval_controls_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Thiết lập nút để so sánh thuật toán
        self.compare_button = tk.Button(eval_controls_frame, text="Compare Algorithms", 
                                       command=self.compare_algorithms, 
                                       bg="#3498DB", fg="white", 
                                       font=("Arial", 11, "bold"), padx=10)
        self.compare_button.pack(pady=5)
        
        self.clear_eval_button = tk.Button(eval_controls_frame, text="Clear Evaluation Data", 
                                          command=self.clear_evaluation_data, 
                                          bg="#E74C3C", fg="white", 
                                          font=("Arial", 11, "bold"), padx=10)
        self.clear_eval_button.pack(pady=5)
        
        self.save_eval_button = tk.Button(eval_controls_frame, text="Save Evaluation Data", 
                                         command=self.save_evaluation_data, 
                                         bg="#27AE60", fg="white", 
                                         font=("Arial", 11, "bold"), padx=10)
        self.save_eval_button.pack(pady=5)
        
        self.chart_button = tk.Button(eval_controls_frame, text="Show Performance Chart", 
                                     command=self.show_performance_chart, 
                                     bg="#9B59B6", fg="white", 
                                     font=("Arial", 11, "bold"), padx=10)
        self.chart_button.pack(pady=5)
        
        # Cân bằng các cột
        eval_subframe.grid_columnconfigure(0, weight=3)
        eval_subframe.grid_columnconfigure(1, weight=1)

    def update_puzzle_display(self, state):
        for i, value in enumerate(state):
            text = "" if value == 0 else str(value)
            bg_color = "#F5F5F5" if value == 0 else "#4682B4"
            fg_color = "#333333" if value == 0 else "#FFFFFF"
            self.tiles[i].config(text=text, bg=bg_color, fg=fg_color, relief="raised", borderwidth=2)

    def display_belief_states(self, belief_states):
        for canvas in self.belief_canvases:
            canvas.delete("all")
        for idx, state in enumerate(belief_states):
            if idx >= len(self.belief_canvases):
                break
            canvas = self.belief_canvases[idx]
            for i in range(3):
                for j in range(3):
                    value = state[i * 3 + j]
                    x, y = j * 30 + 15, i * 30 + 15
                    bg_color = "#F5F5F5" if value == 0 else "#4682B4"
                    fg_color = "#333333" if value == 0 else "#FFFFFF"
                    canvas.create_rectangle(x, y, x + 30, y + 30, fill=bg_color, outline="#000000")
                    if value != 0:
                        canvas.create_text(x + 15, y + 15, text=str(value), font=("Arial", 12), fill=fg_color)

    def display_path(self, path):
        self.result_text.delete(1.0, tk.END)
        if not path:
            self.result_text.insert(tk.END, "No solution found or this is already the goal state!\n")
            return
        self.result_text.insert(tk.END, "Solution path:\n\n")
        for i, state in enumerate(path):
            self.result_text.insert(tk.END, f"Step {i + 1}:\n")
            for row in range(3):
                row_str = " ".join(str(state[row * 3 + col]) if state[row * 3 + col] != 0 else " " for col in range(3))
                self.result_text.insert(tk.END, f"{row_str}\n")
            self.result_text.insert(tk.END, "\n")

    def update_stats(self, duration, steps, stats):
        self.stats_labels["time"].config(text=f"{duration:.4f} seconds")
        self.stats_labels["steps"].config(text=str(steps))
        self.stats_labels["nodes_expanded"].config(text=str(stats["nodes_expanded"]))
        self.stats_labels["max_fringe_size"].config(text=str(stats["max_fringe_size"]))
        self.stats_labels["states_visited"].config(text=str(stats["states_visited"]))

    def on_algorithm_select(self, event):
        self.cancel_current_job()
        selected_algo = self.combo.get()
        if selected_algo in ["DFS", "Iterative Deepening", "IDA*"]:
            self.initial_state = self.fixed_initial_state
        else:
            difficult_algorithms = [
            "Hill Climbing (Simple)", "Hill Climbing (Steepest)", "Hill Climbing (Random)","Stochastic Hill Climbing",
            "Q-learning", "SARSA", "Genetic Algorithm", "Belief State Search", "Search with Partial Observation",
            "AND-OR Tree Search","Search with Partial Observation","Beam Search","Simulated Annealing","Genetic Algorithm",
            ]
            if selected_algo in difficult_algorithms:
                self.initial_state = self.other_initial_state
            else:
                self.initial_state = self.fixed_initial_state
        self.puzzle = Puzzle8(self.initial_state, self.goal_state)
        self.update_puzzle_display(self.initial_state)

        # Khi chọn Belief State Search hoặc Search with Partial Observation, tạo và hiển thị 3 trạng thái niềm tin
        if selected_algo in ["Belief State Search", "Search with Partial Observation"]:
            def generate_belief_state_from_goal(steps=3):
                state = list(self.goal_state)
                for _ in range(steps):
                    neighbors = self.puzzle.get_neighbors(tuple(state))
                    if not neighbors:
                        break
                    idx = random.randint(0, len(neighbors) - 1)
                    state = list(neighbors[idx])
                return tuple(state)

            belief_states = set()
            belief_states.add(self.initial_state)
            while len(belief_states) < 3:
                new_state = generate_belief_state_from_goal(random.randint(2, 4))
                if new_state != self.initial_state and self.puzzle.is_solvable(new_state):
                    belief_states.add(new_state)
            self.current_belief_states = list(belief_states)
            self.display_belief_states(self.current_belief_states)
        else:
            self.current_belief_states = []
            self.display_belief_states([])

        self.result_text.delete(1.0, tk.END)
        self.update_stats(0, 0, {"nodes_expanded": 0, "max_fringe_size": 0, "states_visited": 0})

    def cancel_current_job(self):
        if self.solving and self.current_job_id is not None:
            self.root.after_cancel(self.current_job_id)
            self.current_job_id = None
            self.solving = False
            for job in self.animation_jobs:
                self.root.after_cancel(job)
            self.animation_jobs = []
            self.start_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.tree_button.config(state=tk.NORMAL)
            self.result_text.insert(tk.END, "Algorithm execution cancelled.\n")

    def reset_puzzle(self):
        self.cancel_current_job()
        selected_algo = self.combo.get()
        if selected_algo in [
                "Hill Climbing (Simple)", "Hill Climbing (Steepest)", "Hill Climbing (Random)","Stochastic Hill Climbing",
            "Q-learning", "SARSA", "Genetic Algorithm", "Belief State Search", "Search with Partial Observation",
            "AND-OR Tree Search","Search with Partial Observation","Beam Search","Simulated Annealing","Genetic Algorithm"
            ]:
            self.initial_state = self.other_initial_state
        else:
            self.initial_state = self.fixed_initial_state
        self.puzzle = Puzzle8(self.initial_state, self.goal_state)
        self.update_puzzle_display(self.initial_state)
        self.display_belief_states([])
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Puzzle reset.\n")
        self.update_stats(0, 0, {"nodes_expanded": 0, "max_fringe_size": 0, "states_visited": 0})

    def run_algorithm(self):
        self.cancel_current_job()
        self.solving = True
        self.start_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.tree_button.config(state=tk.DISABLED)
        selected_algo = self.combo.get()
        self.puzzle = Puzzle8(self.initial_state, self.goal_state)
        if not self.puzzle.is_solvable(self.initial_state):
            messagebox.showerror("Error", "The initial state cannot be solved!")
            self.solving = False
            self.start_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.tree_button.config(state=tk.NORMAL)
            return
        # Set custom delay for DFS
        animation_delay = 150 if selected_algo == ["DFS","Min-conflict"] else 350  # Faster for DFS (200ms)
        self.current_job_id = self.root.after(100, self._solve_puzzle, selected_algo, animation_delay)
        
    def _solve_puzzle(self, selected_algo, animation_delay):
        try:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Solving with {selected_algo}...\n")
            self.root.update()

            # Gán belief_states cho Puzzle8 trước khi gọi solve
            if selected_algo in ["Belief State Search", "Search with Partial Observation"]:
                self.puzzle.belief_states = self.other_initial_state

            result = self.puzzle.solve(selected_algo)
            paths, duration, steps, stats, belief_states = result
        
            if selected_algo == "Belief State Search" and belief_states:
                self.display_belief_states(belief_states)
                self.result_text.tag_configure("bold", font=("Arial", 14, "bold"))
                self.result_text.insert(tk.END, "Generated Belief States:\n", "bold")
                for i, state in enumerate(belief_states, 1):
                    self.result_text.insert(tk.END, f"Belief State {i}:\n")
                    for row in range(3):
                        row_str = " ".join(str(state[row * 3 + col]) if state[row * 3 + col] != 0 else " " for col in range(3))
                        self.result_text.insert(tk.END, f"{row_str}\n")
                    self.result_text.insert(tk.END, "\n")
            else:
                self.display_belief_states([])
        
            if not self.solving:
                return
        
            if paths is None:
                reason = "Timeout or algorithm failed to converge" if duration >= self.puzzle.max_runtime else "No solution found"
                messagebox.showinfo("Result", f"{reason} using {selected_algo}!")
            else:
                if selected_algo == "Belief State Search":
                    self.display_path(paths[0])
                else:
                    self.display_path(paths)
                self.update_stats(duration, steps, stats)
                self.add_evaluation_data(selected_algo, duration, steps, stats)
                self.animate_solution(paths, delay=animation_delay, belief_states=belief_states if selected_algo == "Belief State Search" else [])
                messagebox.showinfo("Success", f"Solved with {selected_algo}!\nSteps: {steps}\nTime: {duration:.4f} seconds")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.solving = False
            self.current_job_id = None
            self.start_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.tree_button.config(state=tk.NORMAL)

    def animate_solution(self, paths, delay=500, belief_states=None):
        if not paths or not self.solving:
            return
        
        # Nếu là Belief State Search, mô phỏng cả 3 trạng thái niềm tin
        if belief_states and isinstance(paths, list) and len(paths) == len(belief_states):
            # Đảm bảo tất cả đường đi có cùng độ dài (đồng bộ hóa)
            max_steps = max(len(path) for path in paths)
            for i in range(max_steps):
                for idx, path in enumerate(paths):
                    # Lấy trạng thái tại bước i, hoặc trạng thái cuối nếu vượt quá độ dài đường đi
                    state = path[min(i, len(path) - 1)]
                    canvas = self.belief_canvases[idx]
                    # Xóa canvas trước khi vẽ lại
                    canvas.delete("all")
                    # Vẽ trạng thái
                    for row in range(3):
                        for col in range(3):
                            value = state[row * 3 + col]
                            x, y = col * self.tile_size + 15, row * self.tile_size + 15
                            bg_color = "#F5F5F5" if value == 0 else "#4682B4"
                            fg_color = "#333333" if value == 0 else "#FFFFFF"
                            canvas.create_rectangle(x, y, x + self.tile_size, y + self.tile_size, fill=bg_color, outline="#000000")
                            if value != 0:
                                canvas.create_text(x + self.tile_size / 2, y + self.tile_size / 2, text=str(value), font=("Arial", 12), fill=fg_color)
                # Cập nhật bảng chính với trạng thái ban đầu
                self.update_puzzle_display(paths[0][min(i, len(paths[0]) - 1)])
                self.root.update()
                # Thêm job để kiểm soát thời gian
                job = self.root.after(delay, lambda: None)  # Placeholder để hủy nếu cần
                self.animation_jobs.append(job)
                time.sleep(delay / 1000)  # Đợi đồng bộ
        else:
            # Mô phỏng mặc định cho các thuật toán khác
            self.update_puzzle_display(self.initial_state)
            self.root.update()
            self.animation_jobs = []
            for i, state in enumerate(paths):
                job = self.root.after(delay * (i + 1), lambda s=state: self.update_puzzle_display(s))
                self.animation_jobs.append(job)


    def draw_state_space_tree(self):
        if self.solving or not self.puzzle.visited_states:
            return
        if self.tree_window is not None:
            self.tree_window.destroy()
        self.tree_window = tk.Toplevel(self.root)
        self.tree_window.title("State Space Tree")
        self.tree_window.geometry("2000x1800")
        frame = tk.Frame(self.tree_window)
        frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(frame, bg="#E6F0FA")
        h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.bind("<Button-1>", lambda e: canvas.scan_mark(e.x, e.y))
        canvas.bind("<B1-Motion>", lambda e: canvas.scan_dragto(e.x, e.y, gain=1))
        canvas.bind("<MouseWheel>", lambda event: self._zoom(event, canvas))
        self.draw_tree_on_canvas(canvas, self.puzzle.goal)
        messagebox.showinfo("Success", "State space tree is displayed in a new window. Use mouse wheel to zoom, click and drag to pan.")

    def _zoom(self, event, canvas):
        scale = 1.1 if event.delta > 0 else 1 / 1.1
        canvas.scale("all", event.x, event.y, scale, scale)
        canvas.configure(scrollregion=canvas.bbox("all"))

    def draw_tree_on_canvas(self, canvas, goal_state):
        nodes = self.puzzle.visited_states
        if not nodes:
            canvas.create_text(50, 50, text="No states visited.", font=("Arial", 12))
            return
        edges = []
        nodes_to_include = set()
        nodes_to_include.add(self.puzzle.initial)
        queue = deque([self.puzzle.initial])
        visited = set([self.puzzle.initial])
        while queue:
            state = queue.popleft()
            for neighbor in self.puzzle.get_neighbors(state):
                if neighbor in self.puzzle.visited_states and neighbor not in visited:
                    parent, _ = self.puzzle.parent_map.get(neighbor, (None, None))
                    if parent == state:
                        edges.append((state, neighbor))
                        nodes_to_include.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
        levels = {self.puzzle.initial: 0}
        queue = deque([(self.puzzle.initial, 0)])
        while queue:
            state, level = queue.popleft()
            for neighbor in self.puzzle.get_neighbors(state):
                if neighbor in nodes_to_include and (state, neighbor) in edges:
                    levels[neighbor] = level + 1
                    queue.append((neighbor, level + 1))
        max_level = max(levels.values()) if levels else 0
        nodes_per_level = [[] for _ in range(max_level + 1)]
        for node, level in levels.items():
            nodes_per_level[level].append(node)
        tile_size = self.tile_size
        padding_x = 50
        padding_y = 100
        positions = {}
        max_nodes_per_level = max(len(level) for level in nodes_per_level) if nodes_per_level else 1
        level_width = max_nodes_per_level * (tile_size * 3 + padding_x)
        for level in range(max_level + 1):
            nodes_in_level = nodes_per_level[level]
            for i, state in enumerate(nodes_in_level):
                x = (i - (len(nodes_in_level) - 1) / 2) * (tile_size * 3 + padding_x) + level_width / 2
                y = level * (tile_size * 3 + padding_y)
                positions[state] = (x, y)
        min_x = min(pos[0] for pos in positions.values()) - tile_size * 1.5 - padding_x
        max_x = max(pos[0] for pos in positions.values()) + tile_size * 1.5 + padding_x
        min_y = min(pos[1] for pos in positions.values()) - tile_size * 1.5 - padding_y
        max_y = max(pos[1] for pos in positions.values()) + tile_size * 1.5 + padding_y
        canvas_width = max(max_x - min_x, 2000)
        canvas_height = max(max_y - min_y, 1800)
        canvas.configure(scrollregion=(min_x, min_y, max_x, max_y))
        offset_x = -min_x
        offset_y = -min_y
        for state in positions:
            x, y = positions[state]
            positions[state] = (x + offset_x, y + offset_y)
        for parent, child in edges:
            parent_pos = positions[parent]
            child_pos = positions[child]
            action_idx = self.puzzle.parent_map[child][1]
            color = self.puzzle.action_colors[action_idx]
            mid_x = (parent_pos[0] + child_pos[0]) / 2
            mid_y = (parent_pos[1] + child_pos[1]) / 2 - 20
            canvas.create_line(parent_pos[0], parent_pos[1] + tile_size * 1.5,
                              mid_x, mid_y,
                              child_pos[0], child_pos[1] - tile_size * 1.5,
                              fill=color, width=2, smooth=True)
            canvas.create_text(mid_x, mid_y, text=self.puzzle.action_names[action_idx],
                              font=("Arial", 10), fill=color)
        font_size = max(12, int(tile_size / 2))
        for state in nodes_to_include:
            x, y = positions[state]
            border_color = "#00FF00" if state == goal_state else "#000000"
            for i in range(3):
                for j in range(3):
                    value = state[i * 3 + j]
                    tile_x = x + j * tile_size - tile_size * 1.5
                    tile_y = y + i * tile_size - tile_size * 1.5
                    bg_color = "#F5F5F5" if value == 0 else "#4682B4"
                    fg_color = "#333333" if value == 0 else "#FFFFFF"
                    canvas.create_rectangle(tile_x, tile_y, tile_x + tile_size, tile_y + tile_size,
                                           fill=bg_color, outline=border_color)
                    if value != 0:
                        canvas.create_text(tile_x + tile_size / 2, tile_y + tile_size / 2,
                                          text=str(value), font=("Arial", font_size), fill=fg_color)

    # Các phương thức mới cho đánh giá thuật toán
    def add_evaluation_data(self, algorithm, duration, steps, stats):
        """Thêm dữ liệu đánh giá vào bảng"""
        # Tính toán độ phức tạp không gian (ước lượng)
        space_complexity = stats.get("max_fringe_size", 0) + stats.get("states_visited", 0)
        
        # Tạo một bản ghi đánh giá mới
        eval_record = {
            "algorithm": algorithm,
            "time": duration,
            "states_visited": stats.get("states_visited", 0),
            "nodes_expanded": stats.get("nodes_expanded", 0),
            "path_length": steps,
            "space_complexity": space_complexity
        }
        
        # Thêm vào danh sách dữ liệu
        self.evaluation_data.append(eval_record)
        
        # Cập nhật bảng Treeview
        self.eval_table.delete(*self.eval_table.get_children())
        for record in self.evaluation_data:
            self.eval_table.insert("", "end", values=(
                record["algorithm"],
                f"{record['time']:.4f}",
                record["states_visited"],
                record["nodes_expanded"],
                record["path_length"],
                record["space_complexity"]
            ))

    def compare_algorithms(self):
        """So sánh hiệu suất của các thuật toán đã được đánh giá"""
        if not self.evaluation_data:
            messagebox.showwarning("Warning", "No evaluation data available!")
            return
        
        # Tạo một cửa sổ mới để hiển thị kết quả so sánh
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Algorithm Comparison")
        compare_window.geometry("800x600")
        
        text = scrolledtext.ScrolledText(compare_window, width=90, height=30, font=("Arial", 12))
        text.pack(padx=10, pady=10)
        
        # Thêm tiêu đề
        text.insert(tk.END, "Algorithm Performance Comparison\n\n", "title")
        text.tag_config("title", font=("Arial", 14, "bold"))
        
        # So sánh các chỉ số
        for metric in ["time", "states_visited", "nodes_expanded", "path_length", "space_complexity"]:
            text.insert(tk.END, f"{metric.replace('_', ' ').title()}:\n", "bold")
            text.tag_config("bold", font=("Arial", 12, "bold"))
            best_value = min((record[metric] for record in self.evaluation_data if record[metric] > 0), default=0)
            for record in self.evaluation_data:
                value = record[metric]
                is_best = value == best_value and value > 0
                text.insert(tk.END, f"  {record['algorithm']}: {value:.4f if metric == 'time' else value} "
                                  f"{'(Best)' if is_best else ''}\n")
            text.insert(tk.END, "\n")
        
        text.config(state=tk.DISABLED)

    def clear_evaluation_data(self):
        """Xóa tất cả dữ liệu đánh giá"""
        self.evaluation_data.clear()
        self.eval_table.delete(*self.eval_table.get_children())
        messagebox.showinfo("Success", "Evaluation data cleared!")

    def save_evaluation_data(self):
        """Lưu dữ liệu đánh giá vào file CSV"""
        if not self.evaluation_data:
            messagebox.showwarning("Warning", "No evaluation data to save!")
            return
        
        # Chuyển đổi dữ liệu thành DataFrame và lưu thành file CSV
        df = pd.DataFrame(self.evaluation_data)
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Evaluation data saved to {file_path}")

    def show_performance_chart(self):
        """Hiển thị biểu đồ hiệu suất"""
        if not self.evaluation_data:
            messagebox.showwarning("Warning", "No evaluation data available!")
            return
        
        # Tạo cửa sổ mới cho biểu đồ
        chart_window = tk.Toplevel(self.root)
        chart_window.title("Performance Chart")
        chart_window.geometry("800x600")
        
        # Tạo figure cho matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Lấy dữ liệu để vẽ biểu đồ (ví dụ: Time vs Algorithms)
        algorithms = [record["algorithm"] for record in self.evaluation_data]
        times = [record["time"] for record in self.evaluation_data]
        
        ax.bar(algorithms, times, color='skyblue')
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Algorithm Performance (Time)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        # Thêm canvas để hiển thị biểu đồ trong Tkinter
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()
