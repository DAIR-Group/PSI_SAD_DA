import numpy as np
import torch


def anomaly_detection(
    X: np.ndarray, top_k_percent: float, deepsad_encoder, deepsad_c
) -> np.ndarray:
    model_device = next(deepsad_encoder.parameters()).device
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32, device=model_device)
        x_transformed = deepsad_encoder(x_tensor).cpu().numpy()

    distances = np.linalg.norm(x_transformed - deepsad_c, axis=1)
    top_k = int(top_k_percent * len(distances))
    anomalies_index = np.argpartition(distances, -top_k)[-top_k:]
    return sorted(anomalies_index)

def top_k_normal_indices(X: np.ndarray, top_k_percent: float, deepsad_encoder, deepsad_c) -> np.ndarray:
    model_device = next(deepsad_encoder.parameters()).device
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32, device=model_device)
        x_transformed = deepsad_encoder(x_tensor).cpu().numpy()

    distances = np.linalg.norm(x_transformed - deepsad_c, axis=1)
    top_k = int(top_k_percent * len(distances))
    normal_indices = np.argpartition(distances, top_k)[:top_k]
    return sorted(normal_indices)

import heapq
import math
import numpy as np



def get_ad_intervals(intervals, top_k_percent, deepsad_c, eps=1e-10):
    quad_intervals = []
    for left, right, a, b in intervals:
        u = a - deepsad_c
        A = np.sum(b * b, axis=1)
        B = 2 * np.sum(u * b, axis=1)
        C = np.sum(u * u, axis=1)
        quad_intervals.append((left, right, A, B, C))

    new_intervals = []
    
    

    for left, right, A, B, C in quad_intervals:
        n = len(A)
        top_k = max(1, int(top_k_percent * n))
        boundary = n - top_k  # items at rank >= boundary are in top-k

        def find_next_crossing(i, j, z_min):
            """
            First z in (z_min, right] where d_i crosses from below d_j to above.
            Precondition: d_i(z_min) <= d_j(z_min).
            Each quadratic difference has at most 2 roots, checked in O(1).
            """
            da = float(A[i] - A[j])
            db = float(B[i] - B[j])
            dc = float(C[i] - C[j])
            eps = 1e-10

            if abs(da) < 1e-16:
                roots = [-dc / db] if abs(db) > 1e-16 else []
            else:
                disc = db * db - 4.0 * da * dc
                if disc < 0:
                    return None
                sq = math.sqrt(max(0.0, disc))
                roots = [(-db - sq) / (2 * da), (-db + sq) / (2 * da)]

            for z in sorted(roots):
                tol = abs(z_min) * 1e-9 + 1e-15
                if z <= z_min + tol or z > right + tol:
                    continue
                # Confirm direction: d_i - d_j goes negative → positive here
                z_t = z + eps
                if 2.0 * da * z + db > 0.0:
                    return z
            return None

        # --- Initial sort at z = left ---
        dist0 = A * (left + eps) * (left + eps) + B * (left + eps) + C
        sorted_order = list(np.argsort(dist0))   # sorted_order[r] = item at rank r (asc. distance)
        rank = np.empty(n, dtype=np.intp)
        for r, item in enumerate(sorted_order):
            rank[item] = r

        # --- Seed heap with crossing times for each adjacent pair ---
        # O(n log n) — only n-1 pairs, not O(n²)
        heap = []  # (z_crossing, i, j)  where rank[i]+1 == rank[j] at insertion time
        for r in range(n - 1):
            i, j = sorted_order[r], sorted_order[r + 1]
            z_cross = find_next_crossing(i, j, left + eps)
            if z_cross is not None:
                heapq.heappush(heap, (z_cross, i, j))

        previous = left
        current_topk = set(sorted_order[boundary:])

        # --- Kinetic sweep ---
        while heap:
            z_val, i, j = heapq.heappop(heap)
            if z_val > right:
                break

            if rank[i] + 1 != rank[j]:
                continue

            r = rank[i]

            if r == boundary - 1:
                # i is crossing the boundary with j
                # before swap: sorted_order[r]=i, sorted_order[r+1]=j
                outgoing = sorted_order[boundary]
                incoming = sorted_order[boundary - 1]
                
                new_intervals.append((previous, z_val, current_topk))
                previous = z_val
                current_topk = (current_topk - {outgoing}) | {incoming}

            sorted_order[r], sorted_order[r + 1] = j, i
            rank[j] = r
            rank[i] = r + 1

            for r2 in (r - 1, r, r + 1):
                if 0 <= r2 < n - 1:
                    ii, jj = sorted_order[r2], sorted_order[r2 + 1]
                    z_cross = find_next_crossing(ii, jj, z_val)
                    if z_cross is not None and z_cross < right:
                        heapq.heappush(heap, (z_cross, ii, jj))

        new_intervals.append((previous, right, current_topk))

    return new_intervals


def get_ad_intervals_fast(intervals, top_k_percent, deepsad_c, eps=1e-10):
    return get_ad_intervals(
        intervals=intervals,
        top_k_percent=top_k_percent,
        deepsad_c=deepsad_c,
        eps=eps,
    )
        
def get_top_k_normal_intervals(intervals, top_k_percent, deepsad_c):
    quad_intervals = []
    for left, right, a, b in intervals:
        u = a - deepsad_c
        A = np.sum(b * b, axis=1)
        B = 2 * np.sum(u * b, axis=1)
        C = np.sum(u * u, axis=1)
        quad_intervals.append((left, right, A, B, C))

    new_intervals = []

    for left, right, A, B, C in quad_intervals:
        n = len(A)
        top_k = max(1, int(top_k_percent * n))
        boundary = top_k  # items at rank < boundary are in top-k normal

        def find_next_crossing(i, j, z_min):
            """
            First z in (z_min, right] where d_i crosses from below d_j to above.
            Precondition: d_i(z_min) <= d_j(z_min).
            Each quadratic difference has at most 2 roots, checked in O(1).
            """
            da = float(A[i] - A[j])
            db = float(B[i] - B[j])
            dc = float(C[i] - C[j])
            eps = 1e-10

            if abs(da) < 1e-16:
                roots = [-dc / db] if abs(db) > 1e-16 else []
            else:
                disc = db * db - 4.0 * da * dc
                if disc <= -eps:
                    return None
                sq = math.sqrt(max(0.0, disc))
                roots = [(-db - sq) / (2 * da), (-db + sq) / (2 * da)]

            for z in sorted(roots):
                if z > z_min:
                    return z
            return None

        # --- Initial sort at z = left ---
        dist0 = A * left * left + B * left + C
        sorted_order = list(np.argsort(dist0))   # sorted_order[r] = item at rank r (asc. distance)
        rank = np.empty(n, dtype=np.intp)
        for r, item in enumerate(sorted_order):
            rank[item] = r

        # --- Seed heap with crossing times for each adjacent pair ---
        # O(n log n) — only n-1 pairs, not O(n²)
        heap = []  # (z_crossing, i, j)  where rank[i]+1 == rank[j] at insertion time
        for r in range(n - 1):
            i, j = sorted_order[r], sorted_order[r + 1]
            z_cross = find_next_crossing(i, j, left - 1e-9)
            if z_cross is not None:
                heapq.heappush(heap, (z_cross, i, j))

        previous = left
        current_topk = set(sorted_order[:boundary])

        # --- Kinetic sweep ---
        while heap:
            z_val, i, j = heapq.heappop(heap)
            if z_val > right + 1e-9:
                break

            if rank[i] + 1 != rank[j]:
                continue

            r = rank[i]

            if r == boundary - 1:                      # i is last in top-k, j is first outside
                outgoing = sorted_order[boundary - 1]  # = i, leaving top-k
                incoming = sorted_order[boundary]      # = j, entering top-k

                new_intervals.append((previous, z_val, current_topk))
                previous = z_val
                current_topk = (current_topk - {outgoing}) | {incoming}

            sorted_order[r], sorted_order[r + 1] = j, i
            rank[j] = r
            rank[i] = r + 1

            for r2 in (r - 1, r, r + 1):
                if 0 <= r2 < n - 1:
                    ii, jj = sorted_order[r2], sorted_order[r2 + 1]
                    z_cross = find_next_crossing(ii, jj, z_val)
                    if z_cross is not None and z_cross < right + 1e-9:
                        heapq.heappush(heap, (z_cross, ii, jj))

        new_intervals.append((previous, right, current_topk))

    return new_intervals

def get_j_in_topk_intervals(intervals, top_k_percent, deepsad_c, j):
    quad_intervals = []
    for left, right, a, b in intervals:
        u = a - deepsad_c
        A = np.sum(b * b, axis=1)
        B = 2 * np.sum(u * b, axis=1)
        C = np.sum(u * u, axis=1)
        quad_intervals.append((left, right, A, B, C))
    new_intervals = []
    for left, right, A, B, C in quad_intervals:
        n = len(A)
        top_k = max(1, int(top_k_percent * n))
        boundary = n - top_k  # items at rank >= boundary are in top-k
        
        def get_roots_and_signs(i, j):
            da = float(A[i] - A[j])
            db = float(B[i] - B[j])
            dc = float(C[i] - C[j])
            eps = 1e-10

            if abs(da) < 1e-16:
                # return sign of linear function 
                return [(-dc / db, np.sign(db))] if abs(db) > 1e-16 else []
            else:
                disc = db * db - 4.0 * da * dc
                if disc < 0:
                    return []
                sq = math.sqrt(max(0.0, disc))
                return [((-db - sq) / (2 * da), -1 * np.sign(A[i] - A[j])), ((-db + sq) / (2 * da), np.sign(A[i] - A[j]))]
            
        distances_at_left = A * (left + 1e-9) * (left + 1e-9) + B * (left + 1e-9) + C
        sorted_order = list(np.argsort(distances_at_left))   # sorted_order[r] = item at rank r (asc. distance)
        rank = np.empty(n, dtype=np.intp)
        
        smaller_than_j = 0
        for r, item in enumerate(sorted_order):
            rank[item] = r
            if item == j:
                smaller_than_j = r
        
        z_values = []
        for i in range(n):
            if i == j:
                continue
            z_crossings = get_roots_and_signs(j, i)
            for z, sign in z_crossings:
                if left < z <= right:
                    z_values.append((z, sign))
        previous = left + 1e-9
        for z, sign in sorted(z_values):
            new_intervals.append((previous, z, smaller_than_j >= boundary))
            previous = z
            smaller_than_j += sign
        new_intervals.append((previous, right, smaller_than_j >= boundary))
    return new_intervals


def get_j_in_topk_intervals_v2(intervals, top_k_percent, deepsad_c, j):
    return get_j_in_topk_intervals(
        intervals=intervals,
        top_k_percent=top_k_percent,
        deepsad_c=deepsad_c,
        j=j,
    )
