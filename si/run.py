import numpy as np
import torch

from .detection import (
    anomaly_detection,
    get_j_in_topk_intervals,
    get_o_equal_oobs_intervals,
)
from .dnn.dnn import get_model_intervals as get_model_intervals_cpu
from .dnn_gpu.dnn import get_model_intervals as get_model_intervals_gpu
from .dnn_para.dnn import get_model_intervals as get_model_intervals_para
from .util import (
    build_initial_sign_interval,
    build_test_reference_contrast_path,
    compute_etajTsigmaetaj_a_b,
    gen_data,
    load_models,
    resolve_source_target_test_sizes,
    truncated_cdf,
)


# def run_all(
#     seed: int,
#     delta: float = 0,
#     n: int = 150,
#     mu: float = 0,
#     d: int = 8,
#     anomaly_rate: float = 0.0,
#     top_k_percent: float = 0.05,
#     top_k_normal_percent: float = 0.3,
#     deepsad_encoder=None,
#     deepsad_c=None,
#     device: str = "auto",
#     test_index_class: str = "normal",
#     Sigma: np.ndarray = None,
#     Sigma_source: np.ndarray = None,
#     Sigma_target: np.ndarray = None,
#     Sigma_ref: np.ndarray = None,
#     X_source_obs: np.ndarray = None,
#     rho: float = 0.0,
#     reference_size: int = 100,
#     selection_event: str = "j-in-o",
# ):
#     _ = top_k_normal_percent, Sigma_source, Sigma_ref, X_source_obs
#     selection_event = str(selection_event).strip().lower().replace("_", "-")
#     if selection_event not in {"j-in-o", "o-equal-oobs"}:
#         raise ValueError("selection_event must be either 'j-in-o' or 'o-equal-oobs'.")
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     requested_device = "auto" if device is None else str(device).lower()
#     if requested_device not in {"auto", "cpu", "cuda", "dnn_para"}:
#         raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, cuda, dnn_para.")
#     if requested_device == "cuda" and not torch.cuda.is_available():
#         raise ValueError("CUDA was requested but is not available.")
#     if Sigma_target is not None:
#         Sigma = Sigma_target

#     if deepsad_encoder is None:
#         load_device = None if requested_device == "auto" else requested_device
#         deepsad_encoder, deepsad_c, _ = load_models(device=load_device)

#     model_device = next(deepsad_encoder.parameters()).device
#     target_model_device = requested_device
#     if requested_device in {"auto", "dnn_para"}:
#         target_model_device = model_device.type
#     if target_model_device in {"cpu", "cuda"} and model_device.type != target_model_device:
#         deepsad_encoder = deepsad_encoder.to(target_model_device)
#         model_device = next(deepsad_encoder.parameters()).device

#     X, true_y = gen_data(mu, delta, n, d, anomaly_rate, rho=rho, seed=seed)
#     n, d = X.shape
#     if not np.all(np.isin(true_y, (-1, 1))):
#         raise ValueError("true_y contains invalid labels; expected values in {-1,1}.")

#     O = anomaly_detection(X, top_k_percent, deepsad_encoder, deepsad_c)
#     if test_index_class not in {"normal", "anomaly"}:
#         raise ValueError("test_index_class must be either 'normal' or 'anomaly'.")
#     if test_index_class == "anomaly":
#         candidates = [i for i in O if true_y[i] == -1]
#     else:
#         candidates = [i for i in O if true_y[i] == 1]
#     if len(candidates) == 0:
#         print(f"No '{test_index_class}' points for seed {seed}, skipping...")
#         return []

#     p_values = []
#     for j in candidates:
#         X_normal = gen_data(mu, 0.0, reference_size, d, 0.0, rho=rho, seed=seed + 1)[0]
#         X_mean = np.mean(X_normal, axis=0)
#         X_concat = np.vstack([X, X_normal])

#         etaj = np.zeros(n * d + X_normal.shape[0] * d)
#         test_statistic = 0.0
#         for i in range(d):
#             sign = 1 if X[j, i] - X_mean[i] >= 0 else -1
#             etaj[j * d + i] = sign
#             for u in range(X_normal.shape[0]):
#                 etaj[n * d + u * d + i] = -sign / X_normal.shape[0]
#             test_statistic += sign * (X[j, i] - X_mean[i])

#         etajTx = (etaj.T @ X_concat.reshape(-1, 1)).reshape(1, 1)
#         etajTsigmaetaj, a, b = compute_etajTsigmaetaj_a_b(
#             etaj, etajTx, X_concat, n + X_normal.shape[0], d, S=Sigma
#         )
#         a = a.reshape(n + X_normal.shape[0], d)
#         b = b.reshape(n + X_normal.shape[0], d)

#         postivie_sign = np.sign(X[j, :] - X_mean)
#         itv = [-20 * np.sqrt(etajTsigmaetaj[0][0]), 20 * np.sqrt(etajTsigmaetaj[0][0])]
#         a_mean = a[n:, :].mean(axis=0)
#         b_mean = b[n:, :].mean(axis=0)
#         for i in range(d):
#             new_a = (a[j, i] - a_mean[i]) * postivie_sign[i]
#             new_b = (b[j, i] - b_mean[i]) * postivie_sign[i]
#             if abs(new_b) < 1e-16:
#                 continue
#             z = -new_a / new_b
#             if new_b > 0:
#                 itv = [max(itv[0], z), itv[1]]
#             else:
#                 itv = [itv[0], min(itv[1], z)]
#         itv[0] = itv[0].item() if isinstance(itv[0], np.ndarray) else itv[0]
#         itv[1] = itv[1].item() if isinstance(itv[1], np.ndarray) else itv[1]
#         if etajTx[0][0] > itv[1]:
#             p_values.append(0.0)
#             continue

#         intervals = [(itv[0], itv[1], a[:n], b[:n])]
#         if requested_device == "dnn_para":
#             para_device = "cuda" if torch.cuda.is_available() else "cpu"
#             intervals = get_model_intervals_para(deepsad_encoder, intervals, para_device)
#         else:
#             use_cuda_dnn = model_device.type == "cuda"
#             if requested_device == "cpu":
#                 use_cuda_dnn = False
#             elif requested_device == "cuda":
#                 use_cuda_dnn = True
#             if use_cuda_dnn:
#                 si_dtype = torch.float64
#                 intervals_gpu = [
#                     (
#                         left,
#                         right,
#                         torch.as_tensor(a_i, dtype=si_dtype, device=model_device),
#                         torch.as_tensor(b_i, dtype=si_dtype, device=model_device),
#                     )
#                     for left, right, a_i, b_i in intervals
#                 ]
#                 intervals_gpu = get_model_intervals_gpu(deepsad_encoder, intervals_gpu)
#                 intervals = [
#                     (left, right, a_i.detach().cpu().numpy(), b_i.detach().cpu().numpy())
#                     for left, right, a_i, b_i in intervals_gpu
#                 ]
#             else:
#                 intervals = get_model_intervals_cpu(deepsad_encoder, intervals)

#         if selection_event == "j-in-o":
#             event_intervals = get_j_in_topk_intervals(
#                 intervals,
#                 top_k_percent=top_k_percent,
#                 deepsad_c=deepsad_c,
#                 j=int(j),
#             )
#         else:
#             event_intervals = get_o_equal_oobs_intervals(
#                 intervals,
#                 top_k_percent=top_k_percent,
#                 deepsad_c=deepsad_c,
#                 O_obs=O,
#             )
#         final_intervals = [(left, right, bool(Oz)) for left, right, Oz in event_intervals]
#         cdf = truncated_cdf(
#             0,
#             np.sqrt(etajTsigmaetaj[0][0]),
#             final_intervals,
#             True,
#             etajTx[0][0],
#         )
#         if cdf is None:
#             print(f"Warning: CDF computation failed for seed {seed}. Skipping this run.")
#             continue
#         p_value = 2 * min(cdf, 1 - cdf)
#         print(f"p-value for seed {seed}: {p_value}")
#         p_values.append(p_value)
#     return p_values


def run_one(
    seed: int,
    delta: float = 0,
    n: int = 150,
    target_mu: float = 0,
    source_mu: float = 2,
    d: int = 8,
    anomaly_rate: float = 0.0,
    top_k_percent: float = 0.05,
    top_k_normal_percent: float = 0.3,
    deepsad_encoder=None,
    deepsad_c=None,
    device: str = "auto",
    test_index_class: str = "normal",
    Sigma: np.ndarray = None,
    Sigma_source: np.ndarray = None,
    Sigma_target: np.ndarray = None,
    Sigma_ref: np.ndarray = None,
    X_source_obs: np.ndarray = None,
    target_rho: float = 0.0,
    source_rho: float = 0.0,
    reference_size: int = 100,
    source_test_size: int | None = None,
    target_test_size: int | None = None,
    selection_event: str = "j-in-o",
):
    _ = (
        top_k_normal_percent,
        X_source_obs,
    )
    np.random.seed(seed)
    torch.manual_seed(seed)

    if test_index_class not in {"normal", "anomaly"}:
        raise ValueError("test_index_class must be either 'normal' or 'anomaly'.")
    selection_event = str(selection_event).strip().lower().replace("_", "-")
    if selection_event not in {"j-in-o", "o-equal-oobs"}:
        raise ValueError("selection_event must be either 'j-in-o' or 'o-equal-oobs'.")
    n_total, n_source, n_target = resolve_source_target_test_sizes(
        n,
        source_test_size=source_test_size,
        target_test_size=target_test_size,
    )

    requested_device = "auto" if device is None else str(device).lower()
    if requested_device not in {"auto", "cpu", "cuda", "dnn_para"}:
        raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, cuda, dnn_para.")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    if deepsad_encoder is None:
        load_device = None if requested_device == "auto" else requested_device
        deepsad_encoder, deepsad_c, _ = load_models(device=load_device)

    model_device = next(deepsad_encoder.parameters()).device
    target_model_device = requested_device
    if requested_device in {"auto", "dnn_para"}:
        target_model_device = model_device.type
    if target_model_device in {"cpu", "cuda"} and model_device.type != target_model_device:
        deepsad_encoder = deepsad_encoder.to(target_model_device)
        model_device = next(deepsad_encoder.parameters()).device

    def covariance_from_rho(rho_value: float) -> np.ndarray:
        if abs(float(rho_value)) < 1e-12:
            return np.eye(int(d), dtype=np.float64)
        idx = np.arange(int(d))
        return float(rho_value) ** np.abs(np.subtract.outer(idx, idx))

    if Sigma_target is None and Sigma is not None:
        Sigma_target = np.asarray(Sigma, dtype=np.float64)
    if Sigma_source is None:
        Sigma_source = covariance_from_rho(source_rho)
    if Sigma_target is None:
        Sigma_target = covariance_from_rho(target_rho)
    if Sigma_ref is None:
        Sigma_ref = Sigma_target

    if test_index_class == "normal":
        test_delta = 0.0
        test_anomaly_rate = 0.0
        candidate_label = 1
    else:
        test_delta = float(delta)
        test_anomaly_rate = float(anomaly_rate)
        candidate_label = -1

    # Per spec, FPR uses normal-only source/target; TPR uses shared anomaly rate/delta.
    X_source, y_source = gen_data(
        source_mu, test_delta, n_source, d, test_anomaly_rate, rho=source_rho, seed=seed
    )
    X_target, y_target = gen_data(
        target_mu, test_delta, n_target, d, test_anomaly_rate, rho=target_rho, seed=seed + 1
    )
    X_ref = gen_data(target_mu, 0.0, reference_size, d, 0.0, rho=target_rho, seed=seed + 2)[0]
    if not (np.all(np.isin(y_source, (-1, 1))) and np.all(np.isin(y_target, (-1, 1)))):
        raise ValueError("true_y contains invalid labels; expected values in {-1,1}.")

    X_detect = np.vstack([X_source, X_target])
    with torch.no_grad():
        x_tensor = torch.tensor(X_detect, dtype=torch.float32, device=model_device)
        embeddings = deepsad_encoder(x_tensor).detach().cpu().numpy()
    scores = np.linalg.norm(embeddings - deepsad_c, axis=1)
    top_k = max(1, int(top_k_percent * len(scores)))
    top_k = min(top_k, len(scores))
    O = sorted(np.argpartition(scores, -top_k)[-top_k:])

    candidates = [
        int(i)
        for i in O
        if int(i) >= n_source and y_target[int(i) - n_source] == candidate_label
    ]
    if len(candidates) == 0:
        print(f"No '{test_index_class}' points for seed {seed}, skipping...")
        return []
    j_global = int(np.random.choice(candidates))
    j_target = j_global - n_source

    path = build_test_reference_contrast_path(
        X_test=X_target,
        X_ref=X_ref,
        j=j_target,
        X_source=X_source,
        Sigma_source=Sigma_source,
        Sigma_target=Sigma_target,
        Sigma_ref=Sigma_ref,
    )
    left, right = build_initial_sign_interval(
        path["a_test"],
        path["b_test"],
        j_target,
        path["positive_sign"],
        path["reference_a_mean"],
        path["reference_b_mean"],
        path["sigma"],
    )
    if left > right or not (left <= path["test_statistic"] <= right):
        return [0.0]


    a_detect = np.vstack([path["a_source"], path["a_test"]])
    b_detect = np.vstack([path["b_source"], path["b_test"]])
    intervals = [(left, right, a_detect, b_detect)]
    if requested_device == "dnn_para":
        para_device = "cuda" if torch.cuda.is_available() else "cpu"
        intervals = get_model_intervals_para(deepsad_encoder, intervals, para_device)
    else:
        use_cuda_dnn = model_device.type == "cuda"
        if requested_device == "cpu":
            use_cuda_dnn = False
        elif requested_device == "cuda":
            use_cuda_dnn = True
        if use_cuda_dnn:
            si_dtype = torch.float64
            intervals_gpu = [
                (
                    left,
                    right,
                    torch.as_tensor(a_i, dtype=si_dtype, device=model_device),
                    torch.as_tensor(b_i, dtype=si_dtype, device=model_device),
                )
                for left, right, a_i, b_i in intervals
            ]
            intervals_gpu = get_model_intervals_gpu(deepsad_encoder, intervals_gpu)
            intervals = [
                (left, right, a_i.detach().cpu().numpy(), b_i.detach().cpu().numpy())
                for left, right, a_i, b_i in intervals_gpu
            ]
        else:
            intervals = get_model_intervals_cpu(deepsad_encoder, intervals)

    if selection_event == "j-in-o":
        intervals = get_j_in_topk_intervals(
            intervals, top_k_percent=top_k_percent, deepsad_c=deepsad_c, j=j_global
        )
    else:
        intervals = get_o_equal_oobs_intervals(
            intervals, top_k_percent=top_k_percent, deepsad_c=deepsad_c, O_obs=O
        )
    final_intervals = [(left_i, right_i, bool(Oz)) for left_i, right_i, Oz in intervals]
    cdf = truncated_cdf(0, path["sigma"], final_intervals, True, path["test_statistic"])
    if cdf is None:
        print(f"Warning: CDF computation failed for seed {seed}. Skipping this run.")
        return [None]
    p_value = 2 * min(cdf, 1 - cdf)
    print(f"p-value for seed {seed}: {p_value}")
    return [p_value]
