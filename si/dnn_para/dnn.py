from .si4dnn import InferenceModel


def get_model_intervals(model, intervals, device):
    inference_model = InferenceModel(model, device)
    new_intervals = []
    for left, right, a, b in intervals:
        z = left + 1e-4
        while z < right:
            a_hat, b_hat, itv = inference_model.forward(a, b, z)
            # print(f"Interval: {itv}, z: {z}")
            new_intervals.append((max(z, itv[0]), itv[1], a_hat, b_hat))
            z = itv[1] + 1e-4
    return new_intervals
