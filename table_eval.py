import re


def is_answer_correct(pred: str, label: str) -> bool:
    pred = pred.strip().lower()
    label = label.strip().lower()

    if label in ["yes", "no"]:
        return pred.startswith(label)

    try:
        label_number = float(re.findall(r"[-+]?\d*\.\d+|\d+", label)[0])
    except IndexError:
        return False

    try:
        pred_number = float(re.findall(r"[-+]?\d*\.\d+|\d+", pred)[0])
    except IndexError:
        return False

    return abs(pred_number - label_number) < 1e-2
