"""Mesin prediksi StressGuard AI (tanpa scikit-learn).

Membaca Random Forest yang sudah diekspor ke JSON (stress_model.json) dan
menghitung probabilitas kelas dengan merata-ratakan probabilitas daun
dari semua pohon, sama seperti fungsi predict_label() di versi PHP.

Format tiap pohon (array paralel, indeks = nomor node):
    f : indeks fitur yang dipakai untuk split (-2 = daun)
    t : nilai ambang (threshold)
    l : indeks anak kiri  (nilai fitur <= threshold)
    r : indeks anak kanan (nilai fitur >  threshold)
    v : probabilitas tiap kelas di daun
"""
import json
import os
from typing import Dict, List

LEAF = -2


def find_file(name: str) -> str:
    """Cari file di folder yang sama dengan skrip ini, lalu di subfolder model/."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, name), os.path.join(here, "model", name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} tidak ditemukan di {here} atau {here}/model")


def load_model(path: str = None) -> dict:
    with open(path or find_file("stress_model.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_meta(path: str = None) -> dict:
    with open(path or find_file("model_meta.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _walk(tree: dict, x: List[float]) -> List[float]:
    i = 0
    f, t, l, r = tree["f"], tree["t"], tree["l"], tree["r"]
    while f[i] != LEAF:
        i = l[i] if x[f[i]] <= t[i] else r[i]
    return tree["v"][i]


def predict_proba(model: dict, values: Dict[str, float]) -> Dict[str, float]:
    """Kembalikan {kelas: probabilitas}. `values` berisi nama fitur -> nilai."""
    x = [float(values[name]) for name in model["features"]]
    n_classes = len(model["classes"])
    acc = [0.0] * n_classes
    for tree in model["trees"]:
        leaf = _walk(tree, x)
        for k in range(n_classes):
            acc[k] += leaf[k]
    n = len(model["trees"])
    return {c: acc[k] / n for k, c in enumerate(model["classes"])}


def predict_label(model: dict, values: Dict[str, float]) -> str:
    proba = predict_proba(model, values)
    return max(proba, key=proba.get)
