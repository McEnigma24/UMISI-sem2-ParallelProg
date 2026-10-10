#!/usr/bin/env python3
"""Rysuje wykresy z JSON-a Google Benchmark - czas operacji w funkcji rozmiaru tablicy.

Wolane automatycznie z production.sh po ./bench.bexe, ale dziala tez samodzielnie:
    python3 scripts/bench_plot.py _bench/results/bench.json _bench/results/bench.png

Bierze tylko agregaty (mean/median/stddev/cv), wiec benchmark musi byc zarejestrowany
z ->Repetitions(N) i z argumentem rozmiaru (->Range() / ->DenseRange()).
"""

import json
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")  # w kontenerze nie ma X11 - renderujemy wprost do pliku

import matplotlib.pyplot as plt

WANTED_AGGREGATES = ("mean", "median", "stddev", "cv")


def size_from_run_name(run_name):
    """Z 'BM_x/72/min_time:0.050/repeats:10' wyciaga 72 - pierwszy czlon liczbowy."""
    for part in run_name.split("/")[1:]:
        if part.isdigit():
            return int(part)
    return None


def load_measurements(json_path):
    with open(json_path, encoding="utf-8") as handle:
        raw = json.load(handle)

    families = OrderedDict()  # nazwa benchmarku -> { N -> { agregat -> wartosc } }
    unit = "ns"

    for entry in raw.get("benchmarks", []):
        aggregate = entry.get("aggregate_name")
        if aggregate not in WANTED_AGGREGATES:
            continue  # pojedyncze przebiegi pomijamy, interesuja nas statystyki

        size = size_from_run_name(entry["run_name"])
        if size is None:
            continue  # benchmark bez argumentu rozmiaru nie ma czego odlozyc na osi X

        family = entry["run_name"].split("/")[0]
        unit = entry.get("time_unit", unit)
        families.setdefault(family, {}).setdefault(size, {})[aggregate] = entry["real_time"]

    return families, unit, raw.get("context", {})


def human_bytes(value):
    for suffix in ("B", "KiB", "MiB"):
        if value < 1024 or suffix == "MiB":
            return f"{value:.0f} {suffix}"
        value /= 1024.0
    return f"{value:.0f} MiB"


def context_caption(context):
    """Jednolinijkowy opis maszyny - bez tego wykres nic nie znaczy."""
    bits = []

    if context.get("host_name"):
        bits.append(str(context["host_name"]))

    cpus = context.get("num_cpus")
    mhz = context.get("mhz_per_cpu")
    if cpus and mhz:
        bits.append(f"{cpus} x {float(mhz):.0f} MHz")

    caches = []
    for cache in context.get("caches", []):
        level = cache.get("level")
        kind = str(cache.get("type", ""))
        name = f"L{level}" + ("d" if kind == "Data" else "i" if kind == "Instruction" else "")
        caches.append(f"{name} {human_bytes(float(cache.get('size', 0)))}")
    if caches:
        bits.append(" / ".join(caches))

    if context.get("date"):
        bits.append(str(context["date"]))
    if context.get("cpu_scaling_enabled"):
        bits.append("CPU scaling ON - pomiary zaszumione")

    return "Google Benchmark  ·  " + "  ·  ".join(bits)


def short_name(family):
    return family[3:] if family.startswith("BM_") else family


def print_table(families, unit):
    names = list(families)
    sizes = sorted({size for by_size in families.values() for size in by_size})
    column = 26

    print(f"\nSredni czas jednej operacji [{unit}] - mean +/- stddev z powtorzen\n")
    print("N".ljust(6) + "".join(short_name(name).ljust(column) for name in names))

    for size in sizes:
        row = str(size).ljust(6)
        for name in names:
            stats = families[name].get(size, {})
            cell = "-"
            if "mean" in stats:
                cell = f"{stats['mean']:.3f} +/- {stats.get('stddev', 0.0):.3f}"
            row += cell.ljust(column)
        print(row)


def plot(families, unit, context, png_path):
    figure, (axis_time, axis_noise) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for family, by_size in families.items():
        sizes = sorted(by_size)
        label = short_name(family)

        means = [by_size[size].get("mean", 0.0) for size in sizes]
        stddevs = [by_size[size].get("stddev", 0.0) for size in sizes]
        # cv w JSON-ie to ulamek (stddev/mean), na wykresie chcemy procenty
        cvs = [by_size[size].get("cv", 0.0) * 100.0 for size in sizes]

        axis_time.errorbar(sizes, means, yerr=stddevs, marker="o", capsize=3, label=label)
        axis_noise.plot(sizes, cvs, marker="o", label=label)

    axis_time.set_title("Koszt jednej operacji w funkcji rozmiaru tablicy")
    axis_time.set_xlabel("Rozmiar tablicy N [liczba rekordow]")
    axis_time.set_ylabel(f"Czas - mean +/- stddev [{unit}]")
    axis_time.set_yscale("log")  # bez tego wariant o stalym koszcie zlewa sie z osia

    axis_noise.set_title("Powtarzalnosc pomiaru")
    axis_noise.set_xlabel("Rozmiar tablicy N [liczba rekordow]")
    axis_noise.set_ylabel("cv = stddev / mean [%]")

    for axis in (axis_time, axis_noise):
        axis.grid(True, which="both", linewidth=0.3, alpha=0.5)
        axis.legend()

    figure.suptitle(context_caption(context), fontsize=9, y=0.02)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(png_path, dpi=140)


def main(argv):
    if len(argv) != 3:
        print("uzycie: bench_plot.py <bench.json> <wykres.png>", file=sys.stderr)
        return 2

    json_path, png_path = argv[1], argv[2]
    families, unit, context = load_measurements(json_path)

    if not families:
        print(
            "bench_plot.py - brak agregatow z argumentem rozmiaru "
            "(potrzebne ->Range()/->DenseRange() razem z ->Repetitions())",
            file=sys.stderr,
        )
        return 1

    print_table(families, unit)
    plot(families, unit, context, png_path)
    print(f"\nWykres: {png_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
