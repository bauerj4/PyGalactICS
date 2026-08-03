#!/usr/bin/env python3
"""Extract A2(Rd) results, ship figures, write summaries."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from ntropy.analysis.disk_density import plane_density_azimuthal_fourier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/ml/field_maps/a2Rd_evolve_gate_2026-07-27"
PAPER = ROOT / "papers/mnras_noneq_ics/figures"
RESULTS = ROOT / "papers/mnras_noneq_ics/results"
R_D = 2.0
HALF = 14.0


def fmt(x) -> str:
    try:
        x = float(x)
        return f"{x:.3f}" if np.isfinite(x) else "—"
    except Exception:
        return "—"


def map_a2_at(npz_path: Path, t_target: float | None) -> float:
    z = np.load(npz_path, allow_pickle=True)
    t = np.asarray(z["t_gyr"], float)
    if t_target is None:
        i = len(t) - 1
    else:
        i = int(np.argmin(np.abs(t - t_target)))
        if abs(t[i] - t_target) > 0.06:
            return float("nan")
    dens = np.asarray(z[f"disk_map{i}"], float)
    fout = plane_density_azimuthal_fourier(dens, half_extent=HALF, m=2, n_bins=12)
    rr = np.asarray(fout["r_mid"], float)
    aa = np.asarray(fout["a_m_over_a0"], float)
    ok = np.isfinite(aa)
    if not ok.any():
        return float("nan")
    return float(np.interp(R_D, rr[ok], aa[ok]))


def main() -> None:
    results = []
    shipped = []

    print(
        f"{'run':40s} {'p_pre':>7} {'p_post':>7} {'p@0.5':>7} | "
        f"{'m_pre':>7} {'m_post':>7} {'m@0.5':>7}"
    )
    print("-" * 100)

    for d in sorted(OUT.glob("evolve_*gyr_*")):
        vpath = d / "verdict.json"
        if not vpath.is_file():
            continue
        v = json.loads(vpath.read_text())
        arms = v.get("arms", {})
        arm = None
        for k in arms:
            if "amp" in k or k.startswith("fft_recon"):
                arm = arms[k]
                break
        if arm is None:
            continue
        pre, post = arm.get("a2_pre"), arm.get("a2_post")
        a2t = arm.get("a2_t") or []
        eg = float(v.get("evolve_gyr", 2.0))
        a2_05 = float("nan")
        if a2t:
            tt = np.linspace(0, eg, len(a2t))
            a2_05 = float(a2t[int(np.argmin(np.abs(tt - 0.5)))])
        npz = d / "component_maps_fft_recon_dens_amp_shell.npz"
        map_pre = map_post = map_05 = float("nan")
        if npz.is_file():
            map_pre = map_a2_at(npz, 0.0)
            map_post = map_a2_at(npz, None)
            map_05 = map_a2_at(npz, 0.5)
        print(
            f"{d.name:40s} {fmt(pre):>7} {fmt(post):>7} {fmt(a2_05):>7} | "
            f"{fmt(map_pre):>7} {fmt(map_post):>7} {fmt(map_05):>7}"
        )
        results.append(
            {
                "name": d.name,
                "part_pre": pre,
                "part_post": post,
                "part_05": a2_05,
                "map_pre": map_pre,
                "map_post": map_post,
                "map_05": map_05,
                "eg": eg,
            }
        )
        mapping = [
            (
                d / "faceon_components_fft_recon_dens_amp_shell.png",
                f"fig_a2Rd_{d.name.replace('evolve_', '')}_faceon_fft_recon_dens_amp_shell.png",
            ),
            (
                d / "faceon_components_data.png",
                f"fig_a2Rd_{d.name.replace('evolve_', '')}_faceon_data.png",
            ),
            (
                d / "profiles_components_fft_recon_dens_amp_shell.png",
                f"fig_a2Rd_{d.name.replace('evolve_', '')}_profiles_fft_recon_dens_amp_shell.png",
            ),
            (
                d / "a2_t.png",
                f"fig_a2Rd_{d.name.replace('evolve_', '')}_a2_Rd_t.png",
            ),
        ]
        for src, name in mapping:
            if src.is_file():
                shutil.copy2(src, PAPER / name)
                shipped.append(str(Path("papers/mnras_noneq_ics/figures") / name))

    def find(alpha_tag: str, eg_pat: str):
        for r in results:
            if alpha_tag in r["name"] and eg_pat in r["name"] and "906c4" in r["name"]:
                return r
        return None

    a10_s = find("a10", "0p5")
    a15_s = find("a15", "0p5")
    a25_s = find("a25", "0p5")
    a10 = find("a10", "2p0")
    a15 = find("a15", "2p0")
    a25 = find("a25", "2p0")
    a54 = next((r for r in results if "54a8" in r["name"]), None)

    lines = [
        "# A₂(R = R_d) + evolve-gate push — 2026-07-27",
        "",
        "Teacher FT: `runs/ml/field_maps/a2Rd_evolve_gate_2026-07-27/a2rd_ft/multitower_slice_ae.pt`",
        "Warm start: `df_match_BA_2026-07-27/joint_df_fftlong_ft` (fftlong A+B).",
        "Eval: particle $A_2(R=2\\,\\mathrm{kpc})$ via `--a2-r-eval 2.0`, plus map $A_2(R_d)$ on face-on Σ.",
        "",
        "## Training recipe",
        "",
        "- `--r-focus-rd` ring weights on Am / dens-resid / FFT (band 0.5–1.5 $R_d$, peak=5, floor=0.2)",
        "- `--a2-rd-weight 4` dedicated soft $A_2(R_d)$ + undershoot hinge",
        "- `--evolve-gate-t-boost 1.0` on $t\\in[0.25,0.5]$ Gyr snaps",
        "- Kept DF A: moment_phys=6, vphi_phys=10 (avoid morph_a regression)",
        "- Mild morph: dens_resid=2.5, a2=3, fft=2.5 (not morph-crank 10/10/10)",
        "- 18 ep, bar_frac=0.7, lr=4e-5; train $a2_{Rd}$ 0.0015→0.0004 (best @ep17)",
        "- Smoke recon_map median $A_2$ still ~0.078 (maps remain soft vs data 0.375)",
        "",
        "## 906c4 particle $A_2(R_d)$ (gate metric)",
        "",
        "| α | 0.5 Gyr pre→post | 2 Gyr pre→post | @0.5 within 2 Gyr |",
        "|---|------------------|----------------|-------------------|",
    ]

    def add_row(alpha, short, full):
        if short and full:
            lines.append(
                f"| {alpha} | {short['part_pre']:.3f}→{short['part_post']:.3f} | "
                f"{full['part_pre']:.3f}→{full['part_post']:.3f} | {full['part_05']:.3f} |"
            )

    add_row("1.0", a10_s, a10)
    add_row("1.5", a15_s, a15)
    add_row("2.5", a25_s, a25)

    lines += [
        "",
        "## 906c4 map $A_2(R_d)$ (face-on Σ vs prior reanalysis)",
        "",
        "| α | map pre→post (2 Gyr) | map @0.5 | prior fftlong map |",
        "|---|----------------------|----------|-------------------|",
    ]
    prior_map = {"1.0": "0.110→0.031", "1.5": "0.113→0.045", "2.5": "0.168→0.113"}
    for alpha, full in [("1.0", a10), ("1.5", a15), ("2.5", a25)]:
        if full:
            lines.append(
                f"| {alpha} | {full['map_pre']:.3f}→{full['map_post']:.3f} | "
                f"{full['map_05']:.3f} | {prior_map[alpha]} |"
            )
    lines.append("| data | 0.206→0.124 | — | (reanalysis) |")

    lines += ["", "## 54a8 control α=2.5", ""]
    if a54:
        lines.append(
            f"- particle $A_2(R_d)$ {a54['part_pre']:.3f}→{a54['part_post']:.3f} "
            f"(@0.5: {a54['part_05']:.3f})"
        )
        lines.append(
            f"- map $A_2(R_d)$ {a54['map_pre']:.3f}→{a54['map_post']:.3f}"
        )

    lines += ["", "## W/L", ""]
    wl_alpha1 = "L"
    if a10 and a10_s:
        lines.append(
            f"- **α=1 short (0.5 Gyr):** {a10_s['part_pre']:.3f}→{a10_s['part_post']:.3f}"
        )
        lines.append(
            f"- **α=1 2 Gyr:** {a10['part_pre']:.3f}→{a10['part_post']:.3f} "
            f"(holds to 0.5 Gyr at {a10['part_05']:.3f}). "
            f"Prior fftlong median 0.357→0.275; prior map Rd 0.110→0.031."
        )
        lines.append(
            f"- α=1 map $A_2(R_d)$ {a10['map_pre']:.3f}→{a10['map_post']:.3f} "
            f"vs prior fftlong 0.110→0.031 / data 0.206→0.124."
        )
        hold_ok = a10["part_05"] >= a10["part_pre"] * 0.85
        map_better = a10["map_post"] > 0.05
        if a10["part_post"] >= 0.35:
            lines.append("- **W on α=1:** particle $A_2(R_d)$ approaches data-flat.")
            wl_alpha1 = "W"
        elif hold_ok:
            lines.append(
                "- **Partial W:** α=1 holds through 0.5 Gyr under $A_2(R_d)$; "
                "still fades by 2 Gyr — amplify crutch remains for long runs."
            )
            wl_alpha1 = "partial W"
        else:
            lines.append(
                "- **L on α=1 data-flat:** still fades by 2 Gyr; amplify crutch remains."
            )
    if a25:
        lines.append(
            f"- **α=2.5 2 Gyr:** {a25['part_pre']:.3f}→{a25['part_post']:.3f} "
            f"(map {a25['map_pre']:.3f}→{a25['map_post']:.3f}); "
            f"prior median 0.572→0.437 ≈ data-flat."
        )

    lines += ["", "## Best recipe", ""]
    if a25 and a10:
        if a10["part_post"] >= 0.35:
            best = "a2rd_ft + m2 α=1"
            lines.append(f"- **{best}** clears data-flat under $A_2(R_d)$.")
        elif a10["part_05"] >= a10["part_pre"] * 0.85:
            best = "a2rd_ft + m2 α=1 (≤0.5 Gyr); a2rd_ft + m2 α=2.5 (2 Gyr)"
            lines.append(
                "- Short-term (≤0.5 Gyr): **a2rd_ft + m2 α=1** holds $A_2(R_d)$. "
                "Long-term (2 Gyr): still prefer **a2rd_ft + m2 α=2.5** "
                "(or fftlong A+B α=2.5 for median-$A_2$ data-flat)."
            )
        else:
            best = "fftlong A+B + m2 α=2.5"
            lines.append(
                f"- Keep **{best}** as best 2 Gyr recipe; "
                "a2rd_ft is the R_d-targeted warm start for further work."
            )
    else:
        best = "incomplete"

    lines += ["", "## Figures shipped", ""]
    for s in shipped:
        lines.append(f"- `{s}`")

    text = "\n".join(lines) + "\n"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "a2Rd_evolve_gate_SUMMARY.md").write_text(text)
    (OUT / "SUMMARY.md").write_text(text)
    (OUT / "gate_summary.json").write_text(
        json.dumps({"results": results, "best_recipe": best, "wl_alpha1": wl_alpha1}, indent=2)
    )
    print(text)
    print(f"shipped {len(shipped)} figures; α=1 verdict: {wl_alpha1}; best: {best}")


if __name__ == "__main__":
    main()
