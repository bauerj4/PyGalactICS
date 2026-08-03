#!/usr/bin/env python3
"""Draw latent-θ architecture panels A / B / C (standalone scientific diagrams).

Outputs under ``papers/mnras_noneq_ics/figures/``:

* ``fig_latent_theta_architecture_A.png`` — corpus encode walkthrough (concrete)
* ``fig_latent_theta_corpus_encode.png`` — same Panel A (alias / main-fig name)
* ``fig_latent_theta_architecture_B.png`` — sample IC at fixed θ
* ``fig_latent_theta_architecture_C.png`` — evaluate

An optional compact overview is written as ``fig_latent_theta_architecture.png``
(superseded as a primary; A/B/C are the live story panels).

Panel A uses a real corpus barred dump (906c4 late ``step_003200``) for face-on
dens + multi-component field-map thumbnails, a schematic SliceUNet encode path,
and the library PCA code for that dump.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers/mnras_noneq_ics/figures"
CORPUS = ROOT / "runs/mw_morton_corpus_v2"
EXAMPLE_DUMP = CORPUS / "906c4af73543/evolution/particles/step_003200.npz"
LIB_CODES = ROOT / "runs/ml/field_maps/latent_theta_gen_2026-08-02/feature_library_codes.npz"
EXAMPLE_HASH = "906c4af73543"
EXAMPLE_RD_KPC = 2.0

# ML-paper clean: white bg, black/gray boxes, one muted teal accent.
INK = "#1a1a1a"
MUTED = "#5a5a5a"
FILL = "#fafafa"
FILL2 = "#f0f0ee"
ACCENT = "#2f6f6a"
ACCENT_FILL = "#e8f1ef"
EDGE = "#2a2a2a"
SOFT = "#d8d8d4"
CMAP = "inferno"


def _box(
    ax,
    xy,
    w,
    h,
    text,
    *,
    fc=FILL,
    ec=EDGE,
    lw=1.25,
    fontsize=10,
    weight="normal",
    sub=None,
    subsize=8.0,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.028",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    cy = y + h / 2
    if sub:
        ax.text(
            x + w / 2,
            cy + 0.012,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=INK,
            fontweight=weight,
        )
        ax.text(
            x + w / 2,
            cy - 0.055,
            sub,
            ha="center",
            va="center",
            fontsize=subsize,
            color=MUTED,
        )
    else:
        ax.text(
            x + w / 2,
            cy,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=INK,
            fontweight=weight,
        )
    return (x + w / 2, y + h / 2, x, y, w, h)


def _arrow(ax, p0, p1, *, color=MUTED, rad=0.0, lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
        )
    )


def _title(ax, letter: str, title: str, subtitle: str) -> None:
    ax.text(
        0.02,
        0.94,
        letter,
        ha="left",
        va="center",
        fontsize=22,
        color=ACCENT,
        fontweight="bold",
        fontfamily="serif",
    )
    ax.text(
        0.08,
        0.955,
        title,
        ha="left",
        va="center",
        fontsize=13.5,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.08,
        0.905,
        subtitle,
        ha="left",
        va="center",
        fontsize=9.5,
        color=MUTED,
        style="italic",
    )
    ax.add_patch(Rectangle((0.02, 0.86), 0.96, 0.003, color=SOFT, lw=0))


def _footer(ax, text: str) -> None:
    ax.text(0.02, 0.035, text, ha="left", va="center", fontsize=8, color=MUTED)


def _save(fig, stem: str) -> Path:
    PAPER.mkdir(parents=True, exist_ok=True)
    out = PAPER / f"{stem}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def _log10_norm(img: np.ndarray, pct: float = 98.0) -> np.ndarray:
    """Normalize log10(Σ+ε) into [0, 1] for architecture thumbnails."""
    from galacticsics.campaign.analysis import dens_array_log10

    show, vmin_s, vmax_s, _ = dens_array_log10(img, vmax_pct=pct)
    return np.clip((show - vmin_s) / (vmax_s - vmin_s + 1e-12), 0.0, 1.0)


def _hist2d(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    nbin: int,
    half: float,
) -> np.ndarray:
    H, _, _ = np.histogram2d(
        x,
        y,
        bins=nbin,
        range=[[-half, half], [-half, half]],
        weights=w,
    )
    return H.T.astype(np.float64)


def _load_example_maps(
    dump: Path = EXAMPLE_DUMP,
    *,
    nbin: int = 96,
    half_disk: float = 12.0,
    half_bh: float = 20.0,
) -> dict:
    """Deposit face-on dens (+ disk σ_z) for the example dump in global COM frame."""
    from galacticsics.ml.fields.frame import prepare_shared_frame
    from galacticsics.ml.morton.polygon import _component_ids
    from ntropy.analysis.disk_density import disk_azimuthal_fourier

    with np.load(dump, allow_pickle=True) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        cid = _component_ids(data.get("tags"), data.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)

    maps: dict[str, np.ndarray] = {}
    for name, k, half in (
        ("disk_dens", 0, half_disk),
        ("bulge_dens", 2, half_bh),
        ("halo_dens", 1, half_bh),
    ):
        m = cid == k
        maps[name] = _hist2d(pos[m, 0], pos[m, 1], mass[m], nbin=nbin, half=half)

    # Disk midplane |z|<0.5 σ_z moment channel (compact kin cue).
    m = (cid == 0) & (np.abs(pos[:, 2]) < 0.5)
    w = mass[m]
    vz = vel[m, 2]
    dens = _hist2d(pos[m, 0], pos[m, 1], w, nbin=nbin, half=half_disk)
    mom1 = _hist2d(pos[m, 0], pos[m, 1], w * vz, nbin=nbin, half=half_disk)
    mom2 = _hist2d(pos[m, 0], pos[m, 1], w * vz * vz, nbin=nbin, half=half_disk)
    mean = np.divide(mom1, dens, out=np.zeros_like(mom1), where=dens > 0)
    mean2 = np.divide(mom2, dens, out=np.zeros_like(mom2), where=dens > 0)
    maps["disk_sigz"] = np.sqrt(np.clip(mean2 - mean * mean, 0.0, None))

    disk = cid == 0
    a2 = disk_azimuthal_fourier(
        pos[disk],
        mass[disk],
        m=2,
        r_max=12.0,
        n_bins=24,
        z_max=0.5,
        min_count=10,
        recenter=True,
    )
    R = np.asarray(a2["r_mid"], dtype=np.float64)
    am = np.asarray(a2["a_m_over_a0"], dtype=np.float64)
    i_rd = int(np.argmin(np.abs(R - EXAMPLE_RD_KPC)))
    a2_rd = float(am[i_rd])

    z = None
    lib_a2 = None
    if LIB_CODES.is_file():
        lib = np.load(LIB_CODES, allow_pickle=True)
        paths = np.asarray(lib["paths"])
        hit = [
            i
            for i, p in enumerate(paths)
            if EXAMPLE_HASH in str(p) and dump.name in str(p)
        ]
        if hit:
            z = np.asarray(lib["codes"][hit[0]], dtype=np.float64)
            lib_a2 = float(lib["a2"][hit[0]])

    return {
        "maps": maps,
        "a2_rd": a2_rd,
        "a2_median": float(a2["a_m_over_a0_median"]),
        "lib_a2": lib_a2,
        "z": z,
        "dump": dump,
        "hash": EXAMPLE_HASH,
        "step": dump.stem,
    }


def _imshow_thumb(ax, img: np.ndarray, *, cmap=CMAP) -> None:
    show = _log10_norm(img, pct=98.0)
    ax.imshow(show, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(EDGE)
        spine.set_linewidth(0.8)


def _block_pool(img: np.ndarray, n: int) -> np.ndarray:
    """Average-pool / resize ``img`` onto an ``n×n`` grid (encode-stage proxy).

    Prefer exact block averaging when ``h`` and ``w`` are divisible by ``n``.
    Otherwise resize the *full* FOV (never top-left crop): a 96→64 crop was
    shifting E₁ off-center relative to the face-on disk dens thumbnail.
    """
    from scipy.ndimage import zoom

    x = np.asarray(img, dtype=np.float64)
    h, w = x.shape
    if h == n and w == n:
        return x
    if h % n == 0 and w % n == 0:
        return x.reshape(n, h // n, n, w // n).mean(axis=(1, 3))
    return zoom(x, (n / float(h), n / float(w)), order=1)


def _downsample_pyramid(img: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    return [_block_pool(img, n) for n in sizes]


def _flow_arrow(
    ax,
    x0: float,
    x1: float,
    y: float,
    *,
    color=MUTED,
    lw: float = 1.35,
    head: float = 10.0,
) -> None:
    """Visible horizontal connector on the content spine (shaft + head, no shrink)."""
    if x1 - x0 < 0.006:
        return
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=head,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
            capstyle="butt",
            zorder=3,
        )
    )


def _spine_seg(ax, x0: float, x1: float, y: float, *, color=MUTED, lw: float = 1.05) -> None:
    """Short intra-block spine segment (no arrowhead — avoids floating tips)."""
    if x1 <= x0:
        return
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=3, clip_on=False)


def _inset(ax, x: float, y: float, w: float, h: float):
    return inset_axes(
        ax,
        width="100%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(x, y, w, h),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )


def draw_A() -> Path:
    """A — Concrete corpus encode: example galaxy → maps → U-Net → PCA z → index."""
    ex = _load_example_maps()
    maps = ex["maps"]
    z = ex["z"]
    if z is None:
        z = np.zeros(64, dtype=np.float64)
        z[:8] = [-0.35, 0.37, -1.07, 0.22, 0.0, 0.26, -0.17, 0.43]

    fig_w, fig_h = 14.2, 4.65
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _title(
        ax,
        "A",
        r"Build latent library — encode one corpus galaxy",
        rf"Example {ex['hash'][:5]} late ({ex['step']}) · "
        r"dump $\rightarrow$ field maps $X_k$ $\rightarrow$ SliceUNet $\rightarrow$ "
        r"pool $u$ $\rightarrow$ PCA $z$ $\rightarrow$ index",
    )

    fig_aspect = fig_w / fig_h

    def _sq_h(w: float) -> float:
        return w * fig_aspect

    # Shared content spine — every visual centers here for continuous wiring.
    spine = 0.450
    wire = 0.030

    stage_y = 0.822

    # ---- Geometry (L→R) ----
    # Slightly closer map sizes so the spine reads as one band.
    w1 = 0.092
    h1 = _sq_h(w1)
    x1 = 0.014
    y1 = spine - h1 / 2

    strip_specs = (
        ("disk dens", maps["disk_dens"]),
        (r"disk $\sigma_z$", maps["disk_sigz"]),
        ("bulge dens", maps["bulge_dens"]),
        ("halo dens", maps["halo_dens"]),
    )
    sw, sg = 0.044, 0.005
    sh = _sq_h(sw)
    strip_w = 4 * sw + 3 * sg
    map_pad_x, map_pad_y = 0.008, 0.034
    gx0 = x1 + w1 + wire
    strip_x0 = gx0 + map_pad_x
    strip_y = spine - sh / 2
    frame_w = strip_w + 2 * map_pad_x
    frame_h = sh + 2 * map_pad_y
    frame_y = spine - frame_h / 2

    pyramid = _downsample_pyramid(maps["disk_dens"], [64, 32, 16, 8])
    stage_labs = [r"$E_1$", r"$E_2$", r"$E_3$", r"$B$"]
    stage_subs = ["64²", "32²", "16²", "bottleneck"]
    pw, pg = 0.042, 0.014
    ph = _sq_h(pw)
    n_st = 4
    enc_inner_w = n_st * pw + (n_st - 1) * pg
    enc_pad_x, enc_pad_top, enc_pad_bot = 0.010, 0.072, 0.046
    ux0 = gx0 + frame_w + wire
    uw = enc_inner_w + 2 * enc_pad_x
    uh = ph + enc_pad_top + enc_pad_bot
    uy0 = spine - ph / 2 - enc_pad_bot

    # Stage 4: pool + z inside one light frame (horizontal PCA link).
    pool_w, z_w = 0.088, 0.090
    pca_gap = 0.032
    s4_pad = 0.010
    stage4_inner_h = max(uh * 0.88, ph + 0.06)
    stage4_h = stage4_inner_h + 2 * s4_pad
    s4x = ux0 + uw + wire
    s4w = pool_w + pca_gap + z_w + 2 * s4_pad
    s4y = spine - stage4_h / 2
    px = s4x + s4_pad
    zx = px + pool_w + pca_gap
    pool_h = stage4_inner_h
    pool_y = spine - pool_h / 2
    z_plot_h = stage4_inner_h - 0.042
    z_plot_y = spine - z_plot_h / 2 + 0.006

    ix = s4x + s4w + wire
    iw = max(0.095, 0.985 - ix)
    ih = stage4_h
    iy = spine - ih / 2

    for x, lab in (
        (x1 + w1 / 2, "1 · Example dump"),
        (gx0 + frame_w / 2, r"2 · Field maps $X_k$"),
        (ux0 + uw / 2, "3 · Teacher SliceUNet encode"),
        (s4x + s4w / 2, r"4 · Pool $\rightarrow$ PCA $z$"),
        (ix + iw / 2, "5 · Index"),
    ):
        ax.text(x, stage_y, lab, ha="center", va="center", fontsize=7.4, color=MUTED)

    # ---- 1. Face-on ----
    _imshow_thumb(_inset(ax, x1, y1, w1, h1), maps["disk_dens"])
    ax.text(
        x1 + w1 / 2,
        y1 + h1 + 0.009,
        "disk dens · face-on",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=INK,
    )
    ax.text(
        x1 + w1 / 2,
        y1 - 0.009,
        rf"$A_2(R_d)={ex['a2_rd']:.2f}$ (disk-COM)",
        ha="center",
        va="top",
        fontsize=6.7,
        color=MUTED,
    )

    # ---- 2. Field maps ----
    ax.add_patch(
        FancyBboxPatch(
            (gx0, frame_y),
            frame_w,
            frame_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.9,
            edgecolor=SOFT,
            facecolor=FILL,
            zorder=0,
        )
    )
    ax.text(
        gx0 + frame_w / 2,
        frame_y + frame_h - 0.010,
        "shared COM · dens + vel moments",
        ha="center",
        va="center",
        fontsize=6.1,
        color=MUTED,
        style="italic",
    )
    for i, (lab, img) in enumerate(strip_specs):
        x = strip_x0 + i * (sw + sg)
        _imshow_thumb(_inset(ax, x, strip_y, sw, sh), img)
        ax.text(x + sw / 2, strip_y - 0.006, lab, ha="center", va="top", fontsize=5.9, color=MUTED)

    # ---- 3. U-Net ----
    ax.add_patch(
        FancyBboxPatch(
            (ux0, uy0),
            uw,
            uh,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            linewidth=1.35,
            edgecolor=ACCENT,
            facecolor=ACCENT_FILL,
            zorder=0,
        )
    )
    ax.text(
        ux0 + uw / 2,
        uy0 + uh - 0.018,
        "multitower SliceUNet",
        ha="center",
        va="center",
        fontsize=7.6,
        color=ACCENT,
        fontweight="bold",
    )
    ax.text(
        ux0 + uw / 2,
        uy0 + uh - 0.040,
        "disk tower · skips carry bar",
        ha="center",
        va="center",
        fontsize=6.0,
        color=MUTED,
        style="italic",
    )

    px0 = ux0 + enc_pad_x
    py = spine - ph / 2
    stage_centers: list[float] = []
    for i, (img, lab, sub) in enumerate(zip(pyramid, stage_labs, stage_subs)):
        x = px0 + i * (pw + pg)
        _imshow_thumb(_inset(ax, x, py, pw, ph), img)
        ax.text(
            x + pw / 2,
            py - 0.006,
            lab,
            ha="center",
            va="top",
            fontsize=7.3,
            color=INK,
            fontweight="bold",
        )
        ax.text(x + pw / 2, py - 0.030, sub, ha="center", va="top", fontsize=5.7, color=MUTED)
        stage_centers.append(x + pw / 2)
        if i < n_st - 1:
            _spine_seg(ax, x + pw + 0.002, x + pw + pg - 0.002, spine, color=MUTED, lw=1.25)

    # Skip cue just above the thumbs (below the subtitle).
    skip_y = py + ph + 0.008
    ax.annotate(
        "",
        xy=(stage_centers[-1], skip_y),
        xytext=(stage_centers[0], skip_y),
        arrowprops=dict(
            arrowstyle="-",
            color=ACCENT,
            lw=0.85,
            ls=(0, (2.5, 2.0)),
            connectionstyle="arc3,rad=0.28",
            alpha=0.65,
        ),
    )
    ax.text(
        stage_centers[-1] + 0.012,
        skip_y + 0.018,
        "skips",
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=ACCENT,
        alpha=0.90,
    )

    # ---- 4. Pool → PCA → z (one frame) ----
    ax.add_patch(
        FancyBboxPatch(
            (s4x, s4y),
            s4w,
            stage4_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.9,
            edgecolor=SOFT,
            facecolor=FILL,
            zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (px, pool_y),
            pool_w,
            pool_h,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=1.1,
            edgecolor=ACCENT,
            facecolor=ACCENT_FILL,
            zorder=1,
        )
    )
    ax.text(
        px + pool_w / 2,
        pool_y + pool_h - 0.024,
        r"pool $u$",
        ha="center",
        va="center",
        fontsize=7.6,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        px + pool_w / 2,
        spine + 0.004,
        "AdaptiveAvgPool\n+ concat towers",
        ha="center",
        va="center",
        fontsize=5.6,
        color=MUTED,
    )
    ax.text(
        px + pool_w / 2,
        pool_y + 0.020,
        r"$u\!\in\!\mathbb{R}^{9216}$",
        ha="center",
        va="center",
        fontsize=5.9,
        color=MUTED,
    )

    _flow_arrow(ax, px + pool_w + 0.003, zx - 0.003, spine, color=ACCENT, lw=1.25, head=9.0)
    ax.text(
        0.5 * (px + pool_w + zx),
        spine + 0.026,
        "PCA",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color=ACCENT,
        fontweight="bold",
    )

    zax = _inset(ax, zx, z_plot_y, z_w, z_plot_h)
    n_show = 16
    zz = z[:n_show]
    zax.axhline(0, color=SOFT, lw=0.6)
    zax.vlines(np.arange(n_show), 0, zz, color=ACCENT, lw=1.05)
    zax.plot(np.arange(n_show), zz, "o", ms=1.7, color=INK)
    zax.set_xlim(-0.5, n_show - 0.5)
    ymax = max(0.4, float(np.max(np.abs(zz))) * 1.15)
    zax.set_ylim(-ymax, ymax)
    zax.set_xticks([])
    zax.set_yticks([])
    for sp in zax.spines.values():
        sp.set_color(EDGE)
        sp.set_linewidth(0.7)
    ax.text(
        zx + z_w / 2,
        z_plot_y - 0.006,
        r"$z\in\mathbb{R}^{64}$",
        ha="center",
        va="top",
        fontsize=6.0,
        color=MUTED,
    )

    # ---- 5. Index ----
    ax.add_patch(
        FancyBboxPatch(
            (ix, iy),
            iw,
            ih,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=FILL2,
        )
    )
    ax.text(
        ix + iw / 2,
        iy + ih - 0.024,
        "library entry",
        ha="center",
        va="center",
        fontsize=7.8,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        ix + iw / 2,
        spine + 0.008,
        rf"$(z,\,A_2,\,\theta)$"
        "\n\n"
        rf"$A_2(R_d)={ex['a2_rd']:.2f}$"
        "\n"
        rf"$\theta$: {ex['hash'][:5]}…"
        "\n"
        rf"$z_{{1:3}}=[{z[0]:+.2f}$,"
        "\n"
        rf"${z[1]:+.2f},{z[2]:+.2f}]$",
        ha="center",
        va="center",
        fontsize=6.4,
        color=MUTED,
    )
    ax.text(
        ix + iw / 2,
        iy + 0.018,
        "disk-COM $A_2$",
        ha="center",
        va="center",
        fontsize=5.9,
        color=MUTED,
    )

    # ---- Continuous L→R spine rail + stage arrows ----
    # Full rail behind content (low z) so gaps never look empty; arrowheads mark hops.
    rail_x0 = x1 + w1 + 0.002
    rail_x1 = ix - 0.002
    ax.plot(
        [rail_x0, rail_x1],
        [spine, spine],
        color=SOFT,
        lw=1.6,
        solid_capstyle="butt",
        zorder=0.5,
        clip_on=False,
    )
    # Accent rail through the encode→pool hop.
    ax.plot(
        [ux0 + uw + 0.002, s4x - 0.002],
        [spine, spine],
        color=ACCENT,
        lw=1.7,
        solid_capstyle="butt",
        zorder=0.6,
        alpha=0.55,
        clip_on=False,
    )
    _flow_arrow(ax, x1 + w1 + 0.004, gx0 - 0.004, spine, color=MUTED, lw=1.35, head=11.0)
    _flow_arrow(ax, gx0 + frame_w + 0.004, ux0 - 0.004, spine, color=MUTED, lw=1.35, head=11.0)
    _flow_arrow(ax, ux0 + uw + 0.004, s4x - 0.004, spine, color=ACCENT, lw=1.45, head=11.0)
    _flow_arrow(ax, s4x + s4w + 0.004, ix - 0.004, spine, color=MUTED, lw=1.35, head=11.0)

    ax.annotate(
        "",
        xy=(ix + iw * 0.55, 0.068),
        xytext=(x1 + w1 * 0.35, 0.068),
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=9),
    )
    ax.text(
        0.50,
        0.080,
        r"many corpus systems $\longrightarrow$ stratified library index",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
        style="italic",
    )

    _footer(
        ax,
        rf"Real dump {ex['dump'].relative_to(ROOT)} · "
        r"$z$ from feature_library_codes.npz · "
        r"stage thumbs = dens resolution pyramid (encode schematic) · "
        r"bars live mostly in U-Net skips",
    )

    out_a = _save(fig, "fig_latent_theta_architecture_A")
    for suf in (".png", ".pdf"):
        shutil.copy2(out_a.with_suffix(suf), PAPER / f"fig_latent_theta_corpus_encode{suf}")
    print(f"wrote {PAPER / 'fig_latent_theta_corpus_encode'}.png", flush=True)
    return out_a


def draw_B() -> Path:
    """B — Sample IC at fixed θ (θ → f0; A₂/z target → retrieve → particle_retrieve)."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _title(
        ax,
        "B",
        r"Sample IC at fixed $\theta$",
        r"Condition on mass model $\theta$; retrieve by $A_2$ tier or path-LOO $z$; remass to $f_0(\theta)$",
    )

    # Top rail: conditioning
    y1, h1 = 0.55, 0.22
    b_th = _box(
        ax,
        (0.03, y1),
        0.14,
        h1,
        r"Choose $\theta$",
        sub="structural\nmass model",
        fontsize=10.5,
        weight="bold",
    )
    b_f0 = _box(
        ax,
        (0.21, y1),
        0.18,
        h1,
        r"GalactICS $f_0(\theta)$",
        sub="quiet equilibrium base\nmass totals + DF",
        fontsize=10.5,
    )
    b_tgt = _box(
        ax,
        (0.43, y1),
        0.22,
        h1,
        r"$A_2$ / $z$ target",
        sub="quiet→mild→mod→strong\nor path-LOO neighbor",
        fontsize=10.5,
        fc=ACCENT_FILL,
        ec=ACCENT,
        lw=1.6,
    )
    b_lib = _box(
        ax,
        (0.69, y1),
        0.14,
        h1,
        "Library",
        sub=r"index from A" "\n$(z,A_2)\\mid\\theta$",
        fontsize=10.5,
    )
    b_ret = _box(
        ax,
        (0.86, y1),
        0.11,
        h1,
        "Retrieve",
        sub="same-θ nn\nno eval copy",
        fontsize=10,
        fc=ACCENT_FILL,
        ec=ACCENT,
        lw=1.6,
    )

    for a, b in ((b_th, b_f0), (b_f0, b_tgt), (b_tgt, b_lib), (b_lib, b_ret)):
        _arrow(ax, (a[0] + a[4] / 2 - 0.005, a[1]), (b[0] - b[4] / 2 + 0.005, b[1]), color=INK)

    # Bottom rail: decode
    y2, h2 = 0.14, 0.26
    b_dec = _box(
        ax,
        (0.18, y2),
        0.28,
        h2,
        "particle_retrieve + remass",
        sub=r"load neighbor particles · scale masses $\to f_0(\theta)$" "\noptional LOO disk-moment match",
        fontsize=10.5,
        weight="bold",
    )
    b_ic = _box(
        ax,
        (0.56, y2),
        0.26,
        h2,
        "Particle IC",
        sub="all components · non-eq morph/kin\nlocked to $f_0(\\theta)$ masses",
        fontsize=10.5,
        weight="bold",
        fc=FILL2,
    )

    _arrow(ax, (b_ret[0], b_ret[1] - b_ret[5] / 2), (b_dec[0] + 0.06, b_dec[1] + b_dec[5] / 2), color=ACCENT, rad=0.12)
    _arrow(ax, (b_f0[0], b_f0[1] - b_f0[5] / 2), (b_dec[0] - 0.02, b_dec[1] + b_dec[5] / 2), color=MUTED, rad=-0.08)
    ax.text(0.16, 0.42, r"remass targets", fontsize=7.5, color=MUTED, ha="center", rotation=55)
    ax.text(0.78, 0.42, "query", fontsize=7.5, color=ACCENT, ha="center")
    _arrow(ax, (b_dec[0] + b_dec[4] / 2 - 0.01, b_dec[1]), (b_ic[0] - b_ic[4] / 2 + 0.01, b_ic[1]), color=INK)

    ax.text(
        0.03,
        0.08,
        r"Bar-strength sweep lives here: dial $A_2(R_d)$ target at fixed $\theta$, then retrieve.",
        fontsize=8,
        color=MUTED,
        va="center",
    )
    return _save(fig, "fig_latent_theta_architecture_B")


def draw_C() -> Path:
    """C — Evaluate (t=0 dens+kin, evolve gate, verdict)."""
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _title(
        ax,
        "C",
        "Evaluate",
        r"$t{=}0$ dens+kin gates $\rightarrow$ evolve $A_2(R_d;t)$ $\rightarrow$ MATCH / FADE MATCH / OOD",
    )

    y, h = 0.28, 0.42
    b_ic = _box(
        ax,
        (0.04, y),
        0.16,
        h,
        "Particle IC",
        sub="from panel B",
        fontsize=10.5,
        weight="bold",
        fc=FILL2,
    )
    b_t0 = _box(
        ax,
        (0.26, y),
        0.22,
        h,
        r"$t{=}0$ dens + kin",
        sub="vs dump (path-LOO)\nand vs quiet $f_0$",
        fontsize=10.5,
    )
    b_ev = _box(
        ax,
        (0.53, y),
        0.22,
        h,
        r"Evolve gate",
        sub=r"$A_2(R_d;t)$, COM drift" "\ngpu_bh · disk $=10^6$",
        fontsize=10.5,
    )
    b_ver = _box(
        ax,
        (0.80, y),
        0.16,
        h,
        "Verdict",
        sub="MATCH /\nFADE MATCH /\nOOD transfer",
        fontsize=10.5,
        weight="bold",
        fc=ACCENT_FILL,
        ec=ACCENT,
        lw=1.6,
    )

    _arrow(ax, (b_ic[0] + b_ic[4] / 2 - 0.005, b_ic[1]), (b_t0[0] - b_t0[4] / 2 + 0.005, b_t0[1]), color=INK)
    _arrow(ax, (b_t0[0] + b_t0[4] / 2 - 0.005, b_t0[1]), (b_ev[0] - b_ev[4] / 2 + 0.005, b_ev[1]), color=INK)
    _arrow(ax, (b_ev[0] + b_ev[4] / 2 - 0.005, b_ev[1]), (b_ver[0] - b_ver[4] / 2 + 0.005, b_ver[1]), color=INK)
    _arrow(
        ax,
        (b_ic[0], b_ic[1] - b_ic[5] / 2 + 0.02),
        (b_ev[0] - 0.04, b_ev[1] - b_ev[5] / 2 + 0.02),
        color=MUTED,
        rad=0.18,
        lw=1.0,
    )

    _footer(
        ax,
        r"$A_2(R_d)$ uses disk mass-COM recenter · ID gates $T{=}2\,\mathrm{Gyr}$; OOD coherence $T{=}1\,\mathrm{Gyr}$; sweep $0.5\,\mathrm{Gyr}$",
    )
    return _save(fig, "fig_latent_theta_architecture_C")


def draw_overview_stub() -> Path:
    """Compact A→B→C overview (optional companion; A/B/C panels are primary)."""
    fig, ax = plt.subplots(figsize=(9.5, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.88,
        r"Latent-$\theta$ pipeline (overview)",
        ha="center",
        va="center",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.74,
        r"library retrieve+decode — not free continuous $\mathrm{decode}(z)$",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    y, h = 0.18, 0.42
    ba = _box(ax, (0.06, y), 0.24, h, "A  Build library", sub=r"maps → AE → PCA $z$", fontsize=11, weight="bold", fc=ACCENT_FILL, ec=ACCENT)
    bb = _box(ax, (0.38, y), 0.24, h, r"B  Sample IC$\,\mid\,\theta$", sub=r"$A_2$/$z$ retrieve + remass", fontsize=11, weight="bold", fc=ACCENT_FILL, ec=ACCENT)
    bc = _box(ax, (0.70, y), 0.24, h, "C  Evaluate", sub=r"$t{=}0$ + evolve gate", fontsize=11, weight="bold", fc=FILL2)

    _arrow(ax, (ba[0] + ba[4] / 2 - 0.005, ba[1]), (bb[0] - bb[4] / 2 + 0.005, bb[1]), color=INK)
    _arrow(ax, (bb[0] + bb[4] / 2 - 0.005, bb[1]), (bc[0] - bc[4] / 2 + 0.005, bc[1]), color=INK)
    return _save(fig, "fig_latent_theta_architecture")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        choices=("a", "b", "c", "overview", "all"),
        default="all",
        help="Which panel(s) to draw (default: all).",
    )
    args = p.parse_args()
    if args.only in ("a", "all"):
        draw_A()
    if args.only in ("b", "all"):
        draw_B()
    if args.only in ("c", "all"):
        draw_C()
    if args.only in ("overview", "all"):
        draw_overview_stub()


if __name__ == "__main__":
    main()
