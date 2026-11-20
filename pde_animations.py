import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.special import erf

OUT_DIR = Path("animations")


def configure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def save_mp4(anim, out_path, fps=30):
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
    )
    anim.save(str(out_path), writer=writer)


# =============================================================================
# Initial Profiles (compactly supported on [-1, 1]) & Antiderivatives
# =============================================================================
def tent(x):
    return np.where(np.abs(x) < 1.0, 1.0 - np.abs(x), 0.0)


def tent_antideriv(u):
    return np.where(
        u <= -1.0,
        0.0,
        np.where(
            u <= 0.0,
            0.5 * (u + 1.0) ** 2,
            np.where(u < 1.0, 1.0 - 0.5 * (1.0 - u) ** 2, 1.0),
        ),
    )


def boxcar(x):
    return np.where(np.abs(x) < 1.0, 1.0, np.where(np.abs(x) == 1.0, 0.5, 0.0))


def twin_peak(x):
    # Smooth two-peaked profile on [-1, 1] with max = 1 and total mass = 1 (same as tent)
    return np.where(np.abs(x) < 1.0, 0.5 * (1.0 - np.cos(2.0 * np.pi * x)), 0.0)


# =============================================================================
# PDE Solvers
# =============================================================================
def wave_grid(phi, x, t_vals, c=1.0, psi_antideriv=None, psi_scale=0.0):
    X, T = np.meshgrid(x, t_vals)
    Z = 0.5 * (phi(X - c * T) + phi(X + c * T))
    if psi_antideriv is not None and psi_scale != 0.0:
        Z += (psi_scale / (2.0 * c)) * (
            psi_antideriv(X + c * T) - psi_antideriv(X - c * T)
        )
    return Z


def trapz_last(vals, dx):
    return dx * (
        0.5 * vals[..., 0] + np.sum(vals[..., 1:-1], axis=-1) + 0.5 * vals[..., -1]
    )


def heat_grid(phi, x, t_vals, k=1.0):
    Z = np.empty((len(t_vals), len(x)))
    p = np.linspace(-5.0, 5.0, 400)
    dp = p[1] - p[0]
    s = np.linspace(-1.0, 1.0, 400)
    ds = s[1] - s[0]
    phi_s = phi(s)

    for i, t in enumerate(t_vals):
        if t == 0.0:
            Z[i] = phi(x)
        elif 4.0 * k * t < 0.04:
            # Strauss Sec. 3.5 substitution: avoids 1/sqrt(t) singularity near t = 0
            shifted = x[:, None] - 2.0 * np.sqrt(k * t) * p[None, :]
            integrand = np.exp(-(p[None, :] ** 2)) * phi(shifted)
            Z[i] = trapz_last(integrand, dp) / np.sqrt(np.pi)
        else:
            kernel = np.exp(-((x[:, None] - s[None, :]) ** 2) / (4.0 * k * t))
            Z[i] = trapz_last(kernel * phi_s[None, :], ds) / (
                2.0 * np.sqrt(np.pi * k * t)
            )
    return Z


def boxcar_heat_grid(x, t_vals, k=1.0):
    # Exact analytical convolution of the boxcar on [-1, 1] with the heat kernel
    Z = np.empty((len(t_vals), len(x)))
    for i, t in enumerate(t_vals):
        if t == 0.0:
            Z[i] = boxcar(x)
        else:
            denom = 2.0 * np.sqrt(k * t)
            Z[i] = 0.5 * (erf((x + 1.0) / denom) - erf((x - 1.0) / denom))
    return Z


# =============================================================================
# Animation Renderers (all figsize * dpi give even pixel dimensions for H.264)
# =============================================================================
def render_slice_3d(Z, x, t_vals, title, out_path, z_max=1.15, fps=30):
    X, T = np.meshgrid(x, t_vals)

    fig = plt.figure(figsize=(11, 4.8), dpi=150)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    ax3d.plot_surface(
        X, T, Z, cmap="viridis", alpha=0.75, rstride=2, cstride=2, edgecolor="none"
    )
    cx, cy = np.meshgrid([x[0], x[-1]], [t_vals[0], t_vals[-1]])
    ax3d.plot_surface(
        cx, cy, np.ones_like(cx), color="0.4", alpha=0.14, edgecolor="none"
    )
    ax3d.set(
        xlim=(x[0], x[-1]),
        ylim=(t_vals[0], t_vals[-1]),
        zlim=(-0.05, z_max),
        xlabel="x (space)",
        ylabel="t (time)",
        zlabel="u(x, t)",
        title=f"{title}: Space-Time Surface",
    )
    ax3d.view_init(elev=26, azim=-58)

    (slice_3d,) = ax3d.plot(
        x, np.full_like(x, t_vals[0]), Z[0], color="crimson", lw=2.4, zorder=10
    )
    px, pz = np.meshgrid([x[0], x[-1]], [0.0, z_max])
    plane_ref = [
        ax3d.plot_surface(
            px, np.full_like(px, t_vals[0]), pz, color="crimson", alpha=0.14
        )
    ]

    (slice_2d,) = ax2d.plot(x, Z[0], color="crimson", lw=2.4)
    ax2d.axhline(1.0, color="0.4", ls="--", lw=1.0, label=r"$\max \phi = 1$")
    time_text = ax2d.text(
        0.04, 0.90, f"t = {t_vals[0]:.2f}", transform=ax2d.transAxes, fontsize=11
    )
    ax2d.set(
        xlim=(x[0], x[-1]),
        ylim=(-0.05, z_max),
        xlabel="x",
        ylabel="u(x, t)",
        title="2D Cross-Section at t",
    )
    ax2d.grid(True, alpha=0.3)
    ax2d.legend(loc="upper right")

    def update(frame):
        t = t_vals[frame]
        slice_3d.set_data_3d(x, np.full_like(x, t), Z[frame])
        plane_ref[0].remove()
        plane_ref[0] = ax3d.plot_surface(
            px, np.full_like(px, t), pz, color="crimson", alpha=0.14
        )
        slice_2d.set_ydata(Z[frame])
        time_text.set_text(f"t = {t:.2f}")
        return slice_3d, slice_2d, time_text

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=len(t_vals), interval=1000 / fps)
    save_mp4(anim, out_path, fps=fps)
    plt.close(fig)


def render_overlay_speed_2d(out_path, fps=30):
    x = np.linspace(-5.0, 5.0, 400)
    t_vals = np.linspace(0.0, 3.0 * 2.0, 150 * 2)
    Z_wave = wave_grid(tent, x, t_vals, c=1.0)
    Z_heat = heat_grid(tent, x, t_vals, k=1.0)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    (line_w,) = ax.plot(
        x, Z_wave[0], color="#1f77b4", lw=2.3, label=r"Wave ($c=1, \psi=0$)"
    )
    (line_h,) = ax.plot(x, Z_heat[0], color="#d62728", lw=2.3, label=r"Heat ($k=1$)")
    ax.axhline(1.0, color="0.5", ls="--", lw=1.0)
    time_text = ax.text(
        0.02, 0.78, f"t = {t_vals[0]:.2f}", transform=ax.transAxes, fontsize=11
    )
    ax.set(
        xlim=(-5.0, 5.0),
        ylim=(-0.05, 1.22),
        xlabel="x",
        ylabel="u(x, t)",
        title="Same Initial Profile: Propagation Speed, Smoothness & Decay",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Zoomed tail inset placed above height 0.55 so it never overlaps traveling pulses
    ax_ins = ax.inset_axes([0.56, 0.54, 0.41, 0.42])
    (ins_w,) = ax_ins.plot(x, Z_wave[0], color="#1f77b4", lw=2.0)
    (ins_h,) = ax_ins.plot(x, Z_heat[0], color="#d62728", lw=2.0)
    front_line = ax_ins.axvline(1.0, color="#1f77b4", ls=":", lw=1.2, alpha=0.8)
    ax_ins.set(
        xlim=(1.5, 4.5),
        ylim=(-0.002, 0.045),
        title="Tail Zoom (Heat > 0 immediately; Wave = 0 until x = 1 + t)",
    )
    ax_ins.title.set_fontsize(8.5)
    ax_ins.tick_params(labelsize=8)
    ax_ins.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(ax_ins, edgecolor="0.4", alpha=0.6)

    def update(frame):
        t = t_vals[frame]
        line_w.set_ydata(Z_wave[frame])
        line_h.set_ydata(Z_heat[frame])
        ins_w.set_ydata(Z_wave[frame])
        ins_h.set_ydata(Z_heat[frame])
        front_line.set_xdata([1.0 + t, 1.0 + t])
        time_text.set_text(f"t = {t:.2f}")
        return line_w, line_h, ins_w, ins_h, front_line, time_text

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=len(t_vals), interval=1000 / fps)
    save_mp4(anim, out_path, fps=fps)
    plt.close(fig)


def render_boxcar_singularities_2d(out_path, fps=30):
    x = np.linspace(-5.0, 5.0, 1000)
    t_vals = np.linspace(0.0, 2.5, 150)
    Z_wave = wave_grid(boxcar, x, t_vals, c=1.0)
    Z_heat = boxcar_heat_grid(x, t_vals, k=1.0)

    fig, (ax_w, ax_h) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, sharey=True)
    (line_w,) = ax_w.plot(x, Z_wave[0], color="#1f77b4", lw=2.3)
    (line_h,) = ax_h.plot(x, Z_heat[0], color="#d62728", lw=2.3)

    for ax in (ax_w, ax_h):
        ax.axhline(1.0, color="0.5", ls="--", lw=1.0)
        ax.set(xlim=(-5.0, 5.0), ylim=(-0.05, 1.15), xlabel="x")
        ax.grid(True, alpha=0.3)

    ax_w.set(ylabel="u(x, t)", title="Wave: Jump Discontinuities Preserved")
    ax_h.set(title=r"Heat: Instantly Smoothed ($C^\infty$ for all $t > 0$)")
    time_text = fig.suptitle(
        f"Discontinuous Initial Data   (t = {t_vals[0]:.2f})", fontsize=13
    )

    def update(frame):
        line_w.set_ydata(Z_wave[frame])
        line_h.set_ydata(Z_heat[frame])
        time_text.set_text(f"Discontinuous Initial Data   (t = {t_vals[frame]:.2f})")
        return line_w, line_h, time_text

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    anim = FuncAnimation(fig, update, frames=len(t_vals), interval=1000 / fps)
    save_mp4(anim, out_path, fps=fps)
    plt.close(fig)


def render_long_time_decay_2d(out_path, fps=30):
    x = np.linspace(-13.0, 13.0, 600)
    t_vals = np.linspace(0.0, 10.0, 180)
    Z_wave = wave_grid(tent, x, t_vals, c=1.0)
    Z_heat = heat_grid(tent, x, t_vals, k=1.0)

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    (line_w,) = ax.plot(
        x, Z_wave[0], color="#1f77b4", lw=2.2, label="Wave (constant energy)"
    )
    (line_h,) = ax.plot(
        x, Z_heat[0], color="#d62728", lw=2.4, label=r"Heat ($O(t^{-1/2})$ decay)"
    )
    ax.axhline(0.5, color="#1f77b4", ls=":", lw=1.0, alpha=0.6)
    readout = ax.text(
        0.02,
        0.84,
        "",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.7"
        ),
    )
    ax.set(
        xlim=(-13.0, 13.0),
        ylim=(-0.05, 1.12),
        xlabel="x",
        ylabel="u(x, t)",
        title=r"Long-Time Behavior ($t \to +\infty$): Persistence vs. Decay",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    def update(frame):
        t = t_vals[frame]
        w_cur = Z_wave[frame]
        h_cur = Z_heat[frame]
        line_w.set_ydata(w_cur)
        line_h.set_ydata(h_cur)
        readout.set_text(
            f"t = {t:5.2f}\nmax u_wave = {np.max(w_cur):.3f}\nmax u_heat = {np.max(h_cur):.3f}"
        )
        return line_w, line_h, readout

    update(0)
    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=len(t_vals), interval=1000 / fps)
    save_mp4(anim, out_path, fps=fps)
    plt.close(fig)


def render_information_loss_2d(out_path, fps=30):
    x = np.linspace(-4.5, 4.5, 360)
    t_vals = np.linspace(0.0, 2.5, 150)

    Z_wa = wave_grid(tent, x, t_vals, c=1.0)
    Z_wb = wave_grid(twin_peak, x, t_vals, c=1.0)
    Z_ha = heat_grid(tent, x, t_vals, k=1.0)
    Z_hb = heat_grid(twin_peak, x, t_vals, k=1.0)

    fig, (ax_w, ax_h) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, sharey=True)
    (lw_a,) = ax_w.plot(
        x, Z_wa[0], color="#1f77b4", lw=2.2, label=r"$\phi_1$: single tent (mass 1)"
    )
    (lw_b,) = ax_w.plot(
        x,
        Z_wb[0],
        color="#ff7f0e",
        lw=2.2,
        ls="--",
        label=r"$\phi_2$: twin peak (mass 1)",
    )
    (lh_a,) = ax_h.plot(
        x, Z_ha[0], color="#1f77b4", lw=2.2, label=r"$\phi_1$: single tent (mass 1)"
    )
    (lh_b,) = ax_h.plot(
        x,
        Z_hb[0],
        color="#ff7f0e",
        lw=2.2,
        ls="--",
        label=r"$\phi_2$: twin peak (mass 1)",
    )

    for ax in (ax_w, ax_h):
        ax.set(xlim=(-4.5, 4.5), ylim=(-0.05, 1.12), xlabel="x")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    ax_w.set(ylabel="u(x, t)", title="Wave: Distinct Shapes Transported Intact")
    ax_h.set(title="Heat: Distinct Shapes Wash Together")

    diff_w = ax_w.text(0.03, 0.90, "", transform=ax_w.transAxes, fontsize=10)
    diff_h = ax_h.text(0.03, 0.90, "", transform=ax_h.transAxes, fontsize=10)
    suptitle = fig.suptitle("", fontsize=13)

    def update(frame):
        t = t_vals[frame]
        lw_a.set_ydata(Z_wa[frame])
        lw_b.set_ydata(Z_wb[frame])
        lh_a.set_ydata(Z_ha[frame])
        lh_b.set_ydata(Z_hb[frame])
        dw = np.max(np.abs(Z_wa[frame] - Z_wb[frame]))
        dh = np.max(np.abs(Z_ha[frame] - Z_hb[frame]))
        diff_w.set_text(f"max |u1 - u2| = {dw:.3f}")
        diff_h.set_text(f"max |u1 - u2| = {dh:.3f}")
        suptitle.set_text(
            f"Information: Transported vs. Gradually Lost   (t = {t:.2f})"
        )
        return lw_a, lw_b, lh_a, lh_b, diff_w, diff_h, suptitle

    update(0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    anim = FuncAnimation(fig, update, frames=len(t_vals), interval=1000 / fps)
    save_mp4(anim, out_path, fps=fps)
    plt.close(fig)


def render_bounded_max_principle_3d(out_path, fps=30):
    x = np.linspace(0.0, np.pi, 100)
    t_vals = np.linspace(0.0, np.pi, 120)
    X, T = np.meshgrid(x, t_vals)

    Z_wave = np.sin(X) * np.sin(T)
    Z_heat = np.sin(X) * np.exp(-T)

    fig = plt.figure(figsize=(11, 5.0), dpi=150)
    ax_w = fig.add_subplot(1, 2, 1, projection="3d")
    ax_h = fig.add_subplot(1, 2, 2, projection="3d")

    ax_w.plot_surface(
        X,
        T,
        Z_wave,
        cmap="coolwarm",
        alpha=0.72,
        rstride=2,
        cstride=2,
        edgecolor="none",
    )
    ax_h.plot_surface(
        X, T, Z_heat, cmap="viridis", alpha=0.72, rstride=2, cstride=2, edgecolor="none"
    )

    # Highlight parabolic boundary edges (t = 0, x = 0, x = pi) in solid black
    zeros_x = np.zeros_like(x)
    zeros_t = np.zeros_like(t_vals)
    pi_t = np.full_like(t_vals, np.pi)

    ax_w.plot(x, zeros_x, zeros_x, color="black", lw=2.6, zorder=8)
    ax_w.plot(zeros_t, t_vals, zeros_t, color="black", lw=2.6, zorder=8)
    ax_w.plot(pi_t, t_vals, zeros_t, color="black", lw=2.6, zorder=8)

    ax_h.plot(x, zeros_x, np.sin(x), color="black", lw=2.6, zorder=8)
    ax_h.plot(zeros_t, t_vals, zeros_t, color="black", lw=2.6, zorder=8)
    ax_h.plot(pi_t, t_vals, zeros_t, color="black", lw=2.6, zorder=8)

    # Reference boundary-max planes: z = 0 for wave, z = 1 for heat
    cx, cy = np.meshgrid([0.0, np.pi], [0.0, np.pi])
    ax_w.plot_surface(
        cx, cy, np.zeros_like(cx), color="0.4", alpha=0.14, edgecolor="none"
    )
    ax_h.plot_surface(
        cx, cy, np.ones_like(cx), color="0.4", alpha=0.14, edgecolor="none"
    )

    for ax, title in (
        (ax_w, r"Wave: $u = \sin(x)\sin(t)$" + "\nBoundary = 0, Interior Max = 1"),
        (
            ax_h,
            r"Heat: $u = \sin(x)e^{-t}$" + "\nMax = 1 on $t=0$, Strictly < 1 Inside",
        ),
    ):
        ax.set(
            xlim=(0.0, np.pi),
            ylim=(0.0, np.pi),
            zlim=(-0.05, 1.15),
            xlabel="x",
            ylabel="t",
            zlabel="u(x, t)",
            title=title,
        )
        ax.view_init(elev=25, azim=-55)

    (sw,) = ax_w.plot(x, zeros_x, Z_wave[0], color="crimson", lw=2.5, zorder=10)
    (sh,) = ax_h.plot(x, zeros_x, Z_heat[0], color="crimson", lw=2.5, zorder=10)
    px, pz = np.meshgrid([0.0, np.pi], [0.0, 1.15])
    pw_ref = [ax_w.plot_surface(px, np.zeros_like(px), pz, color="crimson", alpha=0.12)]
    ph_ref = [ax_h.plot_surface(px, np.zeros_like(px), pz, color="crimson", alpha=0.12)]
    suptitle = fig.suptitle(
        f"Bounded Rectangle [0, π] × [0, π]   (t = {t_vals[0]:.2f})", fontsize=13
    )

    def update(frame):
        t = t_vals[frame]
        t_line = np.full_like(x, t)
        sw.set_data_3d(x, t_line, Z_wave[frame])
        sh.set_data_3d(x, t_line, Z_heat[frame])
        pw_ref[0].remove()
        ph_ref[0].remove()
        t_plane = np.full_like(px, t)
        pw_ref[0] = ax_w.plot_surface(px, t_plane, pz, color="crimson", alpha=0.12)
        ph_ref[0] = ax_h.plot_surface(px, t_plane, pz, color="crimson", alpha=0.12)
        suptitle.set_text(f"Bounded Rectangle [0, π] × [0, π]   (t = {t:.2f})")
        return sw, sh, suptitle

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    anim = FuncAnimation(fig, update, frames=len(t_vals), interval=1000 / fps)
    save_mp4(anim, out_path, fps=fps)
    plt.close(fig)


# =============================================================================
# Build Registry & CLI Runner
# =============================================================================
def build_all(selected):
    OUT_DIR.mkdir(exist_ok=True)
    x_core = np.linspace(-5.0, 5.0, 240)
    t_core = np.linspace(0.0, 3.0, 120)

    jobs = {
        1: (
            "pde_01_wave_pluck_3d.mp4",
            lambda p: render_slice_3d(
                wave_grid(tent, x_core, t_core, c=1.0),
                x_core,
                t_core,
                "Wave Equation",
                p,
                z_max=1.15,
            ),
        ),
        2: (
            "pde_02_heat_diffusion_3d.mp4",
            lambda p: render_slice_3d(
                heat_grid(tent, x_core, t_core, k=1.0),
                x_core,
                t_core,
                "Heat Equation",
                p,
                z_max=1.15,
            ),
        ),
        3: (
            "pde_03_wave_max_principle_break_3d.mp4",
            lambda p: render_slice_3d(
                wave_grid(
                    tent,
                    x_core,
                    t_core,
                    c=1.0,
                    psi_antideriv=tent_antideriv,
                    psi_scale=3.0,
                ),
                x_core,
                t_core,
                "Wave Equation (ψ = 3g)",
                p,
                z_max=1.9,
            ),
        ),
        4: ("pde_04_overlay_propagation_speed_2d.mp4", render_overlay_speed_2d),
        5: ("pde_05_boxcar_singularities_2d.mp4", render_boxcar_singularities_2d),
        6: ("pde_06_long_time_decay_2d.mp4", render_long_time_decay_2d),
        7: ("pde_07_information_loss_2d.mp4", render_information_loss_2d),
        8: ("pde_08_bounded_max_principle_3d.mp4", render_bounded_max_principle_3d),
    }

    for idx in selected:
        if idx not in jobs:
            print(f"Skipping unknown clip id: {idx}")
            continue
        fname, fn = jobs[idx]
        out_path = OUT_DIR / fname
        print(f"[{idx}/8] Rendering {out_path} ...")
        fn(out_path)
        print(f"[{idx}/8] Done -> {out_path}")


if __name__ == "__main__":
    configure_ffmpeg()
    if len(sys.argv) > 1:
        ids = [int(arg) for arg in sys.argv[1:]]
    else:
        ids = list(range(1, 9))
    build_all(ids)
