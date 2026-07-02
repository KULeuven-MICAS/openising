import matplotlib.pyplot as plt
import numpy as np
from ising.stages.model.MPPI.environment import plot_environment

def plot_results(env, x_ref, all_x, predicted_traj, savefile="ising_bicycle_mpc_results.png", show=False):
    try:
        fig, ax = plot_environment(env, figsize=(16, 10))
    except Exception as _:
        fig, ax = plt.subplots(figsize=(16, 10))
        if getattr(env, "control_points", None):
            cx = [c[0] for c in env.control_points]
            cy = [c[1] for c in env.control_points]
            ax.plot(cx, cy, "ko", markersize=6, alpha=0.6, label="control points")
        if getattr(env, "start", None) is not None:
            ax.plot([env.start[0]], [env.start[1]], "ro", label="start")
        if getattr(env, "goal_region", None):
            try:
                gx = [g[0] for g in env.goal_region]
                gy = [g[1] for g in env.goal_region]
                ax.plot(gx + [gx[0]], gy + [gy[0]], "g--", alpha=0.6, label="goal")
            except Exception:
                pass
    xs, ys = x_ref[:, 0], x_ref[:, 1]
    ax.plot(xs, ys, "-o", alpha=0.8, markersize=3, color="blue")
    plt.plot(
        [x[0] for x in all_x],
        [x[1] for x in all_x],
        "rs-",
        alpha=0.5,
        label="Optimized by Ising",
        markersize=3,
        linewidth=0.7,
    )

    for coords in predicted_traj:
        coords = coords[:, :2]
        plt.plot([x[0] for x in coords], [x[1] for x in coords], "r-", alpha=0.2)

    # # Also plot the lines connecting all_x and xs
    # for i in range(len(all_x) - 1):
    #     plt.plot([all_x[i][0], xs[i]], [all_x[i][1], ys[i]], "k--", alpha=0.2)

    plt.title("Optimized Trajectory vs Reference Path")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.legend()
    plt.savefig(savefile, dpi=300)
    if show:
        plt.show()


def unitvec_from_heading(theta):
    x = np.cos(theta)
    y = np.sin(theta)
    ds = (x**2 + y**2)**0.5
    return (x/ds, y/ds)


def plot_trajectory(x, y, bch, cx, cy, stepsize=0.1):
    '''
    Plots x-y coords and cx(t)-cy(t) parametric spline
    Plots unit vectors showing the spline headings at boundaries
    Generates c1 and c2 plots showing heading and curvature continuity
    '''

    ts = np.arange(0, len(x)-1+stepsize, stepsize)
    ts_plus = np.arange(ts[0]-.2, ts[-1]+.3, stepsize)

    # Heading constraint unit vectors
    hvec_start = unitvec_from_heading(bch[0])
    hvec_end = unitvec_from_heading(bch[-1])

    # Plot trajectory
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylim(min(y)-1, max(y)+1)
    ax.set_xlim(min(x)-2, max(x)+2)
    ax.plot(x, y, 'o', label='nodes')
    ax.plot(cx(ts_plus), cy(ts_plus), label='spline')
    ax.annotate("", xy=(x[0] + hvec_start[0], y[0] + hvec_start[1]),
                xytext=(x[0], y[0]), arrowprops=dict(arrowstyle="->", color="red"))
    ax.annotate("", xy=(x[-1] + hvec_end[0], y[-1] + hvec_end[1]),
                xytext=(x[-1], y[-1]), arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_aspect('equal')
    ax.set_title('C2 trajectory')
    ax.set_xlabel('x(mu)')
    ax.set_ylabel('y(mu)')

    # Plot heading and curvature
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].set_title('X(mu)')
    ax[0].plot(ts, cx(ts, 1), label='Heading')
    ax[0].plot(ts, cx(ts, 2), label='Curvature')
    ax[0].set_xlabel('mu')
    ax[0].legend()
    ax[1].set_title('Y(mu)')
    ax[1].plot(ts, cy(ts, 1), label='Heading')
    ax[1].plot(ts, cy(ts, 2), label='Curvature')
    ax[1].set_xlabel('mu')
    ax[1].legend()
    return fig, ax
