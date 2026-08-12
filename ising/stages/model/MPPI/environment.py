import numpy as np
import random

import shapely.geometry as geom

from scipy.interpolate import CubicSpline

from matplotlib import pyplot as plt
from matplotlib import patches

#########################
# Bicycle environment
#########################

class Environment:
    def __init__(self, obstacles, start, goal_region, bounds=None):
        self.environment_loaded = False
        self.obstacles = obstacles  # list of lists of tuples
        self.bounds = bounds
        self.start = start  # (x,y)
        self.goal_region = goal_region  # list of tuples defining corner points
        self.control_points = []
        self.calculate_scene_dimensions()

    def add_obstacles(self, obstacles):
        self.obstacles += obstacles
        self.calculate_scene_dimensions()

    def set_goal_region(self, goal_region):
        self.goal_region = goal_region
        self.calculate_scene_dimensions()

    def add_control_points(self, points):
        self.control_points += points
        self.calculate_scene_dimensions()

    def calculate_scene_dimensions(self):
        """Compute scene bounds from obstacles, start, and goal """
        points = []
        for elem in self.obstacles:
            points = points + elem
        if self.start:
            points += [self.start]
        if self.goal_region:
            points += self.goal_region
        if len(self.control_points) > 0:
            points += self.control_points
        mp = geom.MultiPoint(points)
        self.bounds = mp.bounds


def _polygon_patch(poly, **kwargs):
    facecolor = kwargs.pop("facecolor", None)
    if facecolor is None and "fc" in kwargs:
        facecolor = kwargs.pop("fc")
    edgecolor = kwargs.pop("edgecolor", None)
    if edgecolor is None and "ec" in kwargs:
        edgecolor = kwargs.pop("ec")

    if hasattr(poly, "exterior"):
        coords = np.asarray(poly.exterior.coords)
    elif hasattr(poly, "geoms"):
        # MultiPolygon: use the first geometry
        geom0 = next(iter(poly.geoms), None)
        if geom0 is None or not hasattr(geom0, "exterior"):
            raise ValueError("Unsupported polygon geometry")
        coords = np.asarray(geom0.exterior.coords)
    else:
        raise ValueError("Unsupported polygon geometry")

    return patches.Polygon(
        coords,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs,
    )


def plot_ellipse_environment(scene, bounds, figsize):
    '''
    scene - dict from scenarios
    bounds - [[minx, maxx], [miny, maxy]]
    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for obs in scene['obs_list']:
        h, k, a, b, theta = obs
        ellipse = patches.Ellipse(
            (h, k), a, b, angle=theta, fc='orange', ec='k', alpha=0.5, zorder=5)
        ax.add_patch(ellipse)
    # start / goal
    goal_poly = geom.Polygon(scene['goal'])
    ax.add_patch(_polygon_patch(goal_poly, fc='green',
                 ec='green', alpha=0.5, zorder=1))
    start = geom.Point(scene['start']).buffer(0.2, resolution=3)
    ax.add_patch(_polygon_patch(start, fc='red',
                 ec='black', alpha=0.7, zorder=1))
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    ax.set_aspect('equal', adjustable='box')
    return ax


def plot_environment(env, bounds=None, figsize=None, margin=1.0):
    if bounds is None and env.bounds:
        minx, miny, maxx, maxy = env.bounds
        minx -= margin
        miny -= margin
        maxx += margin
        maxy += margin
    elif bounds:
        minx, miny, maxx, maxy = bounds
    else:
        minx, miny, maxx, maxy = (-10, -5, 10, 5)
    max_width, max_height = 12, 5.5
    if figsize is None:
        width, height = max_width, (maxy-miny)*max_width/(maxx-minx)
        if height > 5:
            width, height = (maxx-minx)*max_height/(maxy-miny), max_height
        figsize = (width, height)

    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)
    # obstacles
    for i, obs in enumerate(env.obstacles):
        poly = geom.Polygon(obs)
        patch = _polygon_patch(poly, fc='orange', ec='black',
                             alpha=0.5, zorder=20)
        ax.add_patch(patch)

    # start / goal
    # goal_poly = geom.Polygon(env.goal_region)
    # ax.add_patch(PolygonPatch(goal_poly, fc='green',
    #              ec='green', alpha=0.5, zorder=1))
    start = geom.Point(env.start).buffer(0.2, resolution=3)
    ax.add_patch(_polygon_patch(start, fc='red',
                 ec='black', alpha=0.7, zorder=1))

    # control points
    cx = [c[0] for c in env.control_points]
    cy = [c[1] for c in env.control_points]
    ax.plot(cx, cy, 'ko', markersize=8, alpha=0.8, label='control points')

    # plt.xlim([minx, maxx])
    # plt.ylim([miny, maxy])
    ax.set_aspect('equal', adjustable='box')
    return f, ax

# Helpers for obstacle constraint handling


def centroid(obstacle):
    '''
    Averages all vertices in a given obstacle. Average of x's and y's is
    guaranteed to lie inside polygon
    '''
    x_avg = sum([v[0] for v in obstacle])/len(obstacle)
    y_avg = sum([v[1] for v in obstacle])/len(obstacle)
    return (x_avg, y_avg)


def linear_obstacle_constraints(obs, buffer):
    '''
    Given polygonal obstsacle, returns a list of values for a, b, c
    Constraints take form: cy <= ax + b - buffer + Mz
    Assumes obstacles are given as consecutive ordered list of vertices
    '''
    constraints = []
    cent = centroid(obs)
    for i, v in enumerate(obs):
        v1 = obs[i]
        # get next vertex; loop back to first for last constraint
        v2 = obs[(i+1) % len(obs)]
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]

        if dx == 0:     # vertical constaint case; cy <= ax + b --> x <= b
            c = 0
            if cent[0] <= v1[0]:  # flip constraint
                a, b, c = 1, -v1[0] - buffer, 0

            else:
                a, b, c = -1, v1[0] - buffer, 0

        else:           # non-vertical constraint; cy <= ax + b
            a = dy / dx
            b = v1[1] - a * v1[0]
            if cent[1] < a * cent[0] + b:  # flip constraint
                a, b, c = -a, -b, -1
                a, b, c = calc_offset_coefs((a, b, c), buffer)
            else:
                a, b, c = a, b, 1
                a, b, c = calc_offset_coefs((a, b, c), buffer)
        constraints.append((a, b, c))
    return constraints


def calc_offset_coefs(coefs, offset):
    a, b, c = coefs
    b_new = b - offset*np.sqrt(a**2+1)
    return a, b_new, c


################################
# Trajectory generation
################################

def sample_trajectory(ctrl_pts, bc_headings, v, dt):
    ''' Given control points [(x,y)], boundary condition headings, fixed velocity v,
        return a sampled C2 trajectory every time period dt
    '''
    x = [p[0] for p in ctrl_pts]
    y = [p[1] for p in ctrl_pts]
    cx, cy = calc_c2_traj(x, y, bc_headings)

    total_length = 0
    for i in range(cx.c.shape[1]):
        coeffs_x = np.flip(cx.c[:, i])
        coeffs_y = np.flip(cy.c[:, i])
        slen = calc_spline_length(coeffs_x, coeffs_y)
        total_length += slen

    nsteps = int(total_length/(dt*v))
    # tvec = np.arange(0, len(x)-1+dt, dt)
    tvec = np.linspace(0, len(x)-1, nsteps)
    xs = cx(tvec)
    ys = cy(tvec)

    # calc heading
    dxs = cx(tvec, 1)
    dys = cy(tvec, 1)
    psi = np.arctan2(dys, dxs)
    return xs, ys, psi


def calc_c2_traj(x, y, bc_headings, eps=0.005):
    '''
    Iteratively compute spline coefficients until spline length of first and last segment converges
    '''

    # Start with euclidean dist as slen approx for first and last segments
    slen_start = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    slen_end = np.sqrt((x[-1] - x[-2])**2 + (y[-1] - y[-2])**2)

    while True:
        cx, cy = gen_c2_spline(x, y, bc_headings, slen_start, slen_end)
        coeffs_x_start = np.flip(cx.c[:, 0])
        coeffs_y_start = np.flip(cy.c[:, 0])
        coeffs_x_end = np.flip(cx.c[:, -1])
        coeffs_y_end = np.flip(cy.c[:, -1])

        slen_start_new = calc_spline_length(coeffs_x_start, coeffs_y_start)
        slen_end_new = calc_spline_length(coeffs_x_end, coeffs_y_end)

        if abs(slen_start_new - slen_start) < eps and abs(slen_end_new - slen_end) < eps:
            break
        else:
            slen_start = slen_start_new
            slen_end = slen_end_new
    return cx, cy


def gen_c2_spline(x, y, bc_headings, slen_start, slen_end):
    '''
    Generates a C2 continuous spline using scipy CubicSpline lib
    x: np.array of x-coordinate points
    y: np.array of y-coordinate points
    '''

    # define mu, a virtual path variable of length 1 for each spline segment
    assert(len(x) == len(y))
    mu = np.arange(0, len(x), 1.0)

    # build splines
    cs_x = CubicSpline(mu, x,
                       bc_type=((1, slen_start * np.cos(bc_headings[0])),
                                (1, slen_end * np.cos(bc_headings[1]))))
    cs_y = CubicSpline(mu, y,
                       bc_type=((1, slen_start * np.sin(bc_headings[0])),
                                (1, slen_end * np.sin(bc_headings[1]))))
    return cs_x, cs_y



def calc_spline_length(x_coeffs, y_coeffs, n_ips=20):
    '''
    Returns numerically computed length along cubic spline
    x_coeffs: array of 4 x coefficients
    y_coeffs: array of 4 y coefficients
    '''

    t_steps = np.linspace(0.0, 1.0, n_ips)
    spl_coords = np.zeros((n_ips, 2))

    spl_coords[:, 0] = x_coeffs[0] \
        + x_coeffs[1] * t_steps \
        + x_coeffs[2] * np.power(t_steps, 2) \
        + x_coeffs[3] * np.power(t_steps, 3)
    spl_coords[:, 1] = y_coeffs[0] \
        + y_coeffs[1] * t_steps \
        + y_coeffs[2] * np.power(t_steps, 2) \
        + y_coeffs[3] * np.power(t_steps, 3)

    slength = np.sum(
        np.sqrt(np.sum(np.power(np.diff(spl_coords, axis=0), 2), axis=1)))
    return slength

#################
# Build scene
#################


def create_environment(scene):
    env = Environment(scene["obs_list"], scene["start"], scene["goal"])
    control_pts = scene["control_pts"]
    bc_headings = scene["bc_headings"]
    env.add_control_points(control_pts)
    return env, control_pts, bc_headings


def create_reference_trajectory(env, control_pts, bc_headings, v, dt):
    xs, ys, psi = sample_trajectory(control_pts, bc_headings, v, dt)

    nf = len(xs)

    return np.vstack(
        (
            xs.reshape((1, nf)),
            ys.reshape((1, nf)),
            psi.reshape((1, nf)),
            v * np.ones((1, nf)),
            np.zeros((1, nf)),
        )
    )

# def calc_c2_traj(x, y, bc_headings, eps=0.005):
#     '''
#     Iteratively compute spline coefficients until spline length of first and last segment converges
#     '''

#     # Start with euclidean dist as slen approx for first and last segments
#     slen_start = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
#     slen_end = np.sqrt((x[-1] - x[-2])**2 + (y[-1] - y[-2])**2)

#     while True:
#         cx, cy = gen_c2_spline(x, y, bc_headings, slen_start, slen_end)
#         coeffs_x_start = np.flip(cx.c[:, 0])
#         coeffs_y_start = np.flip(cy.c[:, 0])
#         coeffs_x_end = np.flip(cx.c[:, -1])
#         coeffs_y_end = np.flip(cy.c[:, -1])

#         slen_start_new = calc_spline_length(coeffs_x_start, coeffs_y_start)
#         slen_end_new = calc_spline_length(coeffs_x_end, coeffs_y_end)

#         if abs(slen_start_new - slen_start) < eps and abs(slen_end_new - slen_end) < eps:
#             break
#         else:
#             slen_start = slen_start_new
#             slen_end = slen_end_new
#     return cx, cy

def generate_random_scene(nb_control_points:int = 7, seed=None, ):
    """
    Generate a random scene in the following format:
    {'start':           (0, 0),
    'goal':            [(12, 12), (12, 14), (14, 14), (14, 12)],
    'obs_list':         [[(2, 1), (3, 1), (3, 4), (1, 4)],
                        [(5, 1), (6, 3), (5, 5), (4, 3)],
                        [(1, 6), (6, 6), (6, 7), (2, 9)],
                        [(11, 4), (14, 5), (8, 11), (6, 9)],
                        [(5, 10), (7, 11), (8, 14), (4, 13)]],
    'control_pts':      [(0, 0), (3.5, 1), (4.0, 4.5), (6.4, 6.2),
                        (5.8, 9.4), (8.4, 11.5), (13, 13)],
    'bc_headings':      (np.pi/8, np.pi/8),
    }
    """



    if seed is not None:
        random.seed(seed)

    start = (0, 0)
    goal = [(random.uniform(10, 15), random.uniform(10, 15)) for _ in range(4)]
    obs_list = []

    control_pts = [(0, 0)]

    first_heading = None
    last_heading = None

    for _ in range(nb_control_points):
        # The next control point should be within a certain distance from the last one
        last_cp = control_pts[-1]
        # Generate a random angle
        if len(control_pts) == 1:
            # Angle is random for the first control point
            angle = random.uniform(0, 2 * np.pi)
            first_heading = angle
        else:
            angle_of_last_cp = np.arctan2(last_cp[1], last_cp[0])
            # Generate a random angle around the last control point
            angle = random.uniform(
                angle_of_last_cp - np.pi / 3, angle_of_last_cp + np.pi / 3
            )
        # Generate a random distance
        distance = random.uniform(4, 5)
        # Calculate the next control point
        next_cp = (
            last_cp[0] + distance * np.cos(angle),
            last_cp[1] + distance * np.sin(angle),
        )
        last_heading = angle
        control_pts.append(next_cp)

    bc_headings = (first_heading, last_heading)

    return {
        "start": start,
        "goal": goal,
        "obs_list": obs_list,
        "control_pts": control_pts,
        "bc_headings": bc_headings,
    }
