#!/usr/bin/env python3
"""Pattern match to obtain track candidates using different algorithms"""

import argparse
import logging
from itertools import combinations
from operator import itemgetter
from rtree import index
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import ROOT
from shipunit import um

RESOLUTION = 35 * um

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

ROOT.gInterpreter.Declare(
    """
struct Hit {
    Double_t x,y,z;
    Int_t hit_id, det_id, view;
};
"""
)


class Hit:
    """Hit class for pattern matching purposes"""

    def __init__(self, hit_id):
        pass


class Track:
    """Describe track for pattern matching purposes"""

    def __init__(self, hits, **kwargs):
        self.hits = hits
        self.tracklets = []


class Track2d(Track):
    """Specialisation for 2d tracks"""

    def __init__(self, view, b=0, k=0, **kwargs):
        self.view = view
        self.k = k
        self.b = b
        super().__init__(**kwargs)

    def __add__(self, other):
        if self.view == other.view:
            return Track2d(
                hits=self.hits + other.hits,
                view=self.view,
                b=(self.b + other.b) / 2,
                k=(self.k + other.k) / 2,
            )
        if self.view:
            return other + self
        return Track3d(
            hits=self.hits + other.hits,
            b_x=self.b,
            k_x=self.k,
            b_y=other.b,
            k_y=other.k,
        )

    def to_3d(self):
        return Track3d(
            hits=self.hits,
            b_x=self.b if not self.view else 0,
            k_x=self.k if not self.view else 0,
            b_y=self.b if self.view else 0,
            k_y=self.k if self.view else 0,
        )

    def extrapolate_to(self, z):
        return self.k * z + self.b


class Track3d(Track):
    """Specialisation for 3d tracks"""

    def __init__(self, k_x=0, k_y=0, b_x=0, b_y=0, **kwargs):
        self.k_x = k_x
        self.b_x = b_x
        self.k_y = k_y
        self.b_y = b_y
        super().__init__(**kwargs)


def get_best_seed(x, y, sigma, sample_weight=None):
    """Try to find the best initial guess for k, b and the retina value"""
    best_retina_val = 0
    best_seed_params = [0, 0]

    for i_1 in range(len(x) - 1):
        for i_2 in range(i_1 + 1, len(x)):
            if x[i_1] >= x[i_2]:
                continue

            seed_k = (y[i_2] - y[i_1]) / (x[i_2] - x[i_1] + 10**-6)  # slope
            seed_b = y[i_1] - seed_k * x[i_1]  # intercept

            retina_val = retina_func([seed_k, seed_b], x, y, sigma, sample_weight)

            if retina_val < best_retina_val:
                best_retina_val = retina_val
                best_seed_params = [seed_k, seed_b]

    return best_seed_params


def retina_func(track_prams, x, y, sigma, sample_weight=None):
    """
    Calculates the artificial retina function value.
    Parameters
    ----------
    track_prams : array-like
        Track parameters [k, b].
    x : array-like
        Array of x coordinates of hits.
    y : array-like
        Array of x coordinates of hits.
    sigma : float
        Standard deviation of hit form a track.
    sample_weight : array-like
        Hit weights used during the track fit.
    Retunrs
    -------
    retina : float
        Negative value of the artificial retina function.
    """

    rs = track_prams[0] * x + track_prams[1] - y

    if sample_weight is None:
        exps = np.exp(-((rs / sigma) ** 2))
    else:
        exps = np.exp(-((rs / sigma) ** 2)) * sample_weight

    retina = exps.sum()

    return -retina


def retina_grad(track_prams, x, y, sigma, sample_weight=None):
    """
    Calculates the artificial retina gradient.
    Parameters
    ----------
    track_prams : array-like
        Track parameters [k, b].
    x : array-like
        Array of x coordinates of hits.
    y : array-like
        Array of x coordinates of hits.
    sigma : float
        Standard deviation of hit form a track.
    sample_weight : array-like
        Hit weights used during the track fit.
    Returns
    -------
    retina : float
        Negative value of the artificial retina gradient.
    """

    rs = track_prams[0] * x + track_prams[1] - y

    if sample_weight is None:
        exps = np.exp(-((rs / sigma) ** 2))
    else:
        exps = np.exp(-((rs / sigma) ** 2)) * sample_weight

    dks = -2.0 * rs / sigma**2 * exps * x
    dbs = -2.0 * rs / sigma**2 * exps

    return -np.array([dks.sum(), dbs.sum()])


def hits_split(smeared_hits):
    """
    Split hits into groups of station views.

    Parameters
    ----------
    SmearedHits : list
        Smeared hits. SmearedHits = [{'digiHit': key,
                                      'xtop': xtop, 'ytop': ytop, 'z': ztop,
                                      'xbot': xbot, 'ybot': ybot,
                                      'detID': detID}, {...}, ...]
    Returns
    -------
    hit_dict : dict
        Dictionary of hits indexed by view, column, row
    """

    hits_dict = {
        # view
        0: {
            # column
            0: {
                # row
                0: [],
                1: [],
                2: [],
                3: [],
            },
            1: {
                # row
                0: [],
                1: [],
                2: [],
                3: [],
            },
        },
        1: {
            # column
            0: {
                # row
                0: [],
                1: [],
                2: [],
                3: [],
            },
            1: {
                # row
                0: [],
                1: [],
                2: [],
                3: [],
            },
        },
    }

    for hit in smeared_hits:
        det_id = hit["detID"]
        view = (int(det_id >> 14) + 1) % 2
        column = int(det_id >> 11) % 2
        row = int(det_id >> 12) % 4
        # split by view
        hits_dict[view][column][row].append(hit)

    return hits_dict


def artificial_retina_pattern_recognition(hits):
    """
    Main function of track pattern recognition.

    Parameters:
    -----------
    hits : list
        Hits. hits = [{'digiHit': key,
                                      'xtop': xtop, 'ytop': ytop, 'z': ztop,
                                      'xbot': xbot, 'ybot': ybot,
                                      'detID': detID}, {...}, ...]
    """

    recognized_tracks = {}

    # if len(hits) > 1000:
    #     print("Too many hits in the event!")
    #     return recognized_tracks

    min_hits = 3

    # Separate hits
    hits_dict = hits_split(hits)

    # plt.figure()
    # plt.xlim(-60, 10)
    # plt.ylim(0, 70)
    for view in (0, 1):
        recognized_tracks[view] = {}
        for column in (0, 1):
            recognized_tracks[view][column] = {}
            for row in (0, 1, 2, 3):
                # module = column + 1 + 2 * row
                # color = colors[module % len(colors)]
                hits = hits_dict[view][column][row]
                if not hits:
                    recognized_tracks[view][column][row] = []
                    continue
                # xs = [(h["xtop"] + h["xbot"]) / 2 for h in hits]
                # ys = [(h["ytop"] + h["ybot"]) / 2 for h in hits]
                # if len(hits) > 1:
                #     rect = plt.Rectangle(
                #         (min(xs), min(ys)),
                #         max(xs) - min(xs),
                #         max(ys) - min(ys),
                #         alpha=0.3,
                #         color=color,
                #     )
                #     plt.gca().add_patch(rect)
                # for hit in hits:
                #     rect = plt.Rectangle(
                #         (min(hit["xbot"], hit["xtop"]), min(hit["ybot"], hit["ytop"])),
                #         abs(hit["xtop"] - hit["xbot"]),
                #         abs(hit["ytop"] - hit["ybot"]),
                #         alpha=0.3,
                #         color=color,
                #     )
                #     plt.gca().add_patch(rect)
                # plt.scatter(xs, ys, label=f"{view=}, {module=}", color=color)
                recognized_tracks[view][column][
                    row
                ] = artificial_retina_pat_rec_single_view(
                    hits, min_hits, proj="y" if view else "x"
                )
                # for track in recognized_tracks[view][column][row]:
                # hits = track[f"hits_{'y' if view else 'x'}"]
                # dict_of_hits = {k: [dic[k] for dic in hits] for k in hits[0]}
                # rect = plt.Rectangle(
                #     (min(dict_of_hits["xbot"]), min(dict_of_hits["ybot"])),
                #     max(dict_of_hits["xtop"]) - min(dict_of_hits["ybot"]),
                #     max(dict_of_hits["ytop"]) - min(dict_of_hits["ybot"]),
                #     alpha=0.3,
                #     color=color,
                # )
                # plt.gca().add_patch(rect)
                # xs = [(h["xtop"] + h["xbot"]) / 2 for h in hits]
                # ys = [(h["ytop"] + h["ybot"]) / 2 for h in hits]
                # plt.scatter(xs, ys, label=f"{view=}, {module=}", color=color)

    # plt.legend()
    # plt.show()

    # fig = plt.figure()
    # gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    # ax_xy = fig.add_subplot(gs[1, 0])
    # ax_xz = fig.add_subplot(gs[0, 0], sharex=ax_xy)
    # ax_zy = fig.add_subplot(gs[1, 1], sharey=ax_xy)
    # ax_xy.set_xlim(-60, 10)
    # ax_xy.set_ylim(0, 70)
    # ax_xz.set_ylim(-150, -70)
    # ax_zy.set_xlim(-150, -70)
    # i = 0
    # for view in recognized_tracks:
    #     for column in recognized_tracks[view]:
    #         for row in recognized_tracks[view][column]:
    #             for track in recognized_tracks[view][column][row]:
    #                 hits = track.hits
    #                 dict_of_hits = {k: [dic[k] for dic in hits] for k in hits[0]}
    #                 rect = plt.Rectangle(
    #                     (min(dict_of_hits["xbot"]), min(dict_of_hits["z"])),
    #                     max(dict_of_hits["xtop"]) - min(dict_of_hits["xbot"]),
    #                     max(dict_of_hits["z"]) - min(dict_of_hits["z"]),
    #                     alpha=0.3,
    #                     color=colors[view],
    #                 )
    #                 ax_xz.add_patch(rect)
    #                 rect = plt.Rectangle(
    #                     (min(dict_of_hits["z"]), min(dict_of_hits["ybot"])),
    #                     max(dict_of_hits["z"]) - min(dict_of_hits["z"]),
    #                     max(dict_of_hits["ytop"]) - min(dict_of_hits["ybot"]),
    #                     alpha=0.3,
    #                     color=colors[view],
    #                 )
    #                 ax_zy.add_patch(rect)
    #                 rect = plt.Rectangle(
    #                     (min(dict_of_hits["xbot"]), min(dict_of_hits["ybot"])),
    #                     max(dict_of_hits["xtop"]) - min(dict_of_hits["xbot"]),
    #                     max(dict_of_hits["ytop"]) - min(dict_of_hits["ybot"]),
    #                     alpha=0.3,
    #                     color=colors[view],
    #                 )
    #                 ax_xy.add_patch(rect)
    # plt.plot(
    #     [i, i],
    #     [min(dict_of_hits["z"]), max(dict_of_hits["z"])],
    #     color=colors[view],
    # )
    # i += 1
    # fig.show()

    flat_view_x = []
    flat_view_y = []

    for column in (0, 1):
        for row in (0, 1, 2, 3):
            flat_view_x += recognized_tracks[0][column][row]
            flat_view_y += recognized_tracks[1][column][row]

    # Combine short tracks
    long_tracks_x = match_segments(flat_view_x)
    long_tracks_y = match_segments(flat_view_y)

    # Match tracks between views
    matches = match_tracks(long_tracks_x, long_tracks_y)

    return matches


def merge_tracks(track, other, tolerance=10):
    """Attempt to merge two tracks in 2d or 3d."""
    # Check whether tracks are compatible
    if track.view == other.view:
        # 2d case: extrapolate to end of first track and check whether within tolerance
        proj = "y" if track.view else "x"
        if track.hits[-1]["z"] > other.hits[-1]["z"]:
            return merge_tracks(other, track)
        if (
            abs(
                other.extrapolate_to(track.hits[-1]["z"])
                - (track.hits[-1][f"{proj}top"] + track.hits[-1][f"{proj}bot"]) / 2
            )
            < RESOLUTION * tolerance
        ):
            logging.info(f"Merge 2d: Successful merge of tracks {track}, {other}.")
            return track + other
        logging.info(f"Merge 2d: Tracks {track}, {other} not compatible.")

    else:
        # 3d: extrapolate tracks to centre of other track and check consistency
        if track.view:
            track, other = other, track
        ybots = np.array([hit["ybot"] for hit in track.hits])
        ytops = np.array([hit["ytop"] for hit in track.hits])
        ys = (ytops + ybots) / 2
        dys = np.abs(ytops - ybots) / 2
        zs = np.array([hit["z"] for hit in track.hits])
        middle = int(len(track.hits) / 2)
        o_xbots = np.array([hit["xbot"] for hit in other.hits])
        o_xtops = np.array([hit["xtop"] for hit in other.hits])
        o_xs = (o_xtops + o_xbots) / 2
        o_dxs = np.abs(o_xtops - o_xbots) / 2
        o_zs = np.array([hit["z"] for hit in other.hits])
        o_middle = int(len(other.hits) / 2)
        if (
            o_xs[o_middle] - o_dxs[o_middle] - tolerance * RESOLUTION
            <= track.extrapolate_to(o_zs[o_middle])
            <= o_xs[o_middle] + o_dxs[o_middle] + tolerance * RESOLUTION
        ) and (
            ys[middle] - dys[middle] - tolerance * RESOLUTION
            <= other.extrapolate_to(zs[middle])
            <= ys[middle] + dys[middle] + tolerance * RESOLUTION
        ):
            logging.info(f"Merge 3d: Successful merge of tracks {track}, {other}.")
            return track + other
        logging.info(f"Merge 3d: Tracks {track}, {other} not compatible.")


def match_tracks(tracks_x, tracks_y):
    """Match tracks between views using an R-tree"""
    if not tracks_x and tracks_y:
        logging.warning("Need tracks in both views to attempt matching.")
        return []

    properties = index.Property()
    properties.dimension = 2

    # Create an rtree index (2D : z, x)
    rtree_zx = index.Index(properties=properties)

    # Create an rtree index (2D : z, x)
    rtree_zy = index.Index(properties=properties)

    tracks_2d = tracks_x + tracks_y

    for i, track in enumerate(tracks_2d):
        track.matched = []
        xbots = np.array([hit["xbot"] for hit in track.hits])
        xtops = np.array([hit["xtop"] for hit in track.hits])
        ybots = np.array([hit["ybot"] for hit in track.hits])
        ytops = np.array([hit["ytop"] for hit in track.hits])
        # xs = (xtops + xbots)/2
        # ys = (ytops + ybots)/2
        zs = np.array([hit["z"] for hit in track.hits])

        x_bottom = np.min(np.concatenate((xtops, xbots)))
        x_top = np.max(np.concatenate((xtops, xbots)))
        rtree_zx.insert(i, (zs[0], x_bottom, zs[-1], x_top))

        y_bottom = np.min(np.concatenate((ytops, ybots)))
        y_top = np.max(np.concatenate((ytops, ybots)))
        rtree_zy.insert(i, (zs[0], y_bottom, zs[-1], y_top))

    for i, track in enumerate(tracks_2d):
        xbots = np.array([hit["xbot"] for hit in track.hits])
        xtops = np.array([hit["xtop"] for hit in track.hits])
        zs = np.array([hit["z"] for hit in track.hits])

        x_bottom = np.min(np.concatenate((xtops, xbots)))
        x_top = np.max(np.concatenate((xtops, xbots)))
        for candidate in rtree_zx.intersection((zs[0], x_bottom, zs[-1], x_top)):
            # Select candidates in other view
            if candidate != i:
                candidate_track = tracks_2d[candidate]
                if candidate_track.view == track.view:
                    continue
                if i in candidate_track.matched:
                    print(f"Track {i} already matched to {candidate}")
                    track.matched.append(candidate)
                    continue
                hits = candidate_track.hits
                ybots = np.array([hit["ybot"] for hit in hits])
                ytops = np.array([hit["ytop"] for hit in hits])
                zs = np.array([hit["z"] for hit in hits])
                y_bottom = np.min(np.concatenate((ytops, ybots)))
                y_top = np.max(np.concatenate((ytops, ybots)))
                reverse_match = list(
                    rtree_zy.intersection((zs[0], y_bottom, zs[-1], y_top))
                )
                if i in reverse_match:
                    print(f"Successfully matched {candidate} to {i}")
                    track.matched.append(candidate)
        print(f"Matches for track {i}: {track.matched}")

    tracks_3d = []
    attempted = set()
    for i, track in enumerate(tracks_2d):
        for other in track.matched:
            if (i, other) in attempted or (other, i) in attempted:
                continue
            attempted.add((i, other))
            try:
                if merged := merge_tracks(track, tracks_2d[other]):
                    tracks_3d.append(merged)
            except RuntimeError as e:
                logging.warning(e)
        tracks_3d.append(track.to_3d())

    return reduce_clones_using_one_track_per_hit(tracks_3d)


def hit_in_window(x, y, k_bin, b_bin, window_width=1.0):
    """
    Counts hits in a bin of track parameter space (b, k).

    Parameters
    ---------
    x : array-like
        Array of x coordinates of hits.
    y : array-like
        Array of x coordinates of hits.
    k_bin : float
        Track parameter: y = k_bin * x + b_bin
    b_bin : float
        Track parameter: y = k_bin * x + b_bin

    Return
    ------
    track_inds : array-like
        Hit indexes of a track: [ind1, ind2, ...]
    """

    y_approx = k_bin * x + b_bin

    flag = False
    if np.abs(y_approx - y) <= window_width:
        flag = True

    return flag


def artificial_retina_pat_rec_single_view(hits, min_hits, proj="y"):
    """
    Main function of track pattern recognition.

    Parameters:
    -----------
    SmearedHits : list
        Smeared hits. SmearedHits = [{'digiHit': key,
                                      'xtop': xtop, 'ytop': ytop, 'z': ztop,
                                      'xbot': xbot, 'ybot': ybot,
                                      'detID': detID}, {...}, ...]
    """

    view = 1 if proj == "y" else 0

    long_recognized_tracks = []
    used_hits = np.zeros(len(hits))

    hits_z = np.array([ahit["z"] for ahit in hits])
    hits_p = np.array([(ahit[f"{proj}top"] + ahit[f"{proj}bot"]) / 2 for ahit in hits])

    for i in range(len(hits)):
        hits_z_unused = hits_z[used_hits == 0]
        hits_p_unused = hits_p[used_hits == 0]

        sigma = 1.0 * RESOLUTION
        best_seed_params = get_best_seed(
            hits_z_unused, hits_p_unused, sigma, sample_weight=None
        )

        res = minimize(
            retina_func,
            best_seed_params,
            args=(hits_z_unused, hits_p_unused, sigma, None),
            method="BFGS",
            jac=retina_grad,
            options={"gtol": 1e-6, "disp": False, "maxiter": 5},
        )
        [k_seed_upd, b_seed_upd] = res.x

        track = Track2d(
            view=view,
            hits=[],
        )
        used_stations = []
        hit_ids = []

        # TODO max distance between hits belonging to same track?

        # Add new hits to the seed
        for i_hit3, ahit3 in enumerate(hits):
            if used_hits[i_hit3]:
                continue

            station = np.floor(ahit3["detID"] >> 15)
            if station in used_stations:
                continue

            in_bin = hit_in_window(
                ahit3["z"],
                (ahit3[f"{proj}top"] + ahit3[f"{proj}bot"]) / 2,
                k_seed_upd,
                b_seed_upd,
                window_width=1.4 * RESOLUTION,
            )
            if in_bin:
                track.hits.append(ahit3)
                used_stations.append(station)
                hit_ids.append(i_hit3)

        if len(track.hits) >= min_hits:
            long_recognized_tracks.append(track)
            used_hits[hit_ids] = 1
        else:
            break

    # Remove clones
    recognized_tracks = reduce_clones_using_one_track_per_hit(
        long_recognized_tracks, min_hits
    )

    # Track fit
    for track in recognized_tracks:
        z_coords = [hit["z"] for hit in track.hits]
        p_coords = [(hit[f"{proj}top"] + hit[f"{proj}bot"]) / 2 for hit in track.hits]
        track.k, track.b = np.polyfit(z_coords, p_coords, deg=1)

    return recognized_tracks


def reduce_clones_using_one_track_per_hit(recognized_tracks, min_hits=3):
    """
    Remove clones.

    Parameters
    ----------
    recognized_tracks : list[Track]
    min_hits : int
        Minimum number of hits per track.

    Returns
    -------
    tracks_no_clones : list[Track]
    """
    used_hits = []
    tracks_no_clones = []
    n_hits = [len(track.hits) for track in recognized_tracks]

    for i_track in np.argsort(n_hits)[::-1]:
        track = recognized_tracks[i_track]
        new_track = type(track)(
            hits=[],
            view=track.view if hasattr(track, "view") else None,
            b=track.b if hasattr(track, "b") else None,
            k=track.k if hasattr(track, "k") else None,
            b_x=track.b_x if hasattr(track, "b_x") else None,
            k_x=track.k_x if hasattr(track, "k_x") else None,
            b_y=track.b_y if hasattr(track, "b_y") else None,
            k_y=track.k_y if hasattr(track, "k_y") else None,
        )

        for hit in track.hits:
            if hit["digiHit"] not in used_hits:
                new_track.hits.append(hit)

        if len(new_track.hits) >= min_hits:
            tracks_no_clones.append(new_track)
            for hit in new_track.hits:
                used_hits.append(hit["digiHit"])

    return tracks_no_clones


def match_segments(tracks):
    """Attempt to merge segments of tracks split in z"""
    # TODO use rtrees (inspired by Lardon)
    # rtree nearest neighbours?
    if len(tracks) == 1:
        return tracks
    # DONE sort all tracks in z (they should already be?)
    long_tracks = tracks
    candidate_pairs = []
    # TODO need overlap in x,y, no overlap in z
    z_ordered_tracks = sorted(tracks, key=lambda track: track.hits[0]["z"])
    for track_i, track_j in combinations(z_ordered_tracks, 2):
        sorted_hits_i = sorted(track_i.hits, key=itemgetter("z"))
        sorted_hits_j = sorted(track_j.hits, key=itemgetter("z"))
        if sorted_hits_i[-1]["z"] < sorted_hits_j[0]["z"]:
            candidate_pairs.append((track_i, track_j))
    print(candidate_pairs)
    for pair in candidate_pairs:
        try:
            if merged := merge_tracks(*pair):
                long_tracks.append(merged)
        except RuntimeError as e:
            logging.warning(e)
        # TODO perform merge

    # How to deal with multiple options? chi^2?
    return reduce_clones_using_one_track_per_hit(long_tracks)


def main():
    """Preselect events using second tree with cuts."""
    parser = argparse.ArgumentParser(description="Script for AdvSND analysis.")
    parser.add_argument(
        "-f",
        "--inputfile",
        help="""Simulation results to use as input."""
        """Supports retrieving file from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--geofile",
        help="""Simulation results to use as input. """
        """Supports retrieving files from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        help="""File to write the filtered tree to."""
        """Will be recreated if it already exists.""",
    )
    parser.add_argument(
        "-n",
        "--nEvents",
        help="""Number of Events to process.""",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-D", "--display", help="Visualise pattern matching", action="store_true"
    )
    args = parser.parse_args()
    geofile = ROOT.TFile.Open(args.geofile, "read")
    geo = geofile.FAIRGeom  # noqa: F841
    if not args.outputfile:
        args.outputfile = args.inputfile.removesuffix(".root") + "_PR.root"
    inputfile = ROOT.TFile.Open(args.inputfile, "read")
    tree = inputfile.cbmsim
    if not args.nEvents:
        args.nEvents = tree.GetEntries()
    outputfile = ROOT.TFile.Open(args.outputfile, "recreate")
    out_tree = tree.CloneTree(0)
    track_candidates = ROOT.std.vector("std::vector<int>")()
    out_tree.Branch("track_candidates", track_candidates)
    n = 0
    for event in tqdm(tree, desc="Event loop: ", total=tree.GetEntries()):
        stop = ROOT.TVector3()
        start = ROOT.TVector3()
        hits = [
            {
                "digiHit": i,
                "xtop": stop.x(),
                "ytop": stop.y(),
                "z": stop.z(),
                "xbot": start.x(),
                "ybot": start.y(),
                "detID": hit.GetDetectorID(),
            }
            for i, hit in enumerate(event.Digi_advTargetClusters)
            if (_ := hit.GetPosition(stop, start), True)
        ]
        recognized_tracks = artificial_retina_pattern_recognition(hits)
        ax_xy, ax_xz, ax_zy = None, None, None
        if args.display:
            fig = plt.figure()
            gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
            ax_xy = fig.add_subplot(gs[1, 0])
            ax_xz = fig.add_subplot(gs[0, 0], sharex=ax_xy)
            ax_zy = fig.add_subplot(gs[1, 1], sharey=ax_xy)
            ax_xy.set_xlim(-60, 10)
            ax_xy.set_ylim(0, 70)
            ax_xz.set_ylim(-150, -70)
            ax_zy.set_xlim(-150, -70)
        used_hits = []
        for track in recognized_tracks:
            track_candidate = ROOT.std.vector("int")()
            for hit in track.hits:
                track_candidate.push_back(hit["digiHit"])
            track_candidates.emplace_back(track_candidate)
            if args.display:
                hits = track.hits
                map(used_hits.append, (hit["detID"] for hit in hits))
                z = np.array([hit["z"] for hit in hits])
                x = np.array([(hit["xtop"] + hit["xbot"]) / 2 for hit in hits])
                y = np.array([(hit["ytop"] + hit["ybot"]) / 2 for hit in hits])
                x_err = np.array([abs(hit["xtop"] - hit["xbot"]) / 2 for hit in hits])
                y_err = np.array([abs(hit["ytop"] - hit["ybot"]) / 2 for hit in hits])
                ax_xz.errorbar(x, z, xerr=x_err, fmt=".")
                ax_zy.errorbar(z, y, yerr=y_err, fmt=".")
                ax_xy.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=".")
                if track.b_x:
                    b_x = track.b_x
                    k_x = track.k_x
                    ax_xz.plot(k_x * z + b_x, z, zorder=100)
                if track.b_y:
                    b_y = track.b_y
                    k_y = track.k_y
                    ax_zy.plot(z, k_y * z + b_y, zorder=100)
        if args.display:
            unused_hits = [hit for hit in hits if hit["detID"] not in used_hits]
            z = np.array([hit["z"] for hit in unused_hits])
            x = np.array([(hit["xtop"] + hit["xbot"]) / 2 for hit in unused_hits])
            y = np.array([(hit["ytop"] + hit["ybot"]) / 2 for hit in unused_hits])
            x_err = np.array(
                [abs(hit["xtop"] - hit["xbot"]) / 2 for hit in unused_hits]
            )
            y_err = np.array(
                [abs(hit["ytop"] - hit["ybot"]) / 2 for hit in unused_hits]
            )
            ax_xy.scatter(x, y, marker=".", color="gray", zorder=0.5)
            ax_xz.scatter(x, z, marker=".", color="gray", zorder=0.5)
            ax_zy.scatter(z, y, marker=".", color="gray", zorder=0.5)
            ax_xz.errorbar(x, z, xerr=x_err, fmt=".", color="gray", zorder=0.5)
            ax_zy.errorbar(z, y, yerr=y_err, fmt=".", color="gray", zorder=0.5)
            ax_xy.errorbar(
                x, y, xerr=x_err, yerr=y_err, fmt=".", color="gray", zorder=0.5
            )
            plt.show()
        out_tree.Fill()
        track_candidates.clear()
        if n > args.nEvents:
            break
        n += 1
    out_tree.Write()
    outputfile.Write()


if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    logging.basicConfig(level=logging.INFO)
    main()
