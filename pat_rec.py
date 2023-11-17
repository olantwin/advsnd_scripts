#!/usr/bin/env python3
"""Pattern match to obtain track candidates using different algorithms"""

import argparse
import logging
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

    def __init__(self, hits):
        self.hits = hits
        self.tracklets = []

    def to_genfit(self):
        """Convert to the genfit/sndsw format"""


class Track2d(Track):
    """Specialisation for 2d tracks"""

    def __init__(self, view, b=0, k=0, **kwargs):
        self.view = view
        self.k = k
        self.b = b
        super().__init__(**kwargs)


class Track3d(Track):
    """Specialisation for 3d tracks"""

    def __init__(self, **kwargs):
        self.k_x = 0
        self.b_x = 0
        self.k_y = 0
        self.b_y = 0
        super().__init__(**kwargs)

    def get_zx_track(self):
        return Track2d(view=0, b=self.b_x, k=self.k_x)  # TODO add other args

    def get_zy_track(self):
        return Track2d(view=1, b=self.b_y, k=self.k_y)  # TODO add other args
        pass


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

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    ax_xy = fig.add_subplot(gs[1, 0])
    ax_xz = fig.add_subplot(gs[0, 0], sharex=ax_xy)
    ax_zy = fig.add_subplot(gs[1, 1], sharey=ax_xy)
    ax_xy.set_xlim(-60, 10)
    ax_xy.set_ylim(0, 70)
    ax_xz.set_ylim(-150, -70)
    ax_zy.set_xlim(-150, -70)
    # i = 0
    for view in recognized_tracks:
        for column in recognized_tracks[view]:
            for row in recognized_tracks[view][column]:
                for track in recognized_tracks[view][column][row]:
                    hits = track.hits
                    dict_of_hits = {k: [dic[k] for dic in hits] for k in hits[0]}
                    rect = plt.Rectangle(
                        (min(dict_of_hits["xbot"]), min(dict_of_hits["z"])),
                        max(dict_of_hits["xtop"]) - min(dict_of_hits["xbot"]),
                        max(dict_of_hits["z"]) - min(dict_of_hits["z"]),
                        alpha=0.3,
                        color=colors[view],
                    )
                    ax_xz.add_patch(rect)
                    rect = plt.Rectangle(
                        (min(dict_of_hits["z"]), min(dict_of_hits["ybot"])),
                        max(dict_of_hits["z"]) - min(dict_of_hits["z"]),
                        max(dict_of_hits["ytop"]) - min(dict_of_hits["ybot"]),
                        alpha=0.3,
                        color=colors[view],
                    )
                    ax_zy.add_patch(rect)
                    rect = plt.Rectangle(
                        (min(dict_of_hits["xbot"]), min(dict_of_hits["ybot"])),
                        max(dict_of_hits["xtop"]) - min(dict_of_hits["xbot"]),
                        max(dict_of_hits["ytop"]) - min(dict_of_hits["ybot"]),
                        alpha=0.3,
                        color=colors[view],
                    )
                    ax_xy.add_patch(rect)
                    # plt.plot(
                    #     [i, i],
                    #     [min(dict_of_hits["z"]), max(dict_of_hits["z"])],
                    #     color=colors[view],
                    # )
                    # i += 1
    fig.show()

    match_candidates = track_match(recognized_tracks[0], recognized_tracks[1])
    print(match_candidates)

    # TODO combine short tracks

    # TODO match tracks between modules

    # TODO match tracks between views

    # PatRec in 1D
    # recognized_tracks_x = artificial_retina_pat_rec_single_view(
    #     hits_x, min_hits, proj="x"
    # )
    # recognized_tracks_y = artificial_retina_pat_rec_single_view(
    #     hits_y, min_hits, proj="y"
    # )

    # Try to match hits to other view
    # recognized_tracks_x_matched = artificial_retina_pat_rec_matched(
    #     hits_y, recognized_tracks_x, min_hits, "x"
    # )
    # recognized_tracks_y_matched = artificial_retina_pat_rec_matched(
    #     hits_x, recognized_tracks_y, min_hits, "y"
    # )

    # Combination of tracks fro both views
    # recognized_tracks_combo = track_combination(
    #     recognized_tracks_x_matched, recognized_tracks_y_matched
    # )

    # Prepare output of PatRec
    # recognized_tracks = {}
    # i_track = 0
    # for atrack_combo in recognized_tracks_combo:
    #     hits_y12 = atrack_combo["hits_y12"]
    #     hits_stereo12 = atrack_combo["hits_stereo12"]
    #     hits_y34 = atrack_combo["hits_y34"]
    #     hits_stereo34 = atrack_combo["hits_stereo34"]

    #     if (
    #         len(hits_y12) >= min_hits
    #         and len(hits_stereo12) >= min_hits
    #         and len(hits_y34) >= min_hits
    #         and len(hits_stereo34) >= min_hits
    #     ):
    #         atrack = {
    #             "y12": hits_y12,
    #             "stereo12": hits_stereo12,
    #             "y34": hits_y34,
    #             "stereo34": hits_stereo34,
    #         }
    #         recognized_tracks[i_track] = atrack
    #         i_track += 1

    # merged_tracks = merge_segments(recognized_tracks_x, proj="x")

    return recognized_tracks


def merge_tracks(tracks):
    pass


def track_match(tracks_x, tracks_y):
    """Match tracks between views using an R-tree"""
    if not tracks_x and tracks_y:
        logging.warning("Need tracks in both views to attempt matching.")
        return

    properties = index.Property()
    properties.dimension = 2

    # Create an rtree index (2D : z, x)
    rtree_zx = index.Index(properties=properties)

    # Create an rtree index (2D : z, x)
    rtree_zy = index.Index(properties=properties)

    tracks_2d = []
    # flatten nested structure:
    for column in (0, 1):
        for row in (0, 1, 2, 3):
            tracks_2d += tracks_x[column][row]
            tracks_2d += tracks_y[column][row]

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

    return tracks_2d


def track_match_old(tracks_x, tracks_y):
    """Match tracks between views using an R-tree"""
    if not tracks_x and tracks_y:
        logging.warning("Need tracks in both views to attempt matching.")
        return
    # TODO perform an optimisation to find best set of matches?
    # TODO use hit information in coarse direction to reduce number of possible matches

    properties = index.Property()
    properties.dimension = 2
    # Use rtrees (inspired by Lardon)
    #
    # ztol = 10 * cm
    #
    # Create an rtree index (3D : view, z)
    rtree_idx = index.Index(properties=properties)

    # keep track of indices as sorting will change them!
    idx_to_ID = []

    i = 0
    # fill the index

    tracks_2d = []
    # flatten nested structure:
    for column in (0, 1):
        for row in (0, 1, 2, 3):
            tracks_2d += tracks_x[column][row]
            tracks_2d += tracks_y[column][row]

    for track in tracks_2d:
        start = track.hits[0]["z"]
        stop = track.hits[-1]["z"]

        # if(t.len_straight >= len_min and t.ghost == False):
        rtree_idx.insert(i, (track.view, start, track.view, stop))
        i += 1
        idx_to_ID.append(i)

        # search for the best matching track in the other view

    ID_to_idx = {v: k for k, v in enumerate(idx_to_ID)}

    for track_i in tracks_2d:
        # if(ti.len_straight < len_min):
        #     continue
        track_i.matched = [-1, -1]

        ti_start = track_i.hits[0]["z"]
        ti_stop = track_i.hits[-1]["z"]

        overlaps = []
        for iview in range(2):
            if iview == track_i.view:
                continue
            else:
                overlaps.append(
                    list(rtree_idx.intersection((iview, ti_start, iview, ti_stop)))
                )

        logging.info(overlaps)

        for overlap in overlaps:
            matches = []
            for j_ID in overlap:
                # j_idx = ID_to_idx[j_ID]
                j_idx = j_ID
                track_j = tracks_2d[j_idx]
                tj_view = track_j.view
                # if(ti.module_ini != tj.module_ini):
                #     continue
                # if(ti.module_end != tj.module_end):
                #     continue

                tj_start = track_j.hits[0]["z"]
                tj_stop = track_j.hits[-1]["z"]

                # zmin = max(ti_stop, tj_stop)
                # zmax = min(ti_start, tj_start)
                # qi = np.fabs(ti.charge_in_z_interval(zmin, zmax))
                # qj = np.fabs(tj.charge_in_z_interval(zmin, zmax))

                # TODO use dE/dx to match tracks?
                # try:
                #     balance = math.fabs(qi - qj)/(qi + qj)
                # except ZeroDivisionError:
                #     balance = 9999.
                dmin = min(np.abs(ti_start - tj_start), np.abs(ti_stop - tj_stop))

                # if(balance < qfrac and dmin < ztol):
                #     matches.append( (j_ID, balance, dmin) )
                matches.append((j_ID, dmin))
                # if dmin < ztol:
                #     matches.append((j_ID, dmin))
                # else:
                #     logging.info(f"Match with track {j_ID} failed due to intolerance.")

            if len(matches) > 0:
                # sort matches by distance and take best match
                matches = sorted(matches, key=itemgetter(1))
                track_i.matched[tj_view] = matches[0][0]

                """ now do the matching !"""

    for i_idx, track_i in enumerate(tracks_2d):
        i_ID = idx_to_ID[i_idx]
        trks = [track_i]
        ti_view = track_i.view
        for iview in range(ti_view + 1, 2):
            j_ID = track_i.matched[iview]

            if j_ID > 0:
                j_idx = ID_to_idx[j_ID]
                tj = tracks_2d[j_idx]
                if tj.matched[ti_view] == i_ID:
                    trks.append(tj)
        if len(trks) > 1:
            logging.info(f"Matched the following tracks to each other: {trks}")
            return trks
        return []
        # t3D = complete_trajectories(trks)

        # n_fake = t3D.check_views()
        # if n_fake > 1:
        #     continue

        # t3D.boundaries()

        # # TODO fit track
        # isok = finalize_3d_track(t3D, 10)
        # if isok == False:
        #     continue

        # trk_ID = dc.evt_list[-1].n_tracks3D  # +1
        # t3D.ID_3D = trk_ID

        # dc.tracks3D_list.append(t3D)
        # dc.evt_list[-1].n_tracks3D += 1

        # for t in trks:
        #     for i in range(cf.n_view):
        #         # t.matched[i] = -1
        #         t.match_3D = trk_ID
        #         t.set_match_hits_3D(trk_ID)


def get_zy_projection(z, xtop, ytop, xbot, ybot, k_y, b_y):
    # FIXME: Useless for 90 degree views?
    x = k_y * z + b_y
    k = (ytop - ybot) / (xtop - xbot + 10**-6)
    b = ytop - k * xtop
    y = k * x + b

    return y


def get_zx_projection(z, ytop, xtop, ybot, xbot, k_x, b_x):
    y = k_x * z + b_x
    k = (xtop - xbot) / (ytop - ybot + 10**-6)
    b = xtop - k * ytop
    x = k * y + b

    return x


def artificial_retina_pat_rec_matched(
    SmearedHits_stereo, recognized_tracks_p, min_hits, proj="y"
):
    """Perform pattern recognition for stereo layer"""
    recognized_tracks_stereo = []
    used_hits = []
    other = "x" if proj == "y" else "y"

    for atrack_p in recognized_tracks_p:
        k_p = atrack_p[f"k_{proj}"]
        b_p = atrack_p[f"b_{proj}"]

        # Get hit zx projections
        for ahit in SmearedHits_stereo:
            x_center = get_zy_projection(
                ahit["z"],
                ahit["ytop"],
                ahit["xtop"],
                ahit["ybot"],
                ahit["xbot"],
                k_p,
                b_p,
            )
            ahit[f"z{other}_projection"] = x_center

        long_recognized_tracks_stereo = []
        hits_z = []
        hits_o = []

        for ahit in SmearedHits_stereo:
            if ahit["digiHit"] in used_hits:
                continue
            if abs(ahit[f"z{other}_projection"]) > 300:
                continue
            hits_z.append(ahit["z"])
            hits_o.append(ahit[f"z{other}_projection"])
        hits_z = np.array(hits_z)
        hits_o = np.array(hits_o)

        sigma = 15.0 * RESOLUTION
        best_seed_params = get_best_seed(hits_z, hits_o, sigma, sample_weight=None)

        res = minimize(
            retina_func,
            best_seed_params,
            args=(hits_z, hits_o, sigma, None),
            method="BFGS",
            jac=retina_grad,
            options={"gtol": 1e-6, "disp": False, "maxiter": 5},
        )
        [k_seed_upd, b_seed_upd] = res.x

        atrack_stereo = {}
        atrack_stereo["hits_stereo"] = []
        atrack_stereo_layers = []

        for ahit3 in SmearedHits_stereo:
            if ahit3["digiHit"] in used_hits:
                continue

            if abs(ahit3[f"z{other}_projection"]) > 300:
                continue

            layer3 = np.floor(ahit3["detID"] >> 15)
            if layer3 in atrack_stereo_layers:
                continue

            in_bin = hit_in_window(
                ahit3["z"],
                ahit3[f"z{other}_projection"],
                k_seed_upd,
                b_seed_upd,
                window_width=15.0 * RESOLUTION,
            )
            if in_bin:
                atrack_stereo["hits_stereo"].append(ahit3)
                atrack_stereo_layers.append(layer3)

        if len(atrack_stereo["hits_stereo"]) >= min_hits:
            long_recognized_tracks_stereo.append(atrack_stereo)

        # Remove clones
        max_track = None
        max_n_hits = -999

        for atrack_stereo in long_recognized_tracks_stereo:
            if len(atrack_stereo["hits_stereo"]) > max_n_hits:
                max_track = atrack_stereo
                max_n_hits = len(atrack_stereo["hits_stereo"])

        atrack = {}
        atrack[f"hits_{proj}"] = atrack_p[f"hits_{proj}"]
        atrack[f"k_{proj}"] = atrack_p[f"k_{proj}"]
        atrack[f"b_{proj}"] = atrack_p[f"b_{proj}"]
        atrack["hits_stereo"] = []

        if max_track is not None:
            atrack["hits_stereo"] = max_track["hits_stereo"]
            for ahit in max_track["hits_stereo"]:
                used_hits.append(ahit["digiHit"])

        recognized_tracks_stereo.append(atrack)

    return recognized_tracks_stereo


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
        long_recognized_tracks, min_hits, proj
    )

    # Track fit
    for track in recognized_tracks:
        z_coords = [hit["z"] for hit in track.hits]
        p_coords = [(hit[f"{proj}top"] + hit[f"{proj}bot"]) / 2 for hit in track.hits]
        track.k, track.b = np.polyfit(z_coords, p_coords, deg=1)

    return recognized_tracks


def reduce_clones_using_one_track_per_hit(recognized_tracks, min_hits=3, proj="y"):
    """
    Remove clones.

    Parameters
    ----------
    recognized_tracks : list[Track2d]
    min_hits : int
        Minimum number of hits per track.

    Returns
    -------
    tracks_no_clones : list[Track2d]
    """
    used_hits = []
    tracks_no_clones = []
    n_hits = [len(track.hits) for track in recognized_tracks]

    for i_track in np.argsort(n_hits)[::-1]:
        track = recognized_tracks[i_track]
        new_track = Track2d(
            view=track.view,
            hits=[],
        )

        for hit in track.hits:
            if hit["digiHit"] not in used_hits:
                new_track.hits.append(hit)

        if len(new_track.hits) >= min_hits:
            tracks_no_clones.append(new_track)
            for hit in new_track.hits:
                used_hits.append(hit["digiHit"])

    return tracks_no_clones


def merge_segments(tracks, proj, threshold=0):
    """Attempt to merge segments of tracks split in z"""
    # TODO use rtrees (inspired by Lardon)
    # rtree nearest neighbours?
    if len(tracks) == 1:
        return tracks
    # DONE sort all tracks in z (they should already be?)
    candidate_pairs = []
    z_ordered_tracks = sorted(tracks, key=lambda track: track[f"hits_{proj}"][0]["z"])
    for i, track_i in enumerate(z_ordered_tracks[:-1]):
        sorted_hits_i = sorted(track_i[f"hits_{proj}"], key=itemgetter("z"))
        for j, track_j in enumerate(z_ordered_tracks[1:]):
            if i == j:
                continue
            sorted_hits_j = sorted(track_j[f"hits_{proj}"], key=itemgetter("z"))
            if sorted_hits_i[-1]["z"] < sorted_hits_j[0]["z"]:
                candidate_pairs.append((i, j))
    print(candidate_pairs)
    for pair in candidate_pairs:
        # TODO perform merge
        # How to deal with multiple options? chi^2?
        pass


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
        recognized_tracks_x = []
        recognized_tracks_y = []
        for column in (0, 1):
            for row in (0, 1, 2, 3):
                recognized_tracks_x += recognized_tracks[0][column][row]
                recognized_tracks_y += recognized_tracks[1][column][row]
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
        used_hits_x = []
        used_hits_y = []
        for track_x in recognized_tracks_x:
            track_candidate = ROOT.std.vector("int")()
            for hit in track_x.hits:
                track_candidate.push_back(hit["digiHit"])
            track_candidates.emplace_back(track_candidate)
            if args.display:
                hits_x = track_x.hits
                map(used_hits_x.append, (hit["detID"] for hit in hits_x))
                z = np.array([hit["z"] for hit in hits_x])
                x = np.array([(hit["xtop"] + hit["xbot"]) / 2 for hit in hits_x])
                y = np.array([(hit["ytop"] + hit["ybot"]) / 2 for hit in hits_x])
                ax_xz.scatter(x, z, marker=".")
                ax_zy.scatter(z, y, marker="x")
                ax_xy.scatter(x, y, marker="$x$")
                b_x = track_x.b
                k_x = track_x.k
                ax_xz.plot(k_x * z + b_x, z, zorder=100)
        for track_y in recognized_tracks_y:
            track_candidate = ROOT.std.vector("int")()
            for hit in track_y.hits:
                track_candidate.push_back(hit["digiHit"])
            track_candidates.emplace_back(track_candidate)
            if args.display:
                hits_y = track_y.hits
                map(used_hits_y.append, (hit["detID"] for hit in hits_y))
                z = np.array([hit["z"] for hit in hits_y])
                x = np.array([(hit["xtop"] + hit["xbot"]) / 2 for hit in hits_y])
                y = np.array([(hit["ytop"] + hit["ybot"]) / 2 for hit in hits_y])
                ax_xz.scatter(x, z, marker="x")
                ax_zy.scatter(z, y, marker=".")
                ax_xy.scatter(x, y, marker="$y$")
                b_y = track_y.b
                k_y = track_y.k
                ax_zy.plot(z, k_y * z + b_y, zorder=100)
        if args.display:
            unused_hits = [
                hit
                for hit in hits
                if hit["detID"] not in used_hits_x and hit["detID"] not in used_hits_y
            ]
            z = np.array([hit["z"] for hit in unused_hits])
            x = np.array([(hit["xtop"] + hit["xbot"]) / 2 for hit in unused_hits])
            y = np.array([(hit["ytop"] + hit["ybot"]) / 2 for hit in unused_hits])
            ax_xy.scatter(x, y, marker=".", color="gray", zorder=0.5)
            ax_xz.scatter(x, z, marker=".", color="gray", zorder=0.5)
            ax_zy.scatter(z, y, marker=".", color="gray", zorder=0.5)
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
