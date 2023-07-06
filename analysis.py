#!/usr/bin/env python3

from enum import Enum
import argparse
import ROOT
import rootUtils as ut
from shipunit import cm, um, keV, GeV, mm, MeV
from numpy import sqrt, hypot, array, cross, floor
from numpy.linalg import norm
from particle.pdgid import charge
from particle import Particle
import SNDLHCstyle


DETECTOR_TYPE = Enum("DETECTOR_TYPE", ["PIXEL", "STRIP"])


CONFIGS = {
    "ALICE": {"NAME": "ALICE pixel", "THRESHOLD": 75 * um, "TYPE": DETECTOR_TYPE.PIXEL},
    "CMS": {"NAME": "CMS strip", "THRESHOLD": 300 * um, "TYPE": DETECTOR_TYPE.STRIP},
    "MAPS": {"NAME": "future MAPS", "THRESHOLD": 25 * um, "TYPE": DETECTOR_TYPE.PIXEL},
}


DISTANCES = [7.5 * mm, 10 * mm, 12.5 * mm, 15 * mm]


def track_separation(t1, t2, z, TYPE, **kwargs):
    t1_start = array([t1.GetStartX(), t1.GetStartY()])
    t1_dir = array([t1.GetPx(), t1.GetPy()])
    t1_dir /= t1.GetPz()
    t2_start = array([t2.GetStartX(), t2.GetStartY()])
    t2_dir = array([t2.GetPx(), t2.GetPy()])
    t2_dir /= t2.GetPz()
    delta = (
        t1_start
        + t1_dir * (z - t1.GetStartZ())
        - t2_start
        - t2_dir * (z - t2.GetStartZ())
    )
    return abs(delta[0]) if TYPE == DETECTOR_TYPE.STRIP else max(abs(delta))


def separation_distance(t1, t2, **kwargs):
    start_z = t1.GetStartZ()
    z = start_z + 1 * mm
    while not are_separate(t1, t2, z=z, **kwargs):
        z += 1 * mm
    return z - start_z


def isolation_distance(track, other_tracks, **kwargs):
    return max(separation_distance(track, t, **kwargs) for t in other_tracks)


def are_separate(t1, t2, THRESHOLD, **kwargs):
    return track_separation(t1, t2, **kwargs) >= THRESHOLD


def is_isolated(track, other_tracks, **kwargs):
    return all(are_separate(track, t, **kwargs) for t in other_tracks)


def main():
    parser = argparse.ArgumentParser(description="Script for AdvSND analysis.")
    parser.add_argument(
        "inputfiles",
        help="""Simulation results to use as input. """
        """Supports retrieving files from EOS via the XRootD protocol. Several files can be specified""",
        nargs='+'
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        default="hists.root",
        help="""File to write the flux maps to. """
        """Will be recreated if it already exists.""",
    )
    parser.add_argument("--digi", help="Run studies using digitisation", action='store_true')
    parser.add_argument("--truth", help="Run truth study comparing different scenarii", action='store_true')
    parser.add_argument("--plots", help="Make nice plots as pdf and png", action='store_true')
    args = parser.parse_args()
    ch = ROOT.TChain("cbmsim")
    for inputfile in args.inputfiles:
        ch.Add(inputfile)
    n = ch.GetEntries()

    SNDLHCstyle.init_style()
    h = {}
    ut.bookHist(h, "tau_dz", "tau flight distance;d[cm];", 100, 0, 10)
    if args.truth:
        for key, config in CONFIGS.items():
            ut.bookHist(
                h,
                "tau_isolation_" + key,
                f"Distance for #tau isolation ({config['NAME']});d [mm];",
                100,
                0,
                100,
            )
            ut.bookHist(
                h,
                "tau_isolation_vs_decay_length_" + key,
                f"Distance for #tau isolation vs. decay length ({config['NAME']});d [mm];decay length [mm]",
                100,
                0,
                100,
                100,
                0,
                100,
            )
            ut.bookHist(
                h,
                "tau_isolation_vs_tau_momentum_" + key,
                f"Distance for #tau isolation vs. #tau momentum ({config['NAME']});d [mm];momentum [GeV/c]",
                100,
                0,
                100,
                100,
                0,
                2300,
            )
            for distance in DISTANCES:
                ut.bookHist(
                    h,
                    f"isolated_tracks_{key}_{distance}",
                    f"Isolated tracks per event at {distance} [cm] ({config['NAME']}); Isolated tracks;",
                    50,
                    0,
                    50,
                )
                ut.bookHist(
                    h,
                    f"isolated_track_momentum_{key}_{distance}",
                    f"Isolated track momentum at {distance} [cm] ({config['NAME']}); momentum [GeV/c];",
                    50,
                    0,
                    50,
                )
                ut.bookHist(
                    h,
                    f"isolated_track_charge_{key}_{distance}",
                    f"Isolated track charge at {distance} [cm] ({config['NAME']}); charge;",
                    5,
                    -2.5,
                    2.5,
                )
    ut.bookHist(h, "IP", "IP wrt. to #tau; IP[um];", 1000, 0, 1000)
    ut.bookHist(
        h,
        "IP_charged",
        "IP wrt. to #tau (charged particles only); IP[um];",
        1000,
        0,
        1000,
    )
    ut.bookHist(
        h,
        "multiplicity",
        "multiplicity at primary vertex;multiplicity;",
        100,
        -0.5,
        100.5,
    )
    ut.bookHist(
        h,
        "multiplicity_charged",
        "multiplicity at primary vertex (charged);multiplicity;",
        100,
        -0.5,
        100.5,
    )
    ut.bookHist(
        h, "daughters", "multiplicity at secondary vertex;multiplicity;", 20, -0.5, 20.5
    )
    ut.bookHist(
        h,
        "daughters_charged",
        "multiplicity at secondary vertex (charged);multiplicity;",
        20,
        -0.5,
        20.5,
    )
    ut.bookHist(h, "tau_E", "#tau energy; E [GeV/c^{2}];", 100, 0, 2500)
    ut.bookHist(h, "nu_tau_E", "#nu_{#tau} energy; E [GeV/c^{2}];", 100, 0, 2500)
    if args.digi:
        ut.bookHist(h, "absolute x", "absolute x;x[cm];", 100, -50, 0)
        ut.bookHist(h, "absolute y", "absolute y;y[cm];", 100, 10, 60)
        ut.bookHist(h, "xy", ";x[cm];y[cm]", 100, -50, 0, 100, 10, 60)
        ut.bookHist(h, "x-x_true", "#Delta x;x[um];", 100, -100, 100)
        ut.bookHist(h, "y-y_true", "#Delta y;y[um];", 100, -100, 100)
        ut.bookHist(h, "ELoss", "Energy loss; E [keV/c^{2}];", 1000, 0, 1000)
        ut.bookHist(h, "P", "Momentum at hit; P [GeV];", 100, 0, 100)
        ut.bookHist(h, "P_low", "Momentum at hit; P [GeV];", 100, 0, 0.1)
        ut.bookHist(h, "hits_per_det", "Hits per strip; n;", 20, 0.5, 20.5)
        ut.bookHist(h, "isolated_hits", "Isolated hits per event; n;", 100, 0.5, 100.5)
        ut.bookHist(h, "true_isolated_hits", "Isolated hits per event (true); n;", 100, 0.5, 100.5)
        ut.bookHist(h, "fake_isolated_hits", "Isolated hits per event (fake); n;", 100, 0.5, 100.5)
        ut.bookHist(h, "isolated_hits_after_tau_plane", "Isolated hits per event; n; plane after #tau", 10, 0.5, 10.5, 20, -0.5, 19.5)
        ut.bookHist(h, "true_isolated_hits_after_tau_plane", "Isolated hits per event (true); n; plane after #tau", 10, 0.5, 10.5, 20, -0.5, 19.5)
        ut.bookHist(h, "fake_isolated_hits_after_tau_plane", "Isolated hits per event (fake); n; plane after #tau", 10, 0.5, 10.5, 20, -0.5, 19.5)
        ut.bookHist(h, "hits_per_det_after_tau_layer", "Hits per strip; n; layer after #tau", 20, 0.5, 20.5, 10, -0.5, 9.5)
        ut.bookHist(h, "hits_per_det_after_tau_plane", "Hits per strip; n; plane after #tau", 20, 0.5, 20.5, 10, -0.5, 9.5)
        ut.bookHist(
            h, "ignored", "Hits with no energy deposit per event;count;", 20, -0.5, 20.5
        )
        ut.bookHist(h, "tau_planes", "planes per #tau;planes;", 41, -0.5, 40.5)
        ut.bookHist(h, "tau_layers", "layers per #tau;layers;", 21, -0.5, 20.5)
        ut.bookHist(
            h,
            "multiplicity_seen",
            "multiplicity at primary vertex (at least one hit);multiplicity;",
            100,
            -0.5,
            100.5,
        )
        ut.bookHist(
            h,
            "daughters_seen",
            "multiplicity at secondary vertex (at least one hit);multiplicity;",
            20,
            -0.5,
            20.5,
        )
        ut.bookHist(h, "delta_x", "#Delta x_#tau;x[um];", 100, 0, 1000)
        ut.bookHist(h, "delta_y", "#Delta y_#tau;y[um];", 100, 0, 1000)
        ut.bookHist(h, "delta_L2", "d_#tau;d[um];", 100, 0, 1000)
        ut.bookHist(h, "min_d_x", "min #Delta x_#tau;x[um];", 100, 0, 1000)
        ut.bookHist(h, "min_d_y", "min #Delta y_#tau;y[um];", 100, 0, 1000)
        ut.bookHist(h, "min_d_2", "min d_#tau;d[um];", 100, 0, 1000)
        ut.bookHist(
            h, "hits_both_rel", "hits with both coordinates;% of total hits;", 20, 0, 1
        )
        ut.bookHist(
            h,
            "hits_only_asymmetry",
            "asymmetry (x-y) of hits with only one coordinate;% of total hits;",
            20,
            -1,
            1,
        )

    counter = 0
    N = ch.GetEntries()
    for event in ch:
        if not (counter % 100):
            print(f"{counter}/{N}")
        counter += 1
        layers = 0
        planes = 0
        hits = {}
        ignored = 0
        if args.digi:
            layers_seen = []
            planes_seen = {}
            first_tau_layer = None
            first_tau_plane = None
            link = event.Digi_TargetHits2MCPoints[0]
            detIDs = {}
            strips = {}  # indexed by sensor
            for hit in event.Digi_advTargetHits:
                station = None
                plane = None
                detID = hit.GetDetectorID()
                wlist = link.wList(detID)
                detIDs[detID] = len(wlist)
                h["hits_per_det"].Fill(len(wlist))
                assert len(wlist), f"{detID=}"
                point_indices = [index for index,_ in wlist]
                for index in point_indices:
                    point = event.AdvTargetPoint[index]

                    # if not point.GetEnergyLoss() > 0.:
                    if point.GetEnergyLoss() < 40 * keV:
                        ignored += 1
                        continue
                    h["ELoss"].Fill(point.GetEnergyLoss() * GeV / keV)
                    pdgID = point.PdgCode()
                    plane = point.GetPlane()
                    station = point.GetStation()
                    absolute_plane = plane + station * 2
                    trackID = point.GetTrackID()
                    px = point.GetPx()
                    py = point.GetPy()
                    pz = point.GetPz()
                    pt = hypot(px, py)
                    P = hypot(pz, pt)
                    h["P"].Fill(P)
                    h["P_low"].Fill(P)
                    x = hit.GetX()
                    if x > -100:
                        h["absolute x"].Fill(x)
                        x_true = point.GetX()
                        h["x-x_true"].Fill((x - x_true) * cm / um)
                    y = hit.GetY()
                    if y > -100:
                        h["absolute y"].Fill(y)
                        y_true = point.GetY()
                        h["y-y_true"].Fill((y - y_true) * cm / um)
                    if trackID not in hits:
                        hits[trackID] = {}
                    if station in hits[trackID]:
                        if hits[trackID][station][0] is None:
                            if x > -100:
                                hits[trackID][station][0] = x
                        elif hits[trackID][station][1] is None:
                            if y > -100:
                                hits[trackID][station][1] = y
                    else:
                        if x > -100:
                            hits[trackID][station] = [x, None]
                        elif y > -100:
                            hits[trackID][station] = [None, y]
                        else:
                            assert False
                    if pdgID in (-15, 15):
                        if not first_tau_plane:
                            first_tau_plane = absolute_plane
                        if not first_tau_layer:
                            first_tau_layer = station
                        if station in layers_seen:
                            if not plane in planes_seen[station]:
                                layers += 1
                                planes += 1
                                planes_seen[station].append(plane)
                        else:
                            layers_seen.append(station)
                            planes_seen[station] = [plane]
                            planes += 1
                if station:
                    if station and first_tau_layer:
                        h["hits_per_det_after_tau_layer"].Fill(len(wlist), station - first_tau_layer)
                    else:
                        h["hits_per_det_after_tau_layer"].Fill(len(wlist), 0)
                    if plane and first_tau_plane:
                        h["hits_per_det_after_tau_plane"].Fill(len(wlist), plane + station * 2 - first_tau_plane)
                    else:
                        h["hits_per_det_after_tau_plane"].Fill(len(wlist), 0)

            # Check whether neighbouring strips fired
            isolated_hits = {}
            true_isolated_hits = {}
            fake_isolated_hits = {}
            for detID in detIDs:
                station = floor(detID >> 15)
                plane = (detID >> 14) % 2
                absolute_plane = int(plane + station * 2)
                if (
                        (detID % 768 == 0) or ((detID - 1) not in detIDs)
                ) and (
                    (detID % 768 == 767) or ((detID + 1) not in detIDs)
                ):
                    if absolute_plane not in isolated_hits:
                        isolated_hits[absolute_plane] = 1
                    else:
                        isolated_hits[absolute_plane] += 1
                    if detIDs[detID] == 1:
                        if absolute_plane not in true_isolated_hits:
                            true_isolated_hits[absolute_plane] = 1
                        else:
                            true_isolated_hits[absolute_plane] += 1
                    else:
                        if absolute_plane not in fake_isolated_hits:
                            fake_isolated_hits[absolute_plane] = 1
                        else:
                            fake_isolated_hits[absolute_plane] += 1
            h["isolated_hits"].Fill(sum(isolated_hits.values()))
            h["fake_isolated_hits"].Fill(sum(fake_isolated_hits.values()))
            h["true_isolated_hits"].Fill(sum(true_isolated_hits.values()))
            if first_tau_plane:
                for plane in range(first_tau_plane, first_tau_plane + 10):
                    if plane in isolated_hits:
                        h["isolated_hits_after_tau_plane"].Fill(isolated_hits[plane], plane)
                        if plane in fake_isolated_hits:
                            h["fake_isolated_hits_after_tau_plane"].Fill(fake_isolated_hits[plane], plane)
                        if plane in true_isolated_hits:
                            h["true_isolated_hits_after_tau_plane"].Fill(true_isolated_hits[plane], plane)


        primary_tracks = 0
        primary_tracks_seen = 0
        primary_tracks_charged = 0
        tau_id = 1
        daughter_ids = []
        secondary_tracks_seen = 0
        secondary_tracks_charged = 0
        tau_start = array([None, None, None])
        daughter_start = array([None, None, None])
        tau_E = None
        tau_dz = None
        nu_tau_E = None
        tau_charge = 0
        daughter_charge = 0
        daughter_pids = []
        processes = []
        process_ids = []
        MET = array([0.0, 0.0, 0.0])
        primaries = []
        taus = None
        for id, track in enumerate(event.MCTrack):
            pdgid = track.GetPdgCode()
            if id == 0:
                assert pdgid in (-16, 16), "No tau neutrino present."
                nu_tau_E = track.GetEnergy()
            if pdgid in (-15, 15):
                assert not taus
                tau_id = id
                assert tau_id == 1
                if args.digi:
                    if id in hits:
                        taus = hits[id]
                tau_start = array(
                    [track.GetStartX(), track.GetStartY(), track.GetStartZ()]
                )
                tau_E = track.GetEnergy()
                tau = Particle.from_pdgid(pdgid)
                tau_charge = tau.charge
                MET += array([track.GetPx(), track.GetPy(), track.GetPz()])
            if track.GetMotherId() == 0:
                primary_tracks += 1
                if args.digi:
                    if id in hits:
                        primary_tracks_seen += 1
                if charge(pdgid):
                    primary_tracks_charged += 1
                    primaries.append(track)
            elif track.GetMotherId() == 1:
                daughter_ids.append(id)
                if args.digi:
                    if id in hits:
                        secondary_tracks_seen += 1
                daughter_pids.append(pdgid)
                process = track.GetProcName()
                process_id = track.GetProcID()
                processes.append(process)
                process_ids.append(process_id)
                if process_id != 9:  # Delta ray
                    daughter_charge += charge(pdgid)
                if charge(pdgid):
                    secondary_tracks_charged += 1
                daughter_start = array(
                    [track.GetStartX(), track.GetStartY(), track.GetStartZ()]
                )
                daughter_dir = array([track.GetPx(), track.GetPy(), track.GetPz()])
                IP = norm(cross(tau_start - daughter_start, daughter_dir)) / norm(
                    daughter_dir
                )
                h["IP"].Fill(IP * cm / um)
                if charge(pdgid):
                    h["IP_charged"].Fill(IP * cm / um)
                    MET -= array([track.GetPx(), track.GetPy(), track.GetPz()])
        if tau_charge != daughter_charge:
            print(
                f"{tau_charge=}, {daughter_charge=}, {daughter_pids=}, {processes=}, {process_ids=}, {MET=}, {norm(MET)=}"
            )
        secondary_tracks = len(daughter_ids)

        h["multiplicity"].Fill(primary_tracks)
        h["multiplicity_charged"].Fill(primary_tracks_charged)
        h["daughters"].Fill(secondary_tracks)
        h["daughters_charged"].Fill(secondary_tracks_charged)
        if args.digi:
            h["multiplicity_seen"].Fill(primary_tracks_seen)
            h["daughters_seen"].Fill(secondary_tracks_seen)
            h["tau_planes"].Fill(planes)
            h["tau_layers"].Fill(layers)
        if daughter_start.any() and tau_start.any():
            tau_dz = daughter_start[2] - tau_start[2]
            h["tau_dz"].Fill(tau_dz)
        if tau_E:
            h["tau_E"].Fill(tau_E)
        if nu_tau_E:
            h["nu_tau_E"].Fill(nu_tau_E)

        if args.digi:
            hit_x = 0
            hit_y = 0
            hit_both = 0
            tau_x = None
            tau_y = None
            # TODO Energy cut?
            min_d_x = None
            min_d_y = None
            min_d_2 = None
            for trackID, track_hits in hits.items():
                for station, [x, y] in track_hits.items():
                    if taus and station in taus:
                        tau_x, tau_y = taus[station]
                    if x:
                        hit_x += 1
                        if tau_x and trackID != 1:
                            delta_x = abs(x - tau_x)
                            h["delta_x"].Fill(delta_x * cm / um)
                            if not min_d_x or delta_x < min_d_x:
                                min_d_x = delta_x
                    if y:
                        hit_y += 1
                        if tau_y and trackID != 1:
                            delta_y = abs(y - tau_y)
                            h["delta_y"].Fill(delta_y * cm / um)
                            if not min_d_y or delta_y < min_d_y:
                                min_d_y = delta_y
                    if x and y:
                        hit_both += 1
                        h["xy"].Fill(x, y)
                        if tau_x and tau_y and trackID != 1:
                            delta_x = x - tau_x
                            delta_y = y - tau_y
                            delta_L2 = sqrt(delta_x ** 2 + delta_y ** 2)
                            h["delta_L2"].Fill(delta_L2 * cm / um)
                            if not min_d_2 or delta_L2 < min_d_2:
                                min_d_2 = delta_L2
            if min_d_x:
                h["min_d_x"].Fill(min_d_x * cm / um)
            if min_d_y:
                h["min_d_y"].Fill(min_d_y * cm / um)
            if min_d_2:
                h["min_d_2"].Fill(min_d_2 * cm / um)

            hit_x_only = hit_x - hit_both
            hit_y_only = hit_y - hit_both
            if hit_x and hit_y:
                h["hits_both_rel"].Fill(hit_both / (hit_x_only + hit_y_only + hit_both))
            if hit_x_only and hit_y_only:
                h["hits_only_asymmetry"].Fill(
                    (hit_x_only - hit_y_only) / (hit_x_only + hit_y_only + hit_both)
                )
                h["ignored"].Fill(ignored)

        if args.truth:
            taus = [t for t in primaries if t.GetPdgCode() in (-15, 15)]
            if taus:
                assert len(taus) == 1
            else:
                continue
            tau = taus[0]
            other_primaries = [t for t in primaries if t.GetPdgCode() not in (-15, 15)]
            for key, config in CONFIGS.items():
                tau_isolation = isolation_distance(tau, other_primaries, **config) if other_primaries else 0
                h["tau_isolation_" + key].Fill(tau_isolation / mm)
                assert tau_dz
                h["tau_isolation_vs_decay_length_" + key].Fill(
                    tau_isolation / mm, tau_dz / mm
                )
                h["tau_isolation_vs_tau_momentum_" + key].Fill(
                    tau_isolation / mm, tau.GetP() / GeV
                )
                for distance in DISTANCES:
                    all_primaries = primaries.copy()
                    isolated = 0
                    not_isolated_tracks = []
                    while True:
                        try:
                            primary = all_primaries.pop()
                        except IndexError:
                            # "pop from empty list"
                            break
                        z = primary.GetStartZ() + distance
                        if is_isolated(
                            primary, all_primaries + not_isolated_tracks, z=z, **config
                        ):
                            isolated += 1
                            h[f"isolated_track_momentum_{key}_{distance}"].Fill(
                                primary.GetP() / GeV
                            )
                            h[f"isolated_track_charge_{key}_{distance}"].Fill(
                                charge(primary.GetPdgCode()))
                        else:
                            not_isolated_tracks.append(primary)
                    h[f"isolated_tracks_{key}_{distance}"].Fill(isolated)

    hists = ROOT.TFile.Open(args.outputfile, "recreate")
    for key in h:
        h[key].Write()
        if args.plots:
            c = ROOT.TCanvas("canvas_" + key, key, 800, 600)
            if isinstance(h[key], ROOT.TH2):
                h[key].Draw("Colz")
                c.SetLogz()
            else:
                h[key].Draw()
            c.Draw()
            c.SaveAs("plots/" + key + ".pdf")
            c.SaveAs("plots/" + key + ".png")
    hists.Close()


if __name__ == "__main__":
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    ROOT.gROOT.SetBatch(True)
    main()
