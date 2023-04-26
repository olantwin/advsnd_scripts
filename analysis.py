#!/usr/bin/env python3

import argparse
import ROOT
import rootUtils as ut
from shipunit import cm, um, keV, GeV
from numpy import sqrt, hypot, array, cross, dot
from numpy.linalg import norm
from particle.pdgid import charge
import SNDLHCstyle


def main():
    parser = argparse.ArgumentParser(description="Script for AdvSND analysis.")
    parser.add_argument(
        "inputfile",
        help="""Simulation results to use as input. """
        """Supports retrieving files from EOS via the XRootD protocol.""",
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        default="hists.root",
        help="""File to write the flux maps to. """
        """Will be recreated if it already exists.""",
    )
    args = parser.parse_args()
    ch = ROOT.TChain("cbmsim")
    ch.Add(args.inputfile)
    n = ch.GetEntries()

    SNDLHCstyle.init_style()
    h = {}
    ut.bookHist(h, "absolute x", "absolute x;x[cm];", 100, -50, 0)
    ut.bookHist(h, "absolute y", "absolute y;y[cm];", 100, 10, 60)
    ut.bookHist(h, "tau_dz", "tau flight distance;d[cm];", 100, 0, 10)
    ut.bookHist(h, "xy", ";x[cm];y[cm]", 100, -50, 0, 100, 10, 60)
    ut.bookHist(h, "x-x_true", "#Delta x;x[um];", 100, -100, 100)
    ut.bookHist(h, "y-y_true", "#Delta y;y[um];", 100, -100, 100)
    ut.bookHist(h, "tau_planes", "planes per #tau;planes;", 41, -0.5, 40.5)
    ut.bookHist(h, "tau_layers", "layers per #tau;layers;", 21, -0.5, 20.5)
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
    ut.bookHist(h, "delta_x", "#Delta x_#tau;x[um];", 100, 0, 1000)
    ut.bookHist(h, "delta_y", "#Delta y_#tau;y[um];", 100, 0, 1000)
    ut.bookHist(h, "delta_L2", "d_#tau;d[um];", 100, 0, 1000)
    ut.bookHist(h, "min_d_x", "min #Delta x_#tau;x[um];", 100, 0, 1000)
    ut.bookHist(h, "min_d_y", "min #Delta y_#tau;y[um];", 100, 0, 1000)
    ut.bookHist(h, "min_d_2", "min d_#tau;d[um];", 100, 0, 1000)
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
        "multiplicity_seen",
        "multiplicity at primary vertex (at least one hit);multiplicity;",
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
        "daughters_seen",
        "multiplicity at secondary vertex (at least one hit);multiplicity;",
        20,
        -0.5,
        20.5,
    )
    ut.bookHist(
        h,
        "daughters_charged",
        "multiplicity at secondary vertex (charged);multiplicity;",
        20,
        -0.5,
        20.5,
    )
    ut.bookHist(
        h, "ignored", "Hits with no energy deposit per event;count;", 20, -0.5, 20.5
    )
    ut.bookHist(h, "P", "Momentum at hit; P [GeV];", 100, 0, 100)
    ut.bookHist(h, "P_low", "Momentum at hit; P [GeV];", 100, 0, 0.1)
    ut.bookHist(h, "tau_E", "#tau energy; E [GeV/c^{2}];", 100, 0, 2500)
    ut.bookHist(h, "nu_tau_E", "#nu_{#tau} energy; E [GeV/c^{2}];", 100, 0, 2500)
    ut.bookHist(h, "ELoss", "Energy loss; E [keV/c^{2}];", 1000, 0, 1000)

    counter = 0
    N = ch.GetEntries()
    for event in ch:
        if not (counter % 100):
            print(f"{counter}/{N}")
        link = event.Digi_TargetHits2MCPoints[0]
        layers = 0
        planes = 0
        layers_seen = []
        planes_seen = {}
        taus = None
        hits = {}
        ignored = 0
        for index, hit in enumerate(event.Digi_advTargetHits):
            detID = hit.GetDetectorID()
            point = event.AdvTargetPoint[index]
            # if not point.GetEnergyLoss() > 0.:
            if point.GetEnergyLoss() < 40 * keV:
                ignored += 1
                continue
            h["ELoss"].Fill(point.GetEnergyLoss() * GeV / keV)
            pdgID = point.PdgCode()
            plane = point.GetPlane()
            station = point.GetStation()
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
                if station in layers_seen:
                    if not plane in planes_seen[station]:
                        layers += 1
                        planes += 1
                        planes_seen[station].append(plane)
                else:
                    layers_seen.append(station)
                    planes_seen[station] = [plane]
                    planes += 1

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
        nu_tau_E = None
        tau_charge = 0
        daughter_charge = 0
        daughter_pids = []
        processes = []
        process_ids = []
        MET = array([0.0, 0.0, 0.0])
        for id, track in enumerate(event.MCTrack):
            pdgid = track.GetPdgCode()
            if id == 0:
                assert pdgid in (-16, 16), "No tau neutrino present."
                nu_tau_E = track.GetEnergy()
            if pdgid in (-15, 15):
                assert not taus
                tau_id = id
                assert tau_id == 1
                if id in hits:
                    taus = hits[id]
                tau_start = array(
                    [track.GetStartX(), track.GetStartY(), track.GetStartZ()]
                )
                tau_E = track.GetEnergy()
                tau_charge = charge(pdgid)
                MET += array([track.GetPx(), track.GetPy(), track.GetPz()])
            if track.GetMotherId() == 0:
                primary_tracks += 1
                if id in hits:
                    primary_tracks_seen += 1
                if charge(pdgid):
                    primary_tracks_charged += 1
            elif track.GetMotherId() == 1:
                daughter_ids.append(id)
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
        # print(daughter_ids)

        h["multiplicity"].Fill(primary_tracks)
        h["multiplicity_seen"].Fill(primary_tracks_seen)
        h["multiplicity_charged"].Fill(primary_tracks_charged)
        h["tau_planes"].Fill(planes)
        h["tau_layers"].Fill(layers)
        h["daughters"].Fill(secondary_tracks)
        h["daughters_seen"].Fill(secondary_tracks_seen)
        h["daughters_charged"].Fill(secondary_tracks_charged)
        if daughter_start.any() and tau_start.any():
            h["tau_dz"].Fill(daughter_start[2] - tau_start[2])
        if tau_E:
            h["tau_E"].Fill(tau_E)
        if nu_tau_E:
            h["nu_tau_E"].Fill(nu_tau_E)

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
        # print(f"Number of hits with x: {hit_x}")
        # print(f"Number of hits with y: {hit_y}")
        # print(f"Number of hits with x only: {hit_x_only}")
        # print(f"Number of hits with y only: {hit_y_only}")
        # print(f"Number of hits with two coordinates: {hit_both}")
        if hit_x and hit_y:
            h["hits_both_rel"].Fill(hit_both / (hit_x_only + hit_y_only + hit_both))
        if hit_x_only and hit_y_only:
            h["hits_only_asymmetry"].Fill(
                (hit_x_only - hit_y_only) / (hit_x_only + hit_y_only + hit_both)
            )
            h["ignored"].Fill(ignored)
            counter += 1


    hists = ROOT.TFile.Open(args.outputfile, "recreate")
    for key in h:
        h[key].Write()
        c = ROOT.TCanvas("canvas_" + key, key, 800, 600)
        if isinstance(h[key], ROOT.TH2):
            h[key].Draw("Colz")
            c.SetLogz()
        else:
            h[key].Draw()
        c.Draw()
        c.SaveAs(key + ".pdf")
        c.SaveAs(key + ".png")
    hists.Close()


if __name__ == "__main__":
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    ROOT.gROOT.SetBatch(True)
    main()
