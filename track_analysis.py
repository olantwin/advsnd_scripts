#!/usr/bin/env python3

import argparse
import ROOT
import rootUtils as ut
from shipunit import cm, um, keV, GeV
from numpy import hypot, array, unique, average, median, fabs
from statistics import mode
from math import atan2
import SNDLHCstyle


h = {}
track_residuals_x = []
track_residuals_y = []

ROTATION = array(
    [
        0.999978,
        -0.006606,
        0.0000821516,
        0.00660651,
        0.999901,
        -0.0124347,
        4.69368e-15,
        0.0124349,
        0.999923,
    ]
)
GEOROTATION = ROOT.TGeoRotation("localSND_physCS_rot")
GEOROTATION.SetMatrix(ROTATION)


def residual(a, b):
    diff = a - b
    diff_rot = array([0.0, 0.0, 0.0])
    GEOROTATION.MasterToLocalVect(diff, diff_rot)
    return diff_rot


def cluster_analysis(event, cluster, prefix="", trackpoint=None):
    link = event.Digi_TargetClusterHits2MCPoints[0]
    detID = cluster.GetDetectorID()
    if f"stripmap_{cluster.isVertical()}" not in h:
        ut.bookHist(
            h,
            f"stripmap_{cluster.isVertical()}",
            f"Strip map {cluster.isVertical()=}; #Delta x [um]; #Delta y [um]",
            767,
            -46848,
            46848,
            767,
            -46848,
            46848,
        )
    A, B = ROOT.TVector3(), ROOT.TVector3()
    cluster.GetPosition(A, B)

    hit_pos = array([(A.X() + B.X()) / 2, (A.Y() + B.Y()) / 2, (A.Z() + B.Z()) / 2])
    if trackpoint:
        # (3) Distance between cluster coordinate and fitted track (this is the residual)
        track_pos = array([trackpoint.X(), trackpoint.Y(), trackpoint.Z()])
        resi = residual(track_pos, hit_pos)
        if not cluster.isVertical():
            h[f"{prefix}dx"].Fill((resi[0]) * cm / um)
            track_residuals_x.append(resi[0])
        else:
            h[f"{prefix}dy"].Fill((resi[1]) * cm / um)
            track_residuals_y.append(resi[1])

    wlist = link.wList(detID)
    detIDs = []
    for index, weight in wlist:
        point = event.AdvTargetPoint[index]
        detIDs.append(point.GetDetectorID())
        true_pos = array([point.GetX(), point.GetY(), point.GetZ()])
        # (2) Distance between cluster coordinate and true coordinate
        resi = residual(true_pos, hit_pos)
        if not cluster.isVertical():
            h[f"{prefix}dx_cluster"].Fill((resi[0]) * cm / um)
            if point.GetDetectorID() == detID:
                h[f"{prefix}dx_cluster_same_strip"].Fill((resi[0]) * cm / um)
        else:
            h[f"{prefix}dy_cluster"].Fill((resi[1]) * cm / um)
            if point.GetDetectorID() == detID:
                h[f"{prefix}dy_cluster_same_strip"].Fill((resi[1]) * cm / um)
        if point.GetDetectorID() == detID:
            h[f"stripmap_{cluster.isVertical()}"].Fill(
                (resi[0]) * cm / um, (resi[1]) * cm / um
            )
    cluster_size = len(unique(detIDs))
    # (1) Cluster size, as you have shown
    h[f"{prefix}cluster_size"].Fill(cluster_size)
    # (4) Charge on the leading strip
    for hit in event.Digi_advTargetHits:
        if hit.GetDetectorID() == detID:
            energy_loss = hit.GetSignal()
            h[f"{prefix}energy_loss_leading_hit"].Fill(energy_loss * GeV / keV)
    # (5) Total cluster charge
    energy_loss = cluster.GetSignal()
    h[f"{prefix}energy_loss_cluster"].Fill(energy_loss * GeV / keV)


def main():
    global track_residuals_x
    global track_residuals_y
    parser = argparse.ArgumentParser(description="Script for AdvSND tracking analysis.")
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
    parser.add_argument(
        "-g",
        "--geofile",
        help="""Simulation results to use as input. """
        """Supports retrieving files from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "--plots", help="Make nice plots as pdf and png", action="store_true"
    )
    args = parser.parse_args()
    inputfile = ROOT.TFile.Open(args.inputfile, "read")
    ch = inputfile.cbmsim
    f = ROOT.TFile.Open(args.geofile, "read")
    geo = f.FAIRGeom  # noqa: F841

    SNDLHCstyle.init_style()
    ut.bookHist(h, "dtx", "#Delta #theta_{x}; #Delta #theta_{x} [mrad]", 100, -50, 50)
    ut.bookHist(h, "dty", "#Delta #theta_{y}; #Delta #theta_{y} [mrad]", 100, -50, 50)
    ut.bookHist(
        h,
        "dtx_P",
        "#Delta #theta_{x}; #Delta #theta_{x} [mrad]; P [GeV]",
        100,
        -50,
        50,
        100,
        0,
        2000,
    )
    ut.bookHist(
        h,
        "dty_P",
        "#Delta #theta_{y}; #Delta #theta_{y} [mrad]; P [GeV]",
        100,
        -50,
        50,
        100,
        0,
        2000,
    )
    ut.bookHist(h, "dx", "#Delta x; #Delta x [um]", 100, -2000, 2000)
    ut.bookHist(h, "dy", "#Delta y; #Delta y [um]", 100, -2000, 2000)
    ut.bookHist(h, "dx_hit", "#Delta x (hit vs point); #Delta x [um]", 100, -2000, 2000)
    ut.bookHist(h, "dy_hit", "#Delta y (hit vs point); #Delta y [um]", 100, -2000, 2000)
    ut.bookHist(
        h,
        "dtx_nhits",
        "#Delta #theta_{x} as a function of hits; number of hits; #Delta #theta_{x} [mrad]",
        100,
        0.5,
        100.5,
        100,
        -50,
        50,
    )
    ut.bookHist(
        h,
        "dty_nhits",
        "#Delta #theta_{y} as a function of hits; number of hits; #Delta #theta_{y} [mrad]",
        100,
        0.5,
        100.5,
        100,
        -50,
        50,
    )
    ut.bookHist(
        h,
        "dx_nhits",
        "#Delta x as a function of hits; number of hits; #Delta x [um]",
        100,
        0.5,
        100.5,
        100,
        -2000,
        2000,
    )
    ut.bookHist(
        h,
        "dy_nhits",
        "#Delta y as a function of hits; number of hits; #Delta y [um]",
        100,
        0.5,
        100.5,
        100,
        -2000,
        2000,
    )
    ut.bookHist(h, "chi2ndf", "#chi^{2}/ndf; #chi^{2}/ndf", 50, 0, 10)
    ut.bookHist(h, "nhits", "nhits; nhits", 100, -0.5, 99.5)
    h["efficiency_vs_hits"] = inputfile.n_hits_converged / (
        inputfile.n_hits_converged + inputfile.n_hits_not_converged
    )
    h["efficiency_vs_hits"].SetName("efficiency_vs_hits")
    h["efficiency_vs_hits"].SetTitle(
        "Efficency as a function of hits; Number of hits; converged/total"
    )
    for prefix in ["all_", "track_"]:
        ut.bookHist(h, f"{prefix}dx", "Residual x; #Delta x [um]", 100, -2000, 2000)
        ut.bookHist(h, f"{prefix}dy", "Residual y; #Delta y [um]", 100, -2000, 2000)
        ut.bookHist(
            h,
            f"{prefix}dx_cluster",
            "#Delta x (cluster vs point); #Delta x [um]",
            100,
            -2000,
            2000,
        )
        ut.bookHist(
            h,
            f"{prefix}dy_cluster",
            "#Delta y (cluster vs point); #Delta y [um]",
            100,
            -2000,
            2000,
        )
        ut.bookHist(
            h,
            f"{prefix}dx_cluster_same_strip",
            "#Delta x (cluster vs point); #Delta x [um]",
            100,
            -2000,
            2000,
        )
        ut.bookHist(
            h,
            f"{prefix}dy_cluster_same_strip",
            "#Delta y (cluster vs point); #Delta y [um]",
            100,
            -2000,
            2000,
        )
        ut.bookHist(h, f"{prefix}cluster_size", "Cluster size", 40, 0.5, 40.5)
        ut.bookHist(
            h,
            f"{prefix}energy_loss_leading_hit",
            "Energy deposit leading hit; Energy deposit [keV]",
            100,
            0,
            1000,
        )
        ut.bookHist(
            h,
            f"{prefix}energy_loss_cluster",
            "Energy deposit cluster; Energy deposit [keV]",
            100,
            0,
            1000,
        )
    ut.bookHist(
        h, "track_min_dx", "Min residual (per track) x; #Delta x [um]", 100, 0, 2000
    )
    ut.bookHist(
        h,
        "track_unsigned_average_dx",
        "Average unsigned residual (per track) x; #Delta x [um]",
        100,
        0,
        1000,
    )
    ut.bookHist(
        h,
        "track_unsigned_median_dx",
        "Median unsigned residual (per track) x; #Delta x [um]",
        100,
        0,
        1000,
    )
    ut.bookHist(
        h,
        "track_average_dx",
        "Average residual (per track) x; #Delta x [um]",
        100,
        -1000,
        1000,
    )
    ut.bookHist(
        h,
        "track_median_dx",
        "Median residual (per track) x; #Delta x [um]",
        100,
        -1000,
        1000,
    )
    ut.bookHist(
        h,
        "track_average_v_max_dx",
        "Average residual vs. max (per track) x; #Delta x [um], #Delta x [um]",
        100,
        -1000,
        1000,
        100,
        0,
        1000,
    )
    ut.bookHist(
        h,
        "track_unsigned_average_v_max_dx",
        "Average unsigned residual vs. max (per track) x; #Delta x [um], #Delta x [um]",
        100,
        0,
        1000,
        100,
        0,
        1000,
    )
    ut.bookHist(
        h,
        "track_unsigned_average_v_P_dx",
        "Average unsigned residual x vs. true momentum; #Delta x [um]; P [GeV]",
        100,
        0,
        1000,
        100,
        0,
        2000,
    )
    ut.bookHist(
        h,
        "track_unsigned_average_v_Pt_dx",
        "Average unsigned residual x vs. true momentum; #Delta x [um]; P_{T} [GeV]",
        100,
        0,
        1000,
        100,
        0,
        100,
    )
    ut.bookHist(
        h,
        "track_unsigned_average_v_Pl_dx",
        "Average unsigned residual x vs. true momentum; #Delta x [um]; P_{L} [GeV]",
        100,
        0,
        1000,
        100,
        0,
        2000,
    )
    ut.bookHist(
        h, "track_max_dx", "Max residual (per track) x; #Delta x [um]", 100, 0, 2000
    )
    ut.bookHist(
        h, "track_min_dy", "Min residual (per track) y; #Delta y [um]", 100, 0, 2000
    )
    ut.bookHist(
        h,
        "track_unsigned_average_dy",
        "Average unsigned residual (per track) y; #Delta y [um]",
        100,
        0,
        1000,
    )
    ut.bookHist(
        h,
        "track_unsigned_median_dy",
        "Median unsigned residual (per track) y; #Delta y [um]",
        100,
        0,
        1000,
    )
    ut.bookHist(
        h,
        "track_average_dy",
        "Average residual (per track) y; #Delta y [um]",
        100,
        -1000,
        1000,
    )
    ut.bookHist(
        h,
        "track_median_dy",
        "Median residual (per track) y; #Delta y [um]",
        100,
        -1000,
        1000,
    )
    ut.bookHist(
        h,
        "track_average_v_max_dy",
        "Average residual vs. max (per track) y; #Delta y [um], #Delta y [um]",
        100,
        -1000,
        1000,
        100,
        0,
        1000,
    )
    ut.bookHist(
        h,
        "track_unsigned_average_v_max_dy",
        "Average unsigned residual vs. max (per track) y; #Delta y [um], #Delta y [um]",
        100,
        0,
        1000,
        100,
        0,
        1000,
    )
    ut.bookHist(
        h,
        "track_unsigned_average_v_P_dy",
        "Average unsigned residual y vs. true momentum; #Delta y [um]; P [GeV]",
        100,
        0,
        1000,
        100,
        0,
        2000,
    )
    ut.bookHist(
        h,
        "track_unsigned_average_v_Pt_dy",
        "Average unsigned residual y vs. true momentum; #Delta y [um]; P_{T} [GeV]",
        100,
        0,
        1000,
        100,
        0,
        100,
    )
    ut.bookHist(
        h,
        "track_unsigned_average_v_Pl_dy",
        "Average unsigned residual y vs. true momentum; #Delta y [um]; P_{L} [GeV]",
        100,
        0,
        1000,
        100,
        0,
        2000,
    )
    ut.bookHist(
        h, "track_max_dy", "Max residual (per track) y; #Delta y [um]", 100, 0, 2000
    )
    ut.bookHist(h, "true_momenta", "True track momenta; P [GeV]", 100, 0, 2000)
    ut.bookHist(
        h,
        "track_residual_x_high_P",
        "Residual x (p_{true} > 200 GeV); #delta x [um]",
        100,
        -100,
        100,
    )
    ut.bookHist(
        h,
        "track_residual_y_high_P",
        "Residual y (p_{true} > 200 GeV); #delta x [um]",
        100,
        -100,
        100,
    )

    nev = 0
    track_id = 0
    for event in ch:
        nev += 1
        link = event.Digi_TargetClusterHits2MCPoints[0]
        for track in event.Reco_MuonTracks:
            h["chi2ndf"].Fill(track.getChi2Ndf())

            tx = track.getAngleXZ()
            ty = track.getAngleYZ()

            trackIDs = []
            points = track.getTrackPoints()
            detIDs = track.getRawMeasDetIDs()
            hits = {}  # detID: index
            for i, hit in enumerate(event.Digi_advTargetClusters):
                hits[hit.GetDetectorID()] = i
            assert len(points) == len(detIDs)
            h["nhits"].Fill(len(points))
            for point, detID in zip(points, detIDs):
                hit = event.Digi_advTargetClusters[hits[detID]]
                cluster_analysis(event, hit, prefix="track_", trackpoint=point)
                A, B = ROOT.TVector3(), ROOT.TVector3()
                hit.GetPosition(A, B)

                hit_pos = (A.X() + B.X()) / 2, (A.Y() + B.Y()) / 2, (A.Z() + B.Z()) / 2

                x, y = point.X(), point.Y()
                true_pos = None
                wlist = link.wList(detID)
                for index, weight in wlist:
                    point = event.AdvTargetPoint[index]
                    true_pos = point.GetX(), point.GetY(), point.GetZ()
                    if not hit.isVertical():
                        h["dx_hit"].Fill((true_pos[0] - hit_pos[0]) * cm / um)
                    else:
                        h["dy_hit"].Fill((true_pos[1] - hit_pos[1]) * cm / um)
                    trackID = point.GetTrackID()
                    if trackID != -2:
                        trackIDs.append(trackID)

                assert len(unique(trackIDs)) == 1

                if not hit.isVertical():
                    true_x = hit_pos[0]
                    h["dx"].Fill((x - true_x) * cm / um)
                    h["dx_nhits"].Fill(len(points), (x - true_x) * cm / um)
                else:
                    true_y = hit_pos[1]
                    h["dy"].Fill((y - true_y) * cm / um)
                    h["dy_nhits"].Fill(len(points), (y - true_y) * cm / um)

            # Choose most common track ID for true track
            trackID = mode(trackIDs)
            true_track = event.MCTrack[trackID]

            true_mom = array(
                [true_track.GetPx(), true_track.GetPy(), true_track.GetPz()]
            )

            #  Plot momentum distribution
            h["true_momenta"].Fill(true_track.GetP())

            true_tx = atan2(true_mom[0], true_mom[2])
            true_ty = atan2(true_mom[1], true_mom[2])
            h["dtx"].Fill(1000 * (tx - true_tx))
            h["dtx_P"].Fill(1000 * (tx - true_tx), true_track.GetP())
            h["dtx_nhits"].Fill(len(points), 1000 * (tx - true_tx))
            h["dty"].Fill(1000 * (ty - true_ty))
            h["dty_P"].Fill(1000 * (tx - true_tx), true_track.GetP())
            h["dty_nhits"].Fill(len(points), 1000 * (ty - true_ty))

            if track_residuals_x:
                unsigned_track_residuals_x = fabs(track_residuals_x)
                h["track_min_dx"].Fill((min(unsigned_track_residuals_x)) * cm / um)
                h["track_average_dx"].Fill((average(track_residuals_x)) * cm / um)
                h["track_average_v_max_dx"].Fill(
                    (average(track_residuals_x)) * cm / um,
                    (max(unsigned_track_residuals_x)) * cm / um,
                )
                h["track_unsigned_average_v_max_dx"].Fill(
                    (average(unsigned_track_residuals_x)) * cm / um,
                    (max(unsigned_track_residuals_x)) * cm / um,
                )
                h["track_unsigned_average_v_P_dx"].Fill(
                    (average(unsigned_track_residuals_x)) * cm / um, true_track.GetP()
                )
                h["track_unsigned_average_v_Pl_dx"].Fill(
                    (average(unsigned_track_residuals_x)) * cm / um, true_track.GetPz()
                )
                h["track_unsigned_average_v_Pt_dx"].Fill(
                    (average(unsigned_track_residuals_x)) * cm / um,
                    hypot(true_track.GetPx(), true_track.GetPy()),
                )
                h["track_median_dx"].Fill((median(track_residuals_x)) * cm / um)
                h["track_unsigned_average_dx"].Fill(
                    (average(unsigned_track_residuals_x)) * cm / um
                )
                h["track_unsigned_median_dx"].Fill(
                    (median(unsigned_track_residuals_x)) * cm / um
                )
                h["track_max_dx"].Fill((max(unsigned_track_residuals_x)) * cm / um)
                if true_track.GetP() > 200 * GeV:
                    for resi in track_residuals_x:
                        h["track_residual_x_high_P"].Fill(resi * cm / um)
                track_residuals_x = []
                if average(unsigned_track_residuals_x) > 200 * um:
                    print(nev)
            if track_residuals_y:
                unsigned_track_residuals_y = fabs(track_residuals_y)
                h["track_min_dy"].Fill((min(unsigned_track_residuals_y)) * cm / um)
                h["track_average_dy"].Fill((average(track_residuals_y)) * cm / um)
                h["track_average_v_max_dy"].Fill(
                    (average(track_residuals_y)) * cm / um,
                    (max(unsigned_track_residuals_y)) * cm / um,
                )
                h["track_unsigned_average_v_max_dy"].Fill(
                    (average(unsigned_track_residuals_y)) * cm / um,
                    (max(unsigned_track_residuals_y)) * cm / um,
                )
                h["track_unsigned_average_v_P_dy"].Fill(
                    (average(unsigned_track_residuals_y)) * cm / um, true_track.GetP()
                )
                h["track_unsigned_average_v_Pl_dy"].Fill(
                    (average(unsigned_track_residuals_y)) * cm / um, true_track.GetPz()
                )
                h["track_unsigned_average_v_Pt_dy"].Fill(
                    (average(unsigned_track_residuals_y)) * cm / um,
                    hypot(true_track.GetPx(), true_track.GetPy()),
                )
                h["track_median_dx"].Fill((median(track_residuals_x)) * cm / um)
                h["track_unsigned_average_dx"].Fill(
                    (average(unsigned_track_residuals_x)) * cm / um
                )
                h["track_unsigned_median_dx"].Fill(
                    (median(unsigned_track_residuals_x)) * cm / um
                )
                h["track_median_dy"].Fill((median(track_residuals_y)) * cm / um)
                h["track_unsigned_average_dy"].Fill(
                    (average(unsigned_track_residuals_y)) * cm / um
                )
                h["track_unsigned_median_dy"].Fill(
                    (median(unsigned_track_residuals_y)) * cm / um
                )
                h["track_max_dy"].Fill((max(unsigned_track_residuals_y)) * cm / um)
                if true_track.GetP() > 200 * GeV:
                    for resi in track_residuals_y:
                        h["track_residual_y_high_P"].Fill(resi * cm / um)
                track_residuals_y = []
                if average(unsigned_track_residuals_y) > 200 * um:
                    print(nev)
            track_id += 1

        for cluster in event.Digi_advTargetClusters:
            cluster_analysis(event, cluster, prefix="all_")

    hists = ROOT.TFile.Open(args.outputfile, "recreate")
    for key in h:
        if key in ["dx", "dy", "dtx", "dty"]:
            position = key in ["dx", "dy"]
            hist_range = (-500, 500) if position else (-10, 10)
            core = ROOT.TF1(f"core_{key}", "gaus", *hist_range)
            core.SetParameters(0, 50 if position else 0.5)
            tails = ROOT.TF1(f"tails_{key}", "gaus", *hist_range)
            tails.SetParameters(0, 350 if position else 5)
            ntotal = h[key].GetEntries()
            model_norm = ROOT.TF1NormSum(core, tails, 0.5 * ntotal, 0.5 * ntotal)
            model = ROOT.TF1(  # noqa: F841
                f"model_{key}", model_norm, *hist_range, model_norm.GetNpar()
            )
            h[key].Fit(f"model_{key}", "ME")
        h[key].Write()
        if args.plots:
            ROOT.gStyle.SetOptStat(111110)
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
