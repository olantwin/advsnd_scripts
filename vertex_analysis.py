#!/usr/bin/env python3

import argparse
from tqdm import tqdm
import ROOT
import rootUtils as ut
import SNDstyle

ROOT.gInterpreter.Declare(
    """
bool is_fiducial_x(const TVector3 &vtx) {
    auto x = vtx.X();
    auto vertical = (x < -3 && x > -11) || (x < -15 && x > -23) || (x < -27 && x > -35) || (x < -39 && x > -47);
    auto horizontal = (x < -6 && x > -14) || (x < -15 && x > -23) || (x < -27 && x > -35) || (x < -37 && x > -44);
    return horizontal && vertical;
}
"""
)

ROOT.gInterpreter.Declare(
    """
bool is_fiducial_y(const TVector3 &vtx) {
    auto y = vtx.Y();
    auto vertical = (y < 23 && y > 15) || (y < 33 && y > 25) || (y < 44 && y > 36) || (y < 53 && y > 45);
    auto horizontal = (y < 20 && y > 12) || (y < 32 && y > 24) || (y < 44 && y > 36) || (y < 56 && y > 48);
    return horizontal && vertical;
}
"""
)

ROOT.gInterpreter.Declare(
    """
bool is_fiducial_z(const TVector3 &vtx) {
    return vtx.Z() < -90;  // cm
}
"""
)

ROOT.gInterpreter.Declare(
    """
bool is_fiducial(const TVector3 &vtx) {
    return is_fiducial_x(vtx) && is_fiducial_y(vtx) && is_fiducial_z(vtx);
}
"""
)


def main():
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
    tree = inputfile.cbmsim
    f = ROOT.TFile.Open(args.geofile, "read")
    geo = f.FAIRGeom  # noqa: F841

    if args.plots:
        SNDstyle.init_style()

    h = {}
    ut.bookHist(h, "n_vertices", "Number of vertices", 10, -0.5, 9.5)
    ut.bookHist(
        h, "n_vertices_fiducial", "Number of vertices in fiducial volume", 10, -0.5, 9.5
    )
    # ut.bookHist(h, "n_tracks", "Number of tracks per vertex", 100, 1.5, 102.5)
    ut.bookHist(h, "n_tracks", "Number of tracks per vertex", 100, -1, -1)
    ut.bookHist(
        h, "vertex_xy", "Vertex position; x [cm]; y [cm]", 100, -60, 10, 100, 0, 70
    )
    ut.bookHist(
        h,
        "vertex_dxy",
        "Vertex residual; #deltax [cm]; #deltay [cm]",
        100,
        -10,
        10,
        100,
        -10,
        10,
    )
    ut.bookHist(h, "vertex_z", "Vertex position; z [cm]", 100, -150, -70)
    ut.bookHist(h, "vertex_dz", "Vertex residual; #deltaz [cm]", 100, -10, -10)

    for event in tqdm(tree, desc="Event loop: ", total=tree.GetEntries()):
        true_PV = ROOT.TVector3()
        for true_track in event.MCTrack:
            if true_track.GetMotherId() == 0:
                # Find primary muon
                if abs(true_track.GetPdgCode()) == 13:
                    # Find true primary vertex
                    true_PV = ROOT.TVector3(
                        true_track.GetStartX(),
                        true_track.GetStartY(),
                        true_track.GetStartZ(),
                    )

        # Count vertices
        h["n_vertices"].Fill(len(event.RAVE_vertices))
        PV = None
        n_fiducial = 0
        for vertex in event.RAVE_vertices:
            pos = vertex.getPos()
            # Fiducial cut
            if ROOT.is_fiducial(pos):
                n_fiducial += 1
            else:
                continue
            if not PV or pos.Z() < PV.Z():
                PV = pos
            h["vertex_xy"].Fill(pos.X(), pos.Y())
            h["vertex_z"].Fill(pos.Z())
            h["n_tracks"].Fill(vertex.getNTracks())
        if PV:
            h["vertex_dxy"].Fill(PV.X() - true_PV.X(), PV.Y() - true_PV.Y())
            h["vertex_dz"].Fill(PV.Z() - true_PV.Z())
        h["n_vertices_fiducial"].Fill(n_fiducial)

    hists = ROOT.TFile.Open(args.outputfile, "recreate")
    for key, hist in h.items():
        hist.Write()
        if args.plots:
            ROOT.gStyle.SetOptStat(111110)
            c = ROOT.TCanvas("canvas_" + key, key, 800, 600)
            if isinstance(hist, ROOT.TH2):
                hist.Draw("Colz")
                c.SetLogz()
            else:
                hist.Draw()
            c.Draw()
            c.SaveAs("plots/" + key + ".pdf")
            c.SaveAs("plots/" + key + ".png")
    hists.Close()


if __name__ == "__main__":
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    ROOT.gROOT.SetBatch(True)
    main()