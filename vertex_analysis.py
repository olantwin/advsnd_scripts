#!/usr/bin/env python3
"""Analyse performance of vertexing."""

import argparse
from tqdm import tqdm
import ROOT
import rootUtils as ut
from truth_match import find_MC_track, match_vertex

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
    return (vtx.Z() < -90) && (vtx.Z() > -150);  // cm
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


def find_true_vertex(track, event):
    """Find true vertex for truth matched track.

    returns TVector3 or None
    """
    track_id = track.getMcTrackId()
    if 0 <= track_id < len(event.MCTrack):
        mc_track = event.MCTrack[track_id]
        true_vertex = ROOT.TVector3()
        mc_track.GetStartVertex(true_vertex)
        return true_vertex
    return None


def impact_parameter(track, vertex):
    """Calculate IP."""
    daughter_start = track.getPos()
    daughter_dir = track.getMom()
    return (vertex - daughter_start).Cross(daughter_dir).Mag() / daughter_dir.Mag()


def main():
    """Analyse performance of vertexing."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        # SNDstyle.init_style()
        pass

    h = {}
    ut.bookHist(h, "n_vertices", "Number of vertices", 10, -0.5, 9.5)
    ut.bookHist(
        h, "n_vertices_fiducial", "Number of vertices in fiducial volume", 10, -0.5, 9.5
    )
    ut.bookHist(h, "n_tracks", "Number of tracks per vertex", 100, -0.5, 99.5)
    ut.bookHist(h, "n_tracks_event", "Number of tracks per event", 100, -0.5, 99.5)
    ut.bookHist(
        h,
        "n_tracks_matched_event",
        "Number of truth matched tracks per event",
        50,
        -0.5,
        49.5,
    )
    ut.bookHist(h, "n_hits_track", "Number of hits per track", 100, -0.5, 99.5)
    ut.bookHist(h, "vertex_ip", "IP wrt. reconstructed vertex; IP [cm]", 100, 0, 1)
    ut.bookHist(
        h,
        "vertex_dist_to_closest_point",
        "Vertex distance to closest hit used for tracking; min(d) [cm]",
        100,
        0,
        10,
    )
    ut.bookHist(
        h,
        "vertex_dist_to_points",
        "Vertex distance to hits used for tracking; d [cm]",
        100,
        0,
        10,
    )
    ut.bookHist(h, "vertex_ip_true", "IP wrt. true vertex; IP [cm]", 100, 0, 1)
    ut.bookHist(
        h, "vertex_xy", "Vertex position; x [cm]; y [cm]", 100, -60, 10, 100, 0, 70
    )
    ut.bookHist(
        h,
        "vertex_dxy",
        "Vertex residual; #deltax [cm]; #deltay [cm]",
        1000,
        -10,
        10,
        1000,
        -10,
        10,
    )
    ut.bookHist(
        h,
        "vertex_dxy_zoom",
        "Vertex residual; #deltax [cm]; #deltay [cm]",
        100,
        -1,
        1,
        100,
        -1,
        1,
    )
    ut.bookHist(h, "vertex_z", "Vertex position; z [cm]", 100, -150, -70)
    ut.bookHist(h, "vertex_chi2", "Vertex #chi^{2}; #chi^{2}", 100, 0, 100)
    ut.bookHist(h, "vertex_ndf", "Vertex NDF; NDF", 100, -0.5, 99.5)
    ut.bookHist(h, "vertex_chi2ndf", "Vertex #chi^{2}/NDF; #chi^{2}/NDF", 100, 0, 10)
    ut.bookHist(h, "vertex_dx", "Vertex residual; #deltax [cm]", 100, -0.1, 0.1)
    ut.bookHist(h, "vertex_dy", "Vertex residual; #deltay [cm]", 100, -0.1, 0.1)
    ut.bookHist(h, "vertex_dz", "Vertex residual; #deltaz [cm]", 100, -1, 1)
    ut.bookHist(h, "vertex_dz_zoom", "Vertex residual; #deltaz [cm]", 100, -0.1, 0.1)

    ut.bookHist(
        h,
        "vertex_matched_dx",
        "Vertex residual (truth matched); #deltax [cm]",
        100,
        -0.1,
        0.1,
    )
    ut.bookHist(
        h,
        "vertex_matched_dy",
        "Vertex residual (truth matched); #deltay [cm]",
        100,
        -0.1,
        0.1,
    )
    ut.bookHist(
        h,
        "vertex_matched_dz",
        "Vertex residual (truth matched); #deltaz [cm]",
        100,
        -1,
        1,
    )
    ut.bookHist(
        h,
        "vertex_track_true_momentum",
        "True momentum of truth matched track for truth matched vertex; P [GeV/c]",
        100,
        -1,
        -1,
    )

    cuts = {
        # "all": 3525,
        # "secondary #mu": 2682,
        "true primary vertex in FV": 568,
        "at least two track candidates": 0,  # initialise
        "at least two tracks": 0,  # initialise
        "at least one vertex": 0,  # initialise
        "good vertex": 0,  # initialise
        "vertex in FV": 0,  # initialise
    }

    for event in tqdm(tree, desc="Event loop: ", total=tree.GetEntries()):
        true_primary_vertex = ROOT.TVector3()
        for true_track in event.MCTrack:
            if true_track.GetMotherId() == 0:
                # Find primary muon
                if abs(true_track.GetPdgCode()) == 13:
                    # Find true primary vertex
                    true_primary_vertex = ROOT.TVector3(
                        true_track.GetStartX(),
                        true_track.GetStartY(),
                        true_track.GetStartZ(),
                    )

        if len(event.track_candidates) >= 2:
            cuts["at least two track candidates"] += 1
        else:
            continue

        if (n_tracks := len(event.genfit_tracks)) >= 2:
            cuts["at least two tracks"] += 1
            h["n_tracks_event"].Fill(n_tracks)
        else:
            continue

        matched = 0
        for track in event.genfit_tracks:
            h["n_hits_track"].Fill(track.getNumPoints())
            track_id = find_MC_track(track, event)
            track.setMcTrackId(track_id)
            if track_id >= 0:
                matched += 1
        h["n_tracks_matched_event"].Fill(matched)

        # Count vertices
        if n_vertices := len(event.RAVE_vertices):
            cuts["at least one vertex"] += 1
            h["n_vertices"].Fill(n_vertices)
        else:
            continue
        primary_vertex = None
        n_fiducial = 0
        n_good_vertices = 0
        for vertex in event.RAVE_vertices:
            chi2 = vertex.getChi2()
            h["vertex_chi2"].Fill(chi2)
            ndf = vertex.getNdf()
            h["vertex_ndf"].Fill(ndf)
            h["vertex_chi2ndf"].Fill(chi2 / ndf)
            if (chi2 / ndf) > 20:
                continue
            n_good_vertices += 1
            pos = vertex.getPos()
            # Fiducial cut
            if ROOT.is_fiducial(pos):
                n_fiducial += 1
            else:
                pass
            if not primary_vertex or pos.Z() < primary_vertex.Z():
                primary_vertex = pos
            h["vertex_xy"].Fill(pos.X(), pos.Y())
            h["vertex_z"].Fill(pos.Z())
            h["n_tracks"].Fill(vertex.getNTracks())
            chi2 = vertex.getChi2()
            h["vertex_chi2"].Fill(chi2)
            ndf = vertex.getNdf()
            h["vertex_chi2ndf"].Fill(chi2 / ndf)
            point_distances = []
            for i in range(vertex.getNTracks()):
                track_pars = vertex.getParameters(i)
                ip = impact_parameter(track_pars, pos)
                h["vertex_ip"].Fill(ip)
                track = track_pars.getTrack()
                if track.getMcTrackId() == -1:
                    continue
                for j in range(track.getNumPoints()):
                    point = track.getFittedState(j).getPos()
                    dist = (point - pos).Mag()
                    point_distances.append(dist)
                    h["vertex_dist_to_points"].Fill(dist)
                true_vertex = find_true_vertex(track, event)
                if true_vertex:
                    true_ip = impact_parameter(track_pars, true_vertex)
                    h["vertex_ip_true"].Fill(true_ip)
                    track_id = track.getMcTrackId()
                    mc_track = event.MCTrack[track_id]
                    h["vertex_track_true_momentum"].Fill(mc_track.GetP())
                h["vertex_dist_to_closest_point"].Fill(min(point_distances))
            if true_vertex := match_vertex(vertex, event):
                h["vertex_matched_dx"].Fill(pos.X() - true_vertex.X())
                h["vertex_matched_dy"].Fill(pos.Y() - true_vertex.Y())
                h["vertex_matched_dz"].Fill(pos.Z() - true_vertex.Z())

        if n_good_vertices:
            cuts["good vertex"] += 1
        if primary_vertex:
            cuts["vertex in FV"] += 1
            h["vertex_dx"].Fill(primary_vertex.X() - true_primary_vertex.X())
            h["vertex_dy"].Fill(primary_vertex.Y() - true_primary_vertex.Y())
            h["vertex_dxy"].Fill(
                primary_vertex.X() - true_primary_vertex.X(),
                primary_vertex.Y() - true_primary_vertex.Y(),
            )
            h["vertex_dxy_zoom"].Fill(
                primary_vertex.X() - true_primary_vertex.X(),
                primary_vertex.Y() - true_primary_vertex.Y(),
            )
            h["vertex_dz"].Fill(primary_vertex.Z() - true_primary_vertex.Z())
            h["vertex_dz_zoom"].Fill(primary_vertex.Z() - true_primary_vertex.Z())
        h["n_vertices_fiducial"].Fill(n_fiducial)
    # Cutflow histogram
    h["cutflow"] = ROOT.TH1F("cutflow", "Cut yields", len(cuts), 0, len(cuts))
    h["cuteff"] = ROOT.TH1F("cuteff", "Cut efficiency", len(cuts), 0, len(cuts))
    h["cutcum"] = ROOT.TH1F("cutcum", "Cuts cum. eff.", len(cuts), 0, len(cuts))
    for i, cutname in enumerate(cuts.keys()):
        if i == 0:
            h["cutflow"].SetAxisRange(0, cuts[cutname] * 1.05, "Y")
        h["cutflow"].GetXaxis().SetBinLabel(i + 1, cutname)
        h["cutflow"].SetBinContent(i + 1, cuts[cutname])
        h["cuteff"].GetXaxis().SetBinLabel(i + 1, cutname)
        h["cuteff"].SetBinContent(
            i + 1,
            (h["cutflow"].GetBinContent(i + 1) / h["cutflow"].GetBinContent(i))
            if i
            else 1,
        )
        h["cutcum"].GetXaxis().SetBinLabel(i + 1, cutname)
        h["cutcum"].SetBinContent(
            i + 1,
            (h["cuteff"].GetBinContent(i + 1) * h["cutcum"].GetBinContent(i))
            if i
            else 1,
        )
        if not cuts[cutname]:
            break
    h["cuteff"].SetAxisRange(0, 1.05, "Y")
    h["cutcum"].SetAxisRange(0, 1.05, "Y")

    hists = ROOT.TFile.Open(args.outputfile, "recreate")
    for key in inputfile.GetListOfKeys():
        if key.GetClassName() in ["TH1D", "TH2D"]:
            hist = key.ReadObj()
            hist.Write()

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
