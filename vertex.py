#!/usr/bin/env python3
"""Standalone vertexing implementation."""

import argparse
import logging
from tqdm import tqdm
import ROOT


def main():
    """Vertex tracks."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    geofile = ROOT.TFile.Open(args.geofile, "read")
    geo = geofile.FAIRGeom  # noqa: F841
    if not args.outputfile:
        args.outputfile = args.inputfile.removesuffix(".root") + "_vertexed.root"
    ROOT.gInterpreter.Declare('#include "TGeoMaterialInterface.h"')
    ROOT.gInterpreter.Declare('#include "MaterialEffects.h"')
    ROOT.gInterpreter.Declare('#include "FieldManager.h"')
    ROOT.gInterpreter.Declare('#include "ConstField.h"')
    geo_mat = ROOT.genfit.TGeoMaterialInterface()
    ROOT.genfit.MaterialEffects.getInstance().init(geo_mat)
    bfield = ROOT.genfit.ConstField(0, 0, 0)
    field_manager = ROOT.genfit.FieldManager.getInstance()
    field_manager.init(bfield)
    ROOT.genfit.MaterialEffects.getInstance().setNoEffects()
    ROOT.gInterpreter.Declare('#include "GFRaveVertexFactory.h"')

    inputfile = ROOT.TFile.Open(args.inputfile, "read")
    tree = inputfile.cbmsim
    if tree.GetBranch("Reco_MuonTracks"):
        tree.SetAlias("genfit_tracks", "Reco_MuonTracks")

    outputfile = ROOT.TFile.Open(args.outputfile, "recreate")
    out_tree = tree.CloneTree(0)
    for key in inputfile.GetListOfKeys():
        if key.GetClassName() in ["TH1D", "TH2D"]:
            hist = key.ReadObj()
            hist.Write()

    vertex_factory = ROOT.genfit.GFRaveVertexFactory()
    vertex_factory.setMethod("avr-smoothing:1")
    vertices = ROOT.std.vector("genfit::GFRaveVertex*")()
    out_tree.Branch("RAVE_vertices", vertices)

    for event in tqdm(tree, desc="Event loop: ", total=tree.GetEntries()):
        vertices.clear()
        if len(event.genfit_tracks) >= 2:
            vertex_factory.findVertices(vertices, event.genfit_tracks)
        out_tree.Fill()
    out_tree.Write()
    outputfile.Write()


if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    logging.basicConfig(level=logging.INFO)
    main()
