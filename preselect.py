#!/usr/bin/env python3
"""Preselect events using second tree with cuts"""

import argparse
from tqdm import tqdm
import ROOT


def main():
    """Preselect events using second tree with cuts"""
    parser = argparse.ArgumentParser(description="Script for AdvSND analysis.")
    parser.add_argument(
        "-f",
        "--inputfile",
        help="""Simulation results to use as input."""
        """Supports retrieving file from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cutfile",
        help="""File with precalculated cuts."""
        """Supports retrieving file from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        help="""File to write the filtered tree to."""
        """Will be recreated if it already exists.""",
    )
    args = parser.parse_args()
    if not args.outputfile:
        args.outputfile = args.inputfile.removesuffix(".root") + "_selected.root"
    inputfile = ROOT.TFile.Open(args.inputfile, "read")
    data_tree = inputfile.cbmsim
    nentries = data_tree.GetEntries()
    cutfile = ROOT.TFile.Open(args.cutfile, "read")
    cut_tree = cutfile.cbmsim
    outputfile = ROOT.TFile.Open(args.outputfile, "recreate")
    out_tree = data_tree.CloneTree(0)
    selected = 0
    for i, cuts in zip(tqdm(range(nentries)), cut_tree):
        data_tree.GetEntry(i)
        if cuts.fiducial_cut and cuts.CC_cut:
            selected += 1
            out_tree.Fill()
    out_tree.Write()
    outputfile.Write()
    print(f"Selected {selected} events out of {nentries}.")


if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    main()
