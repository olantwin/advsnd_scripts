#!/usr/bin/env python3
"""Standalone truth matching"""

import argparse
import logging
from collections import Counter
from tqdm import tqdm
import ROOT


def find_MC_track(track, event):
    """Match track to MC track.

    returns MC track index or -1
    """
    link = event.Digi_TargetClusterHits2MCPoints[0]
    points = track.getPoints()
    track_ids = []
    for p in points:
        wlist = link.wList(p.getRawMeasurement().getDetId())
        for index, _ in wlist:
            point = event.AdvTargetPoint[index]
            track_id = point.GetTrackID()
            if track_id == -2:
                continue
            track_ids.append(track_id)
    if not track_ids:
        return -1
    most_common_track, count = Counter(track_ids).most_common(1)[0]
    if count >= len(points) * 0.7:
        # truth match if ≥ 70 % of hits are related to a single MCTrack
        return most_common_track
    return -1


def match_vertex(vertex, event):
    """Match vertex to start of its matched MC tracks.

    returns TVector3 or None
    """
    tracks = [vertex.getParameters(i).getTrack() for i in range(vertex.getNTracks())]
    matched_tracks = [track for track in tracks if track.getMcTrackId() >= 0]
    if len(matched_tracks) < 2:
        return None
    mc_tracks = [event.MCTrack[track.getMcTrackId()] for track in matched_tracks]
    mother_ids = [track.GetMotherId() for track in mc_tracks]
    most_common_mother, count = Counter(mother_ids).most_common(1)[0]
    if count >= len(matched_tracks) * 0.7:
        # truth match if ≥ 70 % of hits are related to a single MCTrack
        for mc_track in mc_tracks:
            if mc_track.GetMotherId() == most_common_mother:
                true_vertex = ROOT.TVector3()
                mc_track.GetStartVertex(true_vertex)
                return true_vertex
    return None


def main():
    """Truth match tracks and vertices"""
    parser = argparse.ArgumentParser(
        description="Script for truth matching for AdvSND."
    )
    parser.add_argument(
        "-f",
        "--inputfile",
        help="""Simulation results to use as input."""
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
        args.outputfile = args.inputfile.removesuffix(".root") + "_MCTruth.root"

    inputfile = ROOT.TFile.Open(args.inputfile, "read")
    tree = inputfile.cbmsim

    outputfile = ROOT.TFile.Open(args.outputfile, "recreate")
    out_tree = tree.CloneTree(0)

    for event in tqdm(tree, desc="Event loop: ", total=tree.GetEntries()):
        for track in event.genfit_tracks:
            track_id = find_MC_track(track, event)
            track.setMcTrackId(track_id)
        out_tree.Fill()
    out_tree.Write()
    outputfile.Write()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
