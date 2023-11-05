#!/usr/bin/env python3

import argparse
import math
import ROOT
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Script for AdvSND tracking studies.")
    parser.add_argument(
        "inputfile",
        help="""Simulation results to use as input. """
        """Supports retrieving files from EOS via the XRootD protocol.""",
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
        "--outputdir",
        default="plots",
        help="""Dir to write displays to. """,
    )
    args = parser.parse_args()
    f = ROOT.TFile.Open(args.geofile, "read")
    geo = f.FAIRGeom  # noqa: F841
    ch = ROOT.TChain("cbmsim")
    ch.Add(args.inputfile)
    n = ch.GetEntries()
    digits = int(math.log10(n)) + 1
    i = 0
    for event in ch:
        xs = []
        ys = []
        zxs = []
        zys = []
        xls = []
        yls = []
        zlxs = []
        zlys = []
        xrs = []
        yrs = []
        zrxs = []
        zrys = []
        left = ROOT.TVector3()
        right = ROOT.TVector3()
        coords = []
        for track in event.Reco_MuonTracks:
            points = track.getTrackPoints()
            _coords = [(p.X(), p.Y(), p.Z()) for p in points]
            coords.append(list(zip(*_coords)))
        for hit in event.Digi_advTargetClusters:
            vert = hit.isVertical()
            x = hit.GetX()
            y = hit.GetY()
            z = hit.GetZ()
            hit.GetPosition(left, right)
            if not vert:
                xs.append(x)
                zxs.append(z)
                xls.append(left.X())
                zlxs.append(left.Z())
                xrs.append(right.X())
                zrxs.append(right.Z())
            else:
                ys.append(y)
                zys.append(z)
                yls.append(left.Y())
                zlys.append(left.Z())
                yrs.append(right.Y())
                zrys.append(right.Z())
        if zxs or zys:
            print(f"{i:0{digits}d}/{n}", end="\r")
            fig, ax1 = plt.subplots()
            plt.title(f"Event {i}")
            for tl in ax1.get_yticklabels():
                tl.set_color("r")
            plt.ylim(-60, 0)
            plt.xlim(-150, 0)
            plt.scatter(zxs, xs, color="red", marker=".", label=r"$x_{\mathrm{true}}$")
            plt.scatter(
                zlxs, xls, color="red", marker="1", label=r"$x_{\mathrm{left}}$"
            )
            plt.scatter(
                zrxs, xrs, color="red", marker="2", label=r"$x_{\mathrm{right}}$"
            )
            for track in coords:
                plt.plot(track[2], track[0], label="track zx")
            plt.xlabel("z [cm]")
            plt.ylabel("x [cm]", color="red")
            ax2 = ax1.twinx()
            plt.ylim(0, 60)
            plt.ylabel("y [cm]", color="blue")
            for tl in ax2.get_yticklabels():
                tl.set_color("b")
            plt.scatter(zys, ys, color="blue", marker=".", label=r"$y_{\mathrm{true}}$")
            plt.scatter(
                zlys, yls, color="blue", marker="1", label=r"$y_{\mathrm{left}}$"
            )
            plt.scatter(
                zrys, yrs, color="blue", marker="2", label=r"$y_{\mathrm{right}}$"
            )
            for track in coords:
                plt.plot(track[2], track[1], label="track zy")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc=0)
            plt.savefig(args.outputdir + f"/event_display_{i:0{digits}d}.pdf")
            plt.close(fig)
        i += 1
        if i > 100:
            break
    print("\nDone.")


if __name__ == "__main__":
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    ROOT.gROOT.SetBatch(True)
    main()
