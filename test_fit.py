import ROOT


def main():
    ROOT.gInterpreter.Declare('#include "MaterialEffects.h"')
    ROOT.gInterpreter.Declare('#include "TGeoMaterialInterface.h"')
    ROOT.gInterpreter.Declare('#include "FieldManager.h"')
    ROOT.gInterpreter.Declare('#include "ConstField.h"')
    ROOT.TGeoManager("Geometry", "Geane geometry")
    ROOT.TGeoManager.Import("geofile_full.Ntuple-TGeant4.root")
    ROOT.genfit.MaterialEffects.getInstance().init(ROOT.genfit.TGeoMaterialInterface())
    ROOT.genfit.FieldManager.getInstance().init(ROOT.genfit.ConstField(0.0, 10.0, 0.0))

    # init event display
    display = ROOT.genfit.EventDisplay.getInstance()

    # init fitter
    fitter = ROOT.genfit.DAF()
    # particle pdg code; muon
    pdg = 13

    # start values for the fit, e.g. from pattern recognition
    pos = ROOT.TVector3(0, 0, 0)
    mom = ROOT.TVector3(0, 0, 3)

    # trackrep
    rep = ROOT.genfit.RKTrackRep(pdg)

    # create track
    fitTrack = ROOT.genfit.Track(rep, pos, mom)

    detId = 0
    planeId = 0
    hitId = 0

    detectorResolution = 0.001
    hitCov = ROOT.TMatrixDSym(2)
    hitCov.UnitMatrix()
    hitCov *= detectorResolution**2

    # add some planar hits to track with coordinates I just made up
    hitCoords = ROOT.TVectorD(2)
    hitCoords[0] = 0
    hitCoords[1] = 0
    measurement = ROOT.genfit.PlanarMeasurement(
        hitCoords, hitCov, detId, hitId, ROOT.nullptr
    )
    hitId += 1
    measurement.setPlane(
        ROOT.genfit.SharedPlanePtr(
            ROOT.genfit.DetPlane(
                ROOT.TVector3(0, 0, 0), ROOT.TVector3(1, 0, 0), ROOT.TVector3(0, 1, 0)
            )
        ),
        planeId,
    )
    planeId += 1
    fitTrack.insertPoint(ROOT.genfit.TrackPoint(measurement, fitTrack))

    hitCoords[0] = -0.15
    hitCoords[1] = 0
    measurement = ROOT.genfit.PlanarMeasurement(
        hitCoords, hitCov, detId, hitId, ROOT.nullptr
    )
    hitId += 1
    measurement.setPlane(
        ROOT.genfit.SharedPlanePtr(
            ROOT.genfit.DetPlane(
                ROOT.TVector3(0, 0, 0), ROOT.TVector3(1, 0, 0), ROOT.TVector3(0, 1, 0)
            )
        ),
        planeId,
    )
    planeId += 1
    fitTrack.insertPoint(ROOT.genfit.TrackPoint(measurement, fitTrack))

    hitCoords[0] = -0.4
    hitCoords[1] = 0
    measurement = ROOT.genfit.PlanarMeasurement(
        hitCoords, hitCov, detId, hitId, ROOT.nullptr
    )
    hitId += 1
    measurement.setPlane(
        ROOT.genfit.SharedPlanePtr(
            ROOT.genfit.DetPlane(
                ROOT.TVector3(0, 0, 0), ROOT.TVector3(1, 0, 0), ROOT.TVector3(0, 1, 0)
            )
        ),
        planeId,
    )
    planeId += 1
    fitTrack.insertPoint(ROOT.genfit.TrackPoint(measurement, fitTrack))

    # check
    fitTrack.checkConsistency()

    # do the fit
    fitter.processTrack(fitTrack)

    # print fit result
    fitTrack.getFittedState().Print()

    # check
    fitTrack.checkConsistency()

    display.addEvent(fitTrack)

    # delete fitter;

    # open event display
    display.open()


main()
