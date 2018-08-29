#!/usr/bin/env python

import vtk
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtk.util.vtkConstants import VTK_DOUBLE
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def cart2sph(vec):
    hxy = np.hypot(vec[0], vec[1])
    r = np.hypot(hxy, vec[2])
    el = np.arctan2(vec[2], hxy)
    az = np.arctan2(vec[1], vec[0])
    return np.array([az, el, r]).flatten()

class vtkTimerCallback():
    def __init__(self):
        self.timer_count = 0

    def execute(self, obj, event):
        self.slicerB.SetNormal(sample_spherical(1))
        self.optimizer.Update()

        if self.timer_count % 5 == 0:
            self.slicerA.SetNormal(sample_spherical(1))


        self.tfFilter.SetTransform(self.optimizer.icp)
        self.tfFilter.Update()

        # normal = self.slicerA.GetNormal()
        # spherical = cart2sph(normal)
        # self.slicerActorA.SetOrientation(0,0,0)
        # self.slicerActorA.RotateY(spherical[1] * 180 / np.pi)
        # self.slicerActorA.RotateZ(-spherical[0] * 180 / np.pi)

        # normal = self.slicerB.GetNormal()
        # spherical = cart2sph(normal)
        # self.slicerActorB.SetOrientation(0,0,0)
        # self.slicerActorB.RotateY(spherical[1] * 180 / np.pi)
        # self.slicerActorB.RotateZ(-spherical[0] * 180 / np.pi)

        iren = obj
        iren.GetRenderWindow().Render()
        self.timer_count += 1

class SlicerAlgorithm(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=2,
            nOutputPorts=1, outputType='vtkPolyData')

        self._normalLine = vtk.vtkPolyLineSource()
        self._normalLine.SetNumberOfPoints(2)
        self._normalLine.SetPoint(0, 0, 0, 0)
        self._normalLine.SetPoint(1, 0, 0, 0)
        self.SetInputConnection(1, self._normalLine.GetOutputPort())
  
    def RequestData(self, request, inInfo, outInfo):
        inp = vtk.vtkDataSet.GetData(inInfo[0])
        vec = vtk.vtkDataSet.GetData(inInfo[1])
        opt = vtk.vtkPolyData.GetData(outInfo)
        origin = np.array(self._normalLine.GetPoints().GetPoint(0))
        normal = np.array(self._normalLine.GetPoints().GetPoint(1)) - origin

        if np.allclose(normal, [0,0,0]):
            centerFilter = vtk.vtkCenterOfMass()
            centerFilter.SetInputData(inp)
            centerFilter.SetUseScalarsAsWeights(False)
            centerFilter.Update()
            center = centerFilter.GetCenter()

            bounds = inp.GetBounds()

            origin = [np.random.triangular(bounds[0], center[0], bounds[1]),
                      np.random.triangular(bounds[2], center[1], bounds[3]),
                      np.random.triangular(bounds[4], center[2], bounds[5])]

            normal = sample_spherical(1)
            self.SetOrigin(origin)
            self.SetNormal(normal)

        plane = vtk.vtkPlane()
        plane.SetNormal(normal[0], normal[1], normal[2])
        plane.SetOrigin(origin[0], origin[1], origin[2])

        #create cutter
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(inp)
        cutter.Update()

        opt.ShallowCopy(cutter.GetOutput())

        return 1

    def SetNormal(self, normal):
        normalArray = np.array(normal).flatten()
        assert(len(normalArray) == 3)
        origin = self.GetOrigin()
        normalNew = normalArray + origin
        self._normalLine.SetPoint(1, normalNew[0], normalNew[1], normalNew[2])

    def GetNormal(self):
        origin = self.GetOrigin()
        return np.array(self._normalLine.GetPoints().GetPoint(1)) - origin

    def SetOrigin(self, origin):
        originArray = np.array(origin).flatten()
        assert(len(origin) == 3)
        originOld = np.array(self._normalLine.GetPoints().GetPoint(0))
        normalOld = np.array(self._normalLine.GetPoints().GetPoint(1)) - originOld
        normalNew = normalOld + originArray
        self._normalLine.SetPoint(0, originArray[0], originArray[1], originArray[2])
        self._normalLine.SetPoint(1, normalNew[0], normalNew[1], normalNew[2])
        return

    def GetOrigin(self):
        return np.array(self._normalLine.GetPoints().GetPoint(0))

class SlicerOptimizer(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=2,
            nOutputPorts=1, outputType='vtkPolyData')

        
        self.icp = vtk.vtkIterativeClosestPointTransform()

    def RequestData(self, request, inInfo, outInfo):
        target = vtk.vtkDataSet.GetData(inInfo[0])
        source = vtk.vtkDataSet.GetData(inInfo[1])
        opt = vtk.vtkPolyData.GetData(outInfo)
        self.icp.SetSource(source)
        self.icp.SetTarget(target)
        # self.icp.GetLandmarkTransform().SetModeToRigidBody()
        self.icp.GetLandmarkTransform().SetModeToSimilarity ()
        self.icp.SetMaximumNumberOfIterations(500)
        self.icp.StartByMatchingCentroidsOn()
        self.icp.Modified()
        self.icp.Update()

        # print(self.icp.GetMatrix())

        tfFilter = vtk.vtkTransformPolyDataFilter()
        tfFilter.SetInputData(source)
        tfFilter.SetTransform(self.icp)
        tfFilter.Update()


        opt.ShallowCopy(tfFilter.GetOutput())

        return 1



def main():
    # random.seed=1
    inputFilename = "Hulk.stl"# , numberOfCuts = get_program_parameters()

    colors = vtk.vtkNamedColors()

    reader= vtk.vtkSTLReader()
    reader.SetFileName(inputFilename)
    reader.Update()

    slicerA = SlicerAlgorithm()
    slicerA.SetInputConnection(reader.GetOutputPort())

    slicerA.Update()

    slicerMapperA = vtk.vtkPolyDataMapper()
    slicerMapperA.SetInputConnection(slicerA.GetOutputPort())
         
    #create cutter actor
    slicerActorA = vtk.vtkActor()
    slicerActorA.GetProperty().SetColor(1.0,1,0)
    slicerActorA.GetProperty().SetLineWidth(2)
    slicerActorA.SetMapper(slicerMapperA)

    slicerB = SlicerAlgorithm()
    slicerB.SetInputConnection(0, reader.GetOutputPort())
    slicerB.Update()

    optimizer = SlicerOptimizer()
    optimizer.SetInputConnection(0, slicerA.GetOutputPort())
    optimizer.SetInputConnection(1, slicerB.GetOutputPort())
    optimizer.Update()

    slicerMapperB = vtk.vtkPolyDataMapper()
    slicerMapperB.SetInputConnection(optimizer.GetOutputPort())
         
    #create cutter actor
    slicerActorB = vtk.vtkActor()
    slicerActorB.GetProperty().SetColor(0,1,1)
    slicerActorB.GetProperty().SetLineWidth(2)
    slicerActorB.SetMapper(slicerMapperB)

    # Create the model actor
    modelMapper= vtk.vtkPolyDataMapper()
    modelMapper.SetInputConnection( reader.GetOutputPort())

    modelActor = vtk.vtkActor()
    modelActor.GetProperty().SetColor(colors.GetColor3d("Flesh"))
    modelActor.SetMapper(modelMapper)

    tfFilter = vtk.vtkTransformPolyDataFilter()
    tfFilter.SetInputData(reader.GetOutput())
    tfFilter.SetTransform(optimizer.icp)
    tfFilter.Update()

    modelMapper2 = vtk.vtkPolyDataMapper()
    modelMapper2.SetInputConnection(tfFilter.GetOutputPort())

    modelActor2 = vtk.vtkActor()
    modelActor2.GetProperty().SetColor(colors.GetColor3d("Flesh"))
    modelActor2.SetMapper(modelMapper2)

    # Create renderers and add the cutter and model actors.
    renderer = vtk.vtkRenderer()
    renderer.AddActor(slicerActorA)
    renderer.AddActor(slicerActorB)
    renderer.AddActor(modelActor)
    renderer.AddActor(modelActor2)

    # Add renderer to renderwindow and render
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(600, 600)

    interactor= vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    renderer.SetBackground(colors.GetColor3d("Burlywood"))
    renderer.GetActiveCamera().SetParallelProjection(True)
    renderer.GetActiveCamera().SetPosition(-1, 0, 0)
    renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
    renderer.GetActiveCamera().SetViewUp(0, 0, 1)
    # renderer.GetActiveCamera().Azimuth(30)
    # renderer.GetActiveCamera().Elevation(30)

    renderer.ResetCamera()
    renderWindow.Render()

    interactor.Initialize()

    # Sign up to receive TimerEvent
    cb = vtkTimerCallback()
    cb.slicerActorA = slicerActorA
    cb.slicerActorB = slicerActorB
    cb.slicerA = slicerA
    cb.slicerB = slicerB
    cb.tfFilter = tfFilter
    cb.optimizer = optimizer
    interactor.AddObserver('TimerEvent', cb.execute)
    interactor.CreateRepeatingTimer(10)

    interactor.Start()

def get_program_parameters():
    import argparse
    description = 'Cutting a surface model of the skin with a series of planes produces contour lines.'
    epilogue = '''
    Lines are wrapped with tubes for visual clarity.
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filename1', help='Torso.vtp.')
    parser.add_argument('-n', type=int, default=20, help='Number of cuts.')
    args = parser.parse_args()
    return args.filename1, args.n

if __name__ == '__main__':
    main()