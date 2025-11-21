import numpy as np

from vispy import app, scene

canvas = scene.SceneCanvas(show = True)
view = canvas.central_widget.add_view()
camera = scene.cameras.TurntableCamera()
view.camera = camera

inertial_node = scene.node.Node(parent = view.scene)
inertial_frame = scene.visuals.Arrow(
    pos = np.array([[0,0,0], [1,0,0], [0,0,0], [0,1,0], [0,0,0], [0,0,1]]),
    color = [[1,0,0,1], [1,0,0,1], [0,1,0,1], [0,1,0,1], [0,0,1,1], [0,0,1,1]],
    parent = inertial_node)
gridlines = scene.visuals.GridLines(parent = inertial_node)

body_node = scene.node.Node(parent = view.scene)
body_frame = scene.visuals.Arrow(
    pos = np.array([[0,0,0], [1,0,0], [0,0,0], [0,1,0], [0,0,0], [0,0,1]]),
    color = [[1,0,0,1], [1,0,0,1], [0,1,0,1], [0,1,0,1], [0,0,1,1], [0,0,1,1]],
    parent = body_node)
body_mesh = scene.visuals.Cube(
    size = (1,1,1),
    parent = body_node)


app.run()