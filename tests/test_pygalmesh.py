import pygalmesh

ball = pygalmesh.Ball([0,0,0], 1.0)
mesh = pygalmesh.generate_mesh(ball, max_cell_circumradius=0.2)
print(mesh.points.shape)
print(mesh.cells[0].data.shape)
print(mesh.cells[1].data.shape)

