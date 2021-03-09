from mshr import *
from dolfin import *

def mesh_hollow_cylinder(rad, thick, res):
	'''  Hollow cylinder cross section mesh generator,
		takes the values: external radius, thickness and resolution of the mesh
	   	returns the mesh for cross section'''
	
	# Create geometry
	rec1 = Circle(Point(0,0), rad)
	rec2 = Circle(Point(0,0), rad - thick)

	geometry = rec1 - rec2

	# Create mesh
	mesh = generate_mesh(geometry, res)
	return mesh

def mesh_hollow_rectangle(width, thick, res):
	'''  Hollow cylinder cross section mesh generator,
		takes the values: external radius, thickness and resolution of the mesh
	   	returns the mesh for cross section'''
	
	# Create geometry
	rec1 = Rectangle(Point(-0.5,-0.5), Point(0.5,0.5))
	rec2 = Rectangle(Point(-0.5+thick,-0.5+thick), Point(0.5-thick,0.5-thick))

	geometry = rec1 - rec2

	# Create mesh
	mesh = generate_mesh(geometry, res)
	return mesh

def mesh_rectangle(width, thick, res):
	'''  Hollow cylinder cross section mesh generator,
		takes the values: external radius, thickness and resolution of the mesh
	   	returns the mesh for cross section'''
	
	# Create geometry
	rec1 = Rectangle(Point(-0.5,-0.5), Point(0.5,0.5))
	
	geometry = rec1

	# Create mesh
	mesh = generate_mesh(geometry, res)
	return mesh


def mesh_T_profile(width, thick, res):
	'''  Hollow cylinder cross section mesh generator,
		takes the values: external radius, thickness and resolution of the mesh
	   	returns the mesh for cross section'''
	
	# Create geometry
	rec1 = Rectangle(Point(-0.5,-0.5), Point(-0.4,0.5))
	rec2 = Rectangle(Point(-0.4,-0.1), Point(0.5,0.1))
	
	geometry = rec1 + rec2

	# Create mesh
	mesh = generate_mesh(geometry, res)
	return mesh