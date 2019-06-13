#ifndef IBM
#define IBM
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

 __global__ void interpolate_velocities_on_nodes(int total_nodes, double * vel_x, double * vel_y, double * vel_z, double * x, double * y, double * z,
	double3 mesh_origin, double3 mesh_lengths, double delta_x, double4* soln, int total_cells);

 __global__ void interpolate_velocities_on_nodes_cos_kernel(int total_nodes, double * vel_x, double * vel_y, double * vel_z, double * x, double * y, double * z,
	double3 mesh_origin, double3 mesh_lengths, double delta_x, double4* soln, double pi);
 __global__ void update_node_forces(int total_nodes, double * force_x, double * force_y, double * force_z, double * x, double * y, double * z,
	 double * x_ref, double * y_ref, double *  z_ref,
	 double stiffness, double radius, double pi, int object_nodes, double * vel_x, double * vel_y, double * vel_z, double delta_t, double depth);

 __global__ void update_node_positions(int total_nodes, double * vel_x, double * vel_y, double * vel_z, double * x, double * y, double * z,
	 double delta_t, int num_nodes, int rk);

 __global__ void update_node_positions_rk4(int total_nodes, double * vel_x, double * vel_y, double * vel_z, double * x, double * y, double * z,
	 double delta_t, int num_nodes, double * x0, double * y0, double * z0);

 __global__ void spread_forces_on_structured_grid(int total_nodes, double * node_force_x, double * node_force_y, double * node_force_z, double * x, double * y, double * z,
	 double3 mesh_origin, double3 mesh_lengths, double delta_x,
	 double * force_x, double * force_y, double * force_z, int total_cells);

 __global__ void spread_forces_on_structured_grid_cos_kernel(int total_nodes, double * node_force_x, double * node_force_y, double * node_force_z, double * x, double * y, double * z,
	 double3 mesh_origin, double3 mesh_lengths, double delta_x,
	 double * force_x, double * force_y, double * force_z, int total_cells, double pi);
#endif