#include "immersed_boundary_method.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include "common_kernels.hpp"


__global__ void interpolate_velocities_on_nodes(int total_nodes, double * vel_x, double * vel_y, double * vel_z, double * x, double * y, double * z,
	double3 mesh_origin, double3 mesh_lengths, double delta_x, double4* soln, int total_cells) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ int x_cells;
	x_cells = mesh_lengths.x / delta_x;
	__shared__ int y_cells;
	y_cells = mesh_lengths.y / delta_x;
	__shared__ double inverse_delta_x_3;
	inverse_delta_x_3 = 1 / delta_x / delta_x / delta_x;
	//loop through lagrangian nodes
	for (int n = index; n < total_nodes; n += stride) {

		if (n < total_nodes) {

			vel_x[n] = 0.0;
			vel_y[n] = 0.0;
			vel_z[n] = 0.0;

			double x_ref, y_ref, z_ref;
			double4 cell_soln;
			int x_min, x_max, y_min, y_max, z_min, z_max;

			int t = 0;
			int cell_index;
			/// assume 2 point stencil for now

			//round  x_ref to nearest vertice

			x_ref = (x[n] - mesh_origin.x) / delta_x;

			x_min = (int)round(x_ref) - 1;
			x_max = x_min + 1;

			y_ref = fabs((y[n] - mesh_origin.y) / delta_x); // ANSYS mesh sorting

			y_min = (int)round(y_ref) - 1;
			y_max = y_min + 1;

			z_ref = fabs((z[n] - mesh_origin.z) / delta_x); // ANSYS mesh sorting

			z_min = (int)round(z_ref) - 1;
			z_max = z_min + 1;

			///check assumption that each cell is equal to 1
			/// otherwise need to account for this
			//weighting must add up to 1

			for (int i = x_min; i <= x_max; i++) {
				for (int j = y_min; j <= y_max; j++) {
					for (int k = z_min; k <= z_max; k++) {
						cell_index = i + (j)* x_cells + k * y_cells*x_cells;
						if (cell_index < total_cells && cell_index >= 0) {

							double dist_x = (i + 0.5 - x_ref); //distance between cell_index and node reference
							double dist_y = (j + 0.5 - y_ref); // using ANSYS Mesh sorting have to reverse Y origin
							double dist_z = (k + 0.5 - z_ref);

							double weight_x = 1 - abs(dist_x);
							double weight_y = 1 - abs(dist_y);
							double weight_z = 1 - abs(dist_z);

							cell_soln = soln[cell_index];

							vel_x[n] += cell_soln.x * weight_x*weight_y*weight_z*inverse_delta_x_3;
							vel_y[n] += cell_soln.y * weight_x*weight_y*weight_z*inverse_delta_x_3;
							vel_z[n] += cell_soln.z* weight_x*weight_y*weight_z*inverse_delta_x_3;
						}
					}
				}
			}

		}
	}

}


__global__ void interpolate_velocities_on_nodes_cos_kernel(int total_nodes, double * vel_x, double * vel_y, double * vel_z, double * x, double * y, double * z,
	double3 mesh_origin, double3 mesh_lengths, double delta_x, double4* soln, double pi) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ int x_cells;
	x_cells = mesh_lengths.x / delta_x;
	__shared__ double inverse_delta_x_3;
	inverse_delta_x_3 = 1 / delta_x / delta_x;
	//loop through lagrangian nodes
	for (int n = index; n < total_nodes; n += stride) {

		if (n < total_nodes) {

			vel_x[n] = 0.0;
			vel_y[n] = 0.0;
			vel_z[n] = 0.0;

			double x_ref, y_ref;
			double4 cell_soln;
			int x_min, x_max, y_min, y_max;

			int t = 0;
			int cell_index;
			/// assume 2 point stencil for now

			//round  x_ref to nearest vertice

			x_ref = (x[n] - mesh_origin.x) / delta_x;

			x_min = (int)round(x_ref) - 2;
			x_max = x_min + 3;

			y_ref = fabs((y[n] - mesh_origin.y) / delta_x); // ANSYS mesh sorting

			y_min = (int)round(y_ref) - 2;
			y_max = y_min + 3;

			///check assumption that each cell is equal to 1
			/// otherwise need to account for this
			//weighting must add up to 1

			for (int i = x_min; i <= x_max; i++) {
				for (int j = y_min; j <= y_max; j++) {
					cell_index = i + (j)* x_cells;
					double dist_x = (i + 0.5 - x_ref); //distance between cell_index and node reference
					double dist_y = (j + 0.5 - y_ref); // using ANSYS Mesh sorting have to reverse Y origin

					double weight_x = 0.25*(1 + cos(pi* dist_x / 2));
					double weight_y = 0.25*(1 + cos(pi* dist_y / 2));
					double weight_z = 1;

					cell_soln = soln[cell_index];

					vel_x[n] += cell_soln.x * weight_x*weight_y*weight_z*inverse_delta_x_3;
					vel_y[n] += cell_soln.y * weight_x*weight_y*weight_z*inverse_delta_x_3;
					vel_z[n] += cell_soln.z* weight_x*weight_y*weight_z*inverse_delta_x_3;

				}
			}

		}
	}

}


__global__ void update_node_forces(int total_nodes, double * force_x, double * force_y, double * force_z, double * x, double * y, double * z,
	double * x_ref, double * y_ref, double *  z_ref,
	double stiffness, double radius, double pi, int object_nodes, double * vel_x, double * vel_y, double * vel_z, double delta_t, double depth) {


	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	__shared__ double area;
	area = 2 * pi * radius / object_nodes * depth;

	for (int i = index; i < total_nodes; i += stride) {

		if (i < total_nodes) {

			// add in different methods for types of deformable objects later
			/// assume no slip boundary in this test case
			force_x[i] = (-stiffness * area*(x[i] - x_ref[i]));
			force_y[i] = (-stiffness * area* (y[i] - y_ref[i]));
			force_z[i] = (-stiffness * area* (z[i] - z_ref[i]));


			////direct forcing 
			/*force_x[i] = -1*vel_x[i]  /delta_t;
			force_y[i] = -1 * vel_y[i]/delta_t ;
			force_z[i] = 0.0;*/


		}
	}
}


__global__ void update_node_positions(int total_nodes, double * vel_x, double * vel_y, double * vel_z, double * x, double * y, double * z,
	double delta_t, int num_nodes, int rk) {


	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ double centre_x;
	__shared__ double centre_y;


	for (int i = index; i < total_nodes; i += stride) {
		double alpha[4] = { 1.0 , 0.5, 0.5,1.0 };

		if (i < total_nodes) {
			/*if (threadIdx.x == 0) {

				centre_x = 0.0;
				centre_y = 0.0;
			}

			__syncthreads();*/

			// add in different methods for types of deformable objects later
			//assume constant timestep
			x[i] += vel_x[i] * delta_t * alpha[rk];
			y[i] += vel_y[i] * delta_t * alpha[rk];
			z[i] += vel_z[i] * delta_t * alpha[rk];

			/*x[i] += 0.0;
			y[i] += 0.0;
			z[i] += 0.0;*/

			//myatomicAdd(&centre_x, (x[i] /num_nodes));
			//myatomicAdd(&centre_y, (y[i] / num_nodes));

			//__syncthreads();
			//if (threadIdx.x == 0) {
			//

			//	printf("x: %4.2f %4.2f \n", centre_x, centre_y);
			//}

		}
	}


	return;
}


__global__ void update_node_positions_rk4(int total_nodes, double * vel_x, double * vel_y, double * vel_z, double * x, double * y, double * z,
	double delta_t, int num_nodes, double * x0, double * y0, double * z0) {


	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ double centre_x;
	__shared__ double centre_y;


	for (int i = index; i < total_nodes; i += stride) {

		if (i < total_nodes) {
			/*if (threadIdx.x == 0) {

				centre_x = 0.0;
				centre_y = 0.0;
			}

			__syncthreads();*/

			// add in different methods for types of deformable objects later
			//assume constant timestep
			x[i] = x0[i] + vel_x[i] * delta_t;
			y[i] = y0[i] + vel_y[i] * delta_t;
			z[i] = z0[i] + vel_z[i] * delta_t;

			/*x[i] += 0.0;
			y[i] += 0.0;
			z[i] += 0.0;*/
			//reset xo
			x0[i] = x[i];
			y0[i] = y[i];
			z0[i] = z[i];

			/*	myatomicAdd(&centre_x, (x[i] / num_nodes));
				myatomicAdd(&centre_y, (y[i] / num_nodes));

			__syncthreads();
			if (threadIdx.x == 0) {


				printf("RK4 x: %4.2f %4.2f \n", centre_x, centre_y);
			}*/

		}
	}


	return;
}



__global__ void spread_forces_on_structured_grid(int total_nodes, double * node_force_x, double * node_force_y, double * node_force_z, double * x, double * y, double * z,
	double3 mesh_origin, double3 mesh_lengths, double delta_x,
	double * force_x, double * force_y, double * force_z, int total_cells) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ int x_cells;
	x_cells = mesh_lengths.x / delta_x;
	__shared__ int y_cells;
	y_cells = mesh_lengths.y / delta_x;
	__shared__ double inverse_delta_x_3;
	inverse_delta_x_3 = 1 / delta_x / delta_x / delta_x;
	//loop through lagrangian nodes
	for (int n = index; n < total_nodes; n += stride) {

		if (n < total_nodes) {


			double x_ref, y_ref, z_ref;
			int x_min, x_max, y_min, y_max, z_min, z_max;

			double val;
			int t = 0;
			int cell_index;
			/// assume 2 point stencil for now

			//round  x_ref to nearest vertice

			x_ref = (x[n] - mesh_origin.x) / delta_x;

			x_min = (int)round(x_ref) - 1;
			x_max = x_min + 1;

			y_ref = fabs((y[n] - mesh_origin.y) / delta_x); // ANSYS mesh sorting

			y_min = (int)round(y_ref) - 1;
			y_max = y_min + 1;

			z_ref = fabs((z[n] - mesh_origin.z) / delta_x); // ANSYS mesh sorting

			z_min = (int)round(z_ref) - 1;
			z_max = z_min + 1;


			for (int i = x_min; i <= x_max; i++) {
				for (int j = y_min; j <= y_max; j++) {
					for (int k = z_min; k <= z_max; k++) {

						cell_index = i + (j)* x_cells + k * y_cells*x_cells;
						if (cell_index < total_cells && cell_index >= 0) {


							double dist_x = (i + 0.5 - x_ref); //distance between cell_index and node reference
							double dist_y = (j + 0.5 - y_ref); // using ANSYS Mesh sorting have to reverse Y origin
							double dist_z = (k + 0.5 - z_ref);

							double weight_x = 1 - abs(dist_x);
							double weight_y = 1 - abs(dist_y);
							double weight_z = 1 - abs(dist_z);
							val = node_force_x[n] * weight_x*weight_y*weight_z*inverse_delta_x_3;
							myatomicAdd(&force_x[cell_index], val);
							val = node_force_y[n] * weight_x*weight_y*weight_z*inverse_delta_x_3;
							myatomicAdd(&force_y[cell_index], val);
							val = node_force_z[n] * weight_x*weight_y*weight_z*inverse_delta_x_3;
							myatomicAdd(&force_z[cell_index], val);

							t++;
						}
					}
				}
			}

		}
	}

}



__global__ void spread_forces_on_structured_grid_cos_kernel(int total_nodes, double * node_force_x, double * node_force_y, double * node_force_z, double * x, double * y, double * z,
	double3 mesh_origin, double3 mesh_lengths, double delta_x,
	double * force_x, double * force_y, double * force_z, int total_cells, double pi) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ int x_cells;
	x_cells = mesh_lengths.x / delta_x;
	__shared__ double inverse_delta_x_2;
	inverse_delta_x_2 = 1 / delta_x / delta_x;
	//loop through lagrangian nodes
	for (int n = index; n < total_nodes; n += stride) {

		if (n < total_nodes) {

			double x_ref, y_ref;
			int x_min, x_max, y_min, y_max;
			double val;
			int t = 0;
			int cell_index;
			/// assume 2 point stencil for now

			//round  x_ref to nearest vertice

			x_ref = (x[n] - mesh_origin.x) / delta_x;

			x_min = (int)round(x_ref) - 2;
			x_max = x_min + 3;

			y_ref = fabs((y[n] - mesh_origin.y) / delta_x); // ANSYS mesh sorting

			y_min = (int)round(y_ref) - 2;
			y_max = y_min + 3;

			for (int i = x_min; i <= x_max; i++) {
				for (int j = y_min; j <= y_max; j++) {
					cell_index = i + (j)* x_cells;
					if (cell_index < total_cells) {


						double dist_x = (i + 0.5 - x_ref); //distance between cell_index and node reference
						double dist_y = (j + 0.5 - y_ref); // using ANSYS Mesh sorting have to reverse Y origin

						double weight_x = 0.25*(1 + cos(pi* dist_x / 2));
						double weight_y = 0.25*(1 + cos(pi* dist_y / 2));
						double weight_z = 1;
						val = node_force_x[n] * weight_x*weight_y*weight_z*inverse_delta_x_2;
						myatomicAdd(&force_x[cell_index], val);
						val = node_force_y[n] * weight_x*weight_y*weight_z*inverse_delta_x_2;
						myatomicAdd(&force_y[cell_index], val);
						val = node_force_z[n] * weight_x*weight_y*weight_z*inverse_delta_x_2;
						myatomicAdd(&force_z[cell_index], val);

						t++;
					}

				}
			}

		}
	}

}

