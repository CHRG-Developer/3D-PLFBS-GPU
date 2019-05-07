
#include "Solver_gpu.h"
#include <math.h>
#include <cmath>
#include "Solver.h"

#include "vector_var.h"
#include <iostream>
#include "Solution.h"
#include <fstream>
#include "global_variables.h"
#include "residuals.h"
#include <cstdio>
#include <ctime>
#include "artificial_dissipation.h"
#include <boost/math/special_functions/sign.hpp>
#include <limits>
#include "RungeKutta.h"
#include "tecplot_output.h"
#include "gradients.h"
#include <string>
#include <sstream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <cuda_profiler_api.h>


using namespace std;

__device__ double3 operator+(const double3 &a, const double3 &b) {

	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);

}
__device__ double3 operator-(const double3 &a, const double3 &b) {

	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);

}


__device__ double dot_product(const double3 &a, const double3 &b) {

	return (a.x * b.x + a.y * b.y + a.z * b.z);

}



__device__ double myatomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}

__constant__ double lattice_weight[15];

__device__ void populate_lattice_macros_uniform(double u_lattice[], double v_lattice[],
	double w_lattice[], double rho_lattice[], double3 cell_1, double3 cell_2,
	double3 interface_node, int i, int neighbour,
	double3 grad_u_1, double3 grad_u_2, double3 grad_v_1, double3  grad_v_2, double3  grad_w_1, double3  grad_w_2, double3  grad_rho_1, double3  grad_rho_2,
	double4 owner_soln, double4 neighbour_soln, int n_cells, double3 cell_normal, double dt, int bcs_rho_type[], int bcs_vel_type[], double4 bcs_macros[]) {

	double3 temp1, temp2;

	///   case 0: // center node

	temp1 = cell_1 - interface_node;
	temp2 = cell_2 - interface_node;

	double rho_i, rho_nb, u_i, u_nb, v_i, v_nb, w_i, w_nb;
	double3 grad_rho, grad_u, grad_v, grad_w;
	int nb;
	nb = neighbour - n_cells;

	if (neighbour > n_cells) {
		double4 bcs = bcs_macros[nb];
		// vboundary = v_i + grad_boundary * distance _ib

		if (bcs_rho_type[nb] == 1) {
			rho_lattice[0] = bcs.w;

		}
		else {
			rho_lattice[0] = owner_soln.w - dot_product(temp1, grad_rho_2);
		}

		if (bcs_vel_type[nb] == 1) {
			u_lattice[0] = bcs.x;
			v_lattice[0] = bcs.y;
			w_lattice[0] = bcs.z;

		}
		else {
			u_lattice[0] = owner_soln.x - dot_product(temp1, grad_u_2);
			v_lattice[0] = owner_soln.y - dot_product(temp1, grad_v_2);
			w_lattice[0] = owner_soln.z - dot_product(temp1, grad_w_2);
		}

		grad_rho = grad_rho_2;
		grad_u = grad_u_2;
		grad_v = grad_v_2;
		grad_w = grad_w_2;



	}
	else {

		rho_i = owner_soln.w - dot_product(temp1, grad_rho_1);
		rho_nb = neighbour_soln.w - dot_product(temp2, grad_rho_2);
		rho_lattice[0] = (rho_i + rho_nb)*0.5;

		u_i = owner_soln.x - dot_product(temp1, grad_u_1);
		u_nb = neighbour_soln.x - dot_product(temp2, grad_u_2);
		u_lattice[0] = (u_i + u_nb)*0.5;

		v_i = owner_soln.y - dot_product(temp1, grad_v_1);
		v_nb = neighbour_soln.y - dot_product(temp2, grad_v_2);
		v_lattice[0] = (v_i + v_nb)*0.5;

		w_i = owner_soln.z - dot_product(temp1, grad_w_1);
		w_nb = neighbour_soln.z - dot_product(temp2, grad_w_2);
		w_lattice[0] = (w_i + w_nb)*0.5;

	



		if ((u_lattice[0] * cell_normal.x + v_lattice[0] * cell_normal.y + w_lattice[0] * cell_normal.z) > 0.01) {


			grad_u = grad_u_1;
			grad_v = grad_v_1;
			grad_w = grad_w_1;
			grad_rho = grad_rho_1;
		}else if ((u_lattice[0] * cell_normal.x + v_lattice[0] * cell_normal.y + w_lattice[0] * cell_normal.z) < -0.01){
			grad_u = grad_u_2;
			grad_v = grad_v_2;
			grad_w = grad_w_2;
			grad_rho = grad_rho_2;
		}
		else {
			grad_u.x = (grad_u_1.x + grad_u_2.x)*0.5;
			grad_u.y = (grad_u_1.y + grad_u_2.y)*0.5;
			grad_u.z = (grad_u_1.z + grad_u_2.z)*0.5;

			grad_v.x = (grad_v_1.x + grad_v_2.x)*0.5;
			grad_v.y = (grad_v_1.y + grad_v_2.y)*0.5;
			grad_v.z = (grad_v_1.z + grad_v_2.z)*0.5;

			grad_w.x = (grad_w_1.x + grad_w_2.x)*0.5;
			grad_w.y = (grad_w_1.y + grad_w_2.y)*0.5;
			grad_w.z = (grad_w_1.z + grad_w_2.z)*0.5;

			grad_rho.x = (grad_rho_1.x + grad_rho_2.x)*0.5;
			grad_rho.y = (grad_rho_1.y + grad_rho_2.y)*0.5;
			grad_rho.z = (grad_rho_1.z + grad_rho_2.z)*0.5;

		}
		


	}

	//set i and nb to cell interface values

	rho_i = rho_lattice[0];
	rho_nb = rho_lattice[0];

	u_i = u_lattice[0];
	u_nb = u_lattice[0];

	v_i = v_lattice[0];
	v_nb = v_lattice[0];

	w_i = w_lattice[0];
	w_nb = w_lattice[0];
	//merge gradients





	///  case 1:west_node


	rho_lattice[1] = rho_nb - grad_rho.x* dt;
	u_lattice[1] = u_nb - grad_u.x* dt;
	v_lattice[1] = v_nb - grad_v.x *dt;
	w_lattice[1] = w_nb - grad_w.x *dt;



	///  case 2: // east_node

	rho_lattice[2] = rho_nb + grad_rho.x* dt;
	u_lattice[2] = u_nb + grad_u.x* dt;
	v_lattice[2] = v_nb + grad_v.x *dt;
	w_lattice[2] = w_nb + grad_w.x *dt;



	///   case 3: // bottom node

	rho_lattice[3] = rho_nb - grad_rho.y* dt;
	u_lattice[3] = u_nb - grad_u.y* dt;
	v_lattice[3] = v_nb - grad_v.y *dt;
	w_lattice[3] = w_nb - grad_w.y *dt;



	///   case 4: // top node


	rho_lattice[4] = rho_nb + grad_rho.y* dt;
	u_lattice[4] = u_nb + grad_u.y* dt;
	v_lattice[4] = v_nb + grad_v.y *dt;
	w_lattice[4] = w_nb + grad_w.y *dt;

	///   case 5: // back node

	rho_lattice[5] = rho_nb - grad_rho.z* dt;
	u_lattice[5] = u_nb - grad_u.z* dt;
	v_lattice[5] = v_nb - grad_v.z *dt;
	w_lattice[5] = w_nb - grad_w.z *dt;

	///   case 6: // front node

	rho_lattice[6] = rho_nb + grad_rho.z* dt;
	u_lattice[6] = u_nb + grad_u.z* dt;
	v_lattice[6] = v_nb + grad_v.z *dt;
	w_lattice[6] = w_nb + grad_w.z *dt;



	/// case 7: back bottom west

	rho_lattice[7] = rho_nb - grad_rho.x* dt
		- grad_rho.y* dt
		- grad_rho.z* dt;
	u_lattice[7] = u_nb - grad_u.x* dt
		- grad_u.y* dt
		- grad_u.z* dt;
	v_lattice[7] = v_nb - grad_v.x* dt
		- grad_v.y* dt
		- grad_v.z* dt;
	w_lattice[7] = w_nb - grad_w.x* dt
		- grad_w.y* dt
		- grad_w.z* dt;

	/// case 9: front bottom west

	rho_lattice[9] = rho_nb - grad_rho.x* dt
		- grad_rho.y* dt
		+ grad_rho.z* dt;
	u_lattice[9] = u_nb - grad_u.x* dt
		- grad_u.y* dt
		+ grad_u.z* dt;
	v_lattice[9] = v_nb - grad_v.x* dt
		- grad_v.y* dt
		+ grad_v.z* dt;
	w_lattice[9] = w_nb - grad_w.x* dt
		- grad_w.y* dt
		+ grad_w.z* dt;




	///  case 11: back top west
	rho_lattice[11] = rho_nb - grad_rho.x* dt
		+ grad_rho.y* dt
		- grad_rho.z* dt;
	u_lattice[11] = u_nb - grad_u.x* dt
		+ grad_u.y* dt
		- grad_u.z* dt;
	v_lattice[11] = v_nb - grad_v.x* dt
		+ grad_v.y* dt
		- grad_v.z* dt;
	w_lattice[11] = w_nb - grad_w.x* dt
		+ grad_w.y* dt
		- grad_w.z* dt;




	/// case 14: front top west

	rho_lattice[14] = rho_nb - grad_rho.x* dt
		+ grad_rho.y* dt
		+ grad_rho.z* dt;
	u_lattice[14] = u_nb - grad_u.x* dt
		+ grad_u.y* dt
		+ grad_u.z* dt;
	v_lattice[14] = v_nb - grad_v.x* dt
		+ grad_v.y* dt
		+ grad_v.z* dt;
	w_lattice[14] = w_nb - grad_w.x* dt
		+ grad_w.y* dt
		+ grad_w.z* dt;




	/// case 8: front top east

	rho_lattice[8] = rho_nb + grad_rho.x* dt
		+ grad_rho.y* dt
		+ grad_rho.z* dt;
	u_lattice[8] = u_nb + grad_u.x* dt
		+ grad_u.y* dt
		+ grad_u.z* dt;
	v_lattice[8] = v_nb + grad_v.x* dt
		+ grad_v.y* dt
		+ grad_v.z* dt;
	w_lattice[8] = w_nb + grad_w.x* dt
		+ grad_w.y* dt
		+ grad_w.z* dt;



	/// case 10 Back Top East

	rho_lattice[10] = rho_nb + grad_rho.x* dt
		+ grad_rho.y* dt
		- grad_rho.z* dt;
	u_lattice[10] = u_nb + grad_u.x* dt
		+ grad_u.y* dt
		- grad_u.z* dt;
	v_lattice[10] = v_nb + grad_v.x* dt
		+ grad_v.y* dt
		- grad_v.z* dt;
	w_lattice[10] = w_nb + grad_w.x* dt
		+ grad_w.y* dt
		- grad_w.z* dt;


	/// case 12 Front Bottom East

	rho_lattice[12] = rho_nb + grad_rho.x* dt
		- grad_rho.y* dt
		+ grad_rho.z* dt;
	u_lattice[12] = u_nb + grad_u.x* dt
		- grad_u.y* dt
		+ grad_u.z* dt;
	v_lattice[12] = v_nb + grad_v.x* dt
		- grad_v.y* dt
		+ grad_v.z* dt;
	w_lattice[12] = w_nb + grad_w.x* dt
		- grad_w.y* dt
		+ grad_w.z* dt;


	/// case 13 Back Bottom East

	rho_lattice[13] = rho_nb + grad_rho.x* dt
		- grad_rho.y* dt
		- grad_rho.z* dt;
	u_lattice[13] = u_nb + grad_u.x* dt
		- grad_u.y* dt
		- grad_u.z* dt;
	v_lattice[13] = v_nb + grad_v.x* dt
		- grad_v.y* dt
		- grad_v.z* dt;
	w_lattice[13] = w_nb + grad_w.x* dt
		- grad_w.y* dt
		- grad_w.z* dt;


}



__device__ void populate_lattice_macros(double u_lattice[], double v_lattice[],
	double w_lattice[], double rho_lattice[], double3 cell_1, double3 cell_2,
	double3 interface_node, int i, int neighbour,
	double3 grad_u_1, double3 grad_u_2, double3 grad_v_1, double3  grad_v_2, double3  grad_w_1, double3  grad_w_2, double3  grad_rho_1, double3  grad_rho_2,
	double4 owner_soln, double4 neighbour_soln, int n_cells, double3 cell_normal, double dt, int bcs_rho_type[], int bcs_vel_type[], double4 bcs_macros[],
	double d1, double d173) {

	double3 temp1, temp2,temp3;

	///   case 0: // center node

	temp1 = cell_1 - interface_node;
	temp2 = cell_2 - interface_node;
	temp3 = cell_2 - cell_1;

	double rho_i, rho_nb, u_i, u_nb, v_i, v_nb, w_i, w_nb;
	double3 grad_rho_3, grad_u_3, grad_v_3, grad_w_3; // interpolated gradients for cell interface
	double c1, c2, c_mag,c_mag1, c_mag2;
	
	int nb;
	nb = neighbour - n_cells;
	/*double d_1 = d1 * dt;
	double d_173 = d173 * dt;*/
	double d_1 = 0.0001 * dt;
	double d_173 = 0.0001 * dt;

	if (neighbour > n_cells) {
		double4 bcs = bcs_macros[nb];
		// vboundary = v_i + grad_boundary * distance _ib

		if (bcs_rho_type[nb] == 1) {
			rho_lattice[0] = bcs.w;
			

		}else {
			rho_lattice[0] = owner_soln.w - dot_product(temp1, grad_rho_2);
		}

		if (bcs_vel_type[nb] == 1) {
			u_lattice[0] = bcs.x;
			v_lattice[0] = bcs.y;
			w_lattice[0] = bcs.z;


		}
		else {
			u_lattice[0] = owner_soln.x - dot_product(temp1, grad_u_2);
			v_lattice[0] = owner_soln.y - dot_product(temp1, grad_v_2);
			w_lattice[0] = owner_soln.z - dot_product(temp1, grad_w_2);
		}

		rho_i = rho_lattice[0];
		rho_nb = rho_lattice[0];

		u_i = u_lattice[0];
		u_nb = u_lattice[0];

		v_i = v_lattice[0];
		v_nb = v_lattice[0];

		w_i = w_lattice[0];
		w_nb = w_lattice[0];
		grad_rho_3 = grad_rho_2;
		grad_u_3 = grad_u_2;
		grad_v_3 = grad_v_2;
		grad_w_3 = grad_w_2;
		
		grad_rho_1 = grad_rho_2;
		grad_u_1 = grad_u_2;
		grad_v_1 = grad_v_2;
		grad_w_1 = grad_w_2;


	}
	else {

		rho_i = owner_soln.w - dot_product(temp1, grad_rho_1);
		rho_nb = neighbour_soln.w - dot_product(temp2, grad_rho_2);
		rho_lattice[0] = (rho_i + rho_nb)*0.5;

		u_i = owner_soln.x - dot_product(temp1, grad_u_1);
		u_nb = neighbour_soln.x - dot_product(temp2, grad_u_2);
		u_lattice[0] = (u_i + u_nb)*0.5;

		v_i = owner_soln.y - dot_product(temp1, grad_v_1);
		v_nb = neighbour_soln.y - dot_product(temp2, grad_v_2);
		v_lattice[0] = (v_i + v_nb)*0.5;

		w_i = owner_soln.z - dot_product(temp1, grad_w_1);
		w_nb = neighbour_soln.z - dot_product(temp2, grad_w_2);
		w_lattice[0] = (w_i + w_nb)*0.5;

		//interpolate gradients for cell interface gradients
		grad_u_3.x = (grad_u_1.x + grad_u_2.x)*0.5;
		grad_u_3.y = (grad_u_1.y + grad_u_2.y)*0.5;
		grad_u_3.z = (grad_u_1.z + grad_u_2.z)*0.5;

		grad_v_3.x = (grad_v_1.x + grad_v_2.x)*0.5;
		grad_v_3.y = (grad_v_1.y + grad_v_2.y)*0.5;
		grad_v_3.z = (grad_v_1.z + grad_v_2.z)*0.5;

		grad_w_3.x = (grad_w_1.x + grad_w_2.x)*0.5;
		grad_w_3.y = (grad_w_1.y + grad_w_2.y)*0.5;
		grad_w_3.z = (grad_w_1.z + grad_w_2.z)*0.5;

		grad_rho_3.x = (grad_rho_1.x + grad_rho_2.x)*0.5;
		grad_rho_3.y = (grad_rho_1.y + grad_rho_2.y)*0.5;
		grad_rho_3.z = (grad_rho_1.z + grad_rho_2.z)*0.5;

		/*grad_rho_3.x = c2 * grad_rho_1.x + c1 * grad_rho_2.x;
		grad_u_3.x = c2 * grad_rho_1.x + c1 * grad_rho_2.x;
		grad_v_3.x = c2 * grad_rho_1.x + c1 * grad_rho_2.x;
		grad_w_3.x = c2 * grad_rho_1.x + c1 * grad_rho_2.x;

		grad_rho_3.y = c2 * grad_rho_1.y + c1 * grad_rho_2.y;
		grad_u_3.y = c2 * grad_rho_1.y + c1 * grad_rho_2.y;
		grad_v_3.y = c2 * grad_rho_1.y + c1 * grad_rho_2.y;
		grad_w_3.y = c2 * grad_rho_1.y + c1 * grad_rho_2.y;
			
		grad_rho_3.z = c2 * grad_rho_1.z + c1 * grad_rho_2.z;
		grad_u_3.z = c2 * grad_rho_1.z + c1 * grad_rho_2.z;
		grad_v_3.z = c2 * grad_rho_1.z + c1 * grad_rho_2.z;
		grad_w_3.z = c2 * grad_rho_1.z + c1 * grad_rho_2.z;*/

	}


	///  case 1 and 2 :west_node and east_node

	if (cell_normal.x > d_1) {
		rho_lattice[1] = rho_i - grad_rho_1.x* dt;
		u_lattice[1] = u_i - grad_u_1.x* dt;
		v_lattice[1] = v_i - grad_v_1.x *dt;
		w_lattice[1] = w_i - grad_w_1.x *dt;

		rho_lattice[2] = rho_nb + grad_rho_2.x* dt;
		u_lattice[2] = u_nb + grad_u_2.x* dt;
		v_lattice[2] = v_nb + grad_v_2.x *dt;
		w_lattice[2] = w_nb + grad_w_2.x *dt;
	}
	else if (cell_normal.x < -d_1){

		rho_lattice[1] = rho_nb - grad_rho_2.x* dt;
		u_lattice[1] = u_nb - grad_u_2.x* dt;
		v_lattice[1] = v_nb - grad_v_2.x *dt;
		w_lattice[1] = w_nb - grad_w_2.x *dt;

		rho_lattice[2] = rho_i + grad_rho_1.x* dt;
		u_lattice[2] = u_i + grad_u_1.x* dt;
		v_lattice[2] = v_i + grad_v_1.x *dt;
		w_lattice[2] = w_i + grad_w_1.x *dt;
	}
	else {
		rho_lattice[1] = rho_lattice[0] - grad_rho_3.x* dt;
		u_lattice[1] = u_lattice[0] - grad_u_3.x* dt;
		v_lattice[1] = v_lattice[0] - grad_v_3.x *dt;
		w_lattice[1] = w_lattice[0] - grad_w_3.x *dt;

		rho_lattice[2] = rho_lattice[0] + grad_rho_3.x* dt;
		u_lattice[2] = u_lattice[0] + grad_u_3.x* dt;
		v_lattice[2] = v_lattice[0] + grad_v_3.x *dt;
		w_lattice[2] = w_lattice[0] + grad_w_3.x *dt;

	}

	///   case 3 and 4: // bottom node and top node

	if (cell_normal.y > d_1) {
		
		rho_lattice[3] = rho_i - grad_rho_1.y* dt;
		u_lattice[3] = u_i - grad_u_1.y* dt;
		v_lattice[3] = v_i - grad_v_1.y *dt;
		w_lattice[3] = w_i - grad_w_1.y *dt;

		rho_lattice[4] = rho_nb + grad_rho_2.y* dt;
		u_lattice[4] = u_nb + grad_u_2.y* dt;
		v_lattice[4] = v_nb + grad_v_2.y *dt;
		w_lattice[4] = w_nb + grad_w_2.y *dt;
		
	
	}
	else if (cell_normal.y < -d_1) {
		rho_lattice[3] = rho_nb - grad_rho_2.y* dt;
		u_lattice[3] = u_nb - grad_u_2.y* dt;
		v_lattice[3] = v_nb - grad_v_2.y *dt;
		w_lattice[3] = w_nb - grad_w_2.y *dt;

		rho_lattice[4] = rho_i + grad_rho_1.y* dt;
		u_lattice[4] = u_i + grad_u_1.y* dt;
		v_lattice[4] = v_i + grad_v_1.y *dt;
		w_lattice[4] = w_i + grad_w_1.y *dt;

	}
	else {
		rho_lattice[3] = rho_lattice[0] - grad_rho_3.y* dt;
		u_lattice[3] = u_lattice[0] - grad_u_3.y* dt;
		v_lattice[3] = v_lattice[0] - grad_v_3.y *dt;
		w_lattice[3] = w_lattice[0] - grad_w_3.y *dt;

		rho_lattice[4] = rho_lattice[0] + grad_rho_3.y* dt;
		u_lattice[4] = u_lattice[0] + grad_u_3.y* dt;
		v_lattice[4] = v_lattice[0] + grad_v_3.y *dt;
		w_lattice[4] = w_lattice[0] + grad_w_3.y *dt;

	}




	///   case 5 and 6: // back node and front node

	if ( cell_normal.z > d_1) {

		rho_lattice[5] = rho_i - grad_rho_1.z* dt;
		u_lattice[5] = u_i - grad_u_1.z* dt;
		v_lattice[5] = v_i - grad_v_1.z *dt;
		w_lattice[5] = w_i - grad_w_1.z *dt;

		rho_lattice[6] = rho_nb + grad_rho_2.z* dt;
		u_lattice[6] = u_nb + grad_u_2.z* dt;
		v_lattice[6] = v_nb + grad_v_2.z *dt;
		w_lattice[6] = w_nb + grad_w_2.z *dt;

	}
	else if (cell_normal.z < -d_1){
		rho_lattice[5] = rho_nb - grad_rho_2.z* dt;
		u_lattice[5] = u_nb - grad_u_2.z* dt;
		v_lattice[5] = v_nb - grad_v_2.z *dt;
		w_lattice[5] = w_nb - grad_w_2.z *dt;

		rho_lattice[6] = rho_i + grad_rho_1.z* dt;
		u_lattice[6] = u_i + grad_u_1.z* dt;
		v_lattice[6] = v_i + grad_v_1.z *dt;
		w_lattice[6] = w_i + grad_w_1.z *dt;
	}
	else {
		rho_lattice[5] = rho_lattice[0] - grad_rho_3.z* dt;
		u_lattice[5] = u_lattice[0] - grad_u_3.z* dt;
		v_lattice[5] = v_lattice[0] - grad_v_3.z *dt;
		w_lattice[5] = w_lattice[0] - grad_w_3.z *dt;

		rho_lattice[6] = rho_lattice[0] + grad_rho_3.z* dt;
		u_lattice[6] = u_lattice[0] + grad_u_3.z* dt;
		v_lattice[6] = v_lattice[0] + grad_v_3.z *dt;
		w_lattice[6] = w_lattice[0] + grad_w_3.z *dt;

	}


	///  case 7: back bottom west and case 8: front top east
	if (( cell_normal.x +  cell_normal.y +  cell_normal.z) > d_173) {

		rho_lattice[7] = rho_i - grad_rho_1.x* dt
			- grad_rho_1.y* dt
			- grad_rho_1.z* dt;
		u_lattice[7] = u_i - grad_u_1.x* dt
			- grad_u_1.y* dt
			- grad_u_1.z* dt;
		v_lattice[7] = v_i - grad_v_1.x* dt
			- grad_v_1.y* dt
			- grad_v_1.z* dt;
		w_lattice[7] = w_i - grad_w_1.x* dt
			- grad_w_1.y* dt
			- grad_w_1.z* dt;

		rho_lattice[8] = rho_nb + grad_rho_2.x* dt
			+ grad_rho_2.y* dt
			+ grad_rho_2.z* dt;
		u_lattice[8] = u_nb + grad_u_2.x* dt
			+ grad_u_2.y* dt
			+ grad_u_2.z* dt;
		v_lattice[8] = v_nb + grad_v_2.x* dt
			+ grad_v_2.y* dt
			+ grad_v_2.z* dt;
		w_lattice[8] = w_nb + grad_w_2.x* dt
			+ grad_w_2.y* dt
			+ grad_w_2.z* dt;

	}
	else if ((cell_normal.x + cell_normal.y + cell_normal.z) < -d_173){

		rho_lattice[7] = rho_nb - grad_rho_2.x* dt
			- grad_rho_2.y* dt
			- grad_rho_2.z* dt;
		u_lattice[7] = u_nb - grad_u_2.x* dt
			- grad_u_2.y* dt
			- grad_u_2.z* dt;
		v_lattice[7] = v_nb - grad_v_2.x* dt
			- grad_v_2.y* dt
			- grad_v_2.z* dt;
		w_lattice[7] = w_nb - grad_w_2.x* dt
			- grad_w_2.y* dt
			- grad_w_2.z* dt;


		rho_lattice[8] = rho_i + grad_rho_1.x* dt
			+ grad_rho_1.y* dt
			+ grad_rho_1.z* dt;
		u_lattice[8] = u_i + grad_u_1.x* dt
			+ grad_u_1.y* dt
			+ grad_u_1.z* dt;
		v_lattice[8] = v_i + grad_v_1.x* dt
			+ grad_v_1.y* dt
			+ grad_v_1.z* dt;
		w_lattice[8] = w_i + grad_w_1.x* dt
			+ grad_w_1.y* dt
			+ grad_w_1.z* dt;
	}
	else {
		rho_lattice[7] = rho_lattice[0] - grad_rho_3.x* dt
			- grad_rho_3.y* dt
			- grad_rho_3.z* dt;
		u_lattice[7] = u_lattice[0] - grad_u_3.x* dt
			- grad_u_3.y* dt
			- grad_u_3.z* dt;
		v_lattice[7] = v_lattice[0] - grad_v_3.x* dt
			- grad_v_3.y* dt
			- grad_v_3.z* dt;
		w_lattice[7] = w_lattice[0] - grad_w_3.x* dt
			- grad_w_3.y* dt
			- grad_w_3.z* dt;


		rho_lattice[8] = rho_lattice[0] + grad_rho_3.x* dt
			+ grad_rho_3.y* dt
			+ grad_rho_3.z* dt;
		u_lattice[8] = u_lattice[0] + grad_u_3.x* dt
			+ grad_u_3.y* dt
			+ grad_u_3.z* dt;
		v_lattice[8] = v_lattice[0] + grad_v_3.x* dt
			+ grad_v_3.y* dt
			+ grad_v_3.z* dt;
		w_lattice[8] = w_lattice[0] + grad_w_3.x* dt
			+ grad_w_3.y* dt
			+ grad_w_3.z* dt;

	}


	/// case 9: front bottom west and case 10 Back Top East
	if (( cell_normal.x +  cell_normal.y + -1 * cell_normal.z) > d_173) {
		
		rho_lattice[9] = rho_i - grad_rho_1.x* dt
			- grad_rho_1.y* dt
			+ grad_rho_1.z* dt;
		u_lattice[9] = u_i - grad_u_1.x* dt
			- grad_u_1.y* dt
			+ grad_u_1.z* dt;
		v_lattice[9] = v_i - grad_v_1.x* dt
			- grad_v_1.y* dt
			+ grad_v_1.z* dt;
		w_lattice[9] = w_i - grad_w_1.x* dt
			- grad_w_1.y* dt
			+ grad_w_1.z* dt;
		
		rho_lattice[10] = rho_nb + grad_rho_2.x* dt
			+ grad_rho_2.y* dt
			- grad_rho_2.z* dt;
		u_lattice[10] = u_nb + grad_u_2.x* dt
			+ grad_u_2.y* dt
			- grad_u_2.z* dt;
		v_lattice[10] = v_nb + grad_v_2.x* dt
			+ grad_v_2.y* dt
			- grad_v_2.z* dt;
		w_lattice[10] = w_nb + grad_w_2.x* dt
			+ grad_w_2.y* dt
			- grad_w_2.z* dt;

	}
	else if ((cell_normal.x + cell_normal.y + -1 * cell_normal.z) < -d_173){

		rho_lattice[9] = rho_nb - grad_rho_2.x* dt
		- grad_rho_2.y* dt
		+ grad_rho_2.z* dt;
		u_lattice[9] = u_nb - grad_u_2.x* dt
			- grad_u_2.y* dt
			+ grad_u_2.z* dt;
		v_lattice[9] = v_nb - grad_v_2.x* dt
			- grad_v_2.y* dt
			+ grad_v_2.z* dt;
		w_lattice[9] = w_nb - grad_w_2.x* dt
			- grad_w_2.y* dt
			+ grad_w_2.z* dt;

		rho_lattice[10] = rho_i + grad_rho_1.x* dt
			+ grad_rho_1.y* dt
			- grad_rho_1.z* dt;
		u_lattice[10] = u_i + grad_u_1.x* dt
			+ grad_u_1.y* dt
			- grad_u_1.z* dt;
		v_lattice[10] = v_i + grad_v_1.x* dt
			+ grad_v_1.y* dt
			- grad_v_1.z* dt;
		w_lattice[10] = w_i + grad_w_1.x* dt
			+ grad_w_1.y* dt
			- grad_w_1.z* dt;
	}
	else {

		rho_lattice[9] = rho_lattice[0] - grad_rho_3.x* dt
			- grad_rho_3.y* dt
			+ grad_rho_3.z* dt;
		u_lattice[9] = u_lattice[0] - grad_u_3.x* dt
			- grad_u_3.y* dt
			+ grad_u_3.z* dt;
		v_lattice[9] = v_lattice[0] - grad_v_3.x* dt
			- grad_v_3.y* dt
			+ grad_v_3.z* dt;
		w_lattice[9] = w_lattice[0] - grad_w_3.x* dt
			- grad_w_3.y* dt
			+ grad_w_3.z* dt;

		rho_lattice[10] = rho_lattice[0] + grad_rho_3.x* dt
			+ grad_rho_3.y* dt
			- grad_rho_3.z* dt;
		u_lattice[10] = u_lattice[0] + grad_u_3.x* dt
			+ grad_u_3.y* dt
			- grad_u_3.z* dt;
		v_lattice[10] = v_lattice[0] + grad_v_3.x* dt
			+ grad_v_3.y* dt
			- grad_v_3.z* dt;
		w_lattice[10] = w_lattice[0] + grad_w_3.x* dt
			+ grad_w_3.y* dt
			- grad_w_3.z* dt;


	}

	/// case 11: back top west and case 12 Front Bottom East
	if ((cell_normal.x + -1 * cell_normal.y + cell_normal.z) > d_173) {

		rho_lattice[11] = rho_i - grad_rho_1.x* dt
			+ grad_rho_1.y* dt
			- grad_rho_1.z* dt;
		u_lattice[11] = u_i - grad_u_1.x* dt
			+ grad_u_1.y* dt
			- grad_u_1.z* dt;
		v_lattice[11] = v_i - grad_v_1.x* dt
			+ grad_v_1.y* dt
			- grad_v_1.z* dt;
		w_lattice[11] = w_i - grad_w_1.x* dt
			+ grad_w_1.y* dt
			- grad_w_1.z* dt;

		rho_lattice[12] = rho_nb + grad_rho_2.x* dt
			- grad_rho_2.y* dt
			+ grad_rho_2.z* dt;
		u_lattice[12] = u_nb + grad_u_2.x* dt
			- grad_u_2.y* dt
			+ grad_u_2.z* dt;
		v_lattice[12] = v_nb + grad_v_2.x* dt
			- grad_v_2.y* dt
			+ grad_v_2.z* dt;
		w_lattice[12] = w_nb + grad_w_2.x* dt
			- grad_w_2.y* dt
			+ grad_w_2.z* dt;

	}
	else if ((cell_normal.x + -1 * cell_normal.y + cell_normal.z) < -d_173) {
		
		rho_lattice[11] = rho_nb - grad_rho_2.x* dt
			+ grad_rho_2.y* dt
			- grad_rho_2.z* dt;
		u_lattice[11] = u_nb - grad_u_2.x* dt
			+ grad_u_2.y* dt
			- grad_u_2.z* dt;
		v_lattice[11] = v_nb - grad_v_2.x* dt
			+ grad_v_2.y* dt
			- grad_v_2.z* dt;
		w_lattice[11] = w_nb - grad_w_2.x* dt
			+ grad_w_2.y* dt
			- grad_w_2.z* dt;
		
		rho_lattice[12] = rho_i + grad_rho_1.x* dt
			- grad_rho_1.y* dt
			+ grad_rho_1.z* dt;
		u_lattice[12] = u_i + grad_u_1.x* dt
			- grad_u_1.y* dt
			+ grad_u_1.z* dt;
		v_lattice[12] = v_i + grad_v_1.x* dt
			- grad_v_1.y* dt
			+ grad_v_1.z* dt;
		w_lattice[12] = w_i + grad_w_1.x* dt
			- grad_w_1.y* dt
			+ grad_w_1.z* dt;
	}
	else {
		rho_lattice[11] = rho_lattice[0] - grad_rho_3.x* dt
			+ grad_rho_3.y* dt
			- grad_rho_3.z* dt;
		u_lattice[11] = u_lattice[0] - grad_u_3.x* dt
			+ grad_u_3.y* dt
			- grad_u_3.z* dt;
		v_lattice[11] = v_lattice[0] - grad_v_3.x* dt
			+ grad_v_3.y* dt
			- grad_v_3.z* dt;
		w_lattice[11] = w_lattice[0] - grad_w_3.x* dt
			+ grad_w_3.y* dt
			- grad_w_3.z* dt;

		rho_lattice[12] = rho_lattice[0] + grad_rho_3.x* dt
			- grad_rho_3.y* dt
			+ grad_rho_3.z* dt;
		u_lattice[12] = u_lattice[0] + grad_u_3.x* dt
			- grad_u_3.y* dt
			+ grad_u_3.z* dt;
		v_lattice[12] = v_lattice[0] + grad_v_3.x* dt
			- grad_v_3.y* dt
			+ grad_v_3.z* dt;
		w_lattice[12] = w_lattice[0] + grad_w_3.x* dt
			- grad_w_3.y* dt
			+ grad_w_3.z* dt;
	}



	/// case 13 Back Bottom East and  case 14: front top west
	if ((-1 * cell_normal.x + cell_normal.y + cell_normal.z) > d_173) {

		rho_lattice[13] = rho_i + grad_rho_1.x* dt
			- grad_rho_1.y* dt
			- grad_rho_1.z* dt;
		u_lattice[13] = u_i + grad_u_1.x* dt
			- grad_u_1.y* dt
			- grad_u_1.z* dt;
		v_lattice[13] = v_i + grad_v_1.x* dt
			- grad_v_1.y* dt
			- grad_v_1.z* dt;
		w_lattice[13] = w_i + grad_w_1.x* dt
			- grad_w_1.y* dt
			- grad_w_1.z* dt;

		rho_lattice[14] = rho_nb - grad_rho_2.x* dt
			+ grad_rho_2.y* dt
			+ grad_rho_2.z* dt;
		u_lattice[14] = u_nb - grad_u_2.x* dt
			+ grad_u_2.y* dt
			+ grad_u_2.z* dt;
		v_lattice[14] = v_nb - grad_v_2.x* dt
			+ grad_v_2.y* dt
			+ grad_v_2.z* dt;
		w_lattice[14] = w_nb - grad_w_2.x* dt
			+ grad_w_2.y* dt
			+ grad_w_2.z* dt;

	}
	else if ((-1 * cell_normal.x + cell_normal.y + cell_normal.z) < -d_173) {
		
		rho_lattice[13] = rho_nb + grad_rho_2.x* dt
			- grad_rho_2.y* dt
			- grad_rho_2.z* dt;
		u_lattice[13] = u_nb + grad_u_2.x* dt
			- grad_u_2.y* dt
			- grad_u_2.z* dt;
		v_lattice[13] = v_nb + grad_v_2.x* dt
			- grad_v_2.y* dt
			- grad_v_2.z* dt;
		w_lattice[13] = w_nb + grad_w_2.x* dt
			- grad_w_2.y* dt
			- grad_w_2.z* dt;


		rho_lattice[14] = rho_i - grad_rho_1.x* dt
			+ grad_rho_1.y* dt
			+ grad_rho_1.z* dt;
		u_lattice[14] = u_i - grad_u_1.x* dt
			+ grad_u_1.y* dt
			+ grad_u_1.z* dt;
		v_lattice[14] = v_i - grad_v_1.x* dt
			+ grad_v_1.y* dt
			+ grad_v_1.z* dt;
		w_lattice[14] = w_i - grad_w_1.x* dt
			+ grad_w_1.y* dt
			+ grad_w_1.z* dt;
	}
	else {
		rho_lattice[13] = rho_lattice[0] + grad_rho_3.x* dt
			- grad_rho_3.y* dt
			- grad_rho_3.z* dt;
		u_lattice[13] = u_lattice[0] + grad_u_3.x* dt
			- grad_u_3.y* dt
			- grad_u_3.z* dt;
		v_lattice[13] = v_lattice[0] + grad_v_3.x* dt
			- grad_v_3.y* dt
			- grad_v_3.z* dt;
		w_lattice[13] = w_lattice[0] + grad_w_3.x* dt
			- grad_w_3.y* dt
			- grad_w_3.z* dt;


		rho_lattice[14] = rho_lattice[0] - grad_rho_3.x* dt
			+ grad_rho_3.y* dt
			+ grad_rho_3.z* dt;
		u_lattice[14] = u_lattice[0] - grad_u_3.x* dt
			+ grad_u_3.y* dt
			+ grad_u_3.z* dt;
		v_lattice[14] = v_lattice[0] - grad_v_3.x* dt
			+ grad_v_3.y* dt
			+ grad_v_3.z* dt;
		w_lattice[14] = w_lattice[0] - grad_w_3.x* dt
			+ grad_w_3.y* dt
			+ grad_w_3.z* dt;

	}



}



__device__ void populate_feq(double u_lattice[], double v_lattice[],
	double w_lattice[], double rho_lattice[], 
	double feq_lattice[], int k, double pre_conditioned_gamma) {

	///d3q15 velocity set

	double uu2, vv2, u2v2w2, uv, uu, vv, ww2, uw, vw, ww;

	uu2 = u_lattice[k] * u_lattice[k] * pre_conditioned_gamma;
	vv2 = v_lattice[k] * v_lattice[k] * pre_conditioned_gamma;
	ww2 = w_lattice[k] * w_lattice[k] * pre_conditioned_gamma;
	u2v2w2 = (uu2 + vv2 + ww2) * 1.5;

	uv = u_lattice[k] * v_lattice[k] * 9.0 * pre_conditioned_gamma;
	uw = u_lattice[k] * w_lattice[k] * 9.0 * pre_conditioned_gamma;
	vw = v_lattice[k] * w_lattice[k] * 9.0 * pre_conditioned_gamma;

	uu = u_lattice[k];
	vv = v_lattice[k];
	ww = w_lattice[k];

	switch (k) {

	case 0:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] * (1.0

			- u2v2w2);
		break;

	case 1:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 + 3.0*uu + 4.5*uu2 - u2v2w2);
		break;
	case 2:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 - 3.0*uu + 4.5*uu2 - u2v2w2);
		break;
	case 3:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 + 3.0*vv + 4.5*vv2 - u2v2w2);
		break;
	case 4:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 - 3.0*vv + 4.5*vv2 - u2v2w2);
		break;
	case 5:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 + 3.0*ww + 4.5*ww2 - u2v2w2);
		break;
	case 6:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 - 3.0*ww + 4.5*ww2 - u2v2w2);
		break;
	case 7:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 + 3.0*uu + 3.0*vv + 3.0*ww + uv + uw + vw +
				4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
		break;
	case 8:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 - 3.0*uu - 3.0*vv - 3.0*ww + uv + uw + vw +
				4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
		break;
	case 9:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 + 3.0*uu + 3.0*vv - 3.0*ww + uv - uw - vw +
				4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
		break;
	case 10:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 - 3.0*uu - 3.0*vv + 3.0*ww + uv - uw - vw +
				4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
		break;
	case 11:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 + 3.0*uu - 3.0*vv + 3.0*ww - uv + uw - vw +
				4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
		break;
	case 12:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 - 3.0*uu + 3.0*vv - 3.0*ww - uv + uw - vw +
				4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
		break;
	case 13:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 - 3.0*uu + 3.0*vv + 3.0*ww - uv - uw + vw +
				4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
		break;
	case 14:
		feq_lattice[k] = lattice_weight[k] * rho_lattice[k] *
			(1.0 + 3.0*uu - 3.0*vv - 3.0*ww - uv - uw + vw +
				4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
		break;
	}


}



__device__ void calculate_flux_at_interface(double3 u_interface, double dt, double pre_conditioned_gamma,
	double rho_interface,  double4 &cell_flux, int i,
	double* feq_lattice, double3 cell_normal, double interface_area, double* local_fneq, double visc) {

	double uu2, vv2, ww2, u2v2w2, uu, vv, ww, uv, uw, vw, fneq_tau;
	double feq_interface[15];
	double4 x_flux, y_flux, z_flux;

	uu2 = u_interface.x * u_interface.x * pre_conditioned_gamma;
	vv2 = u_interface.y * u_interface.y * pre_conditioned_gamma;
	ww2 = u_interface.z *u_interface.z * pre_conditioned_gamma;

	u2v2w2 = (uu2 + vv2 + ww2) * 1.5;

	uu = u_interface.x;
	vv = u_interface.y;
	ww = u_interface.z;

	uv = uu * vv*9.0 * pre_conditioned_gamma;
	uw = uu * ww*9.0 * pre_conditioned_gamma;
	vw = vv * ww *9.0 * pre_conditioned_gamma;


	fneq_tau = (visc * 3 / dt * pre_conditioned_gamma);
	local_fneq[i] = fneq_tau;


	feq_interface[1] = lattice_weight[1] * rho_interface*
		(1.0 + 3.0*uu + 4.5*uu2 - u2v2w2);
	feq_interface[1] = feq_interface[1]
		- fneq_tau * (feq_interface[1] - feq_lattice[1]);

	feq_interface[2] = lattice_weight[2] * rho_interface*
		(1.0 - 3.0*uu + 4.5*uu2 - u2v2w2);
	feq_interface[2] = feq_interface[2]
		- fneq_tau * (feq_interface[2] - feq_lattice[2]);

	feq_interface[3] = lattice_weight[3] * rho_interface*
		(1.0 + 3.0*vv + 4.5*vv2 - u2v2w2);
	feq_interface[3] = feq_interface[3]
		- fneq_tau * (feq_interface[3] - feq_lattice[3]);

	feq_interface[4] = lattice_weight[4] * rho_interface*
		(1.0 - 3.0*vv + 4.5*vv2 - u2v2w2);
	feq_interface[4] = feq_interface[4]
		- fneq_tau * (feq_interface[4] - feq_lattice[4]);

	feq_interface[5] = lattice_weight[5] * rho_interface*
		(1.0 + 3.0*ww + 4.5*ww2 - u2v2w2);
	feq_interface[5] = feq_interface[5]
		- fneq_tau * (feq_interface[5] - feq_lattice[5]);

	feq_interface[6] = lattice_weight[6] * rho_interface*
		(1.0 - 3.0*ww + 4.5*ww2 - u2v2w2);
	feq_interface[6] = feq_interface[6]
		- fneq_tau * (feq_interface[6] - feq_lattice[6]);

	feq_interface[7] = lattice_weight[7] * rho_interface*
		(1.0 + 3.0*uu + 3.0*vv + 3.0*ww + uv + uw + vw +
			4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
	feq_interface[7] = feq_interface[7]
		- fneq_tau * (feq_interface[7] - feq_lattice[7]);

	feq_interface[8] = lattice_weight[8] * rho_interface*
		(1.0 - 3.0*uu - 3.0*vv - 3.0*ww + uv + uw + vw +
			4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
	feq_interface[8] = feq_interface[8]
		- fneq_tau * (feq_interface[8] - feq_lattice[8]);

	feq_interface[9] = lattice_weight[9] * rho_interface*
		(1.0 + 3.0*uu + 3.0*vv - 3.0*ww + uv - uw - vw +
			4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
	feq_interface[9] = feq_interface[9]
		- fneq_tau * (feq_interface[9] - feq_lattice[9]);

	feq_interface[10] = lattice_weight[10] * rho_interface*
		(1.0 - 3.0*uu - 3.0*vv + 3.0*ww + uv - uw - vw +
			4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
	feq_interface[10] = feq_interface[10]
		- fneq_tau * (feq_interface[10] - feq_lattice[10]);

	feq_interface[11] = lattice_weight[11] * rho_interface*
		(1.0 + 3.0*uu - 3.0*vv + 3.0*ww - uv + uw - vw +
			4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
	feq_interface[11] = feq_interface[11]
		- fneq_tau * (feq_interface[11] - feq_lattice[11]);

	feq_interface[12] = lattice_weight[12] * rho_interface*
		(1.0 - 3.0*uu + 3.0*vv - 3.0*ww - uv + uw - vw +
			4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
	feq_interface[12] = feq_interface[12]
		- fneq_tau * (feq_interface[12] - feq_lattice[12]);

	feq_interface[13] = lattice_weight[13] * rho_interface*
		(1.0 - 3.0*uu + 3.0*vv + 3.0*ww - uv - uw + vw +
			4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
	feq_interface[13] = feq_interface[13]
		- fneq_tau * (feq_interface[13] - feq_lattice[13]);

	feq_interface[14] = lattice_weight[14] * rho_interface*
		(1.0 + 3.0*uu - 3.0*vv - 3.0*ww - uv - uw + vw +
			4.5*uu2 + 4.5*vv2 + 4.5*ww2 - u2v2w2);
	feq_interface[14] = feq_interface[14]
		- fneq_tau * (feq_interface[14] - feq_lattice[14]);


	x_flux.w = (feq_interface[1] - feq_interface[2]
		+ feq_interface[7] - feq_interface[8]
		+ feq_interface[9] - feq_interface[10] + feq_interface[11]
		- feq_interface[12] - feq_interface[13]
		+ feq_interface[14]);

	x_flux.x =
		(feq_interface[1] + feq_interface[2]
			+ feq_interface[7] + feq_interface[8]
			+ feq_interface[9] + feq_interface[10] + feq_interface[11]
			+ feq_interface[12] + feq_interface[13]
			+ feq_interface[14]);

	x_flux.y =
		feq_interface[7] + feq_interface[8]
		+ feq_interface[9] + feq_interface[10] - feq_interface[11] - feq_interface[12]
		- feq_interface[13]
		- feq_interface[14];

	x_flux.z =
		feq_interface[7] + feq_interface[8]
		- feq_interface[9] - feq_interface[10] + feq_interface[11] + feq_interface[12]
		- feq_interface[13]
		- feq_interface[14];



	y_flux.w = (feq_interface[3] - feq_interface[4]
		+ feq_interface[7] - feq_interface[8]
		+ feq_interface[9] - feq_interface[10] - feq_interface[11]
		+ feq_interface[12] + feq_interface[13]
		- feq_interface[14]);


	y_flux.x = x_flux.y;

	y_flux.y = (feq_interface[3] + feq_interface[4]
		+ feq_interface[7] + feq_interface[8]
		+ feq_interface[9] + feq_interface[10] + feq_interface[11]
		+ feq_interface[12] + feq_interface[13]
		+ feq_interface[14]);

	y_flux.z = feq_interface[7] + feq_interface[8]
		- feq_interface[9] - feq_interface[10] - feq_interface[11] - feq_interface[12]
		+ feq_interface[13]
		+ feq_interface[14];

	z_flux.w = (feq_interface[5] - feq_interface[6]
		+ feq_interface[7] - feq_interface[8]
		- feq_interface[9] + feq_interface[10] + feq_interface[11]
		- feq_interface[12] + feq_interface[13]
		- feq_interface[14]);
	z_flux.x = x_flux.z;
	z_flux.y = y_flux.z;
	z_flux.z = (feq_interface[5] + feq_interface[6]
		+ feq_interface[7] + feq_interface[8]
		+ feq_interface[9] + feq_interface[10] + feq_interface[11]
		+ feq_interface[12] + feq_interface[13]
		+ feq_interface[14]);


	cell_flux.w = (x_flux.w*cell_normal.x + y_flux.w* cell_normal.y +
		z_flux.w *cell_normal.z)*interface_area;
	cell_flux.x = (x_flux.x*cell_normal.x +
		y_flux.x * cell_normal.y +
		z_flux.x*cell_normal.z)*interface_area;

	cell_flux.y = (x_flux.y*cell_normal.x +
		y_flux.y * cell_normal.y +
		z_flux.y*cell_normal.z)*interface_area;


	cell_flux.z = (x_flux.z*cell_normal.x +
		y_flux.z * cell_normal.y +
		z_flux.z*cell_normal.z)*interface_area;
}


__global__ void add_test(int n, double* delta_t, double3* area) {

	int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < n; i += stride) {
		double3 tmp;
		tmp = area[i];

		delta_t[i] = tmp.x + tmp.y + tmp.z;

	}
	return;
}


// update bc nodes to allow for changes in solution

/// bc boundary conditions only ported for dirichlet and neumann conditions
__global__ void update_unstructured_bcs(int n_bc_cells, int n_neighbours, int n_cells, int* mesh_owner, int* bcs_rho_type, int* bcs_vel_type, double4* input_soln,
	 double4* bc_arr) {



		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (int i = index; i < n_bc_cells; i += stride) {

			if (i < n_bc_cells) {



				int j, face, nb;

				j = i + n_cells;
				face = i + n_neighbours;
				nb = mesh_owner[face];

				double4 nb_soln = input_soln[nb];
				double4 temp;
				double4 bc = bc_arr[i];


				///NEEDS to be modified for non-uniform solver
				// 1 = dirichlet, 2 = neumann, 3 = periodic
				if (bcs_rho_type[i] == 1) {
					temp.w = bc.w - (nb_soln.w - bc.w);
				}
				else if (bcs_rho_type[i] == 2) {

					temp.w = nb_soln.w ;
				}

				if (bcs_vel_type[i] == 1) {

					temp.x = bc.x - (nb_soln.x - bc.x);
					temp.y = bc.y - (nb_soln.y - bc.y);
					temp.z = bc.z - (nb_soln.z - bc.z);
				}
				else if (bcs_vel_type[i] == 2) {

					temp.x = nb_soln.x;
					temp.y = nb_soln.y;
					temp.z = nb_soln.z ;
				}

				input_soln[j] = temp;

		}

	}



}



/// bc boundary conditions only ported for dirichlet and neumann conditions
__global__ void get_bc_gradients(int n_bc_cells, int n_neighbours, int n_cells, int* mesh_owner, int* bcs_rho_type, int* bcs_vel_type,  double4* input_soln,
	double3* face_normal_arr , double3* centroid, double4* bc_arr,
	double3* grad_rho_arr, double3* grad_u_arr, double3* grad_v_arr, double3* grad_w_arr) {



	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_bc_cells; i += stride) {

		if (i < n_bc_cells) {

			double3 cell_to_face = make_double3(0.0, 0.0, 0.0);
			double3 temp, neighbour, face_normal;


			int face = n_neighbours + i;
			int neighbour_id = mesh_owner[face];
			//periodic nodes get LS treatment
			int j = i + n_cells;

			face_normal = face_normal_arr[face];

			double3 centroid_nb = centroid[neighbour_id];
			double3 centroid_bc = centroid[j];
			double dx, dy, dz, d_mag, d_u, d_v, d_w, d_rho;
			double ni, nj, nk;

			double4 bc = bc_arr[i];
			double4 soln = input_soln[neighbour_id];
			
			dx = centroid_bc.x - centroid_nb.x;
			dy = centroid_bc.y - centroid_nb.y;
			dz = centroid_bc.z - centroid_nb.z;

			d_mag = sqrt(pow(dx, 2.0) + pow(dy, 2.0) + pow(dz, 2.0));

			//get unit vectors
			ni = dx / d_mag;
			nj = dy / d_mag;
			nk = dz / d_mag;


			d_u = 2 * (bc.x- soln.x);
			d_v = 2 * (bc.y - soln.y);
			d_w = 2 * (bc.z - soln.z);
			d_rho = 2 * (bc.w - soln.w);

			//dirichlet
			if (bcs_rho_type[i] == 1) {

				temp.x = d_rho / d_mag *ni;
				temp.y = d_rho / d_mag *nj;
				temp.z = d_rho / d_mag * nk;

				//neumann
			}
			else if (bcs_rho_type[i] == 2) {
				neighbour = grad_rho_arr[neighbour_id];
				temp.x = neighbour.x *(1 - fabs(face_normal.x));
				temp.y = neighbour.y*(1 - fabs(face_normal.y));
				temp.z = neighbour.z*(1 - fabs(face_normal.z));

			}
			grad_rho_arr[j] = temp;


			//dirichlet
			if (bcs_vel_type[i] == 1) {
				temp.x = d_u / d_mag*ni;
				temp.y = d_u / d_mag*nj;
				temp.z = d_u / d_mag*nk;
				grad_u_arr[j] = temp;

				temp.x = d_v / d_mag*ni;
				temp.y = d_v / d_mag*nj;
				temp.z = d_v / d_mag*nk;
				grad_v_arr[j] = temp;

				temp.x = d_w / d_mag*ni;
				temp.y = d_w / d_mag*nj;
				temp.z = d_w / d_mag*nk;
				grad_w_arr[j] = temp;
			}
			else if (bcs_vel_type[i] == 2) {

				// page 612 of Moukallad
				neighbour = grad_u_arr[neighbour_id];
				temp.x = neighbour.x *(1 - fabs(face_normal.x));
				temp.y = neighbour.y*(1 - fabs(face_normal.y));
				temp.z = neighbour.z*(1 - fabs(face_normal.z));
				grad_u_arr[j] = temp;

				neighbour = grad_v_arr[neighbour_id];
				temp.x = neighbour.x *(1 - fabs(face_normal.x));
				temp.y = neighbour.y*(1 - fabs(face_normal.y));
				temp.z = neighbour.z*(1 - fabs(face_normal.z));
				grad_v_arr[j] = temp;

				neighbour = grad_w_arr[neighbour_id];
				temp.x = neighbour.x *(1 - fabs(face_normal.x));
				temp.y = neighbour.y*(1 - fabs(face_normal.y));
				temp.z = neighbour.z*(1 - fabs(face_normal.z));
				grad_w_arr[j] = temp;

			}

		}
	}
}


__device__ void add_LS_contributions(double3 &RHS_rho, double3 &RHS_u, double3 &RHS_v, double3 &RHS_w, double4* src, int i1, int i, int i_nb, double3* RHS_array) {

	double d_u, d_v, d_w, d_rho;

	//boundary condition, find gradient at shared face
	double4 cell_1, cell_2;
	cell_1 = src[i1];
	cell_2 = src[i];
	double3 RHS;
	RHS = RHS_array[i * 6 + i_nb];

	d_u = (cell_1.x -cell_2.x);
	d_v = (cell_1.y - cell_2.y);
	d_w = (cell_1.z - cell_2.z);
	d_rho = (cell_1.w - cell_2.w);

	RHS_rho.x = RHS_rho.x + RHS.x * d_rho;
	RHS_rho.y = RHS_rho.y + RHS.y  * d_rho;
	RHS_rho.z = RHS_rho.z + RHS.z * d_rho;

	RHS_u.x = RHS_u.x + RHS.x  * d_u;
	RHS_u.y = RHS_u.y + RHS.y  * d_u;
	RHS_u.z = RHS_u.z + RHS.z * d_u;

	RHS_v.x = RHS_v.x + RHS.x  * d_v;
	RHS_v.y = RHS_v.y + RHS.y * d_v;
	RHS_v.z = RHS_v.z + RHS.z * d_v;

	RHS_w.x = RHS_w.x + RHS.x * d_w;
	RHS_w.y = RHS_w.y + RHS.y * d_w;
	RHS_w.z = RHS_w.z + RHS.z* d_w;

}



__global__ void time_integration(int n_cells, int rk, int rk_max_t, double* delta_t_local, double4* soln_t0, double4* soln_t1, double4* soln,
	double* res_rho, double* res_u, double* res_v, double* res_w) {



	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_cells; i += stride) {
		double f1, f2, f3, f4;
		double alpha[4] = { 1.0 , 0.5, 0.5,1.0 };
		double beta[4] = { 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 };

		if (i < n_cells) {

			double4 t0 = soln_t0[i];
			// update intermediate macroscopic variables for next Runge Kutta Time Step
			f1 = t0.w + res_rho[i] * delta_t_local[i] * alpha[rk];
			f2 = t0.x + (res_u[i]) *delta_t_local[i] * alpha[rk];
			f3 = t0.y  + res_v[i] * delta_t_local[i] * alpha[rk];
			f4 = t0.z + res_w[i] * delta_t_local[i] * alpha[rk];

			// change momentum to velocity
			f2 = f2 / f1;
			f3 = f3 / f1;
			f4 = f4 / f1;

			//add contributions to
			double4 t1 = soln_t1[i];
			t1.w = t1.w + delta_t_local[i] * beta[rk] * res_rho[i];
			t1.x = t1.x + delta_t_local[i] * beta[rk] * res_u[i];
			t1.y = t1.y + delta_t_local[i] * beta[rk] * res_v[i];
			t1.z = t1.z + delta_t_local[i] * beta[rk] * res_w[i];
			soln_t1[i] = t1; //add update

			double4 solution = soln[i];

			if (rk == (rk_max_t -1)) {  // assume rk4
				f1 = t1.w;
				f2 = t1.x / t1.w;
				f3 = t1.y / t1.w;
				f4 = t1.z / t1.w;

				solution.w = f1;
				solution.x = f2;
				solution.y = f3;
				solution.z = f4;

			}
			else {
				solution.w = f1;
				solution.x = f2;
				solution.y = f3;
				solution.z = f4;
			}
			soln[i] = solution;


		}

	}



}




__global__ void get_interior_gradients(int n_cells, int* gradient_cells, double4* input_soln, double3* RHS_array,
	double* LHS_xx, double* LHS_xy, double* LHS_xz,
	double* LHS_yx, double* LHS_yy, double* LHS_yz,
	double* LHS_zx, double* LHS_zy, double* LHS_zz,
	double3* grad_rho_arr, double3* grad_u_arr, double3* grad_v_arr, double3* grad_w_arr) {



	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_cells; i += stride) {

		if (i < n_cells) {
			int i1;

			double3 grad_u, grad_v, grad_w, grad_rho;

			double3 RHS_u = make_double3(0, 0, 0);
			double3 RHS_v = make_double3(0, 0, 0);
			double3 RHS_w = make_double3(0, 0, 0);
			double3 RHS_rho = make_double3(0, 0, 0);

			//change from CPU code, assume 6 surfaces i.e. hex only.
			// no need for hanging nodes etc. in remainder of project.
			//changed as vectors of vectors is not allowable in CUDA
			for (int i_nb = 0; i_nb < 6; i_nb++) {
				i1 = gradient_cells[i * 6 + i_nb];

				add_LS_contributions(RHS_rho, RHS_u, RHS_v, RHS_w, input_soln, i1, i, i_nb, RHS_array);


			}


			grad_rho.x = LHS_xx[i] * RHS_rho.x + LHS_xy[i] * RHS_rho.y + LHS_xz[i] * RHS_rho.z;
			grad_rho.y = LHS_yx[i] * RHS_rho.x + LHS_yy[i] * RHS_rho.y + LHS_yz[i] * RHS_rho.z;
			grad_rho.z = LHS_zx[i] * RHS_rho.x + LHS_zy[i] * RHS_rho.y + LHS_zz[i] * RHS_rho.z;

			grad_u.x = LHS_xx[i] * RHS_u.x + LHS_xy[i] * RHS_u.y + LHS_xz[i] * RHS_u.z;
			grad_u.y = LHS_yx[i] * RHS_u.x + LHS_yy[i] * RHS_u.y + LHS_yz[i] * RHS_u.z;
			grad_u.z = LHS_zx[i] * RHS_u.x + LHS_zy[i] * RHS_u.y + LHS_zz[i] * RHS_u.z;

			grad_v.x = LHS_xx[i] * RHS_v.x + LHS_xy[i] * RHS_v.y + LHS_xz[i] * RHS_v.z;
			grad_v.y = LHS_yx[i] * RHS_v.x + LHS_yy[i] * RHS_v.y + LHS_yz[i] * RHS_v.z;
			grad_v.z = LHS_zx[i] * RHS_v.x + LHS_zy[i] * RHS_v.y + LHS_zz[i] * RHS_v.z;

			grad_w.x = LHS_xx[i] * RHS_w.x + LHS_xy[i] * RHS_w.y + LHS_xz[i] * RHS_w.z;
			grad_w.y = LHS_yx[i] * RHS_w.x + LHS_yy[i] * RHS_w.y + LHS_yz[i] * RHS_w.z;
			grad_w.z = LHS_zx[i] * RHS_w.x + LHS_zy[i] * RHS_w.y + LHS_zz[i] * RHS_w.z;

			grad_rho_arr[i] = grad_rho;
			grad_u_arr[i] = grad_u;
			grad_v_arr[i] = grad_v;
			grad_w_arr[i] = grad_w;

		}




	}

}









__global__ void
calc_face_flux(int n, double4* input_soln, double* cell_volume, double* surface_area, int* mesh_owner, int* mesh_neighbour, double3* centroid, double3* face_centroid, double3* face_normal,
	 double* streaming_dt,
	double3* grad_rho_arr, double3* grad_u_arr, double3* grad_v_arr, double3* grad_w_arr, int n_cells,  double pre_conditioned_gamma, double* local_fneq, double visc,
	double* res_rho, double* res_u, double* res_v, double* res_w, double4* res_face,
	int bcs_rho_type [], int bcs_vel_type[], double4 bcs_arr [], double PI) {

	//loop through cells

	int index = blockIdx.x * blockDim.x + threadIdx.x;;
	int stride = blockDim.x * gridDim.x;

	__shared__ double d_1;
	d_1 =  sin(PI / 36); // distance comparisons to see if lattice nodes are within 5 degrees of the shared face
	__shared__ double d_173;
	d_173 = sin(PI / 36)*sqrt(3.0); // distance comparisons to see if lattice nodes are within 5 degrees of the shared face



	for (int face = index; face < n; face += stride) {

		if (face < n) {

			int i = mesh_owner[face];
			double interface_area = surface_area[face];
			double3 cell_1 = centroid[i]; //get current cell centre
			double3 interface_node = face_centroid[face];
			int neighbour = mesh_neighbour[face];
			double3 cell_normal = face_normal[face];
			double3 cell_2 = centroid[neighbour];
			double4 interface_macrovariables = make_double4(0, 0, 0, 0);
			double dt = streaming_dt[face]; // dt for the cell interface

			double3 grad_rho_1 = grad_rho_arr[i];
			double3 grad_rho_2 = grad_rho_arr[neighbour];
			double3 grad_u_1 = grad_u_arr[i];
			double3 grad_u_2 = grad_u_arr[neighbour];
			double3 grad_v_1 = grad_v_arr[i];
			double3 grad_v_2 = grad_v_arr[neighbour];
			double3 grad_w_1 = grad_w_arr[i];
			double3 grad_w_2 = grad_w_arr[neighbour];

			double4 owner_soln = input_soln[i];
			double4 neighbour_soln = input_soln[neighbour];

			double rho_interface;
			double3 u_interface;

			double4 cell_flux =make_double4(0,0,0,0);

			double u_lattice[15], v_lattice[15], w_lattice[15], rho_lattice[15], feq_lattice[15];



					//populate macro variables
			/*populate_lattice_macros_uniform(u_lattice, v_lattice, w_lattice, rho_lattice, cell_1, cell_2,
				interface_node, i, neighbour, grad_u_1, grad_u_2, grad_v_1, grad_v_2 , grad_w_1, grad_w_2, grad_rho_1, grad_rho_2, owner_soln, neighbour_soln, n_cells,cell_normal,dt,
				 bcs_rho_type,  bcs_vel_type, bcs_arr);*/

			 populate_lattice_macros(u_lattice, v_lattice, w_lattice, rho_lattice, cell_1, cell_2,
				interface_node, i, neighbour, grad_u_1, grad_u_2, grad_v_1, grad_v_2 , grad_w_1, grad_w_2, grad_rho_1, grad_rho_2, owner_soln, neighbour_soln, n_cells,cell_normal,dt,
				bcs_rho_type,  bcs_vel_type, bcs_arr, d_1, d_173);
			//populate macro variables


			//get initial feqs
			for (int k = 0; k < 15; k++) {
				populate_feq(u_lattice, v_lattice, w_lattice, rho_lattice, 
					feq_lattice, k, pre_conditioned_gamma);
			}
			
			// get macroscopic values at cell interface
			rho_interface = feq_lattice[0] + feq_lattice[1] + feq_lattice[2] + feq_lattice[3]
				+ feq_lattice[4] + feq_lattice[5] + feq_lattice[6] + feq_lattice[7] + feq_lattice[8]
				+ feq_lattice[9] + feq_lattice[10] + feq_lattice[11] + feq_lattice[12] + feq_lattice[13]
				+ feq_lattice[14];

			u_interface.x = 1 / rho_interface * (feq_lattice[1] - feq_lattice[2]
				+ feq_lattice[7] - feq_lattice[8]
				+ feq_lattice[9] - feq_lattice[10] + feq_lattice[11] - feq_lattice[12] - feq_lattice[13]
				+ feq_lattice[14]);

			u_interface.y = 1 / rho_interface * (feq_lattice[3] - feq_lattice[4]
				+ feq_lattice[7] - feq_lattice[8]
				+ feq_lattice[9] - feq_lattice[10] - feq_lattice[11] + feq_lattice[12] + feq_lattice[13]
				- feq_lattice[14]);

			u_interface.z = 1 / rho_interface * (feq_lattice[5] - feq_lattice[6]
				+ feq_lattice[7] - feq_lattice[8]
				- feq_lattice[9] + feq_lattice[10] + feq_lattice[11] - feq_lattice[12] + feq_lattice[13]
				- feq_lattice[14]);

			calculate_flux_at_interface(u_interface, dt, pre_conditioned_gamma, rho_interface,
				cell_flux, i, feq_lattice, cell_normal, interface_area, local_fneq, visc);
			

			res_face[face] = cell_flux;

			//atomic adds due to lack of double in cuda10, use custom function
			//// add density flux to current cell and neighbouring cell
			myatomicAdd(&res_rho[i],  -cell_flux.w / cell_volume[i]);
			myatomicAdd(&res_rho[neighbour], cell_flux.w / cell_volume[neighbour]);
			//// add x momentum
			myatomicAdd(&res_u[i], -cell_flux.x / cell_volume[i]);
			myatomicAdd(&res_u[neighbour], cell_flux.x / cell_volume[neighbour]);

			//// add y momentum
			myatomicAdd(&res_v[i], -cell_flux.y / cell_volume[i]);
			myatomicAdd(&res_v[neighbour], cell_flux.y / cell_volume[neighbour]);
			//// add z momentum
			myatomicAdd(&res_w[i], -cell_flux.z / cell_volume[i]);
			myatomicAdd(&res_w[neighbour], cell_flux.z / cell_volume[neighbour]);


		}

	}

}



__global__ void clone_a_to_b(int n_cells, double4* a, double4* b ) {

	//loop through cells

	int index = blockIdx.x * blockDim.x + threadIdx.x;;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_cells; i += stride) {

		if (i < n_cells) {
			double4 tmp1 = a[i];
			double4 tmp2;

			tmp2.x = tmp1.x;
			tmp2.y = tmp1.y;
			tmp2.z = tmp1.z;
			tmp2.w = tmp1.w;

			b[i] = tmp2;

		}
	}
}

	__global__ void fill_zero(int n_cells, double* a) {

		//loop through cells

		int index = blockIdx.x * blockDim.x + threadIdx.x;;
		int stride = blockDim.x * gridDim.x;

		for (int i = index; i < n_cells; i += stride) {
			if (i < n_cells) {
				a[i] = 0;

			}
		}
	}


	__global__ void square(int n_cells, double* a) {

		//loop through cells

		int index = blockIdx.x * blockDim.x + threadIdx.x;;
		int stride = blockDim.x * gridDim.x;

		for (int i = index; i < n_cells; i += stride) {
			if (i < n_cells) {
				a[i] = a[i]* a[i];

			}
		}
	}


	__global__ void calc_total_residual(int n_cells, double4 res_tot , double* res_rho, double* res_u, double* res_v, double* res_w ) {

		//loop through cells

		int index = blockIdx.x * blockDim.x + threadIdx.x;;
		int stride = blockDim.x * gridDim.x;

		for (int i = index; i < n_cells; i += stride) {
			if (i < n_cells) {
				myatomicAdd(&res_tot.w, res_rho[i]* res_rho[i]);
				myatomicAdd(&res_tot.x, res_u[i]* res_u[i]);
				myatomicAdd(&res_tot.y, res_v[i]* res_v[i]);
				myatomicAdd(&res_tot.z, res_w[i]* res_w[i]);

			}
		}
	}



	template <unsigned int blockSize>
	__device__ void warpReduce(volatile double *sdata, unsigned int tid)
	{ if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}

	template <unsigned int blockSize>
	__global__ void reduce6(double *g_idata, double *g_odata, unsigned int n) {
		extern __shared__ double sdata[2 * blockSize];
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*(blockSize * 2) + tid;
		unsigned int gridSize = blockSize * 2 * gridDim.x;
		sdata[tid] = 0;
		while (i < n) {
			sdata[tid] += g_idata[i] + g_idata[i + blockSize];
			i += gridSize;
		}
		__syncthreads();
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
		if (tid < 32) warpReduce< blockSize> (sdata, tid);
		if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	}

	template <unsigned int BLOCK_SIZE>
__global__ void total(double * input, double * output, int len) {
	//@@ Load a segment of the input vector into shared memory
	__shared__ double partialSum[2 * BLOCK_SIZE];
	unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
	if (start + t < len)
		partialSum[t] = input[start + t];
	else
		partialSum[t] = 0;
	if (start + BLOCK_SIZE + t < len)
		partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
	else
		partialSum[BLOCK_SIZE + t] = 0;
	//@@ Traverse the reduction tree
	for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	}
	//@@ Write the computed sum of the block to the output vector at the
	//@@ correct index
	if (t == 0)
		output[blockIdx.x] = partialSum[0];
}

__global__ void check_error(int n_cells,  double4* soln) {

	//loop through cells

	int index = blockIdx.x * blockDim.x + threadIdx.x;;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_cells; i += stride) {
		if (i < n_cells) {
			double4 tmp = soln[i];

			if (std::isnan(tmp.w) || std::isnan(tmp.x)) {
				printf("nan failure");

				/*printf << "nan failure" << endl;
				cout << t << endl;
				cout << i << endl;*/

				asm("trap;");
			}

			if (tmp.w >1000) {
				printf("rho failure");
			/*	cout << "rho failure" << endl;
				cout << t << endl;
				cout << i << endl;*/

				asm("trap;");
			}

		}
	}
}








__global__ void get_cfl_device(int n, double4* input_soln, double* cell_volume, double* delta_t_local,  double3* cfl_areas, double factor,
	double max_velocity, double pre_conditioned_gamma, double visc,int gpu_time_stepping) {

	//loop through cells

	int index = blockIdx.x * blockDim.x + threadIdx.x;;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride) {
		double effective_speed_of_sound;
		double area_x_eigen, visc_eigen;
		double visc_constant;
		visc_constant = 4;
		double  temp;
		double3 area;
		double4 soln;
		area = cfl_areas[i];
		soln = input_soln[i];		//effective_speed_of_sound = 1/sqrt(3);
		effective_speed_of_sound = max_velocity * sqrt(1 - pre_conditioned_gamma +
			pow(pre_conditioned_gamma / sqrt(3.0) / max_velocity, 2));

		if (i < n) {


			// eigen values as per Zhaoli guo(2004) - preconditioning

			  //estimation of spectral radii s per Jiri Blasek: CFD Principles and Application Determination of Max time Step
			area_x_eigen = cell_volume[i];
			area_x_eigen = 0;
			area_x_eigen = (soln.x + effective_speed_of_sound)*area.x
				+ (soln.y + effective_speed_of_sound)*area.y
				+ (soln.z + effective_speed_of_sound)*area.z;

			area_x_eigen = area_x_eigen / pre_conditioned_gamma;

			//reducing preconditioning increases viscous flux - increases eigenvalue
			visc_eigen = 2 * visc / pre_conditioned_gamma / soln.w / cell_volume[i];
			visc_eigen = visc_eigen * (area.x * area.x + area.y * area.y + area.z * area.z);

			area_x_eigen = area_x_eigen + visc_constant * visc_eigen;

			// use smallest time step allowed
			temp = factor * cell_volume[i] / area_x_eigen;

			//fully local time stepping
			delta_t_local[i] = temp;
			//delta_t_local[i] = 1;

			//if (gpu_time_stepping == 1 || gpu_time_stepping == 4) {
			//	delta_t_local[i] = temp;

			//}
			//else { //constant user defined time step
			//	delta_t_local[i] = factor;

			//}

		}
	}

	return;
}




gpu_solver::gpu_solver()
{
	//ctor
}

gpu_solver::~gpu_solver()
{
	//dtor
}


void gpu_solver::cell_interface_initialiser(double &rho_interface, vector_var &rho_u_interface,
	flux_var &x_flux, flux_var &y_flux) {
	// initialise variables
	 // add in reset function
	rho_interface = 0;

	rho_u_interface.x = 0;
	rho_u_interface.y = 0;
	rho_u_interface.z = 0;

	x_flux.P = 0;
	x_flux.momentum_x = 0;
	x_flux.momentum_y = 0;
	x_flux.momentum_z = 0;



	y_flux.P = 0;
	y_flux.momentum_x = 0;
	y_flux.momentum_y = 0;
	y_flux.momentum_z = 0;

}


double gpu_solver::feq_calc_incomp(double weight, vector_var e_alpha, vector_var u_lattice, double u_magnitude,
	double cs, double rho_lattice, double rho_0, int k) {
	double feq;


	feq = e_alpha.Dot_Product(u_lattice) *3.0;
	feq = feq + (pow(e_alpha.Dot_Product(u_lattice), 2) - pow((u_magnitude* cs), 2))
		*4.5;
	feq = feq * weight *rho_0;
	feq = feq + weight * rho_lattice;

	return feq;

}


double gpu_solver::feq_calc(double weight, vector_var e_alpha, vector_var u_lattice, double u_magnitude,
	double cs, double rho_lattice) {
	double feq;


	feq = 1.0;
	feq = feq
		+ e_alpha.Dot_Product(u_lattice) *3.0;
	feq = feq + (pow(e_alpha.Dot_Product(u_lattice), 2) - pow((u_magnitude* cs), 2))
		*4.5;
	feq = feq * weight *rho_lattice;

	return feq;

}


//get CFL numbers for inviscid and viscous matrices
// see what time stepping results
void gpu_solver::populate_cfl_areas(Solution &cfl_areas, unstructured_mesh &Mesh) {

	double area_x, area_y, area_z;
	int face;

	for (int i = 0; i < Mesh.get_n_cells(); i++) {
		area_x = 0;
		area_y = 0;
		area_z = 0;

		// time step condition as per OpenFoam calcs
		for (int f = 0; f < Mesh.gradient_faces[i].size(); f++) {
			face = Mesh.gradient_faces[i][f];

			// eigen values as per Zhaoli guo(2004) - preconditioning

			//method as per Jiri Blasek: CFD Principles and Application Determination of Max time Step

			// need to calulate correct direction of face vector

			area_x = area_x + fabs(Mesh.get_face_i(face)*Mesh.get_face_area(face));
			area_y = area_y + fabs(Mesh.get_face_j(face)*Mesh.get_face_area(face));
			area_z = area_z + fabs(Mesh.get_face_k(face)*Mesh.get_face_area(face));

		}

		cfl_areas.add_u(i, area_x / 2);
		cfl_areas.add_u(i, area_y / 2);
		cfl_areas.add_u(i, area_z / 2);

	}

	return;
}



void gpu_solver::General_Purpose_Solver(unstructured_mesh &Mesh, Solution &soln, Boundary_Conditions &bcs,
	external_forces &source, global_variables &globals, domain_geometry &domain,
	initial_conditions &init_conds, unstructured_bcs &quad_bcs_orig, int mg,
	Solution &residual, int fmg, post_processing &pp)
{

	///Declarations
	RungeKutta rk4;

	Solution residual_worker(Mesh.get_total_cells()); // stores residuals
	Solution vortex_error(Mesh.get_total_cells());
	Solution real_error(Mesh.get_total_cells());
	Solution wall_shear_stress(Mesh.get_n_wall_cells());
	Solution cfl_areas(Mesh.get_total_cells());
	gradients grads(Mesh.get_total_cells());
	flux_var RK;

	//solution based GPU variables
	double4 *temp_soln , *soln_t0,* soln_t1;

	//mesh related GPU variables
	double3 *d_cfl_areas;
	double3 *cell_centroid;
	double3 *face_normal;
	double3 *face_centroid;
	double *cell_volume;
	double *surface_area;
	int* gradient_stencil;
	int* mesh_owner;
	int* mesh_neighbour;
	double *streaming_dt;

	//residual related GPU variables
	double *res_rho, * res_u, *res_v, *res_w;

	///gradient related GPU variables
	double3 *RHS_arr;
	double3 *grad_rho_arr;
	double3 *grad_u_arr;
	double3 *grad_v_arr;
	double3 *grad_w_arr;
	double4 *res_face;
	double *LHS_xx;
	double *LHS_xy;
	double *LHS_xz;
	double *LHS_yx;
	double *LHS_yy;
	double *LHS_yz;
	double *LHS_zx;
	double *LHS_zy;
	double *LHS_zz;

	//bcs related GPU variables
	double4 *bcs_arr;
	int* bcs_rho_type;
	int* bcs_vel_type;


	double4* cell_flux_arr;

	double delta_t = globals.time_marching_step;

	double *d_delta_t_local;
	double *local_fneq;
	double * delta_t_local;
	int *delta_t_frequency;

	//assign memory allocations

	delta_t_local = new double[Mesh.get_n_cells()];
	if (delta_t_local == NULL) exit(1);
	delta_t_frequency = new int[Mesh.get_n_cells()];
	if (delta_t_frequency == NULL) exit(1);

	//solution related allocations
	temp_soln = new double4[Mesh.get_total_cells()];
	if (temp_soln == NULL) exit(1);
	soln_t0 = new double4[Mesh.get_total_cells()];
	if (soln_t0 == NULL) exit(1);
	soln_t1 = new double4[Mesh.get_total_cells()];
	if (soln_t1 == NULL) exit(1);

	d_delta_t_local = new double[Mesh.get_n_cells()];
	if (d_delta_t_local == NULL) exit(1);
	local_fneq = new double[Mesh.get_total_cells()];
	if (local_fneq == NULL) exit(1);

	//residuals related allocations
	res_rho = new double[Mesh.get_n_cells()];
	if (res_rho == NULL) exit(1);
	res_u = new double[Mesh.get_n_cells()];
	if (res_u == NULL) exit(1);
	res_v = new double[Mesh.get_n_cells()];
	if (res_v == NULL) exit(1);
	res_w = new double[Mesh.get_n_cells()];
	if (res_w == NULL) exit(1);
	res_face = new double4[Mesh.get_n_faces()];
	if (res_face == NULL) exit(1);

	//Mesh related allocations
	cell_volume = new double[Mesh.get_total_cells()];
	if (cell_volume == NULL) exit(1);
	surface_area = new double[Mesh.get_n_faces()];
	if (surface_area == NULL) exit(1);
	gradient_stencil = new int[Mesh.get_n_cells() * 6];
	if (gradient_stencil == NULL) exit(1);
	mesh_owner = new int[Mesh.get_n_faces()];
	if (mesh_owner == NULL) exit(1);
	mesh_neighbour= new int[Mesh.get_n_faces()];
	if (mesh_neighbour == NULL) exit(1);
	d_cfl_areas = new double3[Mesh.get_total_cells()];
	if (d_cfl_areas == NULL) exit(1);
	cell_centroid = new double3[Mesh.get_total_cells()];
	if (cell_centroid == NULL) exit(1);
	face_centroid = new double3[Mesh.get_n_faces()];
	if (face_centroid == NULL) exit(1);
	face_normal = new double3[Mesh.get_n_faces()];
	if (face_normal == NULL) exit(1);
	streaming_dt = new double[Mesh.get_total_cells()];
	if (streaming_dt == NULL) exit(1);

	cell_flux_arr = new double4[Mesh.get_n_faces()];
	if (cell_flux_arr == NULL) exit(1);

	//bcs related GPU variables

	bcs_arr = new double4[Mesh.get_num_bc()];
	if (bcs_arr == NULL) exit(1);
	bcs_rho_type = new int[Mesh.get_num_bc()];
	if (bcs_rho_type == NULL) exit(1);
	bcs_vel_type = new int[Mesh.get_num_bc()];
	if (bcs_vel_type == NULL) exit(1);

	//Gradient related allocations

	RHS_arr = new double3[Mesh.get_n_cells()* 6];
	if (RHS_arr == NULL) exit(1);
	grad_rho_arr = new double3[Mesh.get_total_cells()];
	if (grad_rho_arr == NULL) exit(1);
	grad_u_arr = new double3[Mesh.get_total_cells()];
	if (grad_u_arr == NULL) exit(1);
	grad_v_arr = new double3[Mesh.get_total_cells()];
	if (grad_v_arr == NULL) exit(1);
	grad_w_arr = new double3[Mesh.get_total_cells()];
	if (grad_w_arr == NULL) exit(1);
	LHS_xx = new double[Mesh.get_n_cells()];
	if (LHS_xx == NULL) exit(1);
	LHS_xy = new double[Mesh.get_n_cells()];
	if (LHS_xy == NULL) exit(1);
	LHS_xz = new double[Mesh.get_n_cells()];
	if (LHS_xz == NULL) exit(1);
	LHS_yx = new double[Mesh.get_n_cells()];
	if (LHS_yx == NULL) exit(1);
	LHS_yy = new double[Mesh.get_n_cells()];
	if (LHS_yy == NULL) exit(1);
	LHS_yz = new double[Mesh.get_n_cells()];
	if (LHS_yz == NULL) exit(1);
	LHS_zx = new double[Mesh.get_n_cells()];
	if (LHS_zx == NULL) exit(1);
	LHS_zy = new double[Mesh.get_n_cells()];
	if (LHS_zy == NULL) exit(1);
	LHS_zz = new double[Mesh.get_n_cells()];
	if (LHS_zz == NULL) exit(1);


	double local_tolerance;

	//host lattice wegihts
	double* h_lattice_weight;
	h_lattice_weight = new double[15];
	if (h_lattice_weight == NULL) exit(1);

	double time;


	double output_residual_threshold = 0;
	double visc;

	double angular_freq, wom_cos, force;

	double td; // taylor vortex decay time
	double drag_t1; //drag co-efficients

	std::ofstream error_output, vortex_output, max_u, debug_log;
	std::string output_dir, decay_dir, max_u_dir;
	output_dir = globals.output_file + "/error.txt";
	vector_var cell_1, cell_2, interface_node, lattice_node, delta_u, delta_v, delta_w, delta_rho;
	vector_var relative_interface;
	vector_var  vel_lattice, rho_u_interface, u_interface;
	vector_var delta_u1, delta_v1, delta_w1, delta_rho1;
	vector_var cell_normal;
	vector_var flux_e_alpha[9];
	vector_var u, v, w, rho;
	std::vector<vector_var> e_alpha;
	std::vector<int> cell_nodes;

	// vector_var flux_e_alpha;
	residuals convergence_residual;
	flux_var x_flux, y_flux, z_flux;
	flux_var cell_flux;

	flux_var debug[4], debug_flux[4], arti_debug[4];
	flux_var dbug[4];
	flux_var int_debug[4];


	int timesteps;
	int wall = 0;




	tecplot_output<double> tecplot;

	///Output related parameters and file locations

	dt = domain.dt; // timestepping for streaming // non-dim equals 1
	c = 1; // assume lattice spacing is equal to streaming timestep
	cs = c / sqrt(3);
	visc = (globals.tau - 0.5) / 3 * domain.dt;

	local_tolerance = globals.tolerance;
	delta_t = 1;
	timesteps = ceil(globals.simulation_length);
	output_dir = globals.output_file + "/error.txt";
	decay_dir = globals.output_file + "/vortex_error.txt";
	max_u_dir = globals.output_file + "/max_u.txt";
	error_output.open(output_dir.c_str(), ios::out);
	output_dir = globals.output_file + "/residual_log.txt";
	debug_log.open(output_dir.c_str(), ios::out);
	vortex_output.open(decay_dir.c_str(), ios::out);
	max_u.open(max_u_dir.c_str(), ios::out);
	time = 0;
	angular_freq = visc * pow(globals.womersley_no, 2) / pow(Mesh.get_Y() / 2, 2);
	force = -init_conds.pressure_gradient;
	time = 0;
	td = 100000000000000000;

	/// get LHD and RHS coefficients for Hybrid Least Squares Method
	grads.pre_fill_LHS_and_RHS_matrix(bcs, Mesh, domain, soln, globals);
	
	// get cell areas for CFL calcs
	populate_cfl_areas(cfl_areas, Mesh);

	// debug output (optional)
	debug_log << "t,rk,i,res_rho,res_u,res_v,res_w,x,y,z, dt,visc,rho,u,v,ux,uy,uz,vx,vy,vz" << endl;

/// CUDA checks***********************************//////////////////////
	cudaDeviceProp deviceProp;
	int argc;
	const char *argv = " ";
	/*int devID = findCudaDevice(argc, (const char **)argv);

	if (devID < 0) {
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}*/

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

	// Statistics about the GPU device
	printf(
		"> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
	checkCudaErrors(cudaSetDevice(0));


	// num bloacks for different gpu kernels
	int blockSize = 256;
	int numBlocks = (Mesh.get_total_cells() + blockSize - 1) / blockSize;
	int n_Cell_Blocks = (Mesh.get_n_cells() + blockSize - 1) / blockSize;
	int n_bc_Blocks = (Mesh.get_num_bc() + blockSize - 1) / blockSize;
	int n_face_Blocks = (Mesh.get_n_faces() + blockSize - 1) / blockSize;
	double4 convergence;
	convergence.w = 100000000000;

	double *res_rho_block;
	res_rho_block = new double[n_Cell_Blocks];
	double *res_u_block;
	res_u_block = new double[n_Cell_Blocks];
	double *res_v_block;
	res_v_block = new double[n_Cell_Blocks];
	double *res_w_block;
	res_w_block = new double[n_Cell_Blocks];

	//arrrays for CUDA
	checkCudaErrors(cudaMallocManaged(&res_rho_block, n_Cell_Blocks * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&res_u_block, n_Cell_Blocks * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&res_v_block, n_Cell_Blocks * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&res_w_block, n_Cell_Blocks * sizeof(double)));

	checkCudaErrors(cudaMallocManaged(&d_delta_t_local, Mesh.get_total_cells() * sizeof(double)));

	checkCudaErrors(cudaMallocManaged(&d_cfl_areas, Mesh.get_total_cells() * sizeof(double3)));
	checkCudaErrors(cudaMallocManaged(&temp_soln, Mesh.get_total_cells() * sizeof(double4)));
	checkCudaErrors(cudaMallocManaged(&soln_t0, Mesh.get_total_cells() * sizeof(double4)));
	checkCudaErrors(cudaMallocManaged(&soln_t1, Mesh.get_total_cells() * sizeof(double4)));
	checkCudaErrors(cudaMallocManaged(&cell_volume, Mesh.get_total_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&gradient_stencil, Mesh.get_n_cells() * sizeof(int)*6));
	checkCudaErrors(cudaMallocManaged(&mesh_owner, Mesh.get_n_faces() * sizeof(int) ));
	checkCudaErrors(cudaMallocManaged(&mesh_neighbour, Mesh.get_n_faces() * sizeof(int)));
	checkCudaErrors(cudaMallocManaged(&cell_centroid, Mesh.get_total_cells() * sizeof(double3)));
	checkCudaErrors(cudaMallocManaged(&face_centroid, Mesh.get_n_faces() * sizeof(double3)));
	checkCudaErrors(cudaMallocManaged(&face_normal, Mesh.get_n_faces() * sizeof(double3)));
	checkCudaErrors(cudaMallocManaged(&surface_area, Mesh.get_n_faces() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&streaming_dt, Mesh.get_n_faces() * sizeof(double)));

	checkCudaErrors(cudaMallocManaged(&cell_flux_arr, Mesh.get_n_faces() * sizeof(double4)));

	checkCudaErrors(cudaMallocManaged(&res_rho, Mesh.get_total_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&res_u, Mesh.get_total_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&res_v, Mesh.get_total_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&res_w, Mesh.get_total_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&res_face, Mesh.get_n_faces() * sizeof(double4)));

	checkCudaErrors(cudaMallocManaged(&local_fneq, Mesh.get_total_cells() * sizeof(double)));

	//arrays for bcs
	checkCudaErrors(cudaMallocManaged(&bcs_arr, Mesh.get_num_bc() * sizeof(double4)));
	checkCudaErrors(cudaMallocManaged(&bcs_rho_type, Mesh.get_num_bc() * sizeof(int)));
	checkCudaErrors(cudaMallocManaged(&bcs_vel_type, Mesh.get_num_bc() * sizeof(int)));


	//arrrays for CUDA Gradient
	checkCudaErrors(cudaMallocManaged(&grad_rho_arr, Mesh.get_total_cells() * sizeof(double3)));
	checkCudaErrors(cudaMallocManaged(&grad_u_arr, Mesh.get_total_cells() * sizeof(double3)));
	checkCudaErrors(cudaMallocManaged(&grad_v_arr, Mesh.get_total_cells() * sizeof(double3)));
	checkCudaErrors(cudaMallocManaged(&grad_w_arr, Mesh.get_total_cells() * sizeof(double3)));
	checkCudaErrors(cudaMallocManaged(&RHS_arr, Mesh.get_n_cells() * sizeof(double3)*6 ));
	checkCudaErrors(cudaMallocManaged(&LHS_xx, Mesh.get_n_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&LHS_xy, Mesh.get_n_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&LHS_xz, Mesh.get_n_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&LHS_yx, Mesh.get_n_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&LHS_yy, Mesh.get_n_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&LHS_yz, Mesh.get_n_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&LHS_zx, Mesh.get_n_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&LHS_zy, Mesh.get_n_cells() * sizeof(double)));
	checkCudaErrors(cudaMallocManaged(&LHS_zz, Mesh.get_n_cells() * sizeof(double)));

	// get lattice velocities and transfer to GPU
	populate_e_alpha(e_alpha, h_lattice_weight, c, globals.PI, 15);
	checkCudaErrors(cudaMemcpyToSymbol(lattice_weight, h_lattice_weight, 15 * sizeof(double)));

	/// Sync before CUDA array used
	cudaDeviceSynchronize();
	//transfer CFL areas to device
	populate_cfl_areas(d_cfl_areas, Mesh);


	// transfer class members to arrays for CUDA
	soln_to_double(temp_soln, soln, Mesh.get_total_cells());

	mesh_to_array(cell_volume, Mesh, Mesh.get_total_cells(), "volume");
	mesh_to_array(gradient_stencil, Mesh, Mesh.get_n_cells(), "gradient_stencil");
	mesh_to_array(mesh_owner, Mesh, Mesh.get_n_faces(), "mesh_owner");
	mesh_to_array(mesh_neighbour, Mesh, Mesh.get_n_faces(), "mesh_neighbour");
	mesh_to_array(surface_area, Mesh, Mesh.get_n_faces(), "surface_area");
	mesh_to_array(streaming_dt, Mesh, Mesh.get_n_faces(), "streaming_dt");
	mesh_to_array_double(face_normal, Mesh, Mesh.get_n_faces(), "face_normal");
	mesh_to_array_double(cell_centroid, Mesh, Mesh.get_total_cells(), "cell_centroid");
	mesh_to_array_double(face_centroid, Mesh, Mesh.get_n_faces(), "face_centroid");

	gradients_to_array(LHS_xx, grads, Mesh.get_n_cells(), "LHS_xx");
	gradients_to_array(LHS_xy, grads, Mesh.get_n_cells(), "LHS_xy");
	gradients_to_array(LHS_xz, grads, Mesh.get_n_cells(), "LHS_xz");
	gradients_to_array(LHS_yx, grads, Mesh.get_n_cells(), "LHS_yx");
	gradients_to_array(LHS_yy, grads, Mesh.get_n_cells(), "LHS_yy");
	gradients_to_array(LHS_yz, grads, Mesh.get_n_cells(), "LHS_yz");
	gradients_to_array(LHS_zx, grads, Mesh.get_n_cells(), "LHS_zx");
	gradients_to_array(LHS_zy, grads, Mesh.get_n_cells(), "LHS_zy");
	gradients_to_array(LHS_zz, grads, Mesh.get_n_cells(), "LHS_zz");
	gradients_to_array_double(RHS_arr, grads, Mesh.get_n_cells(), "RHS_array");

	bcs_to_array_double(bcs_arr, bcs, Mesh.get_num_bc(), "bcs");
	bcs_to_array(bcs_rho_type, bcs, Mesh.get_num_bc(), "rho_type");
	bcs_to_array(bcs_vel_type, bcs, Mesh.get_num_bc(), "vel_type");
	cudaProfilerStart();

	clone_a_to_b << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), temp_soln, soln_t1); // soln_t0 holds macro variable solution at start of time step

	// loop in time
	for (int t = 0; t < timesteps; t++) {
		// soln_t0 is the solution at the start of every
		// RK step.(rk = n) Temp_soln holds the values at end of
		// step.(rk = n+1)
		clone_a_to_b << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), soln_t1, soln_t0);// soln_t0 holds macro variable solution at start of time step
		post_kernel_checks();

		//local timestepping calculation
		get_cfl_device <<< n_Cell_Blocks, blockSize >>> (Mesh.get_n_cells(),  temp_soln, cell_volume, d_delta_t_local,  d_cfl_areas, globals.time_marching_step,
			globals.max_velocity,globals.pre_conditioned_gamma, globals.visc, globals.gpu_time_stepping);
		post_kernel_checks();

		for (int rk = 0; rk < rk4.timesteps; rk++) {

			drag_t1 = 0.0;

			//update temp_soln boundary conditions
			update_unstructured_bcs << < n_bc_Blocks, blockSize >> > (Mesh.get_num_bc(), Mesh.get_n_neighbours(), Mesh.get_n_cells(), mesh_owner, bcs_rho_type, bcs_vel_type, temp_soln, bcs_arr);

			//set residuals to zeros
			fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_rho);
			post_kernel_checks();
			fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_u);
			post_kernel_checks();
			fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_v);
			post_kernel_checks();
			fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_w);
			post_kernel_checks();

			get_interior_gradients <<< n_Cell_Blocks, blockSize >>> ( Mesh.get_n_cells(), gradient_stencil, temp_soln,
				RHS_arr,LHS_xx, LHS_xy, LHS_xz,LHS_yx, LHS_yy, LHS_yz,LHS_zx, LHS_zy, LHS_zz,
				grad_rho_arr, grad_u_arr, grad_v_arr, grad_w_arr);
			post_kernel_checks();


			//get boundary condition gradients
			get_bc_gradients << < n_bc_Blocks, blockSize >> > (Mesh.get_num_bc(), Mesh.get_n_neighbours(), Mesh.get_n_cells(), mesh_owner, bcs_rho_type, bcs_vel_type, temp_soln,
				face_normal, cell_centroid, bcs_arr,
				grad_rho_arr, grad_u_arr, grad_v_arr, grad_w_arr);
			post_kernel_checks();

			cudaDeviceSynchronize();
			//get flux at each face
			calc_face_flux << < n_face_Blocks, blockSize >> > (Mesh.get_n_faces(), temp_soln, cell_volume, surface_area,  mesh_owner,  mesh_neighbour,  cell_centroid, face_centroid,  face_normal,
				streaming_dt, grad_rho_arr, grad_u_arr,  grad_v_arr, grad_w_arr, Mesh.get_n_cells(),  (1/ globals.pre_conditioned_gamma),  local_fneq, globals.visc,
				res_rho, res_u,res_v,res_w,res_face,
				bcs_rho_type, bcs_vel_type, bcs_arr,globals.PI);
			post_kernel_checks();
			cudaDeviceSynchronize();

			//Update  solutions  //update RK values

			time_integration << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), rk, rk4.timesteps, d_delta_t_local, soln_t0,  soln_t1, temp_soln,
				 res_rho, res_u,  res_v, res_w);
			post_kernel_checks();

		}
		

		//get square of residuals
		square << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_rho);
		square << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_u);
		square << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_v);
		square << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_w);

		//reduce add residuals
		total<256> << < n_Cell_Blocks, blockSize >> > (res_rho, res_rho_block, Mesh.get_n_cells());
		total<256> << < n_Cell_Blocks, blockSize >> > (res_u, res_u_block, Mesh.get_n_cells());
		total<256> << < n_Cell_Blocks, blockSize >> > (res_v, res_v_block, Mesh.get_n_cells());
		total<256> << < n_Cell_Blocks, blockSize >> > (res_w, res_w_block, Mesh.get_n_cells());
		post_kernel_checks();
		cudaDeviceSynchronize();
		convergence_residual.reset();
		//get L2 norm of the residual
		convergence_residual.l2_norm_rms_moukallad(globals, res_rho_block, res_u_block, res_v_block, res_w_block, n_Cell_Blocks, Mesh.get_n_cells());

		time = t * delta_t;

		//output tecplot output and residuals every output step
		if (mg == 0 && t%globals.output_step == 1) {

			soln.clone(temp_soln);

			error_output << t << ", " << convergence_residual.max_error() << ", " <<
				convergence_residual.rho_rms << ", " << convergence_residual.u_rms << ", " <<
				convergence_residual.v_rms << ", " <<
				convergence_residual.w_rms << " , FMG cycle: " << fmg << endl;
			cout << "time t=" << time << " error e =" << convergence_residual.max_error()
				<< " delta_t:" << delta_t << std::endl;
			//max_u << t << "," << soln.get_u(center_node) << "," << force << endl;
			cout << "drag: " << drag_t1 << endl;

			//only output at decreasing order of magnitudes - save space on hard drive
			if (convergence_residual.max_error() < pow(10, output_residual_threshold)) {
				tecplot.tecplot_output_unstructured_soln(globals, Mesh, soln, bcs, time, pp, residual_worker, delta_t_local, local_fneq);
				output_residual_threshold = output_residual_threshold - 1;
				soln.output(globals.output_file, globals, domain);
				cudaProfilerStop();
			}
			//soln.output_centrelines(globals.output_file,globals,Mesh,time);

		}

		//check for convergence
		if (convergence_residual.max_error() < local_tolerance || time > td) {
			if (mg == 0) {
				soln.clone(temp_soln);
				cout << "convergence" << endl;
				cout << "time t=" << time << " error e =" << convergence_residual.max_error()
					<< " delta_t:" << delta_t << std::endl;
				error_output.close();
				debug_log.close();
				vortex_output.close();
				max_u.close();

				// vortex calcs
				soln.update_unstructured_bcs(bcs, Mesh, domain, t);
				grads.Get_LS_Gradients(bcs, Mesh, domain, soln, globals);
				pp.cylinder_post_processing(Mesh, globals, grads, bcs, soln, domain, wall_shear_stress);
				tecplot.tecplot_output_unstructured_soln(globals, Mesh, soln, bcs, time, pp, residual_worker, delta_t_local, local_fneq);
				
			}
			cudaProfilerStop();

			return;
		}


	}



	cudaProfilerStop();
	soln.clone(temp_soln);
	cout << "out of time" << endl;
	error_output.close();
	vortex_output.close();
	debug_log.close();
	max_u.close();
	pp.cylinder_post_processing(Mesh, globals, grads, bcs, soln, domain, wall_shear_stress);
	tecplot.tecplot_output_unstructured_soln(globals, Mesh, soln, bcs, time, pp, residual_worker, delta_t_local, local_fneq);


	//no destruction of memory as end of programme

}


void gpu_solver::get_weighted_average(gradients &grads, int i, int neighbour, double m1, double m2,
	vector_var &u, vector_var &v, vector_var &w, vector_var &rho, unstructured_mesh &mesh)
{
	double a, b, x, y, z;

	//check for boundary condition

	//use boundary cell gradients as these are at cell face
	if (neighbour > mesh.get_n_cells()) {
		x = grads.get_u(neighbour).x;
		y = grads.get_u(neighbour).y;
		z = grads.get_u(neighbour).z;
		u.set_equal(x, y, z);

		x = grads.get_v(neighbour).x;
		y = grads.get_v(neighbour).y;
		z = grads.get_v(neighbour).z;
		v.set_equal(x, y, z);


		x = grads.get_w(neighbour).x;
		y = grads.get_w(neighbour).y;
		z = grads.get_w(neighbour).z;
		w.set_equal(x, y, z);


		x = grads.get_rho(neighbour).x;
		y = grads.get_rho(neighbour).y;
		z = grads.get_rho(neighbour).z;
		rho.set_equal(x, y, z);


	}
	else {

		a = m1 + m2;
		b = m2 / a;
		a = m1 / a;


		x = grads.get_u(i).x * a + grads.get_u(neighbour).x *b;
		y = grads.get_u(i).y * a + grads.get_u(neighbour).y *b;
		z = grads.get_u(i).z * a + grads.get_u(neighbour).z *b;
		u.set_equal(x, y, z);

		x = grads.get_v(i).x * a + grads.get_v(neighbour).x *b;
		y = grads.get_v(i).y * a + grads.get_v(neighbour).y *b;
		z = grads.get_v(i).z * a + grads.get_v(neighbour).z *b;
		v.set_equal(x, y, z);


		x = grads.get_w(i).x * a + grads.get_w(neighbour).x *b;
		y = grads.get_w(i).y * a + grads.get_w(neighbour).y *b;
		z = grads.get_w(i).z * a + grads.get_w(neighbour).z *b;
		w.set_equal(x, y, z);


		x = grads.get_rho(i).x * a + grads.get_rho(neighbour).x *b;
		y = grads.get_rho(i).y * a + grads.get_rho(neighbour).y *b;
		z = grads.get_rho(i).z * a + grads.get_rho(neighbour).z *b;
		rho.set_equal(x, y, z);
	}


}


vector_var gpu_solver::get_e_alpha(int k, double &lattice_weight, double c, double PI) {

	vector_var temp;
	int x, y, z;
	//get e_alpha again
	if (k > 0 && k < 5) { //

		x = round(cos((k - 1)*PI / 2) * c);
		y = round(sin((k - 1)*PI / 2)* c);
		z = 0; //update in 3D
		lattice_weight = 1.0 / 9.0;
	}
	else if (k > 4) {

		x = round(sqrt(2) * cos((k - 5)*PI / 2 + PI / 4) * c);
		y = round(sqrt(2) * sin((k - 5)*PI / 2 + PI / 4) * c);
		z = 0; //update in 3D
		lattice_weight = 1.0 / 36.0;

	}
	else {
		x = 0;
		y = 0;
		z = 0;
		lattice_weight = 4.0 / 9.0;
	}
	temp.x = x;
	temp.y = y;
	temp.z = z;


	return temp;
}

void gpu_solver::populate_e_alpha(vector<vector_var> &e_alpha, double *lattice_weight, double c, double PI, int j) {

	vector_var temp;
	int x[15] = { 0,1,-1,0,0,0,0,1,-1, 1,-1,1,-1,-1,1 };
	int y[15] = { 0,0,0,1,-1,0,0,1,-1,1,-1,-1,1,1,-1 };
	int z[15] = { 0,0,0,0,0,1,-1,1,-1,-1,1,1,-1,1,-1 };
	//get e_alpha again

	for (int k = 0; k < j; k++) {
		if (k > 0 && k < 7) { //

			lattice_weight[k] = 1.0 / 9.0;

		}
		else if (k > 6) {


			lattice_weight[k] = 1.0 / 72.0;

		}
		else {

			lattice_weight[k] = 2.0 / 9.0;
		}



		temp.x = x[k];
		temp.y = y[k];
		temp.z = z[k];

		e_alpha.push_back(temp);


	}



}

void gpu_solver::get_cell_gradients(Mesh &Mesh, int i, int neighbour, int j, Solution &temp_soln,
	vector_var &delta_rho, vector_var &delta_rho1,
	vector_var &delta_u, vector_var &delta_u1,
	vector_var &delta_v, vector_var &delta_v1,
	Boundary_Conditions &bcs) {


	int neighbour_1, neighbour_2;
	vector_var cell_1, cell_2;
	// is it N-S or E-W
	if (j == 2) {


		neighbour_1 = Mesh.get_w_node(i);
		neighbour_2 = Mesh.get_e_node(i);

	}
	else {
		neighbour_1 = Mesh.get_s_node(i);
		neighbour_2 = Mesh.get_n_node(i);

	}

	// get neighbouring cells of cells
	Mesh.get_centroid(neighbour_1, cell_1);
	Mesh.get_centroid(neighbour_2, cell_2);

	delta_rho.Get_Gradient(temp_soln.get_rho(neighbour_1), temp_soln.get_rho(neighbour_2)
		, cell_1, cell_2);
	delta_u.Get_Gradient(temp_soln.get_u(neighbour_1), temp_soln.get_u(neighbour_2)
		, cell_1, cell_2);
	delta_v.Get_Gradient(temp_soln.get_v(neighbour_1), temp_soln.get_v(neighbour_2)
		, cell_1, cell_2);


	// get gradient of neighbouring cell
	if (j == 2) {

		neighbour_1 = Mesh.get_w_node(neighbour);
		neighbour_2 = Mesh.get_e_node(neighbour);

	}
	else {
		neighbour_1 = Mesh.get_s_node(neighbour);
		neighbour_2 = Mesh.get_n_node(neighbour);

	}

	// get neighbouring cells of cells
	Mesh.get_centroid(neighbour_1, cell_1);
	Mesh.get_centroid(neighbour_2, cell_2);

	delta_rho1.Get_Gradient(temp_soln.get_rho(neighbour_1), temp_soln.get_rho(neighbour_2)
		, cell_1, cell_2);
	delta_u1.Get_Gradient(temp_soln.get_u(neighbour_1), temp_soln.get_u(neighbour_2)
		, cell_1, cell_2);
	delta_v1.Get_Gradient(temp_soln.get_v(neighbour_1), temp_soln.get_v(neighbour_2)
		, cell_1, cell_2);

}

void gpu_solver::cell_interface_variables(int j, int i, vector_var &interface_node, int &neighbour, double &interface_area,
	vector_var &cell_normal, Boundary_Conditions &boundary_conditions, bc_var &bc,
	Mesh &Mesh, vector_var &cell_2) {

	switch (j) {

	case 0: // West
		interface_node.x = Mesh.get_west_x(i);
		interface_node.y = Mesh.get_west_y(i);
		interface_node.z = Mesh.get_west_z(i);
		neighbour = Mesh.get_w_node(i);
		interface_area = Mesh.get_w_area(i);
		cell_normal.x = Mesh.get_w_i(i);
		cell_normal.y = Mesh.get_w_j(i);
		cell_normal.z = Mesh.get_w_k(i);
		break;

	case 1: // South
		interface_node.x = Mesh.get_south_x(i);
		interface_node.y = Mesh.get_south_y(i);
		interface_node.z = Mesh.get_south_z(i);
		neighbour = Mesh.get_s_node(i);
		interface_area = Mesh.get_s_area(i);
		cell_normal.x = Mesh.get_s_i(i);
		cell_normal.y = Mesh.get_s_j(i);
		cell_normal.z = Mesh.get_s_k(i);

		break;
	case 2: // East
		interface_node.x = Mesh.get_east_x(i);
		interface_node.y = Mesh.get_east_y(i);
		interface_node.z = Mesh.get_east_z(i);
		interface_area = Mesh.get_e_area(i);
		neighbour = Mesh.get_e_node(i);
		cell_normal.x = Mesh.get_e_i(i);
		cell_normal.y = Mesh.get_e_j(i);
		cell_normal.z = Mesh.get_e_k(i);

		break;
	case 3: // North
		interface_node.x = Mesh.get_north_x(i);
		interface_node.y = Mesh.get_north_y(i);
		interface_node.z = Mesh.get_north_z(i);
		neighbour = Mesh.get_n_node(i);
		interface_area = Mesh.get_n_area(i);
		cell_normal.x = Mesh.get_n_i(i);
		cell_normal.y = Mesh.get_n_j(i);
		cell_normal.z = Mesh.get_n_k(i);

		break;
	case 4: // Front
		interface_node.x = Mesh.get_front_x(i);
		interface_node.y = Mesh.get_front_y(i);
		interface_node.z = Mesh.get_front_z(i);
		neighbour = Mesh.get_f_node(i);
		interface_area = Mesh.get_f_area(i);
		cell_normal.x = Mesh.get_f_i(i);
		cell_normal.y = Mesh.get_f_j(i);
		cell_normal.z = Mesh.get_f_k(i);

		break;
	case 5: // Back
		interface_node.x = Mesh.get_back_x(i);
		interface_node.y = Mesh.get_back_y(i);
		interface_node.z = Mesh.get_back_z(i);
		neighbour = Mesh.get_b_node(i);
		interface_area = Mesh.get_b_area(i);
		cell_normal.x = Mesh.get_b_i(i);
		cell_normal.y = Mesh.get_b_j(i);
		cell_normal.z = Mesh.get_b_k(i);
		break;


	}
	//        cell_2.x = Mesh.get_centroid_x(neighbour);
	//        cell_2.y = Mesh.get_centroid_y((neighbour));
	//        cell_2.z = Mesh.get_centroid_z(neighbour);

}



void gpu_solver::cell_interface_variables(int face, int i, vector_var &interface_node, int &neighbour, double &interface_area,
	vector_var &cell_normal, Boundary_Conditions &boundary_conditions, bc_var &bc,
	unstructured_mesh &Mesh, vector_var &cell_2, vector_var &cell_1) {



	interface_node.x = Mesh.get_face_x(face);
	interface_node.y = Mesh.get_face_y(face);
	interface_node.z = Mesh.get_face_z(face);

	neighbour = Mesh.get_mesh_neighbour(face);
	interface_area = Mesh.get_face_area(face);
	cell_normal.x = Mesh.get_face_i(face);
	cell_normal.y = Mesh.get_face_j(face);
	cell_normal.z = Mesh.get_face_k(face);


	cell_2.x = Mesh.get_centroid_x(neighbour);
	cell_2.y = Mesh.get_centroid_y((neighbour));
	cell_2.z = Mesh.get_centroid_z(neighbour);


}



void gpu_solver::get_cell_nodes(std::vector<int> &cell_nodes, Boundary_Conditions &bcs, int neighbour,
	Mesh &Mesh, int i, int j) {

	//current cell
	cell_nodes.clear();
	if (bcs.get_bc(i) || bcs.get_bc(neighbour)) {
		cell_nodes.push_back(i);
		cell_nodes.push_back(neighbour);

	}
	else if (j == 2) {
		cell_nodes.push_back(i);
		cell_nodes.push_back(Mesh.get_n_node(i));
		//cell_nodes.push_back(Mesh.get_e_node(i));
		//cell_nodes.push_back(Mesh.get_w_node(i));
		cell_nodes.push_back(Mesh.get_s_node(i));
		cell_nodes.push_back(neighbour);
		cell_nodes.push_back(Mesh.get_n_node(neighbour));
		//cell_nodes.push_back(Mesh.get_e_node(neighbour));
		//cell_nodes.push_back(Mesh.get_w_node(neighbour));
		cell_nodes.push_back(Mesh.get_s_node(neighbour));
	}
	else {
		cell_nodes.push_back(i);
		//cell_nodes.push_back(Mesh.get_n_node(i));
		cell_nodes.push_back(Mesh.get_e_node(i));
		cell_nodes.push_back(Mesh.get_w_node(i));
		// cell_nodes.push_back(Mesh.get_s_node(i));
		cell_nodes.push_back(neighbour);
		//cell_nodes.push_back(Mesh.get_n_node(neighbour));
		cell_nodes.push_back(Mesh.get_e_node(neighbour));
		cell_nodes.push_back(Mesh.get_w_node(neighbour));
		//cell_nodes.push_back(Mesh.get_s_node(neighbour));

	}
}



//get CFL numbers for inviscid and viscous matrices
// see what time stepping results
void gpu_solver::populate_cfl_areas(double3 *cfl_areas, unstructured_mesh &Mesh) {

	double area_x, area_y, area_z;
	int face;
	double3 temp;
	for (int i = 0; i < Mesh.get_n_cells(); i++) {
		area_x = 0;
		area_y = 0;
		area_z = 0;

		// time step condition as per OpenFoam calcs
		for (int f = 0; f < Mesh.gradient_faces[i].size(); f++) {
			face = Mesh.gradient_faces[i][f];

			// eigen values as per Zhaoli guo(2004) - preconditioning

			//method as per Jiri Blasek: CFD Principles and Application Determination of Max time Step

			// need to calulate correct direction of face vector

			area_x = area_x + fabs(Mesh.get_face_i(face)*Mesh.get_face_area(face));
			area_y = area_y + fabs(Mesh.get_face_j(face)*Mesh.get_face_area(face));
			area_z = area_z + fabs(Mesh.get_face_k(face)*Mesh.get_face_area(face));

		}

		temp.x =  area_x / 2;
		temp.y =  area_y / 2;
		temp.z =  area_z / 2;
		cfl_areas[i] = temp;
	}

	return;
}




//get CFL numbers for inviscid and viscous matrices
// see what time stepping results



void gpu_solver::inverse_weighted_distance_interpolation(double &u, double &v, double &rho, Boundary_Conditions &bcs,
	Mesh &Mesh, domain_geometry &domain, Solution &soln, vector_var &interface_node,
	int k, int i, int neighbour, vector<vector_var> &e_alpha, int j, std::vector<int> &cell_nodes) {

	// get interface node
	double w_u, w_v, w_rho, w_sum, w;  // weighted macros

	// get 8 nodes'
	w_u = 0.0;
	w_v = 0.0;
	w_rho = 0.0;
	w_sum = 0.0;

	double r;
	r = 0.0;
	double dt;
	if (j == 2) {
		dt = Mesh.get_delta_t_e(i);
	}
	else {
		dt = Mesh.get_delta_t_n(i);
	}

	//get displacements
	vector_var node_displacement, target_node;

	// get target node
	target_node.x = interface_node.x - e_alpha[k].x * dt;
	target_node.y = interface_node.y - e_alpha[k].y * dt;
	target_node.z = interface_node.z - e_alpha[k].z * dt;

	for (auto &it : cell_nodes) {
		node_displacement.x = Mesh.get_centroid_x(it) - target_node.x;
		node_displacement.y = Mesh.get_centroid_y(it) - target_node.y;
		node_displacement.z = Mesh.get_centroid_z(it) - target_node.z;

		r = node_displacement.Magnitude();

		//
		if (r < 10e-5) {
			u = soln.get_u(it);
			w = soln.get_v(it);
			rho = soln.get_rho(it);
			return;

		}

		//get weight for this cc
		w = pow(1 / r, 2.0);

		// sum weighted cc values
		w_u = w_u + w * soln.get_u(it);
		w_v = w_v + w * soln.get_v(it);
		w_rho = w_rho + w * soln.get_rho(it);
		w_sum = w_sum + w;

	}

	// calc u v rho for target node
	u = w_u / w_sum;
	v = w_v / w_sum;
	rho = w_rho / w_sum;

}

void gpu_solver::find_real_time(double* delta_t_local, double* local_time, bool* calc_face,
	unstructured_mesh &Mesh, bool* calc_cell) {

	// for each cell check cell calc check if time is greater than neighbouring cells;
	int nb;

	for (int i = 0; i < Mesh.get_total_cells(); i++) {
		// first update actual time
		if (calc_cell[i]) {
			local_time[i] = local_time[i] + delta_t_local[i];
		}
	}


	for (int i = 0; i < Mesh.get_total_cells(); i++) {


		calc_cell[i] = true;
		for (int j = 0; j < Mesh.gradient_cells[i].size(); j++) {
			nb = Mesh.gradient_cells[i][j];
			if (local_time[i] > local_time[nb]) {
				calc_cell[i] = false;
				j = Mesh.gradient_cells[i].size();
			}
		}
	}

	// then for each face calc if it should be calculated
	for (int k = 0; k < Mesh.get_n_faces(); k++) {
		calc_face[k] = false;
		if (calc_cell[Mesh.get_mesh_owner(k)] || calc_cell[Mesh.get_mesh_neighbour(k)]) {
			calc_face[k] = true;
		}
	}




}

void gpu_solver::post_kernel_checks() {

	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}


}


template <typename T>
void gpu_solver::bcs_to_array(T* target, Boundary_Conditions &bcs, int total_nodes, std::string name) {

	for (int i = 0; i < total_nodes; i++) {
		if (name.compare("vel_type") == 0) {
			target[i] = bcs.get_vel_type(i);

		}
		else if (name.compare("rho_type") == 0) {
			target[i] = bcs.get_rho_type(i);

		}
	}
}



template <typename T>
void gpu_solver::bcs_to_array_double(T* target, Boundary_Conditions &bcs, int total_nodes, std::string name) {

	for (int i = 0; i < total_nodes; i++) {
		if (name.compare("bcs") == 0) {

			double4 temp;
			temp.x = bcs.get_u(i);
			temp.y = bcs.get_v(i);
			temp.z = bcs.get_w(i);
			temp.w = bcs.get_rho(i);
			target[i] = temp;

		}

	}
}






template <typename T>
void gpu_solver::mesh_to_array(T* target, unstructured_mesh &mesh, int total_nodes, std::string name) {


	for (int i = 0; i < total_nodes; i++) {
		if (name.compare("volume") == 0) {
			target[i] = mesh.get_cell_volume(i);

		}
		else if (name.compare("gradient_stencil") == 0) {
			for (int j = 0; j < 6; j++) {
				target[i * 6 + j] = mesh.gradient_cells[i][j];
			}

		}
		else if(name.compare("mesh_owner") == 0) {
			target[i] = mesh.get_mesh_owner(i);

		}
		else if (name.compare("mesh_neighbour") == 0) {
			target[i] = mesh.get_mesh_neighbour(i);

		}
		else if (name.compare("surface_area") == 0) {
			target[i] = mesh.get_face_area(i);

		}
		else if (name.compare("streaming_dt") == 0) {
			target[i] = mesh.get_delta_t_face(i);

		}


	}

}



template <typename T>
void gpu_solver::mesh_to_array_double(T* target, unstructured_mesh &mesh, int total_nodes, std::string name)
{
	for (int i = 0; i < total_nodes; i++) {
		if (name.compare("cell_centroid") == 0) {

			double3 temp;
			temp.x = mesh.get_centroid_x(i);
			temp.y = mesh.get_centroid_y(i);
			temp.z = mesh.get_centroid_z(i);

			target[i] = temp;

		}else if (name.compare("face_normal") == 0) {

			double3 temp;
			temp.x = mesh.get_face_i(i);
			temp.y = mesh.get_face_j(i);
			temp.z = mesh.get_face_k(i);

			target[i] = temp;
		}
		else if (name.compare("face_centroid") == 0) {

			double3 temp;
			temp.x = mesh.get_face_x(i);
			temp.y = mesh.get_face_y(i);
			temp.z = mesh.get_face_z(i);

			target[i] = temp;
		}

	}
}

template <typename T>
void gpu_solver::gradients_to_array_double(T* target, gradients &grads, int total_nodes, std::string name)
{
	for (int i = 0; i < total_nodes; i++) {
		if (name.compare("RHS_array") == 0) {

			for (int j = 0; j < 6; j++) {


				double3 temp;
				temp.x = double(grads.RHS_x[i * 6 + j]);
				temp.y = double(grads.RHS_y[i * 6 + j]);
				temp.z = double(grads.RHS_z[i * 6 + j]);

				target[i * 6 + j] = temp;

			}


		}
	}
}




template <typename T>
void gpu_solver::gradients_to_array(T* target, gradients &grads, int total_nodes, std::string name)
{



	for (int i = 0; i < total_nodes; i++) {
		if(name.compare("LHS_xx") == 0) {
			target[i] = double(grads.LHS_xx[i]);
		}
		else if (name.compare("LHS_xy") == 0) {
			target[i] = grads.LHS_xy[i];
		}
		else if (name.compare("LHS_xz") == 0) {
			target[i] = grads.LHS_xz[i];
		}
		else if (name.compare("LHS_yx") == 0) {
			target[i] = grads.LHS_yx[i];
		}
		else if (name.compare("LHS_yy") == 0) {
			target[i] = grads.LHS_yy[i];
		}
		else if (name.compare("LHS_yz") == 0) {
			target[i] = grads.LHS_yz[i];
		}
		else if (name.compare("LHS_zx") == 0) {
			target[i] = grads.LHS_zx[i];
		}
		else if (name.compare("LHS_zy") == 0) {
			target[i] = grads.LHS_zy[i];
		}
		else if (name.compare("LHS_zz") == 0) {
			target[i] = grads.LHS_zz[i];
		}
	}
}






void gpu_solver::soln_to_double(double4* target, Solution &soln_a, int total_nodes) {


	for (int i = 0; i < total_nodes; i++) {
		double4 tmp;
		tmp.w = soln_a.get_rho(i);
		tmp.x = soln_a.get_u(i);
		tmp.y = soln_a.get_v(i);
		tmp.z = soln_a.get_w(i);
		target[i] = tmp;
	}


};


//get CFL numbers for inviscid and viscous matrices
// see what time stepping results
void gpu_solver::get_cfl(double &delta_t, Solution &soln
	, unstructured_mesh &Mesh, global_variables &globals, double* delta_t_local, int* delta_t_frequency, Solution &cfl_areas) {


	double factor;


	double area_x_eigen,   visc_eigen;
	factor = globals.time_marching_step;

	double visc_constant;
	visc_constant = 4;

	double min_delta_t, temp;

	double effective_speed_of_sound;
	//effective_speed_of_sound = 1/sqrt(3);
	effective_speed_of_sound = globals.max_velocity* sqrt(1 - globals.pre_conditioned_gamma + pow(globals.pre_conditioned_gamma / sqrt(3) / globals.max_velocity, 2));
	//loop through cells

	min_delta_t = 100000000000;

	for (int i = 0; i < Mesh.get_n_cells(); i++) {
		delta_t_frequency[i] = 1;

		// eigen values as per Zhaoli guo(2004) - preconditioning

		  //estimation of spectral radii s per Jiri Blasek: CFD Principles and Application Determination of Max time Step

		area_x_eigen = 0;
		area_x_eigen = (fabs(soln.get_u(i)) + effective_speed_of_sound)*cfl_areas.get_u(i)
			+ (fabs(soln.get_v(i)) + effective_speed_of_sound)*cfl_areas.get_v(i)
			+ (fabs(soln.get_w(i)) + effective_speed_of_sound)*cfl_areas.get_w(i);

		area_x_eigen = area_x_eigen / globals.pre_conditioned_gamma;


		//reducing preconditioning increases viscous flux - increases eigenvalue
		visc_eigen = 2 * globals.visc / globals.pre_conditioned_gamma / soln.get_rho(i) / Mesh.get_cell_volume(i);
		visc_eigen = visc_eigen * (cfl_areas.get_u(i)*cfl_areas.get_u(i) + cfl_areas.get_v(i)*cfl_areas.get_v(i) + cfl_areas.get_w(i)* cfl_areas.get_w(i));

		area_x_eigen = area_x_eigen + visc_constant * visc_eigen;

		// use smallest time step allowed
		temp = factor * Mesh.get_cell_volume(i) / area_x_eigen;
		if (temp < 0) {
			min_delta_t = temp;


		}
		if (temp < min_delta_t) {
			min_delta_t = temp;
		}

		if (globals.time_stepping == "local" || globals.time_stepping == "talts") {
			delta_t_local[i] = temp;

		}
		else { //constant user defined time step
			delta_t_local[i] = factor;

		}
	}

	if (globals.time_stepping == "min") {
		std::fill_n(delta_t_local, Mesh.get_n_cells(), min_delta_t);
	}


	if (globals.time_stepping == "talts") {

		for (int i = 0; i < Mesh.get_n_cells(); i++) {
			delta_t_frequency[i] = pow(2, floor(log2(delta_t_local[i] / min_delta_t)));
			delta_t_local[i] = min_delta_t * delta_t_frequency[i];
		}

	}

	return;
}
