
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
#include "lagrangian_object.h"
#include "common_kernels.hpp"
#include "LBFS.hpp"
#include "immersed_boundary_method.hpp"

#define IMMERSED_BOUNDARY_METHOD

using namespace std;



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



void gpu_solver::General_Purpose_Solver_mk_i(unstructured_mesh &Mesh, Solution &soln, Boundary_Conditions &bcs,
	external_forces &source, global_variables &globals, domain_geometry &domain,
	initial_conditions &init_conds, unstructured_bcs &quad_bcs_orig, int mg,
	Solution &residual, int fmg, post_processing &pp, std::vector<lagrangian_object> &object_vec)
{

	///Declarations
	RungeKutta rk4;

	Solution residual_worker(Mesh.get_total_cells()); // stores residuals
	Solution vortex_error(Mesh.get_total_cells());
	Solution real_error(Mesh.get_total_cells());
	Solution wall_shear_stress(Mesh.get_n_wall_cells());
	gradients grads(Mesh.get_total_cells());
	Solution cfl_areas(Mesh.get_total_cells());

	/// Declarations and initialisations
	
		flux_var RK;

		double4 *temp_soln, *soln_t0, *soln_t1;
		double *force_x, *force_y, *force_z;

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
		double *res_rho, *res_u, *res_v, *res_w;

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
	
	/// assign memory
		{

			delta_t_local = new double[Mesh.get_n_cells()];
			if (delta_t_local == NULL) exit(1);
			delta_t_frequency = new int[Mesh.get_n_cells()];
			if (delta_t_frequency == NULL) exit(1);



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

			force_x = new double[Mesh.get_n_cells()];
			if (force_x == NULL) exit(1);
			force_y = new double[Mesh.get_n_cells()];
			if (force_y == NULL) exit(1);
			force_z = new double[Mesh.get_n_cells()];
			if (force_z == NULL) exit(1);


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
			mesh_neighbour = new int[Mesh.get_n_faces()];
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

			RHS_arr = new double3[Mesh.get_n_cells() * 6];
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
		}
	//lagrangian object allocations


#if defined IMMERSED_BOUNDARY_METHOD
	// first get total of object nodes for all cells
	//loop through vector
	int total_object_nodes = 0;
	int total_object_springs = 0;
	int total_object_tets = 0;

	for (int i = 0; i < object_vec.size(); i++) {
		total_object_nodes = total_object_nodes + object_vec[i].num_nodes;
		total_object_springs = total_object_springs + object_vec[i].num_springs;
		total_object_tets = total_object_tets + object_vec[i].num_tets;
	}

	double * object_x_ref, *object_y_ref, *object_z_ref;
	double * object_x, *object_y, *object_z;
	double * object_x0, *object_y0, *object_z0;
	double * object_vel_x, *object_vel_y, *object_vel_z;
	double * object_force_x, *object_force_y, *object_force_z;
	int * object_tet_connectivity;

	object_tet_connectivity = new int[total_object_tets *3];
	if (object_tet_connectivity == NULL) exit(1);
	

	object_x_ref = new double[total_object_nodes];
	if (object_x_ref == NULL) exit(1);
	object_y_ref = new double[total_object_nodes];
	if (object_y_ref == NULL) exit(1);
	object_z_ref = new double[total_object_nodes];
	if (object_z_ref == NULL) exit(1);

	object_x = new double[total_object_nodes];
	if (object_x == NULL) exit(1);
	object_y = new double[total_object_nodes];
	if (object_y == NULL) exit(1);
	object_z = new double[total_object_nodes];
	if (object_z == NULL) exit(1);

	object_x0 = new double[total_object_nodes];
	if (object_x0 == NULL) exit(1);
	object_y0 = new double[total_object_nodes];
	if (object_y0 == NULL) exit(1);
	object_z0 = new double[total_object_nodes];
	if (object_z0 == NULL) exit(1);


	object_vel_x = new double[total_object_nodes];
	if (object_vel_x == NULL) exit(1);
	object_vel_y = new double[total_object_nodes];
	if (object_vel_y == NULL) exit(1);
	object_vel_z = new double[total_object_nodes];
	if (object_vel_z == NULL) exit(1);

	object_force_x = new double[total_object_nodes];
	if (object_force_x == NULL) exit(1);
	object_force_y = new double[total_object_nodes];
	if (object_force_y == NULL) exit(1);
	object_force_z = new double[total_object_nodes];
	if (object_force_z == NULL) exit(1);

#endif

	double local_tolerance;
	double3 mesh_lengths, mesh_origin;

	mesh_lengths.x = domain.X;
	mesh_lengths.y = domain.Y;
	mesh_lengths.z = domain.Z;

	mesh_origin.x = domain.origin_x;
	mesh_origin.y = domain.origin_y;
	mesh_origin.z = domain.origin_z;


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

	///Initialisations

	dt = domain.dt; // timestepping for streaming // non-dim equals 1
	c = 1; // assume lattice spacing is equal to streaming timestep
	cs = c / sqrt(3);
	visc = (globals.tau - 0.5) / 3 * domain.dt;


	local_tolerance = globals.tolerance;
	delta_t = globals.time_marching_step;
	timesteps = ceil(globals.simulation_length);
	output_dir = globals.output_file + "/error.txt";
	decay_dir = globals.output_file + "/vortex_error.txt";
	max_u_dir = globals.output_file + "/max_u.txt";
	// error_output.open("/home/brendan/Dropbox/PhD/Test Cases/Couette Flow/error.txt", ios::out);
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

	grads.pre_fill_LHS_and_RHS_matrix(bcs, Mesh, domain, soln, globals);



	populate_cfl_areas(cfl_areas, Mesh);


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
#if defined IMMERSED_BOUNDARY_METHOD
	int n_node_Blocks = (total_object_nodes + blockSize - 1) / blockSize;
#endif
	double delta_x = domain.dt * 2; 
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
	{
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
		checkCudaErrors(cudaMallocManaged(&gradient_stencil, Mesh.get_n_cells() * sizeof(int) * 6));
		checkCudaErrors(cudaMallocManaged(&mesh_owner, Mesh.get_n_faces() * sizeof(int)));
		checkCudaErrors(cudaMallocManaged(&mesh_neighbour, Mesh.get_n_faces() * sizeof(int)));
		checkCudaErrors(cudaMallocManaged(&cell_centroid, Mesh.get_total_cells() * sizeof(double3)));
		checkCudaErrors(cudaMallocManaged(&face_centroid, Mesh.get_n_faces() * sizeof(double3)));
		checkCudaErrors(cudaMallocManaged(&face_normal, Mesh.get_n_faces() * sizeof(double3)));
		checkCudaErrors(cudaMallocManaged(&surface_area, Mesh.get_n_faces() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&streaming_dt, Mesh.get_n_faces() * sizeof(double)));


		checkCudaErrors(cudaMallocManaged(&cell_flux_arr, Mesh.get_n_faces() * sizeof(double4)));

		checkCudaErrors(cudaMallocManaged(&force_x, Mesh.get_total_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&force_y, Mesh.get_total_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&force_z, Mesh.get_total_cells() * sizeof(double)));

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
		checkCudaErrors(cudaMallocManaged(&RHS_arr, Mesh.get_n_cells() * sizeof(double3) * 6));
		checkCudaErrors(cudaMallocManaged(&LHS_xx, Mesh.get_n_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&LHS_xy, Mesh.get_n_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&LHS_xz, Mesh.get_n_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&LHS_yx, Mesh.get_n_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&LHS_yy, Mesh.get_n_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&LHS_yz, Mesh.get_n_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&LHS_zx, Mesh.get_n_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&LHS_zy, Mesh.get_n_cells() * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&LHS_zz, Mesh.get_n_cells() * sizeof(double)));


#if defined IMMERSED_BOUNDARY_METHOD
		//arrays for lagrangian objects

		checkCudaErrors(cudaMallocManaged(&object_tet_connectivity, total_object_tets * 3 * sizeof(int)));

		checkCudaErrors(cudaMallocManaged(&object_x_ref, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_y_ref, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_z_ref, total_object_nodes * sizeof(double)));

		checkCudaErrors(cudaMallocManaged(&object_x, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_y, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_z, total_object_nodes * sizeof(double)));

		checkCudaErrors(cudaMallocManaged(&object_x0, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_y0, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_z0, total_object_nodes * sizeof(double)));

		checkCudaErrors(cudaMallocManaged(&object_vel_x, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_vel_y, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_vel_z, total_object_nodes * sizeof(double)));

		checkCudaErrors(cudaMallocManaged(&object_force_x, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_force_y, total_object_nodes * sizeof(double)));
		checkCudaErrors(cudaMallocManaged(&object_force_z, total_object_nodes * sizeof(double)));
#endif

	}
	

	populate_e_alpha(e_alpha, h_lattice_weight, c, globals.PI, 15);
	checkCudaErrors(cudaMemcpyToSymbol(lattice_weight, h_lattice_weight, 15 * sizeof(double)));

	/// Sync before CUDA array used
	cudaDeviceSynchronize();
	populate_cfl_areas(d_cfl_areas, Mesh);


	// transfer class members to arrays for CUDA
	{
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
	}

#if defined IMMERSED_BOUNDARY_METHOD
	lagrangian_object_to_array(object_vec, object_x_ref, object_y_ref, object_z_ref, object_x, object_y, object_z, object_x0, object_y0, object_z0, object_tet_connectivity);
#endif

	cudaProfilerStart();

	clone_a_to_b << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), temp_soln, soln_t1); // soln_t0 holds macro variable solution at start of time step

	// loop in time
	for (int t = 0; t < timesteps; t++) {
		// soln_t0 is the solution at the start of every
		// RK step.(rk = n) Temp_soln holds the values at end of
		// step.(rk = n+1)
		clone_a_to_b << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), soln_t1, soln_t0);// soln_t0 holds macro variable solution at start of time step
		post_kernel_checks();


		//womersley flow peculiarities
		if (globals.testcase == 4) {
			wom_cos = cos(angular_freq * t * delta_t);
			force = -init_conds.pressure_gradient * wom_cos;
		}

		//local timestepping calculation
		// can be removed for uniform grids and replaced with a single calc
		get_cfl_device <<< n_Cell_Blocks, blockSize >>> (Mesh.get_n_cells(),  temp_soln, cell_volume, d_delta_t_local,  d_cfl_areas, globals.time_marching_step,
			globals.max_velocity,globals.pre_conditioned_gamma, globals.visc, globals.gpu_time_stepping);
		post_kernel_checks();

		#if defined IMMERSED_BOUNDARY_METHOD

				//	need to propogate node position based on final RK4 velocity
				interpolate_velocities_on_nodes << < n_node_Blocks, blockSize >> > (total_object_nodes, object_vel_x, object_vel_y, object_vel_z, object_x, object_y, object_z,
					mesh_origin, mesh_lengths, delta_x, temp_soln, Mesh.get_n_cells());

		/*		interpolate_velocities_on_nodes_cos_kernel << < n_node_Blocks, blockSize >> > (total_object_nodes, object_vel_x, object_vel_y, object_vel_z, object_x, object_y, object_z,
					mesh_origin, mesh_lengths, delta_x, temp_soln, globals.PI);*/

				update_node_positions_rk4 << < n_node_Blocks, blockSize >> > (total_object_nodes, object_vel_x, object_vel_y, object_vel_z, object_x, object_y, object_z, delta_t, object_vec[0].num_nodes,
					object_x0, object_y0, object_z0);

					/*fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), force_x);*/
				fill_double << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), force_x, init_conds.pressure_gradient);
				post_kernel_checks();
				fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), force_y);
				post_kernel_checks();
				fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), force_z);
				post_kernel_checks();

				//for now assume uniform stiffness, radius etc. 
				update_node_forces << < n_node_Blocks, blockSize >> > (total_object_nodes, object_force_x, object_force_y, object_force_z, object_x, object_y, object_z, object_x_ref, object_y_ref, object_z_ref,
					object_vec[0].stiffness, object_vec[0].radius, globals.PI, object_vec[0].num_nodes, object_vel_x, object_vel_y, object_vel_z, delta_t, object_vec[0].depth);

				//assume uniform grid for now, need moving least squares stencil in the future
				spread_forces_on_structured_grid << < n_node_Blocks, blockSize >> > (total_object_nodes, object_force_x, object_force_y, object_force_z, object_x, object_y, object_z,
					mesh_origin, mesh_lengths, delta_x, force_x, force_y, force_z, Mesh.get_n_cells());

				/*spread_forces_on_structured_grid_cos_kernel << < n_node_Blocks, blockSize >> > (total_object_nodes, object_force_x, object_force_y, object_force_z, object_x, object_y, object_z,
					mesh_origin, mesh_lengths, delta_x, force_x, force_y, force_z, Mesh.get_n_cells(), globals.PI);*/

		#endif
		
		for (int rk = 0; rk < rk4.timesteps; rk++) {
									
			drag_t1 = 0.0;
	
			//update temp_soln boundary conditions
			update_unstructured_bcs << < n_bc_Blocks, blockSize >> > (Mesh.get_num_bc(), Mesh.get_n_neighbours(), Mesh.get_n_cells(), mesh_owner, bcs_rho_type, bcs_vel_type, temp_soln, bcs_arr, cell_centroid,domain.Y);

			//set to zeros
			fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_rho);
			post_kernel_checks();
			fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_u);
			post_kernel_checks();
			fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_v);
			post_kernel_checks();
			fill_zero << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_w);
			post_kernel_checks();


		   // time2 = clock();
			get_interior_gradients <<< n_Cell_Blocks, blockSize >>> ( Mesh.get_n_cells(), gradient_stencil, temp_soln,
				RHS_arr,LHS_xx, LHS_xy, LHS_xz,LHS_yx, LHS_yy, LHS_yz,LHS_zx, LHS_zy, LHS_zz,
				grad_rho_arr, grad_u_arr, grad_v_arr, grad_w_arr);
			post_kernel_checks();


			//get boundary condition gradients
			get_bc_gradients << < n_bc_Blocks, blockSize >> > (Mesh.get_num_bc(), Mesh.get_n_neighbours(), Mesh.get_n_cells(), mesh_owner, bcs_rho_type, bcs_vel_type, temp_soln,
				face_normal, cell_centroid, bcs_arr,
				grad_rho_arr, grad_u_arr, grad_v_arr, grad_w_arr);
			post_kernel_checks();

			//time3 = clock();

			//std::cout << "CPU Cycles Gradients:" << double(time3 - time2) << std::endl;
			wall = 0;
			// loop through each cell and exclude the ghost cells
			//using n_cells here rather than total_cells
		
			cudaDeviceSynchronize();
			calc_face_flux << < n_face_Blocks, blockSize >> > (Mesh.get_n_faces(), temp_soln, cell_volume, surface_area,  mesh_owner,  mesh_neighbour,  cell_centroid, face_centroid,  face_normal,
				streaming_dt, grad_rho_arr, grad_u_arr,  grad_v_arr, grad_w_arr, Mesh.get_n_cells(),  (1/ globals.pre_conditioned_gamma),  local_fneq, globals.visc,
				res_rho, res_u,res_v,res_w,res_face,
				bcs_rho_type, bcs_vel_type, bcs_arr,globals.PI);
			post_kernel_checks();
			cudaDeviceSynchronize();

			add << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_u, force_x);
			add << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_v, force_y);
			add << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), res_w, force_z);
			post_kernel_checks();
			cudaDeviceSynchronize();

/*
			for (int i = 0; i < Mesh.get_n_cells(); i++) {

				debug_log << t << ", " << rk << ", " << i << ", " << res_rho[i] << ", " <<
					res_u[i] << ", " << res_v[i] << ", " << res_w[i]
					<< ", " <<
					Mesh.get_centroid_x(i) << " , " << Mesh.get_centroid_y(i) << "," << Mesh.get_centroid_z(i) << "," <<
					delta_t_local[i] << " , " << local_fneq[i] << "," <<
					soln.get_rho(i) << "," << soln.get_u(i) << " , " << soln.get_v(i) << " , " <<
					grad_u_arr[i].x << " , " << grad_u_arr[i].y << " , " << grad_u_arr[i].z << " , " <<
					grad_v_arr[i].x << " , " << grad_v_arr[i].y << " , " << grad_v_arr[i].z << " , " <<
					grad_w_arr[i].x << " , " << grad_w_arr[i].y << "," << grad_w_arr[i].z

					<< endl;

			}*/


			//Update  solutions  //update RK values

			time_integration << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), rk, rk4.timesteps, d_delta_t_local, soln_t0, soln_t1, temp_soln,
				res_rho, res_u, res_v, res_w);

			post_kernel_checks();


			#if defined IMMERSED_BOUNDARY_METHOD

						////			//for now assume uniform stiffness, radius etc. 
						//interpolate_velocities_on_nodes << < n_node_Blocks, blockSize >> > (total_object_nodes, object_vel_x, object_vel_y, object_vel_z, object_x, object_y, object_z,
						//	mesh_origin, mesh_lengths, delta_x, temp_soln);
						///*interpolate_velocities_on_nodes_cos_kernel << < n_node_Blocks, blockSize >> > (total_object_nodes, object_vel_x, object_vel_y, object_vel_z, object_x, object_y, object_z,
						//	mesh_origin, mesh_lengths, delta_x, temp_soln,globals.PI);*/


						//update_node_positions << < n_node_Blocks, blockSize >> > (total_object_nodes, object_vel_x, object_vel_y, object_vel_z, object_x, object_y, object_z, delta_t, object_vec[0].num_nodes,rk);


			#endif


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
		convergence_residual.l2_norm_rms_moukallad(globals, res_rho_block, res_u_block, res_v_block, res_w_block, n_Cell_Blocks, Mesh.get_n_cells());

/*
		calc_total_residual << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), convergence, res_rho, res_u, res_v, res_w);
		post_kernel_checks();*/
		/*check_error << < n_Cell_Blocks, blockSize >> > (Mesh.get_n_cells(), temp_soln);
		post_kernel_checks();*/
		//convergence_residual.ansys_5_iter_rms(t);

		time = t * delta_t;

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
				tecplot.tecplot_output_unstructured_soln(globals, Mesh, soln, bcs, t, pp, residual_worker, delta_t_local, local_fneq);
				tecplot.tecplot_output_lagrangian_object_gpu(object_vel_x, object_vel_y, object_vel_z, object_x, object_y, object_z, object_force_x, object_force_y, object_force_z, globals, domain, t
					,object_vec[0].name, object_vec[0].num_nodes, object_vec[0].depth_nodes, object_vec[0].radial_nodes);
				output_residual_threshold = output_residual_threshold - 1;
				soln.output(globals.output_file, globals, domain);
				cudaProfilerStop();
			}
			//soln.output_centrelines(globals.output_file,globals,Mesh,time);

		}

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
				// pp.calc_vorticity(x_gradients,y_gradients);
				  //pp.calc_streamfunction(Mesh,globals,bcs);
				tecplot.tecplot_output_unstructured_soln(globals, Mesh, soln, bcs, t, pp, residual_worker, delta_t_local, local_fneq);
				tecplot.tecplot_output_lagrangian_object_gpu(object_vel_x, object_vel_y, object_vel_z, object_x, object_y, object_z, object_force_x, object_force_y, object_force_z, globals, domain, timesteps
					, object_vec[0].name, object_vec[0].num_nodes, object_vec[0].depth_nodes, object_vec[0].radial_nodes);
				//soln.output_centrelines(globals.output_file,globals,Mesh,time);
			}
			cudaProfilerStop();

			return;
		}


	}


	//    pp.calc_vorticity(x_gradients,y_gradients);
		//pp.calc_streamfunction(Mesh,globals,bcs);
	cudaProfilerStop();
	soln.clone(temp_soln);
	cout << "out of time" << endl;
	error_output.close();
	vortex_output.close();
	debug_log.close();
	max_u.close();
	pp.cylinder_post_processing(Mesh, globals, grads, bcs, soln, domain, wall_shear_stress);
	tecplot.tecplot_output_unstructured_soln(globals, Mesh, soln, bcs, timesteps, pp, residual_worker, delta_t_local, local_fneq);
	tecplot.tecplot_output_lagrangian_object_gpu(object_vel_x, object_vel_y, object_vel_z, object_x, object_y, object_z, object_force_x, object_force_y, object_force_z, globals, domain, timesteps
		, object_vec[0].name, object_vec[0].num_nodes, object_vec[0].depth_nodes, object_vec[0].radial_nodes);



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

void gpu_solver::lagrangian_object_to_array( std::vector<lagrangian_object> &obj_vec, double* &x_ref, double* &y_ref, double* &z_ref, double* &x, double* &y, double* & z, 
	double* &x0, double* &y0, double* & z0, int * & tet_connectivity) {

	int n = 0;
	for (int i = 0; i < obj_vec.size(); i++) {
		for (int j = 0; j < obj_vec[i].num_nodes; j++) {
			
			x_ref[n] = obj_vec[i].node_x_ref[j];
			x[n] = obj_vec[i].node_x[j];
			x0[n] = obj_vec[i].node_x[j];

			y_ref[n] = obj_vec[i].node_y_ref[j];
			y[n] = obj_vec[i].node_y[j];
			y0[n] = obj_vec[i].node_y[j];

			z_ref[n] = obj_vec[i].node_z_ref[j];
			z[n] = obj_vec[i].node_z[j];
			z0[n] = obj_vec[i].node_z[j];

			n++;
		}

		// transfer tet related variables
		for (int k = 0; k < obj_vec[i].num_tets*3; k++) {

			tet_connectivity[k] = obj_vec[i].tet_connectivity[k];

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
