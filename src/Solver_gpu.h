#include "Mesh.h"
#include "Boundary_Conditions.h"
#include "Solution.h"
#include "vector_var.h"
#include "external_forces.h"
#include "global_variables.h"
#include "initial_conditions.h"
#include "flux_var.h"
#include "vector"
#include "post_processing.h"
#include "unstructured_bcs.h"
#include "gradients.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "lagrangian_object.h"

#ifndef gpu_solver_H
#define gpu_solver_H

using namespace std;

class gpu_solver
{
public:
	gpu_solver();
	virtual ~gpu_solver();

	void General_Purpose_Solver_mk_i(unstructured_mesh &Mesh, Solution &soln, Boundary_Conditions &bc,
		external_forces &source, global_variables &globals, domain_geometry &domain,
		initial_conditions &init_conds, unstructured_bcs &quad_bcs_orig, int mg,
		Solution &residual, int fmg, post_processing &pp, std::vector<lagrangian_object> &object_vec);



	void populate_e_alpha(std::vector<vector_var> &e_alpha, double * lattice_weight, double c, double PI, int k);
	void inverse_weighted_distance_interpolation(double &u, double &v, double &rho, Boundary_Conditions &bcs,
		Mesh &Mesh, domain_geometry &domain, Solution &soln
		, vector_var &cell_interface, int k, int i, int neighbour,
		std::vector<vector_var> &e_alpha, int j, std::vector<int> &cell_nodes);
	void get_cell_nodes(std::vector<int> &cell_nodes, Boundary_Conditions &bcs, int neighbour,
		Mesh &Mesh, int i, int j);
	
	

	void find_real_time(double* delta_t_local, double* local_time, bool* calc_face,
		unstructured_mesh &Mesh, bool* calc_cell);

	// initialise variables
protected:
private:

	double dt;
	double tau;
	double kine_viscosity;
	double c, cs;

	//        struct vector_var {
	//            double x;
	//            double y;
	//            double z;
	//        };
	vector_var get_e_alpha(int k, double &lattice_weight, double c, double PI);

	void get_cell_gradients(Mesh &Mesh, int i, int neighbour, int j, Solution &temp_soln,
		vector_var &delta_rho, vector_var &delta_rho1,
		vector_var &delta_u, vector_var &delta1_u1,
		vector_var &delta_v, vector_var &delta_v1,
		Boundary_Conditions &bcs);


	struct bc_var {

		bool present;
		double rho;
		double u;
		double v;
		int vel_type;
		int rho_type;
		int periodic_node;

	};



	void truncate_flux(flux_var &flux);
	void truncate_flux(double &val);

	void get_gradients_weighted_average(gradients &grads, int i, int neighbour, double m1,
		double m2, vector_var &u, vector_var &v, vector_var &w, vector_var &rho);

	void cell_interface_initialiser(double &rho_interface, vector_var &rho_u_interface,
		flux_var &x_flux, flux_var &y_flux);

	double feq_calc(double weight, vector_var e_alpha, vector_var u_lattice, double u_magnitude,
		double cs, double rho_lattice);

	double feq_calc_incomp(double weight, vector_var e_alpha, vector_var u_lattice, double u_magnitude,
		double cs, double rho_lattice, double rho_0, int k);

	void cell_interface_variables(int j, int i, vector_var &interface_node, int &neighbour, double &interface_area,
		vector_var &cell_normal, Boundary_Conditions &boundary_conditions, bc_var &bc,
		Mesh &Mesh, vector_var &cell_2);
	

	void get_weighted_average(gradients &grads, int i, int neighbour, double m1, double m2,
		vector_var &u, vector_var &v, vector_var &w, vector_var &rho, unstructured_mesh &mesh);
	void get_cfl(double &delta_t, Solution &soln, unstructured_mesh &Mesh, global_variables &globals,
		double* delta_t_local, int* delta_t_frequency, Solution &cfl_areas);
	
	void populate_cfl_areas(Solution &cfl_areas, unstructured_mesh &Mesh);

	void cell_interface_variables(int face, int i, vector_var &interface_node, int &neighbour, double &interface_area,
		vector_var &cell_normal, Boundary_Conditions &boundary_conditions, bc_var &bc,
		unstructured_mesh &Mesh, vector_var &cell_2, vector_var &cell_1);
	void populate_cfl_areas(double3 *cfl_areas, unstructured_mesh &Mesh);
	void soln_to_double(double4* target, Solution &soln, int total_nodes);
	
	template <typename T>
	void mesh_to_array(T* target, unstructured_mesh &mesh, int total_nodes, std::string name);
	template <typename T>
	void mesh_to_array_double(T* target, unstructured_mesh &mesh, int total_nodes, std::string name);

	void post_kernel_checks();


	template <typename T>
	void gradients_to_array(T* target, gradients &grads, int total_nodes, std::string name);
	template <typename T>
	void gradients_to_array_double(T* target, gradients &grads, int total_nodes, std::string name);

	template <typename T>
	void bcs_to_array(T* target, Boundary_Conditions &bcs, int total_nodes, std::string name);
	template <typename T>
	void bcs_to_array_double(T* target, Boundary_Conditions &bcs, int total_nodes, std::string name);


	void lagrangian_object_to_array(std::vector<lagrangian_object> &obj_vec,  double* &x_ref, double* &y_ref, double* &z_ref, double* &x, double* &y, double* & z, double* &x0, double* &y0, double* & z0, int * & tet_connectivity);
	
};

//__global__ void get_cfl(int n, double &delta_t, Solution &soln, unstructured_mesh &Mesh, global_variables &globals,
//	double* delta_t_local, int* delta_t_frequency, Solution &cfl_areas);


#endif // SOLVER_H

