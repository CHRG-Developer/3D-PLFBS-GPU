#pragma once
#include <string>
#include <vector>

using namespace std;
class lagrangian_object
{
public:
	lagrangian_object();
	~lagrangian_object();

	std::string name,mesh_file;
	int num_nodes, num_springs,num_tets;
	int type; // 1 = rigid_cylinder
	double radius,depth;
	double centre_x, centre_y, centre_z;
	double stiffness;
	double * node_x, * node_y, * node_z;
	double * node_x_ref, * node_y_ref, * node_z_ref;
	double * node_vel_x, * node_vel_y, * node_vel_z;
	double * node_force_x, *node_force_y, *node_force_z;
	int depth_nodes, radial_nodes;

	int * tet_connectivity, *spring_connectivity;
	int * num_node_neighbours;
	std::vector <std::vector <int> > node_neighbours;

	void initialise(double PI);

	void populate_node_reference_displacement(double PI);
	void import_network_mesh();
	void mesh_read(std::string inputFileName);
};

