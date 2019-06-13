#include "lagrangian_object.h"
#include <cmath>
#include <iostream>
#include <fstream> 


lagrangian_object::lagrangian_object()
{
}


lagrangian_object::~lagrangian_object()
{
	delete[] node_x_ref;
	node_x_ref = NULL;
	delete[] node_y_ref;
	node_y_ref = NULL;
	delete[] node_z_ref;
	node_z_ref = NULL;
	

}

void lagrangian_object::initialise(double PI) {


	
	
	// rigid 3D Cylinder
	if (type == 1) {
		node_x_ref = new double[num_nodes];
		if (node_x_ref == NULL) exit(1);
		node_y_ref = new double[num_nodes];
		if (node_y_ref == NULL) exit(1);
		node_z_ref = new double[num_nodes];
		if (node_z_ref == NULL) exit(1);

		node_x = new double[num_nodes];
		if (node_x == NULL) exit(1);
		node_y = new double[num_nodes];
		if (node_y == NULL) exit(1);
		node_z = new double[num_nodes];
		if (node_z == NULL) exit(1);

		populate_node_reference_displacement(PI);
	}/// spring network mesh
	else if (type == 2) {
		import_network_mesh();

	}


	node_vel_x = new double[num_nodes];
	if (node_vel_x == NULL) exit(1);
	node_vel_y = new double[num_nodes];
	if (node_vel_y == NULL) exit(1);
	node_vel_z = new double[num_nodes];
	if (node_vel_z == NULL) exit(1);

	node_force_x = new double[num_nodes];
	if (node_force_x == NULL) exit(1);
	node_force_y = new double[num_nodes];
	if (node_force_y == NULL) exit(1);
	node_force_z = new double[num_nodes];
	if (node_force_z == NULL) exit(1);
}

void lagrangian_object::import_network_mesh() {
	std::cout << "Constructing a mesh... " << std::endl;

	mesh_read(mesh_file);

	/*get_nod_max_freedom();
	get_nod_min_freedom();
*/
	std::cout << "\tSuccessfully constructed the mesh!\n" << std::endl;

}
void lagrangian_object::populate_node_reference_displacement(double PI) {

	//rigid cylinder in x direction fully 3d
	if( type == 1 ){
		

		depth_nodes = (int)sqrt(num_nodes);
		radial_nodes = depth_nodes;

		double delta_z = depth / depth_nodes;
		int k = 0;

		
		for (int j = 0; j < depth_nodes; j++) {
			for (int i = 0; i < radial_nodes; i++) {
		
				node_x_ref[k] = centre_x + radius * sin(2 * PI * i / radial_nodes);
				node_y_ref[k] = centre_y + radius * cos(2 * PI * i / radial_nodes);
				node_z_ref[k] = centre_z +depth/2 - j* depth/ depth_nodes - delta_z/2;

				node_x[k] = centre_x + radius * sin(2 * PI * i / radial_nodes);
				node_y[k] = centre_y + radius * cos(2 * PI * i / radial_nodes);
				node_z[k] = centre_z + depth / 2 - j * depth / depth_nodes - delta_z / 2;;

				k = k + 1;

			}
		}

		delta_z = 1;

	}


}


void lagrangian_object::mesh_read(std::string inputFileName) {

	std::cout << "\tReading mesh file..." << std::endl;

	//read in the mesh file.
	std::ifstream inputFile;
	inputFile.open(inputFileName.c_str());
	if (inputFile.is_open()) {

		//std::cout << "Successfully opened the NODES input file! " << std::endl;

		//for nodes.
		inputFile >> num_nodes;
		inputFile >> num_springs;
		inputFile >> num_tets;

		//std::cout << nodN << " " << triN << std::endl;

		//update x_y_z nodes
		node_x = new double[num_nodes];
		if (node_x == NULL) exit(1);
		node_y = new double[num_nodes];
		if (node_y == NULL) exit(1);
		node_z = new double[num_nodes];
		if (node_z == NULL) exit(1);
		node_x_ref = new double[num_nodes];
		if (node_x_ref == NULL) exit(1);
		node_y_ref = new double[num_nodes];
		if (node_y_ref == NULL) exit(1);
		node_z_ref = new double[num_nodes];
		if (node_z_ref == NULL) exit(1);

		//import xyz nodes
		for (int i = 0; i < num_nodes ; i++) {

			inputFile >> node_x[i];
			inputFile >> node_y[i];
			inputFile >> node_z[i];
			node_x_ref[i] = node_x[i];
			node_y_ref[i] = node_y[i];
			node_z_ref[i] = node_z[i];
		}

		//get tet connectivity

		tet_connectivity = new int[num_tets * 3]; {
			for (int i = 0; i < num_tets * 3; i++) {
				int k;
				inputFile >> k;
				tet_connectivity[i] = k - 1;

			}
		}

		//spring connectivity
		spring_connectivity = new int[num_springs * 4]; {
			for (int i = 0; i < num_springs * 4; i++) {
				int k;
				inputFile >> k;
				spring_connectivity[i] = k - 1;
			}
		}

		num_node_neighbours = new int[num_nodes];
		
		//use vectors to collate Node connectivity as it's unstructured
		for (int i = 0; i < num_nodes; i++) {
			node_neighbours.push_back(std::vector <int>());
		}

		for (int i = 0; i < num_nodes; i++) {

			int fdm;
			inputFile >> fdm;
			num_node_neighbours[i] = fdm;

			//nod_nodIdx[i] = new int[fdm + 1];
			for (int j = 0; j < fdm; j++) {

				int nIdx;
				inputFile >> nIdx;
				node_neighbours[i].push_back(nIdx - 1);
			}
			
		}
		}
	
	else {
		std::cout << "Failed to open the mesh input file. Press ENTER to exit!" << std::endl;
		getchar();
		exit(EXIT_FAILURE);
	}
	inputFile.close();



	return;
}