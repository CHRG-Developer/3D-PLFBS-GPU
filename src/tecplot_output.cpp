#include "tecplot_output.h"
#include "TECIO.h"
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include "post_processing.h"

template<typename T>
tecplot_output<T>::tecplot_output(){

}
template<typename T>
tecplot_output<T>::tecplot_output(global_variables &globals, Mesh &Mesh, Solution &Soln,
                             Boundary_Conditions &bcs, int fileType_e, double timestamp,
                             post_processing &pp)
{
    //ctor


    std::string output_location;
    std::string reynolds_text;
    reynolds_text = globals.reynolds_number;
    std::string zone_name;
    std::stringstream ss;
    std::stringstream tt;
    int *valueLocation = nullptr;
    INTEGER4 strandID;
    tt << timestamp;

    if( fileType_e == 1){
        output_location = globals.output_file + "/plt/grid.szplt";
        valueLocation = new int[3];
        for(int i = 0; i < 3;i++){
               valueLocation[i] = 1;
        }
        strandID  = 0;   /* StaticZone */

    }else{
        output_location = globals.output_file + "/plt/" + tt.str() +".szplt";
        valueLocation = new int[8];
        for(int i = 0; i < 8;i++){
               valueLocation[i] = 0;
        }
        strandID  = 1;
    }



   //enum fileType_e { FULL = 0, GRID = 1, SOLUTION = 2 };

     float *nodx, *nody, *z, *p,*u, *v,*y ,*x, *u_err, *u_exact ,*w, *vort, *st;
    int *connectivity;
    double solTime;
    INTEGER4 debug, i, j, k, dIsDouble, vIsDouble, zoneType,  parentZn, isBlock;
    INTEGER4 iCellMax, jCellMax, kCellMax, nFConns, fNMode, shrConn, fileType;


    INTEGER4 fileFormat; // 0 == PLT, 1 == SZPLT
    fileFormat = 1;

    INTEGER4 nNodes, nCells, nFaces, connectivityCount, index;
    int XDIM, YDIM, ZDIM; // nodes
    int t,r;
    int n_ghost;  // number of host cells;

    n_ghost = 1;

    if(globals.testcase == 3){
        n_ghost = 2;
    }

    XDIM = Mesh.get_num_x() +1 -2*n_ghost;
    YDIM = Mesh.get_num_y() +1 -2*n_ghost;
    ZDIM = 2;

    debug     = 1;
    vIsDouble = 0;
    dIsDouble = 0;
    nNodes = XDIM * YDIM * ZDIM;
    nCells = (XDIM - 1) * (YDIM - 1) * (ZDIM - 1);
    nFaces = 6; /* Not used */
    zoneType  = 5;      /* Brick */
    solTime   = timestamp;

    parentZn  = 0;      /* No Parent */
    isBlock   = 1;      /* Block */
    iCellMax  = 0;
    jCellMax  = 0;
    kCellMax  = 0;
    nFConns   = 0;
    fNMode    = 0;
    shrConn   = 0;
    fileType  = fileType_e;
    ss << nCells;
    zone_name = ss.str();
    /*
     * Open the file and write the tecplot datafile
     * header information
     */
     if( fileType_e == 1){
        i = TECINI142((char*) "Couette Flow" ,
                  (char*)"nodx nody z",
                  (char*) output_location.c_str(),
                  (char*) ".",
                  &fileFormat,
                  &fileType,
                  &debug,
                  &vIsDouble);


     }else{
     i = TECINI142((char*) "Couette Flow" ,
                  (char*)"p u v w x y vort stf",
                  (char*) output_location.c_str(),
                  (char*) ".",
                  &fileFormat,
                  &fileType,
                  &debug,
                  &vIsDouble);

     }

    i = TECAUXSTR142("Re" ,  reynolds_text.c_str());

     if( fileType_e == 1){
        nodx  = (float*)calloc(nNodes , sizeof(float));
        nody  = (float*)calloc(nNodes , sizeof(float));
        z  = (float*)calloc(nNodes , sizeof(float));
        t = 0;
        r=0;
        for (k = 0; k < ZDIM; k++){
            r=0;
            for (j = 0; j < Mesh.get_num_y() ; j++){
                for (i = 0; i < Mesh.get_num_x() ; i++)
                {
                    if( i != 0 && j != 0 && i != (n_ghost-1) && j != (n_ghost-1)
                    && i != (Mesh.get_num_x() - (n_ghost-1))
                     && j != (Mesh.get_num_y() - (n_ghost-1))){
                        nodx[t] = (float)(Mesh.get_west_x(r));
                        nody[t] = (float)(Mesh.get_south_y(r));
                        z[t] = (float)(k + 1);
                        t++;
                   }
                    r++;

                }
            }
        }

        connectivityCount = 8 * nCells;
        connectivity = (INTEGER4*)malloc(connectivityCount * sizeof(INTEGER4));
        for (k = 0; k < ZDIM - 1; k++)
            for (j = 0; j < YDIM - 1; j++)
                for (i = 0; i < XDIM - 1; i++)
                {
                    index = ((k * (YDIM - 1) + j) * (XDIM - 1) + i) * 8;
                    connectivity[index] = (k * YDIM + j) * XDIM + i +1;
                    connectivity[index + 1] = connectivity[index] + 1;
                    connectivity[index + 2] = connectivity[index] + XDIM + 1;
                    connectivity[index + 3] = connectivity[index] + XDIM;
                    connectivity[index + 4] = connectivity[index] + XDIM * YDIM;
                    connectivity[index + 5] = connectivity[index + 1] + XDIM * YDIM;
                    connectivity[index + 6] = connectivity[index + 2] + XDIM * YDIM;
                    connectivity[index + 7] = connectivity[index + 3] + XDIM * YDIM;
                }

     }
     else{
         p  = (float*)calloc(nCells , sizeof(float));
        u = (float*)calloc(nCells , sizeof(float));
        v = (float*)calloc(nCells , sizeof(float));
        w = (float*)calloc(nCells , sizeof(float));
        y = (float*)calloc(nCells , sizeof(float));
        x = (float*)calloc(nCells , sizeof(float));
        vort = (float*)calloc(nCells , sizeof(float));
        st = (float*)calloc(nCells , sizeof(float));
        t=0;
        for (i = 0; i < Mesh.get_total_cells(); ++i){
        if( bcs.get_bc(i) == false){
            p[t] = (float) Soln.get_rho(i);
            u[t] = (float) Soln.get_u(i)/globals.max_velocity;
            v[t] = (float) Soln.get_v(i)/globals.max_velocity;
            w[t] = (float) 0.0;
            x[t] = (float) Mesh.get_centroid_x(i)/Mesh.get_X();
            y[t] = (float) Mesh.get_centroid_y(i)/Mesh.get_Y();
            vort[t] = (float) pp.vorticity[i];
            st[t] = (float) pp.streamfunction[i];
            t++;
        }

    }

     }



    t = 0;



       /*
     * Write the zone header information.
     */
     std::cout << zone_name.c_str() << std::endl;
   i = TECZNE142(
                 //(char*) zone_name.c_str(),
                 zone_name.c_str(),
                  &zoneType,
                  &nNodes,
                  &nCells,
                  &nFaces,
                  &iCellMax,
                  &jCellMax,
                  &kCellMax,
                  &solTime,
                  &strandID,
                  &parentZn,
                  &isBlock,
                  &nFConns,
                  &fNMode,
                  0,              /* TotalNumFaceNodes */
                  0,              /* NumConnectedBoundaryFaces */
                  0,              /* TotalNumBoundaryConnections */
                  NULL,           /* PassiveVarList */
                  valueLocation,  /* ValueLocation = Nodal */
                  NULL,           /* SharVarFromZone */
                  &shrConn);
/*
     * Write out the field data.
     */
   
   if( fileType_e == 1){
        i = TECDAT142(&nNodes, nodx, &dIsDouble);
        i = TECDAT142(&nNodes, nody, &dIsDouble);
        i = TECDAT142(&nNodes, z, &dIsDouble);
       i = TECNODE142(&connectivityCount, connectivity);
        free(connectivity);
        free(nodx);
        free(nody);
        free(z);


   }else{
        i = TECDAT142(&nCells, p, &dIsDouble);
        i = TECDAT142(&nCells, u, &dIsDouble);
        i = TECDAT142(&nCells, v, &dIsDouble);
        i = TECDAT142(&nCells, w, &dIsDouble);
        i = TECDAT142(&nCells, x, &dIsDouble);
        i = TECDAT142(&nCells, y, &dIsDouble);
        i = TECDAT142(&nCells, vort, &dIsDouble);
        i = TECDAT142(&nCells, st, &dIsDouble);

        free(p);
        free(u);
        free(v);
        free(x);
        free(y);
        free(w);
        free(vort);
        free(st);

   }
    delete [] valueLocation;
    valueLocation= NULL;

    i = TECEND142();
}

template<typename T>
void tecplot_output<T>::tecplot_output_unstructured_soln(global_variables &globals, unstructured_mesh &Mesh, Solution &Soln,
                             Boundary_Conditions &bcs,  double timestamp,
                             post_processing &pp, Solution &residual, double *local_delta_t, T *local_fneq)
{
    //ctor

    int fileType_e = 2;
    std::string output_location;
    std::string reynolds_text;
    reynolds_text = globals.reynolds_number;
    std::string zone_name;
    std::stringstream ss;
    std::stringstream tt;
    int *valueLocation = nullptr;
    INTEGER4 strandID;
    tt << timestamp;

    output_location = globals.output_file + "/plt/" + tt.str() +".plt";
    valueLocation = new int[16];
    for(int i = 0; i < 16;i++){
           valueLocation[i] = 0;
    }
    strandID  = 1;




   //enum fileType_e { FULL = 0, GRID = 1, SOLUTION = 2 };

     float *nodx, *nody, *z, *p,*u, *v,*y ,*x, *u_err, *u_exact ,*w, *vort, *st, *res_rho, *res_u,*res_v, *res_w, *dt, *fneq;
    int *connectivity;
    double solTime;
    INTEGER4 debug, i, j, k, dIsDouble, vIsDouble, zoneType,  parentZn, isBlock;
    INTEGER4 iCellMax, jCellMax, kCellMax, nFConns, fNMode, shrConn, fileType;


    INTEGER4 fileFormat; // 0 == PLT, 1 == SZPLT
    fileFormat = 0;

    INTEGER4 nNodes, nCells, nFaces, connectivityCount, index;

    int t,r;


    debug     = 1;
    vIsDouble = 0;
    dIsDouble = 0;
    nNodes = Mesh.get_n_vertices();
    nCells = Mesh.get_n_cells();
    nFaces = Mesh.get_n_faces(); /* Not used */
    zoneType  = 7;      /* polyhedral */
    solTime   = timestamp;

    parentZn  = 0;      /* No Parent */
    isBlock   = 1;      /* Block */
    iCellMax  = 0;
    jCellMax  = 0;
    kCellMax  = 0;
    nFConns   = 0;
    fNMode    = 0;
    shrConn   = 0;
    fileType  = fileType_e;
    ss << nCells;
    zone_name = ss.str();

    INTEGER4 NumFaceNodes = nFaces*4;
    INTEGER4 NumBConns  = 0;    /* No Boundary Connections */
    INTEGER4 NumBItems = 0;     /* No Boundary Items */

    /*
     * Open the file and write the tecplot datafile
     * header information
     */

     i = TECINI142((char*) "LBFS" ,
                  (char*)"p u v w x y z vort stf res_rho res_u res_v res_w dt fneq",
                  (char*) output_location.c_str(),
                  (char*) ".",
                  &fileFormat,
                  &fileType,
                  &debug,
                  &vIsDouble);


    i = TECAUXSTR142("Re" ,  reynolds_text.c_str());

        p  = (float*)calloc(nCells , sizeof(float));
        u = (float*)calloc(nCells , sizeof(float));
        v = (float*)calloc(nCells , sizeof(float));
        w = (float*)calloc(nCells , sizeof(float));
        y = (float*)calloc(nCells , sizeof(float));
        x = (float*)calloc(nCells , sizeof(float));
        z = (float*)calloc(nCells , sizeof(float));
        vort = (float*)calloc(nCells , sizeof(float));
        st = (float*)calloc(nCells , sizeof(float));
		res_rho = (float*)calloc(nCells, sizeof(float));
		res_u = (float*)calloc(nCells, sizeof(float));
		res_v = (float*)calloc(nCells, sizeof(float));
		res_w = (float*)calloc(nCells, sizeof(float));
		dt = (float*)calloc(nCells, sizeof(float));
		fneq = (float*)calloc(nCells, sizeof(float));

        t=0;
        for (i = 0; i < Mesh.get_n_cells(); ++i){

            p[t] = (float) Soln.get_rho(i);
            u[t] = (float) Soln.get_u(i)/globals.max_velocity;
            v[t] = (float) Soln.get_v(i)/globals.max_velocity;
            w[t] = (float) Soln.get_w(i)/globals.max_velocity;
            x[t] = (float) Mesh.get_centroid_x(i)/Mesh.get_X();
            y[t] = (float) Mesh.get_centroid_y(i)/Mesh.get_Y();
            z[t] = (float) Mesh.get_centroid_z(i)/Mesh.get_Z();
            vort[t] = (float) pp.vorticity[i];
            st[t] = (float) pp.streamfunction[i];
			res_rho[t] = (float) residual.get_rho(i);
			res_u[t] = (float) residual.get_u(i);
			res_v[t] = (float) residual.get_v(i);
			res_w[t] = (float) residual.get_w(i);
			dt[t] = (float)local_delta_t[i];
			fneq[t] = (float)local_fneq[i];
			t++;
    }

    t = 0;

       /*
     * Write the zone header information.
     */
     std::cout << zone_name.c_str() << std::endl;
   i = TECZNE142(
                 //(char*) zone_name.c_str(),
                 zone_name.c_str(),
                  &zoneType,
                  &nNodes,
                  &nCells,
                  &nFaces,
                  &iCellMax,
                  &jCellMax,
                  &kCellMax,
                  &solTime,
                  &strandID,
                  &parentZn,
                  &isBlock,
                  &nFConns,
                  &fNMode,
                  &NumFaceNodes,              /* TotalNumFaceNodes */
                  &NumBConns,              /* NumConnectedBoundaryFaces */
                  &NumBItems,              /* TotalNumBoundaryConnections */
                  NULL,           /* PassiveVarList */
                  valueLocation,  /* ValueLocation = Nodal */
                  NULL,           /* SharVarFromZone */
                  &shrConn);
/*
     * Write out the field data.
     */

    i = TECDAT142(&nCells, p, &dIsDouble);
    i = TECDAT142(&nCells, u, &dIsDouble);
    i = TECDAT142(&nCells, v, &dIsDouble);
    i = TECDAT142(&nCells, w, &dIsDouble);
    i = TECDAT142(&nCells, x, &dIsDouble);
    i = TECDAT142(&nCells, y, &dIsDouble);
    i = TECDAT142(&nCells, z, &dIsDouble);
    i = TECDAT142(&nCells, vort, &dIsDouble);
    i = TECDAT142(&nCells, st, &dIsDouble);
	i = TECDAT142(&nCells, res_rho, &dIsDouble);
	i = TECDAT142(&nCells, res_u, &dIsDouble);
	i = TECDAT142(&nCells, res_v, &dIsDouble);
	i = TECDAT142(&nCells, res_w, &dIsDouble);
	i = TECDAT142(&nCells, dt, &dIsDouble);
	i = TECDAT142(&nCells, fneq, &dIsDouble);

    free(p);
    free(u);
    free(v);
    free(x);
    free(y);
    free(z);
    free(w);
    free(vort);
    free(st);
	free(res_rho);
	free(res_u);
	free(res_v);
	free(res_w);
	free(dt);
	free(fneq);


    delete [] valueLocation;
    valueLocation= NULL;

    i = TECEND142();
}




template<typename T>
tecplot_output<T>::~tecplot_output()
{
    //dtor
}
