#ifndef TECPLOT_OUTPUT_H
#define TECPLOT_OUTPUT_H
#include "global_variables.h"
#include "Mesh.h"
#include "Solution.h"
#include "Boundary_Conditions.h"
#include "post_processing.h"

template <typename T>
class tecplot_output
{
    public:
        tecplot_output();
        tecplot_output(global_variables &globals, Mesh &Mesh, Solution &Soln,
                             Boundary_Conditions &bcs, int fileType_e, double timestamp,
                             post_processing &pp);
		
        void tecplot_output_unstructured_soln(global_variables &globals, unstructured_mesh &Mesh, Solution &Soln,
                             Boundary_Conditions &bcs,  double timestamp,
                             post_processing &pp, Solution &residual, double * local_delta_t, T * local_fneq);
        virtual ~tecplot_output();

    protected:

    private:
};

#endif // TECPLOT_OUTPUT_H
template class tecplot_output<float>;
template class tecplot_output<double>;