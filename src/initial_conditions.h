#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include "vector_var.h"

class initial_conditions
{
    public:
        initial_conditions();
        virtual ~initial_conditions();
        vector_var rho_gradient, rho_origin_mag, origin_loc; // probably be changed for 3d get up and running first
        double average_rho; // average rho that is kept constant and used for momentum calcs
        vector_var velocity;
        vector_var vel_gradient, vel_origin_mag;
        double pressure_gradient = 0.0;
    protected:
    private:
};

#endif // INITIAL_CONDITIONS_H
