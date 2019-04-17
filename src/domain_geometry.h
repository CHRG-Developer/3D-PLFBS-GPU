#ifndef DOMAIN_GEOMETRY_H
#define DOMAIN_GEOMETRY_H


class domain_geometry
{
    public:
        domain_geometry();
        virtual ~domain_geometry();
        double X,Y,Z,dx,dy,dz;
        double dt ;  // streaming time step
        double cs; // speed of sound in medium
        void initialise();
        void scale_geometries(double scale);

    protected:
    private:
};

#endif // DOMAIN_GEOMETRY_H
