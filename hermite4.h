#include <cassert>
#include <cmath>
#include "vector3.h"
#include "taylor.h"

extern double wtime();

struct Force{
	enum{
		nword = 7,
	};
	dvec3 acc;
	dvec3 jrk;
	double pot;
};

struct Particle{
	enum{
		order = 4,
		flops = 60,
		init_iter = 1,
	};
	dvec3 pos;
	dvec3 vel;
	dvec3 acc;
	dvec3 jrk;
	double mass;
	double pot;
	double t;
	float dt;
	int   id;

	Particle(){
		pos = vel = acc = jrk = dvec3(0.0);
		mass = pot = t = dt = 0.0;
	}
	void prefetch() {}
	void init(double tsys, double dtmin, double dtmax, double eta, const Force &fo){
		acc = fo.acc;
		jrk = fo.jrk;
		pot = fo.pot;
		t  = tsys;
		double dt0 = 0.1 * eta * sqrt(acc.norm2()/jrk.norm2());
		assert(dt0 > dtmin);
		dt0 = std::max(dt0, dtmin);
		dt = dtmax;
		while(dt >= dt0) dt *= 0.5;
	}
	void correct(double dtmin, double dtmax, double eta, const Force &fo){
		double h = 0.5 * dt;
		double hi = 1.0 / h;
		dvec3 Ap = (fo.acc + acc);
		dvec3 Am = (fo.acc - acc);
		dvec3 Jp = (fo.jrk + jrk)*h;
		dvec3 Jm = (fo.jrk - jrk)*h;

		dvec3 vel1 = vel +   h*(Ap - (1./3.)*Jm);
		dvec3 Vp = (vel1 + vel)*hi;
		dvec3 pos1 = pos + h*h*(Vp - (1./3.)*Am);

		pos = pos1;
		vel = vel1;
		acc = fo.acc;
		jrk = fo.jrk;
		pot = fo.pot;
		t += dt;

		dvec3 snp = (0.5*hi*hi) * Jm;
		dvec3 crk = (1.5*hi*hi*hi) * (Jp - Am);
		snp += h * crk;

		double aa = acc.norm2();
		double jj = jrk.norm2();
		double ss = snp.norm2();
		double cc = crk.norm2();
		double t1 = sqrt(aa*ss) + jj;
		double t2 = sqrt(jj*cc) + ss;
		double dt0 = eta * sqrt(t1/t2);
		assert(dt0 > dtmin);
		dt0 = std::max(dt0, dtmin);
		dt = dtmax;
		while(dt >= dt0) dt *= 0.5;
		while(fmod(t, dt) != 0.0) dt *= 0.5;
	}
};

typedef Particle Jparticle;

struct Predictor{
	dvec3 pos;
	dvec3 vel;
	double mass;

	Predictor(){}
	Predictor(double tnow, const Particle &p){
		Taylor <double, dvec3> taylor;
		double dt = tnow - p.t;
		pos = taylor(dt, p.pos, p.vel, p.acc, p.jrk);
		vel = taylor(dt,        p.vel, p.acc, p.jrk);
		mass = p.mass;
	}
	static Predictor *allocate(size_t n){
		return new Predictor[n];
	}
};

void calc_force(
		int ni, 
		int nj, 
		double eps2,
		Predictor ipred[],
		Predictor jpred[],
		Force     force[],
		double &,
		double &,
		double &);

#if 1
void calc_force(
		int ni, 
		int nj, 
		double eps2,
		Predictor ipred[],
		Predictor jpred[],
		Force     force[],
		double &t1,
		double &,
		double &){
	t1 = wtime();
#pragma omp parallel for
	for(int i=0; i<ni; i++){
		double ax=0, ay=0, az=0;
		double jx=0, jy=0, jz=0;
		double pot=0;
		for(int j=0; j<nj; j++){
			double dx = jpred[j].pos.x - ipred[i].pos.x;
			double dy = jpred[j].pos.y - ipred[i].pos.y;
			double dz = jpred[j].pos.z - ipred[i].pos.z;
			double dvx = jpred[j].vel.x - ipred[i].vel.x;
			double dvy = jpred[j].vel.y - ipred[i].vel.y;
			double dvz = jpred[j].vel.z - ipred[i].vel.z;

			double r2 = eps2 + dx*dx + dy*dy + dz*dz;
			double rv = dx*dvx + dy*dvy + dz*dvz;
			
			if(r2 == eps2) continue;
			double rinv2 = 1.0 / r2;
			double rinv1 = sqrt(rinv2);
			rv *= -3.0 * rinv2;
			rinv1 *= jpred[j].mass;
			double rinv3 = rinv1 * rinv2;

			pot += rinv1;
			ax += rinv3 * dx;
			ay += rinv3 * dy;
			az += rinv3 * dz;
			jx += rinv3 * (dvx + rv * dx);
			jy += rinv3 * (dvy + rv * dy);
			jz += rinv3 * (dvz + rv * dz);
		}
		force[i].acc.x = ax;
		force[i].acc.y = ay;
		force[i].acc.z = az;
		force[i].jrk.x = jx;
		force[i].jrk.y = jy;
		force[i].jrk.z = jz;
		force[i].pot   = -pot;
	}
}
#endif
