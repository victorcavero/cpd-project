#include <cassert>
#include <cmath>
#include "vector3.h"
#include "taylor.h"

extern double wtime();

struct Force{
	enum{
		nword = 10,
	};
	dvec3 acc;
	dvec3 jrk;
	dvec3 snp;
	double pot;
};

struct Particle{
	enum{
		order = 6,
		flops = 97,
		init_iter = 2
	};
	dvec3 pos;
	dvec3 vel;
	dvec3 acc;
	dvec3 jrk;
	dvec3 snp;
	dvec3 crk;
	double mass;
	double pot;
	double t;
	float dt;
	int   id;

	Particle(){
		pos = vel = acc = jrk = snp = crka = dvec3(0.0);
		mass = pot = t = dt = 0.0;
	}
	void prefetch() {}
	void init(double tsys, double dtmin, double dtmax, double eta, const Force &fo){
		acc = fo.acc;
		jrk = fo.jrk;
		snp = fo.snp;
		pot = fo.pot;
		t  = tsys;
		double dt0 = 0.1 * eta * pow(acc.norm2()/snp.norm2(), 1./4.);
		assert(dt0 > dtmin);
		dt0 = std::max(dt0, dtmin);
		dt = dtmax;
		while(dt >= dt0) dt *= 0.5;
		/*
		std::cout << id << " "
			      << acc << " "
			      << jrk << " "
			      << snp << " "
				  << dt0 << " "
		          << dt << std::endl;
		*/
	}
	void correct(double dtmin, double dtmax, double eta, const Force &fo){
		double h = 0.5 * dt;
		double hi = 1.0 / h;
		dvec3 Ap = (fo.acc + acc);
		dvec3 Am = (fo.acc - acc);
		dvec3 Jp = (fo.jrk + jrk)*h;
		dvec3 Jm = (fo.jrk - jrk)*h;
		dvec3 Sp = (fo.snp + snp)*(h*h);
		dvec3 Sm = (fo.snp - snp)*(h*h);

		dvec3 vel1 = vel +   h*(Ap - 0.4*Jm + (1./15.)*Sp);
		dvec3 Vp = (vel1 + vel)*hi;
		dvec3 pos1 = pos + h*h*(Vp - 0.4*Am + (1./15.)*Jp);

		pos = pos1;
		vel = vel1;
		acc = fo.acc;
		jrk = fo.jrk;
		snp = fo.snp;
		pot = fo.pot;
		t += dt;

		double hi2 = hi*hi;
		double hi3 = hi2*hi;
		double hi4 = hi3*hi;
		double hi5 = hi4*hi;
		crk = (0.75*hi3)*(-5.*Am + 5.*Jp - Sm);
		dvec3 d4a = ( 1.5*hi4)*(-Jm + Sp);
		dvec3 d5a = ( 7.5*hi5)*(3*Am - 3*Jp + Sm);
		crk += h*(d4a + 0.5*h*d5a);
		d4a += h*d5a;

		double aa = acc.norm2();
		double jj = jrk.norm2();
		double ss = snp.norm2();
		double cc = crk.norm2();
		double pp = d4a.norm2();
		double qq = d5a.norm2();

		double t1 = sqrt(aa*ss) + jj;
		double t2 = sqrt(cc*qq) + pp;
		double dt0 = eta * pow(t1/t2, 1./6.);
		// double t2 = sqrt(jj*cc) + ss;
		// double dt0 = eta * sqrt(t1/t2);
		assert(dt0 > dtmin);
		dt0 = std::max(dt0, dtmin);
		dt = dtmax;
		while(dt >= dt0) dt *= 0.5;
		while(fmod(t, dt) != 0.0) dt *= 0.5;
#if 0
		std::cout << id << " "
			      << t  << " "
				  << dt << std::endl;
#endif
	}
};

typedef Particle Jparticle;

struct Predictor{
	dvec3 pos;
	dvec3 vel;
	dvec3 acc;
	double mass;

	Predictor(){}
	Predictor(double tnow, const Particle &p){
		Taylor <double, dvec3> taylor;
		double dt = tnow - p.t;
		pos = taylor(dt, p.pos, p.vel, p.acc, p.jrk, p.snp, p.crk);
		vel = taylor(dt,        p.vel, p.acc, p.jrk, p.snp, p.crk);
		acc = taylor(dt,               p.acc, p.jrk, p.snp, p.crk);
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
		dvec3 acc(0.), jrk(0.), snp(0.);
		double pot=0;
		for(int j=0; j<nj; j++){
			dvec3 dr = jpred[j].pos - ipred[i].pos;
			dvec3 dv = jpred[j].vel - ipred[i].vel;
			dvec3 da = jpred[j].acc - ipred[i].acc;

			double r2 = eps2 + dr*dr;
			
			if(r2 == eps2) continue;
			double rinv2 = 1.0 / r2;
			double alpha = (dr*dv)*rinv2;
			double beta = (dv*dv + dr*da)*rinv2 + alpha*alpha;
			double rinv1 = sqrt(rinv2);
			rinv1 *= jpred[j].mass;
			double rinv3 = rinv1 * rinv2;

			dvec3 aij = rinv3*dr;
			dvec3 jij = rinv3*dv + (-3.*alpha)*aij;
			dvec3 sij = rinv3*da + (-6.*alpha)*jij + (-3.*beta)*aij;

			pot += rinv1;
			acc += aij;
			jrk += jij;
			snp += sij;
		}
		force[i].acc = acc;
		force[i].jrk = jrk;
		force[i].snp = snp;
		force[i].pot = -pot;
	}
}
#endif
