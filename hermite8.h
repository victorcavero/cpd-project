#include <cassert>
#include <cmath>
#include "vector3.h"
#include "taylor.h"

extern double wtime();

struct Force{
	enum{
		nword = 13,
	};
	dvec3 acc;
	dvec3 jrk;
	dvec3 snp;
	dvec3 crk;
	double pot;
};

struct Particle{
	enum{
		order = 8,
		flops = 144,
		init_iter = 2
	};
	dvec3 pos;
	dvec3 vel;
	dvec3 acc;
	dvec3 jrk;
	dvec3 snp;
	dvec3 crk;
	dvec3 pop;
	dvec3 d5a;
	double mass;
	double pot;
	double t;
	float dt;
	int   id;

	Particle(){
		pos = vel = acc = jrk = snp = crk = pop = d5a = dvec3(0.0);
		mass = pot = t = dt = 0.0;
	}
	void prefetch() {}
	void init(double tsys, double dtmin, double dtmax, double eta, const Force &fo){
		acc = fo.acc;
		jrk = fo.jrk;
		snp = fo.snp;
		crk = fo.crk;
		pot = fo.pot;
		t  = tsys;

		double aa = acc.norm2();
		double jj = jrk.norm2();
		double ss = snp.norm2();
		double cc = crk.norm2();
		double t1 = sqrt(aa*ss) + jj;
		double t2 = sqrt(jj*cc) + ss;
		double dt0 = 0.1 * eta * sqrt(t1/t2);
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
		dvec3 Cp = (fo.crk + crk)*(h*h*h);
		dvec3 Cm = (fo.crk - crk)*(h*h*h);

		dvec3 vel1 = vel +   h*(Ap - (3./7.)*Jm + (2./21.)*Sp - (1./105.)*Cm);
		dvec3 Vp = (vel1 + vel)*hi;
		dvec3 pos1 = pos + h*h*(Vp - (3./7.)*Am + (2./21.)*Jp - (1./105.)*Sm);

		pos = pos1;
		vel = vel1;
		acc = fo.acc;
		jrk = fo.jrk;
		snp = fo.snp;
		crk = fo.crk;
		pot = fo.pot;
		t += dt;

		double hi2 = hi*hi;
		double hi3 = hi2*hi;
		double hi4 = hi3*hi;
		double hi5 = hi4*hi;
		double hi6 = hi5*hi;
		double hi7 = hi6*hi;
		pop       = (hi4 *   24./32.)*(         - 5.*Jm +  5.*Sp - Cm);
		d5a       = (hi5 *  120./32.)*( 21.*Am - 21.*Jp +  8.*Sm - Cp);
		dvec3 d6a = (hi6 *  720./32.)*(              Jm -     Sp + Cm/3.);
		dvec3 d7a = (hi7 * 5040./32.)*( -5.*Am +  5.*Jp -  2.*Sm + Cp/3.);
		Taylor <double ,dvec3> taylor;
		pop = taylor(+h, pop, d5a, d6a, d7a);
		d5a = taylor(+h, d5a, d6a, d7a);
		d6a = taylor(+h, d6a, d7a);

		double s0 = acc.norm2();
		double s1 = jrk.norm2();
		double s2 = snp.norm2();
		double s3 = crk.norm2();
		double s4 = pop.norm2();
		double s5 = d5a.norm2();
		double s6 = d6a.norm2();
		double s7 = d7a.norm2();

#if 0
		double t1 = sqrt(s0*s2) + s1;
		double t2 = sqrt(s5*s7) + s6;
		double dt0 = eta * pow(t1/t2, 1./10.);
#else
		double t1 = sqrt(s0*s2) + s1;
		double t2 = sqrt(s3*s5) + s4;
		double dt0 = eta * pow(t1/t2, 1./6.);
#endif
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
	dvec3 jrk;
	double mass;

	Predictor(){}
	Predictor(double tnow, const Particle &p){
		Taylor <double, dvec3> taylor;
		double dt = tnow - p.t;
		pos = taylor(dt, p.pos, p.vel, p.acc, p.jrk, p.snp, p.crk, p.pop, p.d5a);
		vel = taylor(dt,        p.vel, p.acc, p.jrk, p.snp, p.crk, p.pop, p.d5a);
		acc = taylor(dt,               p.acc, p.jrk, p.snp, p.crk, p.pop, p.d5a);
		jrk = taylor(dt,                      p.jrk, p.snp, p.crk, p.pop, p.d5a);
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
	// typedef double real;
	typedef float real;
	typedef vector3<real> rvec3;
#pragma omp parallel for
	for(int i=0; i<ni; i++){
		dvec3 acc(0.), jrk(0.), snp(0.), crk(0.);
		double pot=0;
		for(int j=0; j<nj; j++){
			rvec3 dr = jpred[j].pos - ipred[i].pos;
			rvec3 dv = jpred[j].vel - ipred[i].vel;
			rvec3 da = jpred[j].acc - ipred[i].acc;
			rvec3 dj = jpred[j].jrk - ipred[i].jrk;

			real r2 = eps2 + dr*dr;
			
			if(r2 == eps2) continue;
			real rinv2 = 1.0 / r2;
			real alpha = (dr*dv)*rinv2;
			real beta = (dv*dv + dr*da)*rinv2 + alpha*alpha;
			real gamma = (3.*(dv*da) + dr*dj) * rinv2 
						 + alpha * (3.*beta - 4.*(alpha*alpha));
			real rinv1 = sqrt(rinv2);
			rinv1 *= jpred[j].mass;
			real rinv3 = rinv1 * rinv2;

			rvec3 aij = rinv3*dr;
			rvec3 jij = rinv3*dv + (real(-3.)*alpha)*aij;
			rvec3 sij = rinv3*da + (real(-6.)*alpha)*jij + (real(-3.)*beta)*aij;
			rvec3 cij = rinv3*dj + (real(-9.)*alpha)*sij + (real(-9.)*beta)*jij + (real(-3.)*gamma)*aij;

			pot += rinv1;
			acc += aij;
			jrk += jij;
			snp += sij;
			crk += cij;
		}
		force[i].acc = acc;
		force[i].jrk = jrk;
		force[i].snp = snp;
		force[i].crk = crk;
		force[i].pot = -pot;
	}
}
#endif
