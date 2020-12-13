#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <vector_functions.h>
// #include <algorithm>
#ifdef __CUDA
namespace std{
	template<typename T>
	T max(const T&a, const T&b){
		return a<b ? b : a;
	}
	template<typename T>
	T min(const T&a, const T&b){
		return a<b ? a : b;
	}
}
#endif
#include "vector3.h"
#include "taylor.h"
#include "float2.h"

#ifndef __CUDA
#include<iostream>
#endif

#ifndef INLINE
#define INLINE __attribute__((always_inline))
#endif

// static const int init_iter = 2; // 1 for the 4th-order scheme, 2 for 6th or 8th

struct Force{
	enum{
		nword = 12,
	};
	dvec3 acc;
	dvec3 jrk;
	dvec3 snp;
	dvec3 crk;
	// double pot;
	Force() : acc(0.0), jrk(0.0), snp(0.0), crk(0.0) {}
};

struct Particle{
	enum{
		order = 8,
		init_iter = 2,
		flops = 144,
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
	// double pot;
	double t;
	double dt;
	int   id;
	int pad;

	Particle(){
		pos = vel = acc = jrk = snp = crk = pop = d5a = dvec3(0.0);
		// mass = pot = t = dt = 0.0;
		mass = t = dt = 0.0;
	}
	void init(double tsys, double dtmin, double dtmax, double eta, const Force &fo){
		acc = fo.acc;
		jrk = fo.jrk;
		snp = fo.snp;
		crk = fo.crk;
		// pot = fo.pot;
		t  = tsys;

		double aa = acc.norm2();
		double jj = jrk.norm2();
		double ss = snp.norm2();
		double cc = crk.norm2();
		double t1 = sqrt(aa*ss) + jj;
		double t2 = sqrt(jj*cc) + ss;
		double dt0 = 0.1 * eta * sqrt(t1/t2);
//		assert(dt0 > dtmin);
		dt0 = std::max(dt0, dtmin);
		dt = dtmax;
		while(dt >= dt0) dt *= 0.5;
#ifndef __CUDA
#if 0
		std::cout << id << " "
			      << acc << " "
			      << jrk << " "
			      << snp << " "
			      << crk << " "
				  << dt0 << " "
		          << dt << std::endl;
#endif
#endif
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
		// pot = fo.pot;
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

		double t1 = sqrt(s0*s2) + s1;

#if 0 // 8th-order crit

		double t2 = sqrt(s5*s7) + s6;
		double dt0 = eta * pow(t1/t2, 1./10.);

#elif 0 // 7th-order crit

		double t2 = sqrt(s4*s6) + s5;
		double dt0 = eta * pow(t1/t2, 1./8.);

#else // 6th-order crit

		double t2 = sqrt(s3*s5) + s4;
		double dt0 = eta * pow(t1/t2, 1./6.);

#endif

//		assert(dt0 > dtmin);
		dt0 = std::max(dt0, dtmin);
		// double dt1 = dtmax;
		double dt1 = dt * 4.0;
		while(dt1 >= dt0) dt1 *= 0.5;
		while(fmod(t, dt1) != 0.0) dt1 *= 0.5;
		dt = dt1;
#if 0
		std::cout << id << " "
			      << t  << " "
				  << dt << std::endl;
#endif
	}
	void prefetch() const{
#if 1
		char *p = (char *)this;
		__builtin_prefetch(p,     0, 0);
		__builtin_prefetch(p+128, 0, 0);
#endif
	}
};


struct Posm{
	float2 pos[3];
	float  mass;
	float  pad;

	Posm(){}
	Posm(const Particle &p){
		pos[0] = float2_split <20> (p.pos.x);
		pos[1] = float2_split <20> (p.pos.y);
		pos[2] = float2_split <20> (p.pos.z);
		mass   = p.mass;
	}
};
#ifdef __CUDA
struct Predictor{
	float4 posH;
	float4 posL;
	float4 vel;
	float4 acc;
	float4 jrk;
};
#else
struct Jparticle{
	typedef float v4sf __attribute__((vector_size(16)));
	static inline v4sf make_v4sf(double a, double b, double c, double d){
		typedef double v2df __attribute__((vector_size(16)));
		return __builtin_ia32_movlhps(
				__builtin_ia32_cvtpd2ps((v2df){a, b}),
				__builtin_ia32_cvtpd2ps((v2df){c, d}));
	}
	static inline v4sf make_v4sf(float a, float b, float c, float d){
		return (v4sf){a, b, c, d};
	}
	v4sf posH;
	v4sf posL;
	v4sf vel;
	v4sf acc;
	v4sf jrk;
	v4sf snp;
	v4sf crk;
	v4sf pop;
	v4sf d5a;
	double t, pad;
	Jparticle() {}
	Jparticle(Particle &p){
		float2 posx = float2_split<20>(p.pos.x);
		float2 posy = float2_split<20>(p.pos.y);
		float2 posz = float2_split<20>(p.pos.z);
		posH = make_v4sf(posx.x,  posy.x,  posz.x,  (float)p.mass);
		posL = make_v4sf(posx.y,  posy.y,  posz.y,  0.f);
		vel  = make_v4sf(p.vel.x, p.vel.y, p.vel.z, 0.0);
		acc  = make_v4sf(p.acc.x, p.acc.y, p.acc.z, 0.0);
		jrk  = make_v4sf(p.jrk.x, p.jrk.y, p.jrk.z, 0.0);
		snp  = make_v4sf(p.snp.x, p.snp.y, p.snp.z, 0.0);
		crk  = make_v4sf(p.crk.x, p.crk.y, p.crk.z, 0.0);
		pop  = make_v4sf(p.pop.x, p.pop.y, p.pop.z, 0.0);
		d5a  = make_v4sf(p.d5a.x, p.d5a.y, p.d5a.z, 0.0);
		t = p.t;
	}
	void prefetch() const{
		char *p = (char *)this;
		__builtin_prefetch(p,     0, 0);
		__builtin_prefetch(p+128, 0, 0);
	}
};
struct Predictor{
	typedef float v4sf __attribute__((vector_size(16)));
	v4sf posH;
	v4sf posL;
	v4sf vel;
	v4sf acc;
	v4sf jrk;
	struct _float{
		float val;
		_float(float _val) : val(_val) {}
		const _float operator*(const _float rhs) const{
			return _float(val * rhs.val);
		}
		friend const v4sf operator*(const _float f, const v4sf v){
			return (v4sf){f.val, f.val, f.val, f.val} * v;
		}
	};
	Predictor(double tnow, const Jparticle &p){
		_float dt = tnow - p.t;
		Taylor<_float, v4sf> taylor;
		posH = p.posH;
		posL = taylor(dt, p.posL, p.vel, p.acc, p.jrk, p.snp, p.crk, p.pop, p.d5a);
		vel  = taylor(dt,         p.vel, p.acc, p.jrk, p.snp, p.crk, p.pop, p.d5a);
		acc  = taylor(dt,                p.acc, p.jrk, p.snp, p.crk, p.pop, p.d5a);
		jrk  = taylor(dt,                       p.jrk, p.snp, p.crk, p.pop, p.d5a);
	}
	static Predictor *allocate(size_t n){
		void *p;
		cudaMallocHost(&p, n * sizeof(Predictor));
		return (Predictor *)p;
	}
	void store(void *p){
		__builtin_ia32_movntps((float *)p +  0, posH);
		__builtin_ia32_movntps((float *)p +  4, posL);
		__builtin_ia32_movntps((float *)p +  8, vel );
		__builtin_ia32_movntps((float *)p + 12, acc );
		__builtin_ia32_movntps((float *)p + 16, jrk );
	}
};
#endif

extern "C" void calc_pot(
		int ni,
		int js,
		int je,
		float eps2,
		Posm posm[],
		double pot[]);


static void potential(
		int ni,
		int js,
		int je,
		double eps2,
		Particle ptcl[],
		double pot[]){
	Posm *posm;
	// cudaMallocHost((void **)&posm, ni * sizeof(Posm));
	void *p;
	cudaMallocHost(&p, ni * sizeof(Posm));
	posm = (Posm *)p;

#pragma omp parallel for
	for(int i=0; i<ni; i++){
		posm[i] = Posm(ptcl[i]);
	}

	calc_pot(ni, js, je, eps2, posm, pot);

	cudaFreeHost(posm);
}

extern "C" void calc_force(
		int ni, 
		int nj, 
		float eps2,
		Predictor ipred[],
		Predictor jpred[],
		Force     force[],
		double &,
		double &,
		double &);

#if 0 // An emulator
void calc_force(
		int ni, 
		int nj, 
		float eps2,
		Predictor ipred[],
		Predictor jpred[],
		Force     force[]){
	for(int i=0; i<ni; i++){
		float2 Ax, Ay, Az, Pot;
		float Jx, Jy, Jz, Sx, Sy, Sz, Cx, Cy, Cz;
		Ax = Ay = Az = Pot = make_float2(0.f, 0.f);
		Jx = Jy = Jz = Sx = Sy = Sz = Cx = Cy = Cz = 0.f;
		for(int j=0; j<nj; j++){
			const Predictor &jp = jpred[j], &ip = ipred[i];
			float dx = (jp.posH.x - ip.posH.x) + (jp.posL.x - ip.posL.x);
			float dy = (jp.posH.y - ip.posH.y) + (jp.posL.y - ip.posL.y);
			float dz = (jp.posH.z - ip.posH.z) + (jp.posL.z - ip.posL.z);

			float dvx = jp.vel.x - ip.vel.x;
			float dvy = jp.vel.y - ip.vel.y;
			float dvz = jp.vel.z - ip.vel.z;

			float dax = jp.acc.x - ip.acc.x;
			float day = jp.acc.y - ip.acc.y;
			float daz = jp.acc.z - ip.acc.z;

			float djx = jp.jrk.x - ip.jrk.x;
			float djy = jp.jrk.y - ip.jrk.y;
			float djz = jp.jrk.z - ip.jrk.z;

			float r2 = eps2 + dx*dx + dy*dy + dz*dz;
			if(r2 == eps2) continue;
			float drdv =  dx*dvx +  dy*dvy +  dz*dvz;
			float dvdv = dvx*dvx + dvy*dvy + dvz*dvz;
			float drda =  dx*dax +  dy*day +  dz*daz;
			float dvda = dvx*dax + dvy*day + dvz*daz;
			float drdj =  dx*djx +  dy*djy +  dz*djz;

			float rinv2 = 1.f / r2;
			float alpha = (drdv)*rinv2;
			float beta = (dvdv + drda)*rinv2 + alpha*alpha;
			float gamma = (3.*dvda + drdj) * rinv2 
						 + alpha * (3.*beta - 4.*(alpha*alpha));

			float rinv1 = sqrtf(rinv2);
			// rinv1 *= jpred[j].mass;
			rinv1 *= jp.posH.w;
			float rinv3 = rinv1 * rinv2;

			float pot = rinv1;
			float ax = rinv3*dx;
			float ay = rinv3*dy;
			float az = rinv3*dz;
			float jx = rinv3*dvx + (-3.f*alpha)*ax;
			float jy = rinv3*dvy + (-3.f*alpha)*ay;
			float jz = rinv3*dvz + (-3.f*alpha)*az;
			float sx = rinv3*dax + (-6.f*alpha)*jx + (-3.*beta)*ax;
			float sy = rinv3*day + (-6.f*alpha)*jy + (-3.*beta)*ay;
			float sz = rinv3*daz + (-6.f*alpha)*jz + (-3.*beta)*az;
			float cx = rinv3*djx + (-9.f*alpha)*sx + (-9.f*beta)*jx + (-3.f*gamma)*ax;
			float cy = rinv3*djy + (-9.f*alpha)*sy + (-9.f*beta)*jy + (-3.f*gamma)*ay;
			float cz = rinv3*djz + (-9.f*alpha)*sz + (-9.f*beta)*jz + (-3.f*gamma)*az;

			Pot = float2_accum(Pot, pot);
			Ax = float2_accum(Ax, ax);
			Ay = float2_accum(Ay, ay);
			Az = float2_accum(Az, az);
			Jx += jx;
			Jy += jy;
			Jz += jz;
			Sx += sx;
			Sy += sy;
			Sz += sz;
			Cx += cx;
			Cy += cy;
			Cz += cz;
		}
		force[i].acc.x = float2_reduce(Ax);
		force[i].acc.y = float2_reduce(Ay);
		force[i].acc.z = float2_reduce(Az);
		force[i].jrk.x = Jx;
		force[i].jrk.y = Jy;
		force[i].jrk.z = Jz;
		force[i].snp.x = Sx;
		force[i].snp.y = Sy;
		force[i].snp.z = Sz;
		force[i].crk.x = Cx;
		force[i].crk.y = Cy;
		force[i].crk.z = Cz;
		force[i].pot = -float2_reduce(Pot);
	}
}
#endif

extern "C" void CUDA_MPI_Init(int);
