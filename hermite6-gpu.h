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

#ifndef INLINE
#define INLINE __attribute__((always_inline))
#endif

// static const int init_iter = 2; // 1 for the 4th-order scheme, 2 for 6th or 8th

struct Force{
	enum{
		nword = 9,
	};
	dvec3 acc;
	dvec3 jrk;
	dvec3 snp;
	// double pot;
	Force() : acc(0.0), jrk(0.0), snp(0.0) {}
};

struct Particle{
	enum{
		order = 6,
		init_iter = 2,
		flops = 97,
	};
	dvec3 pos;
	dvec3 vel;
	dvec3 acc;
	dvec3 jrk;
	dvec3 snp;
	dvec3 crk;
	double mass;
	// double pot;
	double t;
	double dt;
	int   id;
	int pad;

	Particle(){
		pos = vel = acc = jrk = snp = crk = dvec3(0.0);
		// mass = pot = t = dt = 0.0;
		mass = t = dt = 0.0;
	}
	void init(double tsys, double dtmin, double dtmax, double eta, const Force &fo){
		acc = fo.acc;
		jrk = fo.jrk;
		snp = fo.snp;
		// pot = fo.pot;
		t  = tsys;
		double dt0 = 0.1 * eta * pow(acc.norm2()/snp.norm2(), 1./4.);
//		assert(dt0 > dtmin);
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
		// pot = fo.pot;
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

/*
struct Predictor{
	float2 pos[3];
	float  vel[3];
	float  acc[3];
	float  mass;
	float  pad[3];

	Predictor(){}
	INLINE Predictor(double tnow, const Particle &p){
		Taylor <double, dvec3> taylor;
		double dt = tnow - p.t;
		dvec3 dpos = taylor(dt, p.pos, p.vel, p.acc, p.jrk, p.snp, p.crk);
		dvec3 dvel = taylor(dt,        p.vel, p.acc, p.jrk, p.snp, p.crk);
		dvec3 dacc = taylor(dt,               p.acc, p.jrk, p.snp, p.crk);
		pos[0] = float2_split <20> (dpos.x);
		pos[1] = float2_split <20> (dpos.y);
		pos[2] = float2_split <20> (dpos.z);
		vel[0] = dvel.x;
		vel[1] = dvel.y;
		vel[2] = dvel.z;
		acc[0] = dacc.x;
		acc[1] = dacc.y;
		acc[2] = dacc.z;
		mass = p.mass;
	}
	static Predictor *allocate(size_t n){
		void *p;
		cudaMallocHost(&p, n * sizeof(Predictor));
		return (Predictor *)p;
	}
#ifndef __CUDA
	void store(void *dst){
		typedef float v4sf __attribute__((vector_size(16)));
		v4sf src0 = ((v4sf *)this)[0];
		v4sf src1 = ((v4sf *)this)[1];
		v4sf src2 = ((v4sf *)this)[2];
		v4sf src3 = ((v4sf *)this)[3];
		float *dst0 = (float *)dst;
		float *dst1 = dst0 + 4;
		float *dst2 = dst0 + 8;
		float *dst3 = dst0 + 12;
		__builtin_ia32_movntps(dst0, src0);
		__builtin_ia32_movntps(dst1, src1);
		__builtin_ia32_movntps(dst2, src2);
		__builtin_ia32_movntps(dst3, src3);
	}
#endif
};
*/

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
#if 1
		posH = p.posH;
		posL = taylor(dt, p.posL, p.vel, p.acc, p.jrk, p.snp, p.crk);
		vel  = taylor(dt,         p.vel, p.acc, p.jrk, p.snp, p.crk);
		acc  = taylor(dt,                p.acc, p.jrk, p.snp, p.crk);
#else
		__builtin_ia32_movntps((float *)&posH, p.posH);
		__builtin_ia32_movntps((float *)&posL, taylor(dt, p.posL, p.vel, p.acc, p.jrk, p.snp, p.crk));
		__builtin_ia32_movntps((float *)&vel,  taylor(dt,         p.vel, p.acc, p.jrk, p.snp, p.crk));
		__builtin_ia32_movntps((float *)&acc,  taylor(dt,                p.acc, p.jrk, p.snp, p.crk));
#endif
	}
	static Predictor *allocate(size_t n){
		void *p;
		cudaMallocHost(&p, n * sizeof(Predictor));
		return (Predictor *)p;
	}
	void store(void *p){
		__builtin_ia32_movntps((float *)p + 0, posH);
		__builtin_ia32_movntps((float *)p + 4, posL);
		__builtin_ia32_movntps((float *)p + 8, vel );
		__builtin_ia32_movntps((float *)p +12, acc );
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
		float Jx, Jy, Jz, Sx, Sy, Sz;
		Ax = Ay = Az = Pot = make_float2(0.f, 0.f);
		Jx = Jy = Jz = Sx = Sy = Sz = 0.f;
		for(int j=0; j<nj; j++){
			float dx = float2_sub(jpred[j].pos[0], ipred[i].pos[0]);
			float dy = float2_sub(jpred[j].pos[1], ipred[i].pos[1]);
			float dz = float2_sub(jpred[j].pos[2], ipred[i].pos[2]);

			float dvx = jpred[j].vel[0] - ipred[i].vel[0];
			float dvy = jpred[j].vel[1] - ipred[i].vel[1];
			float dvz = jpred[j].vel[2] - ipred[i].vel[2];

			float dax = jpred[j].acc[0] - ipred[i].acc[0];
			float day = jpred[j].acc[1] - ipred[i].acc[1];
			float daz = jpred[j].acc[2] - ipred[i].acc[2];

			float r2 = eps2 + dx*dx + dy*dy + dz*dz;
			if(r2 == eps2) continue;
			float drdv =  dx*dvx +  dy*dvy +  dz*dvz;
			float dvdv = dvx*dvx + dvy*dvy + dvz*dvz;
			float drda =  dx*dax +  dy*day +  dz*daz;

			float rinv2 = 1.f / r2;
			float alpha = (drdv)*rinv2;
			float beta = (dvdv + drda)*rinv2 + alpha*alpha;
			float rinv1 = sqrtf(rinv2);
			rinv1 *= jpred[j].mass;
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
		force[i].pot = -float2_reduce(Pot);
	}
}
#endif

extern "C" void CUDA_MPI_Init(int);
