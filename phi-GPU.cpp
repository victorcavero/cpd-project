/*                               			*/
/* phi-GPU ver 3.0 with MPI & multi-GPU support :-) 	*/
/*                               			*/
/* - eta_BBH - added             			*/
/* - phi-GPU4, 6, 8                   			*/
/*                               			*/
/*                               			*/
/* 10.07.2009 19:12 	 				*/
/*                               			*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <utility>
#include <algorithm>
#include <mpi.h>

#define FOURTH 1
#ifdef FOURTH
#  ifdef GPU
#    include "hermite4-gpu.h"
#  else
#    include "hermite4.h"
#  endif
#elif defined SIXTH
#  ifdef GPU
#    include "hermite6-gpu.h"
#  else
#    include "hermite6.h"
#  endif
#elif defined EIGHTH
#  ifdef GPU
#    include "hermite8-gpu.h"
#  else
#    include "hermite8.h"
#  endif
#else
#  error
#endif

// #define N_MAX      (  4 << 20)
#define N_MAX      (1 << 20)
#define N_MAX_loc  (1 << 20)

static Particle ptcl[N_MAX];
static Jparticle jptcl[N_MAX];
#if 0
std::pair<int, double> t_plus_dt[N_MAX];
template <class T, class S>
struct cmp_pair_second{
	bool operator () (const std::pair<T,S> &p0, const std::pair<T,S> &p1){
		return (p0.second < p1.second);
	}
};
#else
std::pair<double, int> t_plus_dt[N_MAX];
#endif

// static double t_plus_dt[N_MAX];

// static Predictor jpred[N_MAX_loc], ipred[N_MAX];

#if 0
static Predictor *jpred = Predictor::allocate(N_MAX_loc);
static Predictor *ipred = Predictor::allocate(N_MAX);
#else
#endif
static Force force_tmp[N_MAX], force[N_MAX];
static int active_list[N_MAX];

static int nbody, diskstep;

static int myRank, n_proc, name_proc;
static char processor_name[MPI_MAX_PROCESSOR_NAME];

static int jstart, jend;
static double eps2;
static double time_cur, Timesteps=0.0, n_act_sum=0.0, g6_calls=0.0;
static double CPU_time_real0, CPU_time_user0, CPU_time_syst0;
static double CPU_time_real,  CPU_time_user,  CPU_time_syst;

#define PROFILE
#ifdef PROFILE
extern double wtime()
{
	return MPI_Wtime();
}
#else
extern double wtime()
{
	return 0;
}
#endif

static void get_CPU_time(double *time_real, double *time_user, double *time_syst) {
	struct rusage xxx;
	double sec_u, microsec_u, sec_s, microsec_s;

	getrusage(RUSAGE_SELF,&xxx);

	sec_u = xxx.ru_utime.tv_sec;
	sec_s = xxx.ru_stime.tv_sec;

	microsec_u = xxx.ru_utime.tv_usec;
	microsec_s = xxx.ru_stime.tv_usec;

	*time_user = sec_u + microsec_u * 1.0E-06;
	*time_syst = sec_s + microsec_s * 1.0E-06; 

	// *time_real = time(NULL);
	struct timeval tv;
	gettimeofday(&tv, NULL);
	*time_real = tv.tv_sec + 1.0E-06 * tv.tv_usec;
}

static void outputsnap(FILE *out){
	fprintf(out,"%04d \n", diskstep);
	fprintf(out,"%06d \n", nbody);
	fprintf(out,"%.8E \n", time_cur);
	for(int i=0; i<nbody; i++){
		Particle &p = ptcl[i];
	   	fprintf(out,"%06d  %.8E  % .8E % .8E % .8E  % .8E % .8E % .8E \n", 
				p.id, p.mass, p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2]);
	}
}

static void outputsnap(){
	static char out_fname[256];
	sprintf(out_fname,"%04d.dat",diskstep);
	FILE *out = fopen(out_fname,"w");
	assert(out);
	outputsnap(out);
	fclose(out);

	out = fopen("data.con","w");
	assert(out);
	outputsnap(out);
	fclose(out);
}

/*
static void outputsnap_BH()
{
  FILE *out = fopen("bh.dat","a");

  assert(out);

  fprintf(out,"%.8E \n", time_cur);

  int i = 0;
  Particle &p = ptcl[i];
  fprintf(out,"%06d  %.8E  % .8E % .8E % .8E  % .8E % .8E % .8E \n", 
	p.id, p.mass, p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2]);

  i = 1;
  p = ptcl[i];
  fprintf(out,"%06d  %.8E  % .8E % .8E % .8E  % .8E % .8E % .8E \n", 
	p.id, p.mass, p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2]);

  fprintf(out,"\n");

  fclose(out);
}
*/

static void energy(int myRank){
	static bool init_call = true;
	static double einit;

#ifdef GPU
	static double pot_tmp[N_MAX], pot[N_MAX];
	potential(nbody, jstart, jend, eps2, ptcl, pot_tmp);
	MPI_Allreduce(pot_tmp, pot, nbody, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double E_pot = 0.0;
	for(int i=0; i<nbody; i++) E_pot += ptcl[i].mass * pot[i];
#else
	double E_pot = 0.0;
	for(int i=0; i<nbody; i++) E_pot += ptcl[i].mass * ptcl[i].pot;
#endif
	E_pot *= 0.5;

	double E_kin = 0.0;
	for(int i=0; i<nbody; i++) E_kin += ptcl[i].mass * ptcl[i].vel.norm2();
	E_kin *= 0.5;

	assert(E_pot == E_pot);
	assert(E_kin == E_kin);

	double mcm = 0.0;
	dvec3 xcm = 0.0;
	dvec3 vcm = 0.0;
	for(int i=0; i<nbody; i++){
		mcm += ptcl[i].mass;
		xcm += ptcl[i].mass * ptcl[i].pos;
		vcm += ptcl[i].mass * ptcl[i].vel;
	}
	xcm /= mcm;
	vcm /= mcm;

	double rcm_mod = xcm.abs();
	double vcm_mod = vcm.abs();

	dvec3 mom = 0.0;
	for(int i=0; i<nbody; i++) mom += ptcl[i].mass * (ptcl[i].pos % ptcl[i].vel);

	get_CPU_time(&CPU_time_real, &CPU_time_user, &CPU_time_syst);


	if(init_call)
	  {
	  einit = E_pot + E_kin;
	  init_call = false;
  	  }

	double eerr = (E_pot+E_kin-einit)/einit;
	
        einit = E_pot + E_kin;
	

	if(myRank == 0){
		printf("%.4E   %.3E %.3E   % .6E % .6E % .6E % .6E   %.6E \n",
			   time_cur, Timesteps, n_act_sum,
			   E_pot, E_kin, E_pot+E_kin, 
			   eerr,
			   CPU_time_user-CPU_time_user0);
		fflush(stdout);

		FILE *out = fopen("contr.dat","a");
		assert(out);

		fprintf(out,"%.8E  %.8E  %.8E   % .16E % .16E % .16E   % .16E   % .16E % .16E   % .16E % .16E % .16E   %.8E %.8E %.8E \n",
				time_cur, Timesteps, n_act_sum,
				E_pot, E_kin, E_pot+E_kin, 
			eerr, 
				rcm_mod, vcm_mod, mom[0], mom[1], mom[2], 
				CPU_time_real-CPU_time_real0, CPU_time_user-CPU_time_user0, CPU_time_syst-CPU_time_syst0);
		fclose(out);
	}

/*
	if(myRank == 0){

		FILE *out = fopen("bh.dat","a");
		assert(out);

		  fprintf(out,"%.10E \n", time_cur);

		  fprintf(out,"%06d \t %.6E \t % .10E % .10E % .10E \t % .10E % .10E % .10E \t % .10E \t % .10E % .10E % .10E \t % .10E % .10E % .10E \n", 
			0, 
			ptcl[0].mass, 
			ptcl[0].pos[0], ptcl[0].pos[1], ptcl[0].pos[2], 
			ptcl[0].vel[0], ptcl[0].vel[1], ptcl[0].vel[2], 
			pot[0], 
			ptcl[0].acc[0], ptcl[0].acc[1], ptcl[0].acc[2], 
			ptcl[0].jrk[0], ptcl[0].jrk[1], ptcl[0].jrk[2]);

		  fprintf(out,"%06d \t %.6E \t % .10E % .10E % .10E \t % .10E % .10E % .10E \t % .10E \t % .10E % .10E % .10E \t % .10E % .10E % .10E \n", 
			1, 
			ptcl[1].mass, 
			ptcl[1].pos[0], ptcl[1].pos[1], ptcl[1].pos[2], 
			ptcl[1].vel[0], ptcl[1].vel[1], ptcl[1].vel[2], 
			pot[1], 
			ptcl[1].acc[0], ptcl[1].acc[1], ptcl[1].acc[2], 
			ptcl[1].jrk[0], ptcl[1].jrk[1], ptcl[1].jrk[2]);


		  fprintf(out,"\n");

		fclose(out);
	}
*/

#ifdef CMCORR
	for(int i=0; i<nbody; i++)
	{
		ptci[i].pos -= xcm;
		ptci[i].vel -= vcm;
	}
#endif

}

int main(int argc, char *argv[]){
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);	

        /* Define the processors names */
        MPI_Get_processor_name(processor_name, &name_proc);

        /* Print the Rank and the names of processors */
        printf("Rank of the processor %02d on %s \n", myRank, processor_name);
 
#ifdef GPU
	CUDA_MPI_Init(myRank);
#endif

	Predictor *jpred = Predictor::allocate(N_MAX_loc);
	Predictor *ipred = Predictor::allocate(N_MAX);

//	double eps, t_end, dt_disk, dt_contr, eta;
	double eps, t_end, dt_disk, dt_contr, eta, eta_BH;

	// int diskstep, nbody;
	// double time_cur;
	MPI_Barrier(MPI_COMM_WORLD);
	if(myRank == 0){
#ifdef FOURTH
		std::ifstream ifs("phi-GPU4.cfg");
#endif
#ifdef SIXTH
		std::ifstream ifs("phi-GPU6.cfg");
#endif
#ifdef EIGHTH
		std::ifstream ifs("phi-GPU8.cfg");
#endif
		static char inp_fname[256];
//		ifs >> eps >>  t_end >>  dt_disk >>  dt_contr >>  eta >> inp_fname;
		ifs >> eps >>  t_end >>  dt_disk >>  dt_contr >>  eta >> eta_BH >> inp_fname;
		if(ifs.fail()) MPI_Abort(MPI_COMM_WORLD, -1);
		std::ifstream inp(inp_fname);
		// std::cerr << inp_fname << std::endl;
		inp >> diskstep >> nbody >> time_cur;
		// std::cerr << diskstep << " " << nbody << " " << time_cur << std::endl;
		assert(nbody <= N_MAX);
		assert(nbody/n_proc <= N_MAX_loc);
		if(inp.fail()) MPI_Abort(MPI_COMM_WORLD, -2);
		for(int i=0; i<nbody; i++){
			Particle &p = ptcl[i];
			inp >> p.id >> p.mass >> p.pos >> p.vel;
			p.t = time_cur;
		}
		if(inp.fail()) MPI_Abort(MPI_COMM_WORLD, -3);

#ifdef CMCORR
		if(diskstep == 0){
			double msum = 0.0;
			dvec3 xsum = 0.0, vsum=0.0;
			for(int i=0; i<nbody; i++){
				msum += ptcl[i].mass;
				xsum += ptcl[i].mass * ptcl[i].pos;
				vsum += ptcl[i].mass * ptcl[i].vel;
			}
			xsum /= msum;
			vsum /= msum;
			for(int i=0; i<nbody; i++){
				ptcl[i].pos -= xsum;
				ptcl[i].vel -= vsum;
			}
		}
#endif

		printf("\n");
//		printf("Begin the calculation of phi-GPU program on %03d processors\n", n_proc); 
#ifdef FOURTH
    printf("Begin the calculation of phi-GPU4 program on %03d processors\n", n_proc); 
#endif
#ifdef SIXTH
    printf("Begin the calculation of phi-GPU6 program on %03d processors\n", n_proc); 
#endif
#ifdef EIGHTH
    printf("Begin the calculation of phi-GPU8 program on %03d processors\n", n_proc); 
#endif
		printf("\n");
		printf("N       = %06d \t eps      = %.6E \n", nbody, eps);
		printf("t_beg   = %.6E \t t_end    = %.6E \n", time_cur, t_end);
		printf("dt_disk = %.6E \t dt_contr = %.6E \n", dt_disk, dt_contr);
//		printf("eta     = %.6E \n", eta);
	        printf("eta     = %.6E \t eta_BH   = %.6E \n", eta, eta_BH);
		printf("\n"); 

		fflush(stdout);

		if(diskstep == 0)
		{
			outputsnap();
		}
		get_CPU_time(&CPU_time_real0, &CPU_time_user0, &CPU_time_syst0);
    } /* if(myRank == rootRank) */

	MPI_Barrier(MPI_COMM_WORLD);
	// std::cerr << "bcast parameters" << std::endl;

	MPI_Bcast(&nbody,    1, MPI_INT,    0, MPI_COMM_WORLD);
	MPI_Bcast(&eps,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
	MPI_Bcast(&eta,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eta_BH,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&t_end,    1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt_disk,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
	MPI_Bcast(&dt_contr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&time_cur, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double dt_max = 1. / (1 << 3);
	while(dt_max >= std::min(dt_disk, dt_contr)) dt_max *= 0.5;
	double dt_min = 1. / (1 << 23);

	int n_loc = nbody/n_proc;
	double t_disk  = time_cur + dt_disk;
	double t_contr = time_cur + dt_contr;

	MPI_Barrier(MPI_COMM_WORLD);
#if 1
	MPI_Bcast(ptcl, nbody*sizeof(Particle), MPI_CHAR, 0, MPI_COMM_WORLD);
#else
	if(myRank == 0){
		for(int p=1; p<n_proc; p++){
			MPI_Send(ptcl, nbody*sizeof(Particle), MPI_CHAR, p, 0, MPI_COMM_WORLD);
			fprintf(stderr, ".");
		}
		fprintf(stderr, "\n");
	}else{
		MPI_Recv(ptcl, nbody*sizeof(Particle), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
#endif

	jstart = (myRank * nbody) / n_proc; 
	jend   = ((1+myRank) * nbody) / n_proc; 
	n_loc = jend - jstart;
	int nj = n_loc;
	int ni = nbody;
	eps2 = eps*eps;

	MPI_Barrier(MPI_COMM_WORLD);
	if(myRank == 0){
		get_CPU_time(&CPU_time_real0, &CPU_time_user0, &CPU_time_syst0);
	}
	for(int l=0; l<Particle::init_iter; l++){
#pragma omp parallel for
		for(int j=0; j<nj; j++){
			jpred[j] = Predictor(time_cur, Jparticle(ptcl[j+jstart]));
		}
#pragma omp parallel for
		for(int i=0; i<ni; i++){
			ipred[i] = Predictor(time_cur, Jparticle(ptcl[i]));
		}
		double dum;
		calc_force(ni, nj, eps2, ipred, jpred, force_tmp, dum, dum, dum);
		MPI_Allreduce(force_tmp, force, ni*Force::nword, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		for(int i=0; i<ni; i++){

			ptcl[i].init(time_cur, dt_min, dt_max, eta, force[i]);

/*
      if( (i == 0) || (i == 1) ) 	
        ptcl[i].init(time_cur, dt_min, dt_max, eta_BH, force[i]);
      else      		       		
        ptcl[i].init(time_cur, dt_min, dt_max, eta, force[i]);
*/

			// t_plus_dt[i] = ptcl[i].t + ptcl[i].dt;	
			t_plus_dt[i].first = ptcl[i].t + ptcl[i].dt;	
			t_plus_dt[i].second = i;
			jptcl[i] = Jparticle(ptcl[i]);
		}
		// std::sort(t_plus_dt, t_plus_dt + ni, cmp_pair_second<int, double>());
		std::sort(t_plus_dt, t_plus_dt + ni);
	}

	if(myRank == 0){
		get_CPU_time(&CPU_time_real, &CPU_time_user, &CPU_time_syst);
		double Gflops = Particle::flops * 1.e-9 * Particle::init_iter * double(nbody) * nbody 
			            / (CPU_time_real - CPU_time_real0);
		fprintf(stdout, "Initialized, %f Gflops\n", Gflops);
	}

#if 0
	if(myRank == 0){
		for(int i=0; i<nbody; i++){
			std::cout << force[i].acc << " "
				      << force[i].jrk << " "
				      << force[i].snp << " "
					  << std::endl;
		}
	}
	exit(0);
#endif

	energy(myRank);
	if(myRank == 0){
		get_CPU_time(&CPU_time_real0, &CPU_time_user0, &CPU_time_syst0);
	}
	double t_scan = 0.0, t_pred = 0.0, t_jsend = 0.0, t_isend = 0.0, t_force = 0.0, t_recv = 0.0, t_comm = 0.0, t_corr = 0.0;

	while(time_cur <= t_end){
		double t0 = wtime();
#if 0
		double min_t = t_plus_dt[0];
		// for(int i=0; i<nbody; i++) min_t = std::min(min_t, t_plus_dt[i]);
		for(int i=0; i<nbody; i++) min_t = min_t < t_plus_dt[i] ? min_t : t_plus_dt[i];
		int n_act = 0;
		for(int i=0; i<nbody; i++){
			if(t_plus_dt[i] == min_t) active_list[n_act++] = i;
		}
#else
		double min_t = t_plus_dt[0].first;
		int n_act = 0;
		while(t_plus_dt[n_act].first == min_t){
			active_list[n_act] = t_plus_dt[n_act].second;
			n_act++;
		}
#endif

//		fprintf(stderr, "%16.6e %4d\n", time_cur, n_act);
//		fprintf(stderr, "%16.6e %4d\n", min_t, n_act);

		double t1 = wtime();

#pragma omp parallel for
#if 0
		for(int j=0; j<nj; j++){
			Taylor <double, dvec3> taylor;
			ptcl[j+jstart+1].prefetch();
			const Particle &p = ptcl[j+jstart];
			Predictor &pr = jpred[j];
			double dt = min_t - p.t;
			dvec3 dpos = taylor(dt, p.pos, p.vel, p.acc, p.jrk, p.snp, p.crk);
			dvec3 dvel = taylor(dt,        p.vel, p.acc, p.jrk, p.snp, p.crk);
			dvec3 dacc = taylor(dt,               p.acc, p.jrk, p.snp, p.crk);
			pr.pos[0] = float2_split <20> (dpos.x);
			pr.pos[1] = float2_split <20> (dpos.y);
			pr.pos[2] = float2_split <20> (dpos.z);
			pr.vel[0] = dvel.x;
			pr.vel[1] = dvel.y;
			pr.vel[2] = dvel.z;
			pr.acc[0] = dacc.x;
			pr.acc[1] = dacc.y;
			pr.acc[2] = dacc.z;
			pr.mass = p.mass;
		}
#else
		for(int j=0; j<nj; j++){
			// ptcl[j+jstart+1].prefetch();
			jptcl[j+jstart+1].prefetch();
		   	// jpred[j] = Predictor(min_t, ptcl[j+jstart]);
		   	// jpred[j] = Predictor(min_t, Jparticle(ptcl[j+jstart]));

#ifdef GPU
		   	Predictor(min_t, jptcl[j+jstart]).store(&jpred[j]);
#else
		   	jpred[j] = Predictor(min_t, jptcl[j+jstart]);
#endif
		}
#endif
		int ni = n_act;

#pragma omp parallel for
		for(int i=0; i<ni; i++){
			// ipred[i] = Predictor(min_t, ptcl[active_list[i]]);
#if 0
			ptcl[active_list[i+1]].prefetch();
			ipred[i] = Predictor(min_t, Jparticle(ptcl[active_list[i]]));
#else
			jptcl[active_list[i+1]].prefetch();
#ifdef GPU
			Predictor(min_t, jptcl[active_list[i]]).store(&ipred[i]);
#else
			ipred[i] = Predictor(min_t, jptcl[active_list[i]]);
#endif
#endif
		}

		double t2 = wtime();
		double t3;
		calc_force(ni, nj, eps2, ipred, jpred, force_tmp, t3, t_isend, t_recv);
		double t4 = wtime();
		MPI_Allreduce(force_tmp, force, ni*Force::nword, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		double t5 = wtime();

#pragma omp parallel for
		for(int i=0; i<ni; i++){
			ptcl[active_list[i+1]].prefetch();
			// jptcl[active_list[i+1]].prefetch();
			Particle &p = ptcl[active_list[i]];

			p.correct(dt_min, dt_max, eta, force[i]);
/*
      if( (active_list[i] == 0) || (active_list[i] == 1) ) 
        p.correct(dt_min, dt_max, eta_BH, force[i]);
      else 
        p.correct(dt_min, dt_max, eta, force[i]);
*/
			// t_plus_dt[active_list[i]] = p.t + p.dt;
			t_plus_dt[i].second = active_list[i];
			t_plus_dt[i].first  = p.t + p.dt;
#if 0
			int ii = active_list[i];
			if(jstart<=ii && ii < jend){
				jptcl[ii] = Jparticle(p);
			}
#else
			jptcl[active_list[i]] = Jparticle(p);
#endif
		}

		double t6 = wtime();
		// std::sort(t_plus_dt, t_plus_dt + ni, cmp_pair_second<int, double>());
		std::sort(t_plus_dt, t_plus_dt + ni);
		double t7 = wtime();
		t_scan  += t1 - t0;
		t_pred  += t2 - t1;
		t_jsend += t3 - t2;
		t_force += t4 - t3;
		t_comm  += t5 - t4;
		t_corr  += t6 - t5;
		t_scan  += t7 - t6;

		time_cur = min_t;
		Timesteps += 1.0;
		n_act_sum += n_act;

		if(time_cur >= t_contr){
			energy(myRank);
			if(myRank == 0){
//			outputsnap_BH();
			}
			t_contr += dt_contr;
		}
		if(time_cur >= t_disk){
			if(myRank == 0){
				diskstep++;
				outputsnap();
			}
			t_disk += dt_disk;
		}
	} // while(time_cut <= t_end)

	double g6_calls_sum;
	MPI_Reduce(&g6_calls, &g6_calls_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(myRank == 0){
		printf("\n");
		printf("Timesteps = %.0f   Total sum of integrated part. = %.0f   n_act average = %.0f\n", 
				Timesteps, n_act_sum, n_act_sum/Timesteps);
		printf("\n");
		// double Gflops  = 57.0e-9*nbody*n_act_sum/(CPU_time_user - CPU_time_user0);
		// double GflopsT = 57.0e-9*nbody*g6_calls*48/(CPU_time_user - CPU_time_user0);
		double Gflops  = Particle::flops*1.e-9*double(nbody)*double(n_act_sum)/(CPU_time_real - CPU_time_real0);
		// double GflopsT = Particle::flops*1.e-9*nbody*g6_calls*48/(CPU_time_real - CPU_time_real0);
		printf("Real Speed = %.3f GFlops \n", Gflops);

#ifdef PROFILE
		double t_tot = t_scan + t_pred + t_jsend + t_force + t_comm + t_corr;
		t_force -= t_isend + t_recv;
		// printf("Time ratio: %10.2E%10.2E%10.2E%10.2E%10.2E\n", t_scan/t_tot, t_pred/t_tot, t_forc/t_tot, t_comm/t_tot, t_corr/t_tot);
		printf("        sec_tot     usec/step   ratio\n"); 
		printf("scan :%12.4E%12.4E%12.4E\n", t_scan, t_scan/Timesteps*1.e6, t_scan/t_tot);
		printf("pred :%12.4E%12.4E%12.4E\n", t_pred, t_pred/Timesteps*1.e6, t_pred/t_tot);
		printf("jsend:%12.4E%12.4E%12.4E\n", t_jsend, t_jsend/Timesteps*1.e6, t_jsend/t_tot);
		printf("isend:%12.4E%12.4E%12.4E\n", t_isend, t_isend/Timesteps*1.e6, t_isend/t_tot);
		printf("force:%12.4E%12.4E%12.4E\n", t_force, t_force/Timesteps*1.e6, t_force/t_tot);
		printf("recv :%12.4E%12.4E%12.4E\n", t_recv, t_recv/Timesteps*1.e6, t_recv/t_tot);
		printf("comm :%12.4E%12.4E%12.4E\n", t_comm, t_comm/Timesteps*1.e6, t_comm/t_tot);
		printf("corr :%12.4E%12.4E%12.4E\n", t_corr, t_corr/Timesteps*1.e6, t_corr/t_tot);
		printf("tot  :%12.4E%12.4E%12.4E\n", t_tot, t_tot/Timesteps*1.e6, t_tot/t_tot);
		// double t_comm_avr = t_comm / Timesteps * 1.e6;
		// printf("Comm time : %10.2E usec\n", t_comm_avr);
#endif
                fflush(stdout);
	}

	MPI_Finalize();
	return 0;
}
