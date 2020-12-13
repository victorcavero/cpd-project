template <int SCALE>
static float2 float2_split(double x){
#if 0
	return make_float2((float)x, 0.f);
#else
	float2 ret;
	x *= (1<<SCALE);
	double xi = (int)x;
	double xf = x - xi;
	ret.x = xi * (1./(1<<SCALE));
	ret.y = xf * (1./(1<<SCALE));
	return ret;
#endif
}

#ifdef __CUDA
static __device__ float2 float2_accum(float2 acc, float x){
	float tmp = acc.x + x;
	acc.y -= (tmp - acc.x) - x;
	acc.x = tmp;
	return acc;
}

static double float2_reduce(float2 a){
	// fprintf(stderr, "%E\n", a.y/a.x);
	return (double)a.x + (double)a.y;
}

static __device__ __host__ float float2_sub(float2 a, float2 b){
	return (a.x - b.x) + (a.y - b.y);
}
#endif
