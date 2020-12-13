#ifndef INLINE
#define INLINE __attribute__((always_inline))
#endif

template <class REAL, class VECT> 
struct Taylor{
	const VECT operator () (
			const REAL &h, 
			const VECT &a0, 
			const VECT &a1)
	{
		return a0 + h*a1;
	}
	INLINE const VECT operator () (
			const REAL &h, 
			const VECT &a0, 
			const VECT &a1, 
			const VECT &a2)
	{
		REAL h2 = h * REAL(1./2.);
		return a0 + h*(a1 + h2*(a2));
	}
	INLINE const VECT operator () (
			const REAL &h, 
			const VECT &a0, 
			const VECT &a1, 
			const VECT &a2, 
			const VECT &a3) 
	{
		REAL h2 = h * REAL(1./2.);
		REAL h3 = h * REAL(1./3.);
		return a0 + h*(a1 + h2*(a2 + h3*(a3)));
	}
	INLINE const VECT operator () (
			const REAL &h, 
			const VECT &a0, 
			const VECT &a1, 
			const VECT &a2, 
			const VECT &a3, 
			const VECT &a4) 
	{
		REAL h2 = h * REAL(1./2.);
		REAL h3 = h * REAL(1./3.);
		REAL h4 = h * REAL(1./4.);
		return a0 + h*(a1 + h2*(a2 + h3*(a3 + h4*(a4))));
	}
	INLINE const VECT operator () (
			const REAL &h, 
			const VECT &a0, 
			const VECT &a1, 
			const VECT &a2, 
			const VECT &a3, 
			const VECT &a4, 
			const VECT &a5) 
	{
		REAL h2 = h * REAL(1./2.);
		REAL h3 = h * REAL(1./3.);
		REAL h4 = h * REAL(1./4.);
		REAL h5 = h * REAL(1./5.);
		return a0 + h*(a1 + h2*(a2 + h3*(a3 + h4*(a4 + h5*(a5)))));
	}
	INLINE const VECT operator () (
			const REAL &h, 
			const VECT &a0, 
			const VECT &a1, 
			const VECT &a2, 
			const VECT &a3, 
			const VECT &a4, 
			const VECT &a5, 
			const VECT &a6) 
	{
		REAL h2 = h * REAL(1./2.);
		REAL h3 = h * REAL(1./3.);
		REAL h4 = h * REAL(1./4.);
		REAL h5 = h * REAL(1./5.);
		REAL h6 = h * REAL(1./6.);
		return a0 + h*(a1 + h2*(a2 + h3*(a3 + h4*(a4 + h5*(a5 + h6*(a6))))));
	}
	INLINE const VECT operator () (
			const REAL &h, 
			const VECT &a0, 
			const VECT &a1, 
			const VECT &a2, 
			const VECT &a3, 
			const VECT &a4, 
			const VECT &a5, 
			const VECT &a6, 
			const VECT &a7) 
	{
		REAL h2 = h * REAL(1./2.);
		REAL h3 = h * REAL(1./3.);
		REAL h4 = h * REAL(1./4.);
		REAL h5 = h * REAL(1./5.);
		REAL h6 = h * REAL(1./6.);
		REAL h7 = h * REAL(1./7.);
		return a0 + h*(a1 + h2*(a2 + h3*(a3 + h4*(a4 + h5*(a5 + h6*(a6 + h7*(a7)))))));
	}
	INLINE const VECT operator () (
			const REAL &h, 
			const VECT &a0, 
			const VECT &a1, 
			const VECT &a2, 
			const VECT &a3, 
			const VECT &a4, 
			const VECT &a5, 
			const VECT &a6, 
			const VECT &a7, 
			const VECT &a8) 
	{
		REAL h2 = h * REAL(1./2.);
		REAL h3 = h * REAL(1./3.);
		REAL h4 = h * REAL(1./4.);
		REAL h5 = h * REAL(1./5.);
		REAL h6 = h * REAL(1./6.);
		REAL h7 = h * REAL(1./7.);
		REAL h8 = h * REAL(1./8.);
		return a0 + h*(a1 + h2*(a2 + h3*(a3 + h4*(a4 + h5*(a5 + h6*(a6 + h7*(a7 + h8*(a8))))))));
	}
	INLINE const VECT operator () (
			const REAL &h, 
			const VECT &a0, 
			const VECT &a1, 
			const VECT &a2, 
			const VECT &a3, 
			const VECT &a4, 
			const VECT &a5, 
			const VECT &a6, 
			const VECT &a7, 
			const VECT &a8, 
			const VECT &a9)
	{
		REAL h2 = h * REAL(1./2.);
		REAL h3 = h * REAL(1./3.);
		REAL h4 = h * REAL(1./4.);
		REAL h5 = h * REAL(1./5.);
		REAL h6 = h * REAL(1./6.);
		REAL h7 = h * REAL(1./7.);
		REAL h8 = h * REAL(1./8.);
		REAL h9 = h * REAL(1./9.);
		return a0 + h*(a1 + h2*(a2 + h3*(a3 + h4*(a4 + h5*(a5 + h6*(a6 + h7*(a7 + h8*(a8 + h9*(a9)))))))));
	}
};

