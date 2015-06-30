#define ind2d(w,i,j) ((i)*(w)+j)
#define ind3d(h,w,i,j,k) ((i)*(h)*(w)+(j)*(w)+k)
#define ind4d(c,h,w,i,j,k,l) ((i)*(c)*(h)*(w)+(j)*(h)*(w)+(k)*(w)+l)
#define ind5d(c,h,w,d,i,j,k,l,m) ((i)*(c)*(h)*(w)*(d)+(j)*(h)*(w)*(d)+(k)*(w)*(d)+(l)*(d)+m)

#define cat3d(A)    A[0],A[1],A[2]
#define cat4d(A)    A[0],A[1],A[2],A[3]
#define cat5d(A)    A[0],A[1],A[2],A[3],A[4]
#define prod5d(A)   (A[0]*A[1]*A[2]*A[3]*A[4])
#define dims2strides5d(A) A[1]*A[2]*A[3]*A[4],A[2]*A[3]*A[4],A[3]*A[4],A[4],1

