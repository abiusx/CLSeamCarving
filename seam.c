unsigned int getIndex(const int x,const int y, const int width);
double MAX(const double a, const double b);

unsigned int getIndex(const int x,const int y, const int width)
{
	return (y*width+x);
}
double MAX(const double a, const double b)
{
	if (a>b) return a; else return b;
}
__kernel void seam_carving( const int width,const int height, __global double *input,__global double * output)
{
	unsigned int y = get_global_id(0);
	if (y>0) return; //only once per y
	unsigned int x = get_global_id(1);
	if (x!=0)
		mem_fence(CLK_GLOBAL_MEM_FENCE); //wait for previous column to be written before reading
	for (int y=0;y<height;++y)
	{
		if (x==0)
		{
			output[getIndex(x,y,width)]=input[getIndex(x,y,width)];
			continue;
		}
		if (y==0)
			output[getIndex(x,y,width)]=input[getIndex(x,y,width)]+ MAX(output[getIndex(x-1,y,width)],output[getIndex(x-1,y+1,width)]);
		else if (y==height-1)
			output[getIndex(x,y,width)]=input[getIndex(x,y,width)]+ MAX(output[getIndex(x-1,y,width)],output[getIndex(x-1,y-1,width)]);
		else
			output[getIndex(x,y,width)]=input[getIndex(x,y,width)]+ MAX(
					MAX(output[getIndex(x-1,y,width)],output[getIndex(x-1,y-1,width)]),
					output[getIndex(x-1,y+1,width)]
					);

	}
}
