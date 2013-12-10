#Port from Adventures in OpenCL Part1 to PyOpenCL
# http://enja.org/2010/07/13/adventures-in-opencl-part-1-getting-started/
# http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy
import pylab
import matplotlib;
import skimage
import skimage.io
import skimage.transform
import datetime

from operator import itemgetter
import sys
import math
import random
import os


file="bg.png"
demo=False;
reduction=2;
def getTime():
    return datetime.datetime.now();

INF = float("infinity")
class CLSeamCarving:
    '''
    CLSeamCarving class,
    performs the seam carving algorithm on an image to reduce its size without scaling
    its main features, using OpenCL in realtime
    '''
    def __init__(self,energyFunction=None):
        '''
        Inits the opencl environment and parameters
        Can provide a energyFunction to compute the energy using that here
        '''
        #profiling operation times
        self.times={k:datetime.timedelta() for k in ("execute","energy",
                    "backtrack","init","resize","shrink")};
        #show opencl compiler errors
        os.environ["PYOPENCL_COMPILER_OUTPUT"]="1";
        
        t=getTime();
        
        if (energyFunction is not None):
            self.energyFunction=energyFunction
        else:
            self.energyFunction=self._simpleEnergy;
        
        self.ctx = cl.create_some_context(False)
        self.queue = cl.CommandQueue(self.ctx)
        
        self.times["init"]+=getTime()-t;
        
    def loadProgram(self, filename):
        t=getTime();
        
#        f = open(filename, 'r')
#        fstr = "".join(f.readlines())
        fstr="""
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
        """
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()
        self.times["init"]+=getTime()-t;

    def loadImage(self,file):
        '''
        Loads the image from a filename into the inputImage and originalImage properties
        originalImage retains the initial image, for demonstration in show
        inputImage is frequently replaced and manipulated by the code
        '''
        t=getTime();
        self.inputImage = skimage.img_as_float(skimage.io.imread(file))
        self.originalImage=numpy.copy(self.inputImage) 
        self.loadProgram("seam.c")
        self.times["init"]+=getTime()-t;

    def _simpleEnergy(self,img):
        '''
        The simple built-in energy function, which is based on the brightness of pixels
        '''
        return img.mean(axis=2);
    
    def computeEnergy(self):
        self.energy= self.energyFunction(self.inputImage)
        
    def adjustBuffers(self):
        '''
        Recreates the buffers for opencl
        '''
        t=getTime();
        mf= cl.mem_flags
        self.inputBuf=cl.Buffer(self.ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.energy)
        self.outputBuf=cl.Buffer(self.ctx,mf.WRITE_ONLY,self.energy.nbytes)
        self.times["energy"]+=getTime()-t;
        

    def execute(self):
        '''
        Runs the dynamic programming algorithm for seam carving using openCL
        concurrently
        '''
        t=getTime();
        c = numpy.empty_like(self.energy)
#        print "Finding minimum carve...";
        self.program.seam_carving(self.queue, self.energy.shape, None, 
                                  numpy.int32(self.energy.shape[1]), #width
                                  numpy.int32(self.energy.shape[0]), #height
                                   self.inputBuf,  self.outputBuf)
        cl.enqueue_read_buffer(self.queue, self.outputBuf, c).wait()
        
        self.result=c;
        self.times["execute"]+=getTime()-t;

    def backtrack(self):
        '''
        backtracks through the computed array, finding the minimal path
        then removes it from the image
        '''
#        print "Backtrack..."
        t=getTime();
        energy=self.result;
        size=self.inputImage.shape;
        self.outputImage=numpy.ndarray((size[0]-1,size[1],size[2]));
        path=[];
        minIndex=0
        size=self.energy.shape;
        i=size[1]-1;
        for j in range(1,size[0]):  
            if (energy[j][i]<=energy[minIndex][i]):
                minIndex=j;
        path.append(minIndex);
        for i in range(1,size[1]):
            three=(INF if minIndex+y<0 or minIndex+y>=size[0] else energy[minIndex+y][i] for y in (-1,0,1))
            minIndex+= min(enumerate(three), key=itemgetter(1))[0]-1
            path.append(minIndex);
        path.reverse(); 
        self.times["backtrack"]+=getTime()-t;
        self.shrink(path);
    
    def shrink(self,path):
        '''
        removing the minimal path
        '''
        t=getTime();
        size=self.inputImage.shape;
#        newEnergy=numpy.ndarray((size[0]-1,size[1]));
        for i in range(0,size[1]):
            self.outputImage[0:path[i],i,]=self.inputImage[0:path[i],i,];
            self.outputImage[path[i]:,i,]=self.inputImage[path[i]+1:,i,];
#            newEnergy[0:path[i],i]=self.energy[0:path[i],i];
#            newEnergy[path[i]:,i]=self.energy[path[i]+1:,i];
#        self.energy=newEnergy;
        self.times["shrink"]+=getTime()-t;

    def swap(self):
        '''
        swaps the input image with output image, 
        preparing it for the next iteration
        '''
        self.inputImage=self.outputImage
        
    def show(self):
        '''
        shows times and images
        '''
        for i in self.times.keys():
            print i,":", (self.times[i].seconds*1000+self.times[i].microseconds/1000)/1000.0,"seconds"
        f=pylab.figure()
        f.add_subplot(1,2,0);    
        pylab.imshow(self.originalImage,cmap=matplotlib.cm.Greys_r);
        f.add_subplot(1,2,1);
        pylab.imshow(self.outputImage,cmap=matplotlib.cm.Greys_r);
        pylab.title("Seam Carving (by AbiusX)")
        pylab.show();

    def resize(self,img,newWidth,newHeight,show=False):
        '''
        resize an image to newWidth,newHeight
        '''
        t=getTime();
        self.loadImage(img);
        size=self.inputImage.shape;
        heightDifference=size[0]-newHeight;
        assert(heightDifference>=0);
        widthDifference=size[1]-newWidth;
        assert(widthDifference>=0);
        self.times["resize"]+=getTime()-t;
        
        for _ in range(0,heightDifference):
            self.computeEnergy();
            self.adjustBuffers();
            self.execute()
            self.backtrack();
            self.swap();
            
        t=getTime();
        self.swap()
        self.inputImage=numpy.rot90(self.inputImage);
        self.times["resize"]+=getTime()-t;

        for _ in range(0,widthDifference):
            self.computeEnergy();
            self.adjustBuffers();
            self.execute()
            self.backtrack();
            self.swap();
        
        t=getTime();
        self.swap()
        self.outputImage=numpy.rot90(self.inputImage,3);
        self.times["resize"]+=getTime()-t;
        
        if (show): self.show();
        return self.outputImage;


if __name__ == "__main__":
    if (len(sys.argv)>=2):
        if (len(sys.argv)!=4):
            print "Usage:",sys.argv[0]," inputImage.png 640x480 outputImage.png";
            exit(0);
        inputImage=sys.argv[1];
        size=[int(i) for i in sys.argv[2].split("x")];
        outputImage=sys.argv[3];
        carver= CLSeamCarving();
        print "This will take a few seconds...";
        skimage.io.imsave(outputImage,carver.resize(inputImage,size[0],size[1]))
    else:
        carver = CLSeamCarving()
        print "Please wait a few seconds...";
        carver.resize("bg.png", 640,480,True);
