#include <iomanip>
#include <getopt.h>
#include <stdio.h>
#include <string.h>



void testPooling(
        int tdims, int xDims[],
        int pdims, int poolDims[], int poolPad[], int poolStride[],
        int verbose, int compare
    );

void testXcorr(
        int tdims, int xDims[], int wDims[],
        int cdims, int convPad[], int convStride[], int convUpscale[], int verbose, int compare
        );

void cmdLine(int argc, char *argv[], int& mode, int& dims, int& verbose, int& compare){
    /// Command line arguments
    // Default value of the domain sizes
    static struct option long_options[] = {
        {"d", required_argument, 0, 'd'},
        {"m", required_argument, 0, 'm'},
        {"v", no_argument, 0, 'v'},
        {"c", no_argument, 0, 'c'},
    };
    // Process command line arguments
    int ac;
    for(ac=1;ac<argc;ac++) {
        int cmd;
        while ((cmd=getopt_long(argc,argv,"m:d:v:c",long_options,NULL)) != -1){
            switch (cmd) {
                case 'm':
                    mode = atoi(optarg);
                    break;

                case 'd':
                    dims = atoi(optarg);
                    break;

                case 'v':
                    verbose = 1;
                    break;

                case 'c':
                    compare = 1;
                    break;

                default:
                    printf("Usage: a.out [-m <mode>] [-d <dims>] [-v for verbose] [-c for comparison]\n");
                    exit(-1);
            }
        }
    }
}

int main(int argc, char *argv[]){
    int mode=0, dims=5, verbose=0, compare=0;
    cmdLine(argc, argv, mode, dims, verbose, compare);
    printf("%d %d %d\n", dims, verbose, compare);

    if(mode==0){ // xcorr
        printf("mode: xcorr\n");
        if(dims==5){
            int xDims[5] = {10,3,28,28,28};
            int wDims[5] = {2,3,5,5,5};
            int convPad[3] = {2,2,2};
            int convStride[3] = {1,1,1};
            int convUpscale[3] = {1,1,1};
            testXcorr(
                5, xDims, wDims,
                3, convPad, convStride, convUpscale, verbose, compare
                );
        }else if(dims == 4){
            int xDims[4] = {11,3,25,25};
            int wDims[4] = {5,3,5,6};
            int convPad[2] = {3,3};
            int convStride[2] = {1,1};
            int convUpscale[2] = {1,1};
            testXcorr(
                4, xDims, wDims,
                2, convPad, convStride, convUpscale, verbose, compare
                );
        }
    }else if(mode==1){ // max pool
        printf("mode: max pool\n");
        if(dims==4){
            int xDims[4] = {100,3,28,28}; // N C H W D

            int poolDims[2] = {5,5};
            int poolStride[2] = {5,5};
            int poolPad[2] = {0,0};
            testPooling( 4, xDims, 2, poolDims, poolPad, poolStride, verbose, compare);
        }else if(dims == 5){
            int xDims[5] = {100,3,28,28,28}; // N C H W D

            int poolDims[3] = {5,5,5};
            int poolStride[3] = {5,5,5};
            int poolPad[3] = {0,0,0};
            testPooling(5, xDims, 3, poolDims, poolPad, poolStride, verbose, compare);
        }else
            exit(-1);
    }

    return 0;
}

