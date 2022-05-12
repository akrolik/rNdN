# rNdN

rNdN is a runtime-optimized GPU compiler and database system, focusing on end-to-end performance of short-running queries.

## Design

### Compiler

### Assembler

### Runtime

## Installation

rNdN is distributed both as source code (this repository) and pre-built Docker images that accompany conference paper artefacts. Instructions for downloading and running the Docker images are included in the paper appendices.

### Requirements

1. A recent consumer-grade NVIDIA GPU is required for the complete compiler and assembler:
   - Pascal 10-series (sm_61)
   - Ampere 30-series (sm_86)
   - *Assembler support for Maxwell, Turing and Volta requires an updated scheduler profile. Otherwise, `--backend=ptxas` must be used.*
2. CUDA >= 11.3 and associated graphics driver

**Note**: Ubuntu 18.04 and 20.04 have been extensively tested. Other platforms may require additional setup or code changes.

**Build requirements**
1. Compiler supporting C++17
2. flex and bison tools
3. LLVM >= 14.0 (both `llvm-14` and `llvm-14-dev`)
4. CMake >= 3.21
5. `libpcre2-8` (distributed as part of `libpcre2-dev`)

**Data requirements**

For evaluating the [TPC-H benchmark suite](https://www.tpc.org/tpch/) version 2, data must be generated using the dbgen tool. The queries are included as part of this repository.

### Building

```
# Clone repository
git clone <repository path>
cd rNdN

# Setup build environment
mkdir build
cd build
cmake ..

# Build the project
make -j
```

### Usage examples

rNdN executes queries and programs written in HorseIR, and accepts configuration options for debugging, optimization, and data.

```
./rNdN [OPTION...] <filename>
```

**Full list of options**
```
./rNdN --help
```

**TPC-H query data loading**
```
./rNdN --data-scale-tpch=<data scale> --data-load-tpch=<data path> <filename>
```

**Other useful options**
- `--debug-time`: Profile compilation and execution
- `--debug-print`: Output detailed debug logging information
- `--debug-options`: Output current options, including defaults

## Citations

Alexander Krolik, Clark Verbrugge, and Laurie Hendren. r3d3: Optimized query compilation on GPUs. In 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), pages 277–288, 2021. [doi:10.1109/CGO51591.2021.9370323](https://doi.org/10.1109/CGO51591.2021.9370323).

```
@inproceedings{Krolik2021:r3d3,
    author    = {Krolik, Alexander and Verbrugge, Clark and Hendren, Laurie},
    booktitle = {2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)},
    title     = {r3d3: Optimized Query Compilation on GPUs},
    year      = {2021},
    volume    = {},
    number    = {},
    pages     = {277-288},
    doi       = {10.1109/CGO51591.2021.9370323}
}
```

Alexander Krolik. rNdN: Optimized query compilation for GPUs. PhD thesis, McGill University, May 2022. URL: TODO.

```
@phdthesis{Krolik2022:Thesis,
    author = {Krolik, Alexander}, 
    title  = {{rNdN}: Optimized Query Compilation for GPUs},
    school = {McGill University},
    year   = {2022},
    month  = {May},
    url    = {}
}
```

## Acknowledgements

This work was completed at the [School of Computer Science, McGill University](https://cs.mcgill.ca/) under supervision of [Professor Clark Verbrugge](https://www.sable.mcgill.ca/~clump/) and [Professor Laurie Hendren](https://www.sable.mcgill.ca/~hendren/).

> We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC).
> 
> Nous remercions le Conseil de recherches en sciences naturelles et en génie du Canada (CRSNG) de son soutien.
