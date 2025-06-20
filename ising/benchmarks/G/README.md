# MaxCut Benchmark: Gset

Gset is a benchmark suite of the MaxCut problem, created by Stefan E. Karisch in around 2003. The source of the original dataset can be found at: [here](https://web.stanford.edu/~yyye/yyye/Gset/).

Gset consists of the problems G1 to G81 and, due to some skipped numbers, has a total of 71 problems with a graph size ranging from 800 to 20,000 vertices.
There are graphs without weighted edges (all weights are 1) as well as graphs with weighted edges where the weights are either +1 or -1.
Geometrically, the structure of the graphs can be split into three categories:

- Random graphs
- Planar graphs
- Toroidal graphs

Here we also include K2000 (dense graph), so 72 testcases in total.

## Format explanation
In Gset, each file begins with a line defining the amount of nodes (N) and the amount of edges (E).

``
N E
``

From then on each edge is represented, on its own line, as follows:

``
node_i node_j weight_ij
``

Where ``weight_ij`` represents the weight of the edge between node `i` and node `j`.

## Best known values
Below the best known values of each benchmark are presented. For some benchmarks they are the actual global maximum cut values.

| **Benchmark** | **N**  | **E** | **Weight**  | **Type** | **Best Known Cut** |
| :-: | -: | -: | -: | -: | -: |
| G1 | 800 | 19,176 | +1 | random | 11,624 |
| G2 | 800 | 19,176 | +1 | random | 11,620 |
| G3 | 800 | 19,176 | +1 | random | 11,622 |
| G4 | 800 | 19,176 | +1 | random | 11,646 |
| G5 | 800 | 19,176 | +1 | random | 11,631 |
| G6 | 800 | 19,176 | +1, -1 | random | 2,178 |
| G7 | 800 | 19,176 | +1, -1 | random | 2,006 |
| G8 | 800 | 19,176 | +1, -1 | random | 2,005 |
| G9 | 800 | 19,176 | +1, -1 | random | 2,054 |
| G10 | 800 | 19,176 | +1, -1 | random | 2,000 |
| G11 | 800 | 1,600 | +1, -1 | toroidal | 564 |
| G12 | 800 | 1,600 | +1, -1 | toroidal | 556 |
| G13 | 800 | 1,600 | +1, -1 | toroidal | 582 |
| G14 | 800 | 4,694 | +1 | planar | 3,063 |
| G15 | 800 | 4,661 | +1 | planar | 3,050 |
| G16 | 800 | 4,672 | +1 | planar | 3,052 |
| G17 | 800 | 4,667 | +1 | planar | 3,047 |
| G18 | 800 | 4,694 | +1, -1 | planar | 992 |
| G19 | 800 | 4,661 | +1, -1 | planar | 906 |
| G20 | 800 | 4,672 | +1, -1 | planar | 941 |
| G21 | 800 | 4,667 | +1, -1 | planar | 931 |
| G22 | 2,000 | 19,990 | +1 | random | 13,359 |
| G23 | 2,000 | 19,990 | +1 | random | 13,342 |
| G24 | 2,000 | 19,990 | +1 | random | 13,337 |
| G25 | 2,000 | 19,990 | +1 | random | 13,340 |
| G26 | 2,000 | 19,990 | +1 | random | 13,328 |
| G27 | 2,000 | 19,990 | +1, -1 | random | 3,341 |
| G28 | 2,000 | 19,990 | +1, -1 | random | 3,298 |
| G29 | 2,000 | 19,990 | +1, -1 | random | 3,405 |
| G30 | 2,000 | 19,990 | +1, -1 | random | 3,413 |
| G31 | 2,000 | 19,990 | +1, -1 | random | 3,310 |
| G32 | 2,000 | 4,000 | +1, -1 | toroidal | 1,410 |
| G33 | 2,000 | 4,000 | +1, -1 | toroidal | 1,382 |
| G34 | 2,000 | 4,000 | +1, -1 | toroidal | 1,384 |
| G35 | 2,000 | 11,778 | +1 | planar | 7,680 |
| G36 | 2,000 | 11,766 | +1 | planar | 7,675 |
| G37 | 2,000 | 11,785 | +1 | planar | 7,685 |
| G38 | 2,000 | 11,779 | +1 | planar | 7,686 |
| G39 | 2,000 | 11,778 | +1, -1 | planar | 2,407 |
| G40 | 2,000 | 11,766 | +1, -1 | planar | 2,400 |
| G41 | 2,000 | 11,785 | +1, -1 | planar | 2,404 |
| G42 | 2,000 | 11,779 | +1, -1 | planar | 2,475 |
| G43 | 1,000 | 9,990 | +1 | random | 6,660 |
| G44 | 1,000 | 9,990 | +1 | random | 6,650 |
| G45 | 1,000 | 9,990 | +1 | random | 6,654 |
| G46 | 1,000 | 9,990 | +1 | random | 6,649 |
| G47 | 1,000 | 9,990 | +1 | random | 6,657 |
| G48 | 3,000 | 6,000 | +1, -1 | toroidal | 6,000 |
| G49 | 3,000 | 6,000 | +1, -1 | toroidal | 6,000 |
| G50 | 3,000 | 6,000 | +1, -1 | toroidal | 5,880 |
| G51 | 1,000 | 5,909 | +1 | planar | 3,848 |
| G52 | 1,000 | 5,916 | +1 | planar | 3,851 |
| G53 | 1,000 | 5,914 | +1 | planar | 3,849 |
| G54 | 1,000 | 5,916 | +1 | planar | 3,851 |
| G55 | 5,000 | 12,498 | +1 | random | 10,289 |
| G56 | 5,000 | 12,498 | +1, -1 | random | 4,008 |
| G57 | 5,000 | 10,000 | +1, -1 | toroidal | 3,480 |
| G58 | 5,000 | 29,570 | +1 | planar | 19,257 |
| G59 | 5,000 | 29,570 | +1, -1 | planar | 6,067 |
| G60 | 7,000 | 17,148 | +1 | random | 14,168 |
| G61 | 7,000 | 17,148 | +1, -1 | random | 5,777 |
| G62 | 7,000 | 14,000 | +1, -1 | toroidal | 4,844 |
| G63 | 7,000 | 41,459 | +1 | planar | 26,986 |
| G64 | 7,000 | 41,459 | +1, -1 | planar | 8,728 |
| G65 | 8,000 | 16,000 | +1, -1 | toroidal | 5,532 |
| G66 | 9,000 | 18,000 | +1, -1 | toroidal | 6,324 |
| G67 | 10,000 | 20,000 | +1, -1 | toroidal | 6,906 |
| G70 | 10,000 | 9,999 | +1 | random | 9,522 |
| G72 | 10,000 | 20,000 | +1, -1 | toroidal | 7,008 |
| G77 | 14,000 | 28,000 | +1, -1 | toroidal | 9,940 |
| G81 | 20,000 | 40,000 | +1, -1 | toroidal | 14,060 |
|K2000 | 2,000 | 1,999,000 | +1, -1 | dense | 33,337 |

The best-known cut for G72/G77/G81 is reported in the paper [24 May, 2025]: https://arxiv.org/abs/2505.18508. All others are reported in 2019 at [here](https://medium.com/toshiba-sbm/benchmarking-the-max-cut-problem-on-the-simulated-bifurcation-machine-e26e1127c0b0) by using SB solver.

Folder `./ccode` contains the source C code used to generate the Gset and the author's explanation.