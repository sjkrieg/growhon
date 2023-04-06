# GrowHON
![Toy example of growing a HON from sequence input](https://raw.githubusercontent.com/sjkrieg/growhon/master/toyhon.png)
Python implementation of GrowHON for generating a Higher-Order Network (HON) from sequence input. GrowHON accepts any text file as input, and processes each line as a sequence vector. The output is a weighted adjacency list in CSV format.

## Getting Started
GrowHON can either be installed and imported as a Python package, or downloaded and run as a standalone script.

### Core Prerequisites
1. Python 3 (3.7+ for best performance if using multiple cores)
2. numpy (developed on 1.16.2)
3. py-cpuinfo 5.0.0+ (for logging)

### Additional Prerequisites for Parallelism
4. Ray 0.7.1+

## Example 1
Generates a higher-order network with max order 3 from airport_sample_sequences.txt, and writes the resulting network to airport_sample_hon.csv.
```
python growhon.py airport_sample_sequences.txt airport_sample_hon.txt 3
```

## Required (positional) Arguments:
```
  infname               source path + file name
  otfname               destination path + file name
  k                     max order to use in growing the HON
```

Required arguments must be specified in the correct order and immediately following the python command.

## Optional Arguments:
```
  -h, --help            show this help message and exit
  -w NUMCPUS, --numcpus NUMCPUS
                        number of workers (integer; default 1)
  -p INFNUMPREFIXES, --infnumprefixes INFNUMPREFIXES
                        number of prefixes for input vectors (integer; default 1)
                        GrowHON will skip this number of values at the beginning of each line in the input file
                        generally, this should be 1 if each vector has some kind of ID at the beginning
  -di INFDELIMITER, --infdelimiter INFDELIMITER
                        delimiter for entities in input vectors (char; default " ")
                        this is the character by which GrowHON delimits entities in each input vector
  -do OTFDELIMITER, --otfdelimiter OTFDELIMITER
                        delimiter for output network (char; default " ")
  -s SKIPPRUNE, --skipprune SKIPPRUNE
                        whether to skip the prune phase (bool; default False)
  -t TAU, --tau TAU
                        threshold multiplier for determining dependencies (float; default 1.0)
                        higher values mean that dependencies must exceed a higher threshold, so the resulting network will be smaller
  -m MINSUPPORT, --minsupport MINSUPPORT
                        sample frequency for considering whether a pattern is a dependency (int; default 1)
                        higher values mean that dependencies must occur more frequently for their KLD to be considered, so the resulting network will be smaller
  -e DOTFNAME, --dotfname DOTFNAME
                        destination path + file for divergences (string; default None)
                        if a value if is provided, the KLD values for each sequence will be written as a CSV file
                        this will slow execution
  -o LOGNAME, --logname LOGNAME
                        location to write log output (string; default None)
                        if None, all log messages are printed to console
  -lsg LOGISGROW, --logisgrow LOGISGROW
                        logging interval for sequential growth (integer; default 1000)
                        this value determines how often the driver logs its current progress during the grow phase (for sequential mode)
                        as an integer, it represents the number of records processed
  -lpg LOGIPGROW, --logipgrow LOGIPGROW
                        logging interval for parallel growth (float; default 0.2)
                        this value determines how often each worker logs its current progress during the grow phase (for parallel mode)
                        as a float, it represents a fraction of that worker's partition of the input file
  -lpr LOGIPRUNE, --logiprune LOGIPRUNE
                        logging interval for pruning (float; default 0.2)
                        this value determines how often the driver logs its progress during the prune phase
                        as a float, it represents the fraction of nodes at each level in the tree
  -v, --verbose         print more messages for debugging
```

Optional arguments can be utilized by specifying the appropriate flag followed by a space and the desired value. Optional arguments can be used in any order as long as they all come after the required (positional) arguments.

## Example 2
A modification of Example 1 that utilizies 4 cpus (-w 4).
```
python growhon.py airport_sample_sequences.txt airport_sample_hon.txt 3 -w 4
```

## Example 3
A modification of Example 2 that sets the number of input vector prefixes to 0 (-p 0), increases the threshold multiplier to reduce the number of edges (-t 5.0), and writes the log output to "airport.log" (-o airport.log).
```
python growhon.py airport_sample_sequences.txt airport_sample_hon.txt 3 -w 4 -p 0 -t 5.0 -o airport.log
```

Example 3 was used to generate the "airport_sample_hon.txt" file in this repository.

## Importing GrowHON as a Package
You can also import GrowHON into your own script (first save growhon.py locally) like the following:
```
import growhon
t = growhon.HONTree(3) # max order of 3
t.grow('airport_sample_sequences.txt')
t.prune()
t.extract('airport_sample_hon.txt')
```

Alternatively, you can call the grow and prune methods as part of the tree constructor by providing an input file name:
```
import growhon
t = growhon.HONTree(3, 'airport_sample_sequences.txt')
t.extract() # prints to console if no output file is provided
```

## Help

Additional information is provided in the module docstring:
```
>>> import growhon
>>> print(help(growhon))
```

## Authors

* **Steven Krieg** - (skrieg@nd.edu)

## Reference
If you make use of this code, please cite our paper!

* Krieg, Steven J., Peter Kogge, and Nitesh V. Chawla. "GrowHON: A Scalable Algorithm for Growing Higher-order Networks of Sequences." In *International Conference on Complex Networks and Their Applications* (accepted for Dec 2020). Springer, Cham.

Publications that utilize GrowHON include:
* Krieg, Steven J., William C. Burgis, Patrick M. Soga, Nitesh V. Chawla. “Deep ensembles for graphs with higher-order dependencies.” (*International Conference on Learning Representations*, 2023).
* Krieg, Steven J., Daniel H. Robertson, Meeta P. Pradhan, and Nitesh V. Chawla. “Higher-order Networks of Diabetes Comorbidities: Disease Trajectories that Matter.” (*Proceedings of the IEEE International Conference on Healthcare Informatics*, 2020).

This algorithm is based on the node-splitting paradigm introduced by Xu et al., and grows the same type of network but in a more accurate and scalable fashion. Their paper and code repository can be found at:

* Xu, J., Wickramarathne, T. L., & Chawla, N. V. (2016). Representing higher-order dependencies in networks. *Science advances*, 2(5), e1600028.
* https://github.com/xyjprc/hon
