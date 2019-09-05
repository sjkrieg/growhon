#!/usr/bin/env python
"""Module to generate a higher-order network (HON).

Exposed methods:
    grow(): grows a tree from a list of input vectors
    prune(): prunes the insignificant sequences
    extract(): converts the tree into a weighted edgelist
    get_divergences(): returns a list of all divergences

How to use:
    GrowHON can be used in the following ways:
    1)  Executing as a standalone script via:
        $ python growhon.py input_file output_file max_order
        For additional help with arguments, run:
        $ python growhon.py -h
    2) Importing as a module. See API details below.
"""

# =================================================================
# MODULES REQUIRED FOR CORE FUNCTIONALITY
# =================================================================
from collections import defaultdict, deque
from itertools import islice
from math import log, ceil
from multiprocessing import cpu_count
import numpy as np
# =================================================================
# =================================================================
# MODULES REQUIRED FOR PARALLELIZATION
# =================================================================
# ray is currently only available on linux
from importlib.util import find_spec
ray_available = find_spec('ray')
if ray_available:
    import ray
    # import full sys or SMap triggers a pickling error on version_info
    import sys
    from threading import Thread
# =================================================================
# =================================================================
# MODULES USED FOR LOGGING AND DEBUGGING
# =================================================================
from os import getpid
from psutil import Process
from time import perf_counter, strftime, gmtime
import cpuinfo
# import logging last or other modules may adjust the logging level
import logging
# =================================================================

__version__ = '0.1'
__author__ = 'Steven Krieg'
__email__ = 'skrieg@nd.edu'

class HONTree():
    """Main module for growing a HON.
    Exposes three methods: grow(), prune(), extract().
    """
    
    def __init__(self, 
                 max_order, 
                 inf_name=None, 
                 num_cpus=0, 
                 inf_delimiter=' ', 
                 inf_num_prefixes=1, 
                 order_delimiter='|', 
                 prune=True,
                 top_down=False,
                 threshold_multiplier=1.0, 
                 min_support=1, 
                 log_name=None, 
                 seq_grow_log_step=1000, 
                 par_grow_log_interval=0.2, 
                 prune_log_interval=0.2, 
                 verbose=False):
        """Init method for GrowHON.
        
        Description:
            __init__ creates a new instance of HONTree with the given
            parameters. It first creates a root for the tree.
            If inf_name is supplied, it calls grow(). If grow() is 
            called and prune==True, it then calls prune().
        
        Parameters:
            max_order (int): specifies the max order of the HON
            inf_name (str or None): input file, call grow() if provided
            num_cpus (int): specifies the number of worker processes
            inf_delimiter (char): delimiting char for input vectors
            inf_num_prefixes (int): skip this many items in each vector
            order_delimiter (char): delimiter for higher orders
            prune (bool): whether to call prune() during init()
            threshold_multiplier (float): multiplier for KLD in prune()
            min_support (int): min frequency for higher order sequences
            log_name (str or None): where to write log output
            seq_grow_log_step (int): heartbeat interval for logging
            par_grow_log_interval (float): heartbeat interval as %
            prune_log_interval (float): heartbeat interval for prune()
            verbose (bool): print extra messages for debugging
        """
        
        logging.basicConfig(level=logging.INFO, filename=log_name,
                            format='%(levelname)-8s  %(asctime)23s  ' + 
                            '%(runtime)10s  %(rss_size)6dMB  %(message)s')
        self.logger = logging.LoggerAdapter(logging.getLogger(), self._LogMap())
        self.logger.info('Initializing GrowHON on CPU {}'.format(cpuinfo.get_cpu_info()['brand']))
        # root is essentially a dummy node
        self.root = self._HONNode(tuple())
        self.max_order = max_order
        self.nmap = defaultdict(int)
        self.inf_delimiter = inf_delimiter
        self.inf_num_prefixes = inf_num_prefixes
        self.seq_grow_log_step = seq_grow_log_step
        self.par_grow_log_interval = par_grow_log_interval
        self.prune_log_interval = prune_log_interval
        self.min_support = min_support
        self._HONNode.order_delimiter = order_delimiter
        self.threshold_multiplier = threshold_multiplier
        self.verbose = verbose

        self.num_cpus = num_cpus if num_cpus else cpu_count()
        # if ray is unavailable, default to 1 worker
        if self.num_cpus > 1 and not ray_available:
            self.logger.warn('Ray is unavailable; parallel mode disabled.')
            self.num_cpus = 1
        if inf_name: 
            self.grow(inf_name)
            if prune: 
                self.prune(top_down)

    # =================================================================
    # BEGIN EXPOSED FUNCTIONS FOR GROWING, PRUNING, AND EXTRACTING
    # =================================================================
    def grow(self, inf_name, num_cpus=None, inf_delimiter=None, inf_num_prefixes=None):
        """The main method used to grow the HONTree.
        
        Description:
            This method delegates to either _grow_sequential or
            _grow_parallel, depending on the value of num_cpus. It
            first checks for a user-supplied value. If there is none,
            it calls multiprocessing.cpu_count().
            
        Parameters: 
            inf_name (str): input file, required
            num_cpus (int): number of worker processes to initiate
            inf_delimiter (char): delimiter for input vectors
            inf_num_prefixes (char): items to skip in input vectors
        
        Returns:
            None
        """
        # reassign parameters if they were passed explicitly
        if num_cpus: self.num_cpus = num_cpus
        if inf_delimiter: self.inf_delimiter = inf_delimiter
        if inf_num_prefixes: self.inf_num_prefixes = inf_num_prefixes
        
        self.logger.info('Growing tree with {} and max order {}...'.format(str(self.num_cpus) 
                    + ' worker' + ('s' if self.num_cpus > 1 else ''), self.max_order))
        if self.num_cpus > 1:
            self._grow_parallel(inf_name) 
        else:
            self._grow_sequential(inf_name)
        self.logger.info('Growing successfully completed!')
        if self.verbose: self.logger.info('Tree dump:\n{}'.format(self))

    def prune(self, top_down=False, threshold_multiplier=None, min_support=None):
        """Uses statistical methods to prune the grown HONTree.
        
        Description:
            This method checks all nodes in the tree (except those in
            the first and last levels) and uses relative entropy
            to determine whether the higher order or lower order
            sequences should be preserved. Nodes that are not preserved
            have their in-degrees reduced, possibly to 0.
            
        Parameters:
            bottom_up (bool): perform pruning on bottom levels first
            threshold_multiplier (float): multiplier for KLD threshold
            min_support (int): min frequency for higher order sequences
        
        Returns:
            None
        """
        # reassign parameters if they were passed explicitly
        if threshold_multiplier: self.threshold_multiplier = threshold_multiplier
        if min_support: self.min_support = min_support

        self._prune_top_down() if top_down else self._prune_bottom_up()
        if self.verbose: self.logger.info('Tree dump:\n{}'.format(self))
    
    def extract(self, otf_name=None, delimiter=' '):
        """Converts the HONTree into a weighted edgelist.
        
        Description:
            This method uses a recursive helper to traverse the tree
            and convert each node into an edge. Checks are made to
            preserve the integrity of flow in the network. Any orphaned
            edges are "adopted" by having their weight combined with
            another suitable edge.
            
        Parameters: 
            otf_name (str or None): if provided, write edges as a CSV
            delimiter (char): delimiter for the edgelist
        
        Returns:
            A list of strings, each representing a weighted edge.
        """
        self.logger.info('Extracting edgelist...')
        # using a shared dictionary saves time on merging return values
        edgelist = defaultdict(int)
        # __extract_helper modifies edgelist in place
        self.__extract_helper(self.root, edgelist)
        self.logger.info('Extracted {:,d} edges. Formatting...'.format(len(edgelist.keys())))
        
        # edgelist is structured as (u, v): w, where
        # u is the source node, v is the destination node
        # and w is the edge weight
        result = [edge[0] + delimiter + edge[1] + delimiter 
                  + str(weight) for edge, weight in edgelist.items()]

        if otf_name:
            with open(otf_name, 'w+') as otf: otf.write('\n'.join(result))
            self.logger.info('Edgelist successfully written to {}.'.format(otf_name))
        self.logger.info('Edgelist extraction successfully completed.')
        if not otf_name: 
            return result
    
    def get_divergences(self, otf_name=None):
        """Measures the KLD for all nodes in the tree. Not used
        directly by GrowHON, but useful for supplemental information.
  
        Parameters: 
            None
            
        Returns:
            A list of node labels and their KLD values.
        """
        self.logger.info('Calculating divergences...')
        divergences = defaultdict(float)
        self.__get_divergences_helper(self.root, divergences)
        result = [str(node) + ',' + '{:.6f}'.format(d) for node, d in divergences.items()]
        if otf_name:
            with open(otf_name, 'w+') as otf: otf.write('\n'.join(result))
            self.logger.info('Divergences successfully written to {}.'.format(otf_name))
        self.logger.info('Divergence calculations successfully completed.')
        return result
    # =================================================================
    # END MAIN FUNCTIONS
    # =================================================================
    # =================================================================
    # BEGIN SEQUENTIAL MODE FUNCTIONS
    # =================================================================
    # if the child does not already exist, add it
    # returns a reference to the child node
    def _add_child(self, parent, child_label):
        """Insert a sequence into the tree.
        
        Description:
            If the sequence does not yet exist in the tree, we first
            create a new _HONNode object. If it does exist, we can
            simply increment its in-degree and its parent's out-degree.
            All nodes are also referenced in nmap by their unique label
            to enable fast lookup to lower-orders.

        Parameters: 
            parent (_HONNode): reference to the parent of the new child
            
        Returns:
            child (_HONNode): reference to the child node
        """
        if child_label in parent.children:
            child = parent.children[child_label]
        else:
            child = self._HONNode(parent.name + (child_label,), parent)
            parent.children[child_label] = child
            if child.order < self.max_order:
                self.nmap[child.name] = child
        parent.out_deg += 1
        parent.out_freq += 1
        child.in_deg += 1
        child.in_freq += 1
        return child

    def _grow_sequential(self, inf_name):
        """Process an entire input file using 1 worker.
        
        Description:
            This method iterates over each line in an input file,
            and calls the _grow_vector method for each one.

        Parameters:
            inf_name (str): the input file
            
        Returns:
            None (the tree is modified in-place and accessed from root)
        """
        
        with open(inf_name,'r') as inf:
            for i, line in enumerate(inf):
                if not i % self.seq_grow_log_step: 
                    self.logger.info('Processing input line {}...'.format(i))
                self._grow_vector(line)

    def _grow_vector(self, vec):
        """Process a single vector from the input.
        
        Description:
            This method processes a single input vector by generating
            all sequential combinations up to length max_order. It
            does include the "tail" of the sequence, meaning the
            sequences at the very end of the vector which may have
            a shorter length. It passes each sequence to the _add_child
            method to insert into the tree.

        Parameters: 
            line (str): an input vector
            
        Returns:
            None
        """
        # we build the tree 1 level higher than max_order
        # so max_order + 1 is required
        q = deque(maxlen = self.max_order+1)
        vec = vec.strip().split(self.inf_delimiter)[self.inf_num_prefixes:]
        # prime the queue
        # e.g. we have line=='1 2 3 4 5' && max_order==2 -> q=[1,2,3]
        for e in vec[:self.max_order+1]:
            q.append(e)
        # loop through the rest of the sequence
        for e in vec[self.max_order+1:]:
            # for each combination, add the sequence to the tree
            # e.g. q==[1,2,3]
            #   -> insert 1 as child of root
            #   -> insert 2 as child of root->1
            #   -> insert 3 as child of root->1->2
            cur_parent = self.root
            for node in q:
                cur_parent = self._add_child(cur_parent, node)
            # move forward in the sequence
            # e.g. q==[1,2,3] & next item in line==4 -> q==[2,3,4]
            q.popleft()
            q.append(e)
        # add the "tail" of the sequence
        # this could be ommitted if you want to truncate the vectors
        while q:
            cur_parent = self.root
            for node in q:
                cur_parent = self._add_child(cur_parent, node)
            q.popleft()

    # =================================================================
    # END SEQUENTIAL MODE FUNCTIONS
    # =================================================================
    # =================================================================
    # BEGIN PARALLEL MODE FUNCTIONS
    # =================================================================
    def _get_partitions(self, inf_name):
        """Partition the input file for multiple workers.
        
        Description:
            This method reads the entire input file, then divides it
            into equal chunks for each worker.

        Parameters:
            inf_name (str): the input file
            
        Returns:
            inf_partitions (list): a list of slices to input chunks
        """
        # need to read the whole file to know its length
        # equal-sized partitions seem to work best
        with open(inf_name,'r') as inf:
            lines = inf.readlines()
        intervals = [0] + [i for i in range(1, self.num_cpus+1)]
        interval_step = ceil(len(lines) / self.num_cpus)
        inf_partitions = [lines[(intervals[i] * interval_step):(intervals[i+1]*interval_step)]
                          for i in range(len(intervals)-1)]
        return inf_partitions

    def _grow_parallel(self, inf_name):
        """The manager function for growing the tree in parallel.
        
        Description:
            This method oversees the parallel growth process. It
            first partitions the input, then initializes worker
            processes using Ray. It also initializes a listener
            thread to log messages from workers. Once the workers
            have been initialized, this method blocks until it hears
            that one of them is finished. It then receives the
            sequences and starts a separate thread to insert them into
            the tree. Meanwhile, it continues to wait for other
            results. If another worker finishes, this method blocks
            until either a) the growing thread is finished, or b)
            a second worker finishes. If a) this method receives the
            new result and begins growing. If b) this method instructs
            the worker to combine results with another worker. This
            method terminates when no more workers are active and
            all results have been inserted into the tree.

        Parameters:
            inf_name (str): the input file
            
        Returns:
            None (tree is grown in-place and accessed from the root)
        """
        self.logger.info('MGR: Partitioning input file for {} workers...'.format(self.num_cpus))
        inf_partitions = self._get_partitions(inf_name)
        self.logger.info('MGR: Partitioning successfully completed. Partition sizes: {}'
                    .format([len(p) for p in inf_partitions]))
        self.logger.info('MGR: Initializing ray...')
        if not ray.is_initialized():
            ray.init(logging_level=50, logging_format='%(levelname)-8s  %(asctime)23s' + 
                                                      (' ' * 24) + 'Ray: %(message)s')
        self.logger.info('MGR: Ray successfully initialized. Creating jobs for {} workers...'
                    .format(len(inf_partitions)))
        # create 1 worker for each partition
        # this dict is used to map ray actor ID to worker ID as int
        workers = {_SMap.remote(self.max_order, 
                               inf_delimiter=self.inf_delimiter, 
                               inf_num_prefixes=self.inf_num_prefixes, 
                               log_interval=self.par_grow_log_interval
                               ): i + 1 for i, p in enumerate(inf_partitions)}
        # jobs is updated to to track the number of active jobs
        # and which worker is working on them
        jobs = {worker.populate_smap.remote(p): worker
                for worker, p in zip(workers.keys(), inf_partitions)}
        listener = Thread(target=self._signal_listener, args=(jobs, workers), daemon=True)
        listener.start()
        grower = None
        
        # continue to run until all the jobs have completed
        while jobs:
            active_jobs = list(jobs.keys())
            self.logger.info('MGR: Waiting to hear from a worker...')
            # block until there is at least 1 result available
            # this call will only return 1 completed job id
            ready_ids, nready_ids = ray.wait(active_jobs, num_returns=1)
            
            if len(ready_ids):
                # this call gets the ids of all completed jobs
                ready_ids, nready_ids = ray.wait(active_jobs, num_returns=len(active_jobs), 
                                                 timeout=0.0)
                # if only one job is complete, we can grow
                # otherwise we instruct a remote worker to absorb
                # in general it will always be faster to absorb
                # when possible, but this could add a little overhead
                # if the last 2 SMaps are ready at the same time.
                # in that case, the driver will be waiting for a while.
                if len(ready_ids) == 1:
                    # growing is a CPU bound task, so having 2 growers
                    # running simultaneously will not improve runtime
                    # best scenario is to block
                    if grower and grower.is_alive():
                        self.logger.info('MGR: Smap is ready from worker {}, but driver is busy ' 
                                    + 'growing - blocking...'.format(workers[jobs[ready_ids[0]]]))
                        while grower.is_alive() and len(ready_ids) < 2:
                            ready_ids, nready_ids = ray.wait(active_jobs, 
                                                             num_returns=min(2, len(active_jobs)), 
                                                             timeout=0.0)
                    else:
                        self.logger.info('MGR: Smap is ready from worker {} - receiving...'
                                    .format(workers[jobs[ready_ids[0]]]))
                        # read the smap from the ray object store
                        smap = ray.get(ready_ids[0])
                        self.logger.info('MGR: Received smap from worker {} - growing tree ' 
                                    .format(workers[jobs[ready_ids[0]]])
                                    + 'with {:,d} sequences...'.format(len(smap)))
                        # start the grower in a separate thread so the
                        # driver can continue to manage other workers
                        grower = Thread(target=self._grow_from_smap, args=(smap,))
                        grower.start()
                        jobs.pop(ready_ids[0])
                elif len(ready_ids) > 1:
                    self.logger.info('MGR: Smaps are ready from workers {} and {} - combining on '
                                 + 'worker {}...'
                                .format(workers[jobs[ready_ids[0]]], workers[jobs[ready_ids[1]]], 
                                        workers[jobs[ready_ids[1]]]))
                    # the most recently finished worker is most likely
                    # to have a larger smap, so we run on that worker
                    # to potentially save time on serialization
                    worker = jobs.pop(ready_ids[-1])
                    jobs.pop(ready_ids[0])
                    new_job = worker.absorb_smap.remote(ready_ids[0])
                    jobs[new_job] = worker            
        # end while jobs
        
        # in case grower is still running
        if grower and grower.is_alive():
            grower.join()
            
        ray.shutdown()
    # end _grow_parallel
    
    def _grow_from_smap(self, incoming_smap):
        """Grows a tree from a list of sequences from a remote worker.
        
        Description:
            This method receives a list of sequences from a remote
            worker and uses them to grow the tree. The sequences
            are received as a numpy array (for fast serialization)
            and can be an arbitrary length. The last number is the
            frequency (in-degree) and all numbers before that are
            ordered tuples representing the sequence.

        Parameters:
            inf_name (np.array): incoming_smap
            
        Returns:
            None (tree is grown in-place and accessed from the root)
        """
        log_step = self.par_grow_log_interval * len(incoming_smap)
        for i, sequence in enumerate(incoming_smap):
            if not i % log_step:
                self.logger.info('MGR: Growing sequence {:,d}/{:,d}.'.format(i, len(incoming_smap)))
        
            # we receive each sequence as a numpy array
            # we don't know its length, just that
            # the last element contains the degree and
            # everything before that is the sequence itself
            node = tuple(sequence[:-1])
            deg = sequence[-1]
            
            # if the node already exists, increment its degree
            if node in self.nmap:
                obj = self.nmap[node]
                obj.in_deg += deg
                obj.in_freq += deg
            else:
                # the parent MUST already exist before inserting child
                # i.e. if the smap is not sorted, this will fail
                obj = self._HONNode(name=node, 
                              parent=self.nmap[node[:-1]] if len(node) > 1 else self.root)
                obj.in_deg = obj.in_freq = deg
                obj.parent.children[node[-1]] = obj
                self.nmap[node] = obj
            obj.parent.out_deg += deg
            obj.parent.out_freq += deg
        self.logger.info('MGR: Growing from smap successfully completed.')

    def _signal_listener(self, jobs, workers):
        """Listens and logs messages from workers.
        
        Description:
            This method is initiated via a daemon from _grow_parallel.
            It uses Ray's signal API to receive messages from workers
            and write them to the logger. It continues to run until
            the jobs list is empty (managed from _grow_parallel).

        Parameters:
            jobs (list of Ray object IDs): the active jobs
            workers (dict): maps Ray actor IDs to interpretable IDs
            
        Returns:
            None
        """
        try:
            while jobs:
                signal = ray.experimental.signal.receive(list(workers.keys()))
                for actor_id, msg in signal:
                    self.logger.info('W{:02d}: {}'.format(workers[actor_id], msg))
        # cleanest way to terminate this thread seems to be
        # waiting for ray to close the signal socket
        except:
            pass
    
    # =================================================================
    # END PARALLEL FUNCTIONS
    # =================================================================
    # =================================================================
    # BEGIN SUPPORTING (_ and __) FUNCTIONS
    # =================================================================
    def __str__(self):
        """Calls a recursive helper to traverse and print the tree.
        """
        return self.__str_helper(self.root)
    
    def __str_helper(self, node):
        """A recursive DFS helper for printing the tree.
        """
        if node:
            s = node.dump() if node != self.root else ''
            for child in node.children.values():
                s += self.__str_helper(child)
            return s

    def _delete_node(self, node):
        """Reduces the incoming node's in-degree to 0.
        Also reduces its parent's out-degree by the same number.
        The node is not deleted from memory.
  
        Parameters: 
            node (_HONNode): the node to delete
        """
        if self.verbose: self.logger.info('deleting node {}'.format(node))
        node.parent.out_deg -= node.in_deg
        node.in_deg = 0
        
    def __extract_helper(self, node, edgelist):
        """Used by the extract() method to find the edge destination.
        
        Description:
            Each edge must be checked to ensure the integrity of flow
            in the HON is preserved. If an edge destination would
            result in a "dead end," the edge is redirected to the
            original destination's lower-order counterpart.
  
        Parameters: 
            node: the node whose destination we want to verify
            
        Returns:
            The label to be included as the edge destination
        """
        for child in node.children.values():
            self.__extract_helper(child, edgelist)
        # if node.in_deg==0, it was pruned
        # if node.order==0, it is not a sequence destination
        if node.in_deg > 0 and node.order > 0:
            # if node.parent.in_deg==0, node is an orphan
            if node.parent.in_deg > 0:
                edgelist[(node.parent.get_label_full(), self._get_edge_dst(node))] += node.in_deg
            else:
                edgelist[self._get_edge_orphan(node)] += node.in_deg

    def _has_dependency(self, hord_node):
        """Decides whether a higher-order node should be preserved.
        
        Description:
            This method measures the KLD (Kullback-Leibler Divergence
            or relative entropy) of a higher-order node with respect
            to its lower-order counterpart. If the KLD exceeds
            the threshold as defined by _get_divergence_threshold,
            the higher-order node is preserved and the lower-order
            pruned. If not, the higher-order node is pruned. The higher
            order node must also exceed the value of min_support.
  
        Parameters: 
            hord_node (_HONNode): the higher-order node to check
            
        Returns:
            True if the higher-order node should be preserved;
            False otherwise
        """
        if hord_node.in_freq >= self.min_support:
            divergence = self._get_divergence(hord_node)
            threshold = self._get_divergence_threshold(hord_node)
            if self.verbose: 
                self.logger.info('{} -> {}'
                            .format(str(hord_node), str([(k, v.in_deg) 
                            for k, v in hord_node.children.items()])))
                self.logger.info('{} -> {}'
                            .format(str(self._get_lord_match(hord_node)), str([(k, v.in_deg)
                            for k, v in self._get_lord_match(hord_node).children.items()])))
                self.logger.info('dvrg:{} ; thrs: {}'.format(divergence, threshold))
                if divergence > threshold:
                    self.logger.info('{} has a dependency.'.format(str(hord_node)))
                else:
                    self.logger.info('{} does not have a dependency.'.format(str(hord_node)))
                    
            return (divergence > threshold)
        else: return False

    def _get_divergence(self, hord_node):
        """Measures the KLD (relative entropy) of a higher-order node.
  
        Parameters: 
            hord_node (_HONNode): the higher-order node to check
            
        Returns:
            The KLD (Kullback-Leibler divergence or relative entropy)
            of the higher-order node, as a floating point number.
        """
        lord_node = self._get_lord_match(hord_node)
        divergence = 0.0
        for child_label, child in hord_node.children.items():
            hord_conf = child.in_freq / hord_node.out_freq
            divergence += hord_conf * log((hord_conf * lord_node.out_freq)
                                          / lord_node.children[child_label].in_freq, 2)
        return divergence

    def _get_divergence_threshold(self, hord_node):
        """Determines the KLD threshold for a higher-order node.
  
        Parameters: 
            hord_node (_HONNode): the higher-order node to check
            
        Returns:
            The threshold the node's KLD must exceed to be preserved.
        """
        return self.threshold_multiplier * (1 + hord_node.order) / log(1 + hord_node.in_freq, 2)

    def __get_divergences_helper(self, node, divergences):
        """Recursive helper for get_divergences().
  
        Parameters: 
            node (_HONNode): the current node
            divergences (defaultdict): KLD values for all nodes
            
        Returns:
            None (divergences is passed via shared object reference)
        """
        for child in node.children.values():
            self.__get_divergences_helper(child, divergences)
        if node.order < self.max_order and node.order > 0:
            divergences['->'.join([str(n) for n in node.name])] = self._get_divergence(node)

    def _get_edge_dst(self, node):
        """Used by the extract() method to find the edge destination.
        
        Description:
            Each edge must be checked to ensure the integrity of flow
            in the HON is preserved. If an edge destination would
            result in a "dead end," the edge is redirected to the
            original destination's lower-order counterpart.
  
        Parameters: 
            node (_HONNode): the node whose destination we want to find
            
        Returns:
            The label to be included as the edge destination
        """
        if node.out_deg > 0 and node.in_deg > 0:
            return node.get_label_full()
        else:
            dst = None
            i = 1
            start_ord = node.order
            # search lower orders until a valid node is found
            while (not dst or dst.in_deg <= 0 or dst.out_deg <= 0) and (i <= start_ord):
                if node.name[i:] in self.nmap:
                    dst = self.nmap[node.name[i:]]
                i += 1
            return dst.get_label_full()

    def _get_edge_orphan(self, node):
        """Used by the extract() method to adopt orphaned edges.
        
        Description:
            During the pruning phase an edge may be "orphaned" in the
            sense that it has no tree parent, so no flow would reach
            it in the resulting HON. To preserve the integrity of the
            HON, these edges are "adopted" by lower order counterparts.
  
        Parameters: 
            node (_HONNode): the orphan who needs to be adopted
            
        Returns:
            The new edge that represents the orphaned node.
        """
        while node and (node.parent.in_deg <= 0):
            node = self._get_lord_match(node)
        return (node.parent.get_label_full(), self._get_edge_dst(node))

    def _get_lord_match(self, hord_node):
        """Used to find a node's lower-order counterpart.
        
        Description:
            This method is used by several others to find a 
            higher-order node's lower-order counterpart in O(1) time.
            This is done by truncating the first element (oldest
            history) from the higher-order node's label, then finding
            the new label in the nmap.

        Parameters: 
            hord_node (_HONNode): the higher-order node
            
        Returns:
            (_HONNode) the object reference to the lower-order node
        """
        return self.nmap[hord_node.name[1:]] if hord_node.order else None

    def _mark_ancestors(self, node):
        """Used to mark the tree ancestors of preserved nodes.
        
        Description:
            This method is used to tell GrowHON to preserve certain
            nodes. If we want to preserve a node in a high order of
            the tree, we mark its ancestors to ensure that it will
            not later be orphaned. It "climbs" from the preserved node
            and continues marking each subsequent parent until it 
            reaches the root.

        Parameters: 
            node (_HONNode): the node whose ancestors we want to mark
            
        Returns:
            None
        """
        while node != self.root:
            node.marked = True
            node = node.parent

    def _prune_bottom_up(self):
        """Prune the tree from the bottom-up, starting with higher orders.
        
        Description:
            Bottom-up pruning is the default because it is more comprehensive
            than top-down. Its accuracy with respect to top-down pruning is 
            still unverified, so it may be worthwhile to try both.
            
        Parameters: 
            None
            
        Returns:
            None
        """
        self.logger.info('Pruning tree bottom-up...')
        # pruning is performed in bottom-up fashion
        # starting at the second-to-last level of the tree
        for order in range(self.max_order-1, 0, -1):
            # getting nodes from nmap is faster than tree traversal
            cur_order = [n for n in self.nmap.values() if n.order == order]
            log_step = ceil(self.prune_log_interval * len(cur_order))
            self.logger.info('Pruning {:,d} nodes from order {}...'
                             .format(len(cur_order), order + 1))
            for i, node in enumerate(cur_order):
                # add an update to the log every so often
                if not i % log_step:
                    self.logger.info('Pruning node {:,d}/{:,d} ({}%) in order {}...'
                                .format(i, len(cur_order), int(i*100/len(cur_order)), order + 1))
                # the two criteria for preserving a higher-order node are if
                # 1) it has a higher-order descendent (to preserve flow)
                # or
                # 2) it has a dependency, as indicate by its relative entropy
                if node.marked or (self._has_dependency(node)):
                    # prune the lord_rule's children by hord weights
                    self._prune_lord_children(node)
                    # marking ancestors helps prevent orphaning
                    self._mark_ancestors(node)
                else:
                    for c in node.children.values(): self._delete_node(c)
            self.logger.info('Pruning successfully completed on order {}.'.format(order + 1)) 
        self.logger.info('Pruning successfully completed all orders!') 
    
    def _prune_top_down(self):
        """Prune the tree top-down.
        
        Description:
            Top-down pruning will be slightly faster and most likely result in
            a smaller network than bottom-up pruning. This is because lower
            orders are pruned first, so if a dependency exists downstream
            from a non-dependency, the algorithm will never find it.
            
        Parameters: 
            None
            
        Returns:
            None
        """
        self.logger.info('Pruning tree top-down is not supported in this implementation.')
        self._prune_bottom_up()

    def _prune_lord_children(self, hord_node):
        """Used to prune lower-order child nodes.
        
        Description:
            When a higher-order dependency is found, we prune the lower
            order node. Instead of deleting all the nodes, we simply
            reduce the frequency of the lower-order node's children by
            the amount we are preserving in the higher order.
            This is becuase there is a one-to-many relationship from
            a lower-order node to higher orders. The method takes the
            in-degree of each of hord_node's children, then reduces
            the corresponding child in lord_node by the same amount.

        Parameters: 
            hord_node (_HONNode): the node that is being preserved
            
        Returns:
            None
        """
        to_prune = self._get_lord_match(hord_node)
        while to_prune:
            for child_label, hord_child in hord_node.children.items():
                child_to_prune = to_prune.children[child_label]
                if self.verbose: self.logger.info('Reducing child {} weight by {}'
                                             .format(str(child_to_prune), hord_child.in_deg))
                child_to_prune.in_deg -= hord_child.in_deg
                to_prune.out_deg -= hord_child.in_deg
            to_prune = self._get_lord_match(to_prune)
            if self.verbose: self.logger.info('Next prune: {}'.format(str(to_prune)))
        
    # =================================================================
    # END SUPPORTING FUNCTIONS
    # =================================================================
    # =================================================================
    # BEGIN _HONNode DEFINITION
    # =================================================================
    class _HONNode:
        """The definition of all objects inserted into HONTree.
        
        Description:
            Each _HONNode represents a sequence from the input data and
            an edge in the output HON.

        Static Variables:
            order_delimiter (char): for labelling higher order nodes
        """
        # initialize to -1 to discount root node
        order_delimiter = None
        
        def __init__(self, name='', parent=None):
            self.name = name
            self.order = len(self.name) - 1
            self.in_deg = 0
            self.out_deg = 0
            self.in_freq = 0
            self.out_freq = 0
            self.parent = parent
            self.children = {}
            self.marked = False # used during pruning
            
        def __str__(self):
            return ('{}[{}:{}]'.format(self.get_label_full(), self.in_deg, self.out_deg))

        def dump(self):
            return '------->' * (len(self.name) - 1) + str(self) + '\n'
        
        def get_label_full(self):
            return HONTree._HONNode.order_delimiter.join(reversed([str(c) for c in self.name]))
        
        def get_label_short(self):
            return self.name[-1]

    # =========================================================================
    # END _HONNode DEFINITION
    # =========================================================================
    # =========================================================================
    # BEGIN _LogMap DEFINITION
    # =========================================================================
    class _LogMap():
        """Internal class used to standardize logging.
        """

        def __init__(self):
            self.start_time = perf_counter()
            self._info = {}
            self._info['runtime'] = self.get_time
            self._info['rss_size'] = self.get_rss
        
        def __getitem__(self, key):
            return self._info[key]()
        
        def __iter__(self):
            return iter(self._info)
        
        def get_time(self):
            return strftime('%H:%M:%S', gmtime(perf_counter() - self.start_time))
        
        def get_rss(self):
            return(Process(getpid()).memory_info().rss >> 20)
            
    # =========================================================================
    # END _LogMap DEFINITION
    # =========================================================================
# =================================================================
# END HONTree DEFINITION
# =================================================================
# =========================================================================
# BEGIN _SMap DEFINITION
# =========================================================================
if ray_available:
    @ray.remote
    class _SMap():
        """The definition for sequence objects created by workers.
    
        Description:
            Each worker creates one smap, a dict containing all
            sequences and frequencies as key, value pairs.
        """
        def __init__(self, max_order, inf_num_prefixes=1, inf_delimiter=' ', log_interval=0.2):
            self._send_signal('Initializing on CPU {}'.format(cpuinfo.get_cpu_info()['brand']))
            self.max_order = max_order
            self.inf_num_prefixes = inf_num_prefixes
            self.inf_delimiter = inf_delimiter
            self.log_interval = log_interval
            self.smap = defaultdict(int)

        def __str__(self):
            return ('\n'.join(['{0: >{1}} : {2}'.format(str(node), 
                    (8 * self.max_order) + 8, deg) for node, deg in sorted(self.smap.items())]))

        # to_absorb_id is a list of object ids
        def absorb_smap(self, to_absorb):
            """The method used to combine smap with another worker's.
    
            Description:
                If a worker finishes processing its input partition
                while the driver is still busy, the driver may
                instruct it to absorb another worker's smap. This
                method then receives the smap from another worker
                and combines its sequences with those already present
                locally.
            
            Parameters:
                to_absorb (ray ID): ID to retrieve from object store
                
            Returns:
                see get_smap()
            """
            self._send_signal('Received a new smap with {:,d} sequences to absorb...'
                              .format(len(to_absorb)))
            log_step = self.log_interval * len(to_absorb)
            for i, sequence in enumerate(to_absorb):
                if not i % log_step:
                    self._send_signal('Absorbing sequence {:,d}/{:,d}...'
                                      .format(i, len(to_absorb)))
                # we receive each sequence as a numpy array
                # we don't know its length, just that
                # the last element contains the degree and
                # everything before that is the sequence itself
                self.smap[tuple(sequence[:-1])] += sequence[-1]
            self._send_signal('Absorbing successfully completed. Returning smap with '
                              + '{:,d} sequences...'.format(len(self.smap)))
            return self.get_smap()

        def get_smap(self, skip_sort=False):
            """The method used to return an smap to the driver.
    
            Description:
                This method converts the local smap to a numpy array
                and returns it to the ray object store. Ray handles
                numpy arrays much faster than lists of strings.
                In Python 3.7+, dictionaries retain their insertion
                order and sorting is not necessary. In earlier versions
                order is not guaranteed, so we have to sort the items
                to prevent a child sequence from being inserted before
                its parent.
            
            Parameters:
                skip_sort (bool): whether sorting is necessary
                
            Returns:
                (np.array): an smap converted to a numpy array
            """
            # python 3.7+ preserves insert order for dictionaries
            if sys.version_info >= (3, 7) or skip_sort:
                return np.asarray([k + (v,) for k, v in self.smap.items()])
            else:
                self._send_signal('Sorting smap before return. Use Python 3.7+ ' + 
                                  'to avoid this requirement and improve performance.')
                return np.asarray([k + (v,) for k, v in sorted(self.smap.items())])

        def populate_smap(self, lines):
            """The method used to populate the smap dictionary locally.
    
            Description:
                This method is essentially the same as 
                HONTree._grow_sequential, but instead of adding tree
                nodes it simply generates combinations of sequences.
                It processes an entire input partition line by line.
            
            Parameters:
                lines (list of strings): input partition
                
            Returns:
                see get_smap()
            """
            log_step = self.log_interval * len(lines)
            for i, line in enumerate(lines):
                if not i % log_step:
                    self._send_signal('Processing line {:,d}/{:,d} of my partition...'
                                      .format(i, len(lines)))
                self._populate_smap_vector(line)
            
            self._send_signal('Procesing successfully completed on my partition. Returning smap with ' + 
                              '{:,d} sequences...'.format(len(self.smap)))
            self._send_signal('Local mem usage: {:,d}MB'
                              .format(Process(getpid()).memory_info().rss >> 20))
            return self.get_smap()

        def _populate_smap_vector(self, vec):
            """The method used to get sequences from an input vector.
    
            Description:
                This method accepts a single input vector and
                generates sequences with length max_order. It functions
                like HONTree._grow_vector, but instead of creating tree
                nodes it just counts occurrences.
            
            Parameters:
                line (string): a single input vector
                
            Returns:
                None (sequences are modified in place in self.smap)
            """
            q = deque(maxlen = self.max_order+1)
            
            vec = vec.strip().split(self.inf_delimiter)[self.inf_num_prefixes:]
            # prime the queue with first entities in the sequence
            for e in vec[:self.max_order+1]:
                q.append(e)
            for e in vec[self.max_order+1:]:
                for i in range(1, len(q) + 1):
                    self.smap[tuple([int(node) for node in islice(q, i)])] += 1
                q.popleft()
                q.append(e)
            while q:
                for i in range(1, len(q) + 1):
                    self.smap[tuple([int(node) for node in islice(q, i)])] += 1
                q.popleft()

        def _send_signal(self, msg):
            """Send a message to the driver for logging.
            
            Parameters:
                msg (string): message to be sent
                
            Returns:
                None
            """
            ray.experimental.signal.send(msg)
# =========================================================================
# END _SMap DEFINITION
# =========================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infname', help='source path + file name')
    parser.add_argument('otfname', help='destination path + file name')
    parser.add_argument('maxorder', help='max order to use in growing the HON', type=int)
    parser.add_argument('-w', '--numcpus', help='number of workers', type=int, default=1)
    parser.add_argument('-p', '--infnumprefixes', help='number of prefixes for input vectors', 
                        type=int, default=1)
    parser.add_argument('-di', '--infdelimiter', help='delimiter for entities in input vectors', 
                        default=' ')
    parser.add_argument('-do', '--otfdelimiter', help='delimiter for output network', default=' ')
    parser.add_argument('-d', '--topdown', help='enable top-down pruning (default is bottom-up)',
                        action='store_true', default=False)
    parser.add_argument('-t', '--tmult', help='threshold multiplier for determining dependencies', 
                        type=float, default=1.0)
    parser.add_argument('-e', '--dotfname', help='destination path + file for divergences', 
                        default=None)
    parser.add_argument('-o', '--logname', help='location to write log output', 
                        default=None)
    parser.add_argument('-lsg', '--logisgrow', 
                        help='logging interval for sequential growth', type=int, default=1000)
    parser.add_argument('-lpg', '--logipgrow', 
                        help='logging interval for parallel growth', type=float, default=0.2)
    parser.add_argument('-lpr', '--logiprune', 
                        help='logging interval for pruning', type=float, default=0.2)
    parser.add_argument('-v', '--verbose', help='print more messages for debugging', 
                        action='store_true', default=False)
    args = parser.parse_args()
    
    t1 = HONTree(args.maxorder, 
                 args.infname, 
                 inf_num_prefixes=args.infnumprefixes, 
                 inf_delimiter=args.infdelimiter,
                 num_cpus=args.numcpus, 
                 log_name=args.logname, 
                 verbose=args.verbose,
                 top_down=args.topdown,
                 threshold_multiplier=args.tmult,
                 seq_grow_log_step=args.logisgrow, 
                 par_grow_log_interval=args.logipgrow, 
                 prune_log_interval=args.logiprune, 
                 )
    t1.extract(args.otfname, args.otfdelimiter)
    
    if args.dotfname:
        with open(args.dotfname, 'w+') as otf_divergences: 
            otf_divergences.write('\n'.join(t1.get_divergences()))
# =====================================================================
# END growhon.py
# =====================================================================