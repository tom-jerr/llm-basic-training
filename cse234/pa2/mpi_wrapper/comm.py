from mpi4py import MPI
import numpy as np


class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert (
            src_array.size % nprocs == 0
        ), "src_array size must be divisible by the number of processes"
        assert (
            dest_array.size % nprocs == 0
        ), "dest_array size must be divisible by the number of processes"

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        Recursive Doubling (递归加倍) algorithm for Allreduce.

        Algorithm:
        This is a tree-based algorithm that works in log(N) steps, where N is the
        number of processes. In each step k, processes exchange data with their
        partner that differs in the k-th bit of their rank.

        Steps:
        1. In step 0: processes with ranks differing in bit 0 exchange and reduce
           (0↔1, 2↔3, 4↔5, 6↔7)
        2. In step 1: processes with ranks differing in bit 1 exchange and reduce
           (0↔2, 1↔3, 4↔6, 5↔7)
        3. In step 2: processes with ranks differing in bit 2 exchange and reduce
           (0↔4, 1↔5, 2↔6, 3↔7)
        ...and so on for log(N) steps

        After log(N) steps, all processes have the fully reduced result.

        Complexity:
        - Number of steps: O(log N)
        - Data transferred per step: M (message size)
        - Total time: O(log N * (α + M * β))
          where α = latency, β = per-byte transfer time

        Advantages:
        - Optimal complexity O(log N) steps
        - All processes participate in parallel at each step
        - Perfectly balanced load distribution
        - Works best when N is a power of 2

        Note: This implementation handles non-power-of-2 process counts by having
        some processes idle in later steps.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        # Initialize dest_array with src_array
        dest_array[:] = src_array[:]

        # Helper function to apply reduction operation in place
        def apply_op_inplace(local_data, recv_data, op):
            if op == MPI.SUM:
                local_data += recv_data
            elif op == MPI.PROD:
                local_data *= recv_data
            elif op == MPI.MAX:
                np.maximum(local_data, recv_data, out=local_data)
            elif op == MPI.MIN:
                np.minimum(local_data, recv_data, out=local_data)
            else:
                raise ValueError(f"Unsupported operation: {op}")

        # Recursive Doubling: iterate through log(size) steps
        # In each step k, exchange with partner rank ^ (1 << k)
        step = 0
        mask = 1
        dst_bytes = dest_array.itemsize * dest_array.size
        recv_buf = np.empty_like(dest_array)
        partner = rank ^ mask
        while partner < size:
            # Exchange data with partner using Sendrecv
            self.comm.Sendrecv(
                sendbuf=dest_array,
                dest=partner,
                sendtag=step,
                recvbuf=recv_buf,
                source=partner,
                recvtag=step,
            )

            # Reduce the received data with local data
            apply_op_inplace(dest_array, recv_buf, op)

            # Update bytes transferred
            self.total_bytes_transferred += dst_bytes * 2

            mask <<= 1
            step += 1
            partner = rank ^ mask

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.

        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.

        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.

        The total data transferred is updated for each pairwise exchange.
        """
        rank = self.comm.Get_rank()
        nprocs = self.comm.Get_size()

        assert (
            src_array.size % nprocs == 0
        ), "src_array size must be divisible by number of processes"
        assert (
            dest_array.size % nprocs == 0
        ), "dest_array size must be divisible by number of processes"

        segment_sz = src_array.size // nprocs
        segment_bytes = src_array.itemsize * segment_sz

        for dest in range(nprocs):
            if dest == rank:
                dest_array[dest * segment_sz : (dest + 1) * segment_sz] = src_array[
                    dest * segment_sz : (dest + 1) * segment_sz
                ]
            else:
                self.comm.Sendrecv(
                    sendbuf=src_array[dest * segment_sz : (dest + 1) * segment_sz],
                    dest=dest,
                    sendtag=rank,
                    recvbuf=dest_array[
                        dest * segment_sz : (dest + 1) * segment_sz
                    ],  # 从dest接收到属于dest的那份数据，放在dest的位置
                    source=dest,
                    recvtag=dest,
                )
                self.total_bytes_transferred += segment_bytes * 2
