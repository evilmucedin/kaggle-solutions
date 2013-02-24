/* Lockless MPI Profiling Header */

#ifndef PMPI_H
#define PMPI_H
#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

__must_check int PMPI_Comm_test_inter(MPI_Comm comm, int *flag);
__must_check int PMPI_Comm_size(MPI_Comm comm, int *n);
__must_check int PMPI_Comm_remote_size(MPI_Comm comm, int *n);
__must_check int PMPI_Comm_rank(MPI_Comm comm, int *r);
__must_check int PMPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *out);
__must_check int PMPI_Comm_free(MPI_Comm *comm);
__must_check int PMPI_Comm_set_name(MPI_Comm comm, char *name);
__must_check int PMPI_Comm_get_name(MPI_Comm comm, char *name, int *len);
__must_check int PMPI_Comm_group(MPI_Comm comm, MPI_Group *group);
__must_check int PMPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group);
__must_check int PMPI_Comm_dup(MPI_Comm comm, MPI_Comm *out);
__must_check int PMPI_Comm_compare(MPI_Comm c1, MPI_Comm c2, int *result);
__must_check int PMPI_Comm_split(MPI_Comm comm, int split_key, int key, MPI_Comm *out);
__must_check int PMPI_Intercomm_create(MPI_Comm local_comm, int local_leader,
	MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *comm);
__must_check int PMPI_Intercomm_merge(MPI_Comm comm_in, int high, MPI_Comm *comm);
__must_check int PMPI_Group_free(MPI_Group *group);
__must_check int PMPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *out);
__must_check int PMPI_Group_compare(MPI_Group g1, MPI_Group g2, int *result);
__must_check int PMPI_Group_difference(MPI_Group g1, MPI_Group g2, MPI_Group *out);
__must_check int PMPI_Group_excl(MPI_Group group, int n, int *ranks, MPI_Group *out);
__must_check int PMPI_Group_intersection(MPI_Group g1, MPI_Group g2, MPI_Group *out);
__must_check int PMPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *out);
__must_check int PMPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *out);
__must_check int PMPI_Group_rank(MPI_Group group, int *rank);
__must_check int PMPI_Group_size(MPI_Group group, int *size);
__must_check int PMPI_Group_translate_ranks(MPI_Group g1, int n, int *r1, MPI_Group g2,
	int *r2);
__must_check int PMPI_Group_union(MPI_Group g1, MPI_Group g2, MPI_Group *out);
__must_check int PMPI_Topo_test(MPI_Comm comm, int *status);
__must_check int PMPI_Dims_create(int nnodes, int ndims, int *dims);
__must_check int PMPI_Cart_create(MPI_Comm comm, int ndims, int *dims, int *periods,
	int reorder, MPI_Comm *out);
__must_check int PMPI_Cartdim_get(MPI_Comm comm, int *ndims);
__must_check int PMPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods,
	int *coords);
__must_check int PMPI_Cart_rank(MPI_Comm comm, int *coords, int *rank);
__must_check int PMPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int *coords);
__must_check int PMPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rsource,
	int *rdest);
__must_check int PMPI_Cart_sub(MPI_Comm comm, int *remain_dims, MPI_Comm *out);
__must_check int PMPI_Cart_map(MPI_Comm comm, int ndims, int *dims, int *periods, int *newrank);
__must_check int PMPI_Graph_create(MPI_Comm comm, int nnodes, int *index, int *edges,
	int reorder, MPI_Comm *out);
__must_check int PMPI_Graphdims_get(MPI_Comm comm, int *nodes, int *nedges);
__must_check int PMPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int *index,
	int *edges);
__must_check int PMPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors);
__must_check int PMPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors,
	int *neighbors);
__must_check int PMPI_Graph_map(MPI_Comm comm, int nnodes, int *index, int *edges,
	int *newrank);

__must_check int PMPI_Send(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm);
__must_check int PMPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Rsend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm);
__must_check int PMPI_Irsend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Ssend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm);
__must_check int PMPI_Issend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Buffer_attach(void *buffer, int size);
__must_check int PMPI_Buffer_detach(void *buffer_addr, int *size);
__must_check int PMPI_Bsend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm);
__must_check int PMPI_Ibsend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Send_init(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Bsend_init(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Ssend_init(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Rsend_init(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Start(MPI_Request *qd);
__must_check int PMPI_Startall(int count, MPI_Request *qd);
__must_check int PMPI_Cancel(MPI_Request *qd);
__must_check int PMPI_Request_free(MPI_Request *qd);
__must_check int PMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
	int tag, MPI_Comm comm, MPI_Status *status);
__must_check int PMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
	int tag, MPI_Comm comm, MPI_Request *qd);
__must_check int PMPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag,
	MPI_Status *status);
__must_check int PMPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
__must_check int PMPI_Test(MPI_Request *qd, int *flag, MPI_Status *status);
__must_check int PMPI_Testall(int count, MPI_Request *request_array, int *flag,
	MPI_Status *statusarray);
__must_check int PMPI_Testany(int count, MPI_Request *request_array, int *index,
	int *flag, MPI_Status *status);
__must_check int PMPI_Testsome(int count, MPI_Request *request_array, int *out_count,
	int *index_array, MPI_Status *statusarray);
__must_check int PMPI_Test_cancelled(MPI_Status *status, int *flag);
__must_check int PMPI_Wait(MPI_Request *qd, MPI_Status *status);
__must_check int PMPI_Waitall(int count, MPI_Request *request_array,
	MPI_Status *statusarray);
__must_check int PMPI_Waitany(int count, MPI_Request *request_array, int *index,
	MPI_Status *status);
__must_check int PMPI_Waitsome(int count, MPI_Request *request_array, int *out_count,
	int *index_array, MPI_Status *statusarray);
__must_check int PMPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	int dest, int sendtag,
	void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int source, int recvtag, MPI_Comm comm, MPI_Status *status);
__must_check int PMPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype,
	int dest, int sendtag, int source, int recvtag,
	MPI_Comm comm, MPI_Status *status);
__must_check int PMPI_Barrier(MPI_Comm comm);
__must_check int PMPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root,
	MPI_Comm comm);
__must_check int PMPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm);
__must_check int PMPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int *recvcount, int *displs, MPI_Datatype recvtype,
	int root, MPI_Comm comm);
__must_check int PMPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
__must_check int PMPI_Scatterv(void *sendbuf, int *sendcount, int *disp,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm);
__must_check int PMPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
__must_check int PMPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int *recvcount, int *disp, MPI_Datatype recvtype, MPI_Comm comm);
__must_check int PMPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
__must_check int PMPI_Alltoallv(void *sendbuf, int *sendcount, int *sdisp,
	MPI_Datatype sendtype, void *recvbuf, int *recvcount, int *rdisp, MPI_Datatype recvtype,
	MPI_Comm comm);
__must_check int PMPI_Alltoallw(void *sendbuf, int *sendcount, int *sdisp,
	MPI_Datatype *sendtype, void *recvbuf, int *recvcount, int *rdisp,
	MPI_Datatype *recvtype, MPI_Comm comm);
__must_check int PMPI_Reduce(void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
__must_check int PMPI_Allreduce(void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
__must_check int PMPI_Reduce_scatter(void *sendbuf, void *recvbuf, int *counts,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
__must_check int PMPI_Scan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
	MPI_Op op, MPI_Comm comm);
__must_check int PMPI_Exscan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
	MPI_Op op, MPI_Comm comm);
__must_check int PMPI_Init(int *argc, char ***argv);
__must_check int PMPI_Initialized(int *flag);
__must_check int PMPI_Finalize(void);
__must_check int PMPI_Finalized(int *flag);
__must_check int PMPI_Get_version(int *version, int *subversion);
__must_check int PMPI_Op_create(MPI_User_function *func, int commute, MPI_Op *op);
__must_check int PMPI_Op_free(MPI_Op *op);
__must_check int PMPI_Get_count(MPI_Status *status, MPI_Datatype datatype, int *count);
__must_check int PMPI_Get_elements(MPI_Status *status, MPI_Datatype datatype, int *elems);
__must_check int PMPI_Type_size(MPI_Datatype datatype, int *size);
__must_check int PMPI_Type_ub(MPI_Datatype datatype, MPI_Aint *ub);
__must_check int PMPI_Type_lb(MPI_Datatype datatype, MPI_Aint *lb);
__must_check int PMPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent);
__must_check int PMPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent);
__must_check int PMPI_Type_free(MPI_Datatype *datatype);
__must_check int PMPI_Type_commit(MPI_Datatype *datatype);
__must_check int PMPI_Type_struct(int count, int *b_len, MPI_Aint *disp,
	MPI_Datatype *oldtype, MPI_Datatype *newtype);
__must_check int PMPI_Type_create_struct(int count, int *b_len, MPI_Aint *disp,
	MPI_Datatype *oldtype, MPI_Datatype *newtype);
__must_check int PMPI_Type_contiguous(int count, MPI_Datatype old_type, MPI_Datatype *newtype);
__must_check int PMPI_Type_indexed(int count, int *b_lens, int *indices,
	MPI_Datatype oldtype, MPI_Datatype *newtype);
__must_check int PMPI_Type_hindexed(int count, int *b_len, MPI_Aint *disp,
	MPI_Datatype oldtype, MPI_Datatype *newtype);
__must_check int PMPI_Type_create_hindexed(int count, int *b_len, MPI_Aint *disp,
	MPI_Datatype oldtype, MPI_Datatype *newtype);
__must_check int PMPI_Type_vector(int count, int b_len, int stride, MPI_Datatype oldtype,
	MPI_Datatype *newtype);
__must_check int PMPI_Type_hvector(int count, int b_len, MPI_Aint stride, MPI_Datatype oldtype,
	MPI_Datatype *newtype);
__must_check int PMPI_Type_create_hvector(int count, int b_len, MPI_Aint stride, MPI_Datatype oldtype,
	MPI_Datatype *newtype);
__must_check int PMPI_Pack_size(int count, MPI_Datatype datatype, MPI_Comm comm, int *size);
__must_check int PMPI_Pack(void *buf, int count, MPI_Datatype datatype, void *pack_buf,
	int pack_size, int *pos, MPI_Comm comm);
__must_check int PMPI_Unpack(void *pack_buf, int pack_size, int *pos, void *buf, int count,
	MPI_Datatype datatype, MPI_Comm comm);
__must_check int PMPI_Get_processor_name(char *name, int *len);
__must_check int PMPI_Abort(MPI_Comm comm, int errcode);
__must_check int PMPI_Address(void *ptr, MPI_Aint *addr);
__must_check int PMPI_Get_address(void *ptr, MPI_Aint *addr);
__must_check int PMPI_Pcontrol(const int level, ...);
double PMPI_Wtime(void);
double PMPI_Wtick(void);
__must_check int PMPI_Keyval_create(MPI_Copy_function *copy, MPI_Delete_function *del,
	int *keyval, void *extra_state);
__must_check int PMPI_Keyval_free(int *keyval);
__must_check int PMPI_Attr_put(MPI_Comm comm, int keyval, void *attr);
__must_check int PMPI_Attr_get(MPI_Comm comm, int keyval, void *attr, int *flag);
__must_check int PMPI_Attr_delete(MPI_Comm comm, int keyval);
__must_check int PMPI_Errhandler_create(MPI_Handler_function *func, MPI_Errhandler *errh);
__must_check int PMPI_Errhandler_free(MPI_Errhandler *errh);
__must_check int PMPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errh);
__must_check int PMPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errh);
__must_check int PMPI_Error_string(int errcode, char *out, int *len);
__must_check int PMPI_Error_class(int errcode, int *errclass);


static inline MPI_Request PMPI_Request_f2c(MPI_Fint request){return request;}
static inline MPI_Fint PMPI_Request_c2f(MPI_Request request){return request;}
static inline MPI_Comm PMPI_Comm_f2c(MPI_Fint comm){return comm;}
static inline MPI_Fint PMPI_Comm_c2f(MPI_Comm comm){return comm;}
static inline MPI_Datatype PMPI_Type_f2c(MPI_Fint type){return type;}
static inline MPI_Fint PMPI_Type_c2f(MPI_Datatype type){return type;}
static inline MPI_Group PMPI_Group_f2c(MPI_Fint group){return group;}
static inline MPI_Fint PMPI_Group_c2f(MPI_Group group){return group;}
/* static inline MPI_File PMPI_File_f2c(MPI_Fint file){return file;} */
/* static inline MPI_Fint PMPI_File_c2f(MIP_File file){return file;} */
/* static inline MPI_Win PMPI_Win_f2c(MPI_Fint win){return win;} */
/* static inline MPI_Fint PMPI_Win_c2f(MPI_Win win){return win;} */
static inline MPI_Op PMPI_Op_f2c(MPI_Fint op){return op;}
static inline MPI_Fint PMPI_Op_c2f(MPI_Op op){return op;}
/* static inline MPI_Info PMPI_Info_f2c(MPI_Fint info){return info;} */
/* static inline MPI_Fint PMPI_Info_c2f(MPI_Info info){return info;} */
__must_check int PMPI_Status_f2c(MPI_Fint *f_status, MPI_Status *c_status);
__must_check int PMPI_Status_c2f(MPI_Status *c_status, MPI_Fint *f_status);

/* Extensions */
__must_check int PMPI_bufcopy_set(MPI_Aint size);
__must_check int PMPI_bufcopy_get(MPI_Aint *size);

#ifdef __cplusplus
}
#endif

#endif /* PMPI_H */
