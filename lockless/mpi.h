/* Lockless MPI Header */

#ifndef MPI_H
#define MPI_H

#ifdef __cplusplus
extern "C" {
#endif

#define MPI_VERSION		1
#define MPI_SUBVERSION	3

#define MPI_ANY_TAG		-1
#define MPI_ANY_SOURCE	-1
#define MPI_PROC_NULL	-2
#define MPI_ROOT		-3

#define MPI_COMM_NULL	(0 + (1 << 27))
#define MPI_COMM_WORLD	(1 + (1 << 27))
#define MPI_COMM_SELF	(2 + (1 << 27))

#define MPI_IN_PLACE	((void *) 1)
#define MPI_BOTTOM			((void *) (1UL << (sizeof(long)*8 - 1)))

#define MPI_MAX_PROCESSOR_NAME	256
#define MPI_MAX_OBJECT_NAME		256
#define MPI_BSEND_OVERHEAD		32
#define MPI_MAX_ERROR_STRING	1024

#define MPI_DATATYPE_NULL		(0 + (2 << 27))
#define MPI_LB					(1 + (2 << 27))
#define MPI_UB					(2 + (2 << 27))
#define MPI_PACKED				(3 + (2 << 27))

#define MPI_BYTE				(4 + (2 << 27))
#define MPI_CHAR				(5 + (2 << 27))
#define MPI_SHORT				(6 + (2 << 27))
#define	MPI_INT					(7 + (2 << 27))
#define MPI_LONG				(8 + (2 << 27))
#define MPI_LONG_LONG_INT		(9 + (2 << 27))

#define MPI_SIGNED_CHAR			(10 + (2 << 27))
#define MPI_UNSIGNED_CHAR		(11 + (2 << 27))
#define MPI_WCHAR				(12 + (2 << 27))

#define MPI_UNSIGNED_SHORT		(13 + (2 << 27))
#define MPI_UNSIGNED			(14 + (2 << 27))
#define MPI_UNSIGNED_LONG		(15 + (2 << 27))
#define MPI_UNSIGNED_LONG_LONG	(16 + (2 << 27))

#define MPI_INT8_T				(17 + (2 << 27))
#define MPI_INT16_T				(18 + (2 << 27))
#define MPI_INT32_T				(19 + (2 << 27))
#define MPI_INT64_T				(20 + (2 << 27))
#define MPI_INT128_T			(21 + (2 << 27))

#define MPI_UINT8_T				(22 + (2 << 27))
#define MPI_UINT16_T			(23 + (2 << 27))
#define MPI_UINT32_T			(24 + (2 << 27))
#define MPI_UINT64_T			(25 + (2 << 27))
#define MPI_UINT128_T			(26 + (2 << 27))

#define MPI_FLOAT				(27 + (2 << 27))
#define MPI_DOUBLE				(28 + (2 << 27))
#define MPI_LONG_DOUBLE			(29 + (2 << 27))
#define MPI_FLOAT128			(30 + (2 << 27))

#define MPI_C_FLOAT_COMPLEX		(31 + (2 << 27))
#define MPI_C_COMPLEX			MPI_C_FLOAT_COMPLEX
#define MPI_C_DOUBLE_COMPLEX	(32 + (2 << 27))
#define MPI_C_LONG_DOUBLE_COMPLEX (33 + (2 << 27))
#define MPI_C_FLOAT128_COMPLEX	(34 + (2 << 27))

#define MPI_C_BOOL				(35 + (2 << 27))

#define MPI_CHARACTER			(36 + (2 << 27))
#define MPI_LOGICAL				(37 + (2 << 27))
#define MPI_INTEGER				(38 + (2 << 27))

#define MPI_LOGICAL1			(39 + (2 << 27))
#define MPI_LOGICAL2			(40 + (2 << 27))
#define MPI_LOGICAL4			(41 + (2 << 27))
#define MPI_LOGICAL8			(42 + (2 << 27))

#define MPI_INTEGER1			(43 + (2 << 27))
#define MPI_INTEGER2			(44 + (2 << 27))
#define MPI_INTEGER4			(45 + (2 << 27))
#define MPI_INTEGER8			(46 + (2 << 27))
#define MPI_INTEGER16			(47 + (2 << 27))

#define MPI_REAL				(48 + (2 << 27))
#define MPI_DOUBLE_PRECISION	(49 + (2 << 27))
#define MPI_COMPLEX				(50 + (2 << 27))
#define MPI_DOUBLE_COMPLEX		(51 + (2 << 27))
#define MPI_LONG_DOUBLE_COMPLEX	(52 + (2 << 27))

#define MPI_REAL4				(53 + (2 << 27))
#define MPI_REAL8				(54 + (2 << 27))
#define MPI_REAL16				(55 + (2 << 27))

#define MPI_COMPLEX8			(56 + (2 << 27))
#define MPI_COMPLEX16			(57 + (2 << 27))
#define MPI_COMPLEX32			(58 + (2 << 27))

#define MPI_AINT				(59 + (2 << 27))
#define MPI_OFFSET				(60 + (2 << 27))

#define MPI_FLOAT_INT			(61 + (2 << 27))
#define MPI_DOUBLE_INT			(62 + (2 << 27))
#define MPI_LONG_DOUBLE_INT		(63 + (2 << 27))
#define MPI_SHORT_INT			(64 + (2 << 27))
#define MPI_2INT				(65 + (2 << 27))
#define MPI_LONG_INT			(66 + (2 << 27))

#define MPI_2INTEGER			(67 + (2 << 27))
#define MPI_2REAL				(68 + (2 << 27))
#define MPI_2DOUBLE_PRECISION	(69 + (2 << 27))


#define MPI_GROUP_NULL		(0 + (3 << 27))
#define MPI_GROUP_EMPTY 	(1 + (3 << 27))

#define MPI_OP_NULL		(0 + (4 << 27))
#define MPI_MAX			(1 + (4 << 27))
#define MPI_MIN			(2 + (4 << 27))
#define MPI_SUM			(3 + (4 << 27))
#define MPI_PROD		(4 + (4 << 27))
#define MPI_LAND		(5 + (4 << 27))
#define MPI_BAND		(6 + (4 << 27))
#define MPI_LOR			(7 + (4 << 27))
#define MPI_BOR			(8 + (4 << 27))
#define MPI_LXOR		(9 + (4 << 27))
#define MPI_BXOR		(10 + (4 << 27))
#define MPI_MAXLOC		(11 + (4 << 27))
#define MPI_MINLOC		(12 + (4 << 27))

/* Error returns */
#define MPI_SUCCESS			0
#define MPI_ERR_OTHER		1
#define MPI_ERR_COMM		2
#define MPI_ERR_COUNT		3
#define MPI_ERR_TYPE		4
#define MPI_ERR_BUFFER		5
#define MPI_ERR_ROOT		6
#define MPI_ERR_TAG			7
#define MPI_ERR_RANK		8
#define MPI_ERR_ARG			9
#define MPI_ERR_REQUEST		10
#define MPI_ERR_IN_STATUS	11
#define MPI_ERR_PENDING		12
#define MPI_ERR_INTERN		13
#define MPI_ERR_GROUP		14
#define MPI_ERR_CANCEL		15
#define MPI_ERR_OP			16
#define MPI_ERR_TOPOLOGY	17
#define MPI_ERR_DIMS		18
#define MPI_ERR_UNKNOWN		19
#define MPI_ERR_TRUNCATE	20
#define MPI_ERR_KEYVAL		21
#define MPI_ERR_LASTCODE	21

#define MPI_ERRHANDLER_NULL		(0 + (5 << 27))
#define MPI_ERRORS_ARE_FATAL	(1 + (5 << 27))
#define MPI_ERRORS_RETURN		(2 + (5 << 27))
#define MPI_ERRORS_THROW_EXCEPTIONS (3 + (5 << 27))

#define MPI_IDENT			0
#define MPI_CONGRUENT		1
#define MPI_SIMILAR			2
#define MPI_UNEQUAL			3

#define	MPI_CART			0
#define MPI_GRAPH			1

#define MPI_UNDEFINED		-1
#define MPI_REQUEST_NULL	(0 + (6 << 27))
#define MPI_STATUS_IGNORE	((MPI_Status *) 0)
#define MPI_STATUSES_IGNORE	((MPI_Status *) 0)

#define MPI_KEYVAL_INVALID		(0 + (7 << 27))
#define MPI_TAG_UB				(1 + (7 << 27))
#define MPI_HOST				(2 + (7 << 27))
#define	MPI_IO					(3 + (7 << 27))
#define MPI_WTIME_IS_GLOBAL		(4 + (7 << 27))

#ifndef __must_check
#define __must_check
#endif

typedef struct MPI_Status MPI_Status;
struct MPI_Status
{
	unsigned long size;
	int MPI_TAG;
	int MPI_SOURCE;
	int MPI_ERROR;
	int pad;
};

typedef unsigned MPI_Request;
typedef unsigned MPI_Datatype;
typedef unsigned MPI_Group;
typedef unsigned MPI_Comm;
typedef unsigned MPI_Op;
typedef unsigned MPI_Errhandler;

typedef void MPI_User_function(void *inbuf, void *outbuf, int *len, MPI_Datatype *datatype);
typedef void MPI_Handler_function(MPI_Comm *comm, int *err, ...);

typedef int MPI_Copy_function(MPI_Comm comm, int keyval, void *extra_state, void *attr_in,
	void *attr_out, int *flag);
typedef int MPI_Delete_function(MPI_Comm comm, int keyval, void *attr, void *extra_state);
#define MPI_NULL_COPY_FN	((MPI_Copy_function *) 0)
#define MPI_NULL_DELETE_FN	((MPI_Delete_function *) 0)
#define MPI_DUP_FN ((MPI_Copy_function *) 1UL)

typedef long MPI_Offset;
typedef long MPI_Aint;
typedef int MPI_Fint;

__must_check int MPI_Comm_test_inter(MPI_Comm comm, int *flag);
__must_check int MPI_Comm_size(MPI_Comm comm, int *size);
__must_check int MPI_Comm_remote_size(MPI_Comm comm, int *size);
__must_check int MPI_Comm_rank(MPI_Comm comm, int *rank);
__must_check int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *out);
__must_check int MPI_Comm_free(MPI_Comm *comm);
__must_check int MPI_Comm_set_name(MPI_Comm comm, char *name);
__must_check int MPI_Comm_get_name(MPI_Comm comm, char *name, int *len);
__must_check int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
__must_check int MPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group);
__must_check int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *out);
__must_check int MPI_Comm_compare(MPI_Comm c1, MPI_Comm c2, int *result);
__must_check int MPI_Comm_split(MPI_Comm comm, int split_key, int key, MPI_Comm *out);
__must_check int MPI_Intercomm_create(MPI_Comm local_comm, int local_leader,
	MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *comm);
__must_check int MPI_Intercomm_merge(MPI_Comm comm, int high, MPI_Comm *out);
__must_check int MPI_Group_free(MPI_Group *group);
__must_check int MPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *out);
__must_check int MPI_Group_compare(MPI_Group g1, MPI_Group g2, int *result);
__must_check int MPI_Group_difference(MPI_Group g1, MPI_Group g2, MPI_Group *out);
__must_check int MPI_Group_excl(MPI_Group group, int n, int *ranks, MPI_Group *out);
__must_check int MPI_Group_intersection(MPI_Group g1, MPI_Group g2, MPI_Group *out);
__must_check int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *out);
__must_check int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *out);
__must_check int MPI_Group_rank(MPI_Group group, int *rank);
__must_check int MPI_Group_size(MPI_Group group, int *size);
__must_check int MPI_Group_translate_ranks(MPI_Group g1, int n, int *r1, MPI_Group g2,
	int *r2);
__must_check int MPI_Group_union(MPI_Group g1, MPI_Group g2, MPI_Group *out);
__must_check int MPI_Topo_test(MPI_Comm comm, int *status);
__must_check int MPI_Dims_create(int nnodes, int ndims, int *dims);
__must_check int MPI_Cart_create(MPI_Comm comm, int ndims, int *dims, int *periods,
	int reorder, MPI_Comm *out);
__must_check int MPI_Cartdim_get(MPI_Comm comm, int *ndims);
__must_check int MPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods,
	int *coords);
__must_check int MPI_Cart_rank(MPI_Comm comm, int *coords, int *rank);
__must_check int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int *coords);
__must_check int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rsource,
	int *rdest);
__must_check int MPI_Cart_sub(MPI_Comm comm, int *remain_dims, MPI_Comm *out);
__must_check int MPI_Cart_map(MPI_Comm comm, int ndims, int *dims, int *periods, int *newrank);
__must_check int MPI_Graph_create(MPI_Comm comm, int nnodes, int *index, int *edges,
	int reorder, MPI_Comm *out);
__must_check int MPI_Graphdims_get(MPI_Comm comm, int *nodes, int *nedges);
__must_check int MPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int *index,
	int *edges);
__must_check int MPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors);
__must_check int MPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors,
	int *neighbors);
__must_check int MPI_Graph_map(MPI_Comm comm, int nnodes, int *index, int *edges,
	int *newrank);

__must_check int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm);
__must_check int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Rsend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm);
__must_check int MPI_Irsend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Ssend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm);
__must_check int MPI_Issend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Buffer_attach(void *buffer, int size);
__must_check int MPI_Buffer_detach(void *buffer_addr, int *size);
__must_check int MPI_Bsend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm);
__must_check int MPI_Ibsend(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Send_init(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Bsend_init(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Ssend_init(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Rsend_init(void *buf, int count, MPI_Datatype datatype, int dest,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Start(MPI_Request *rq);
__must_check int MPI_Startall(int count, MPI_Request *rq);
__must_check int MPI_Cancel(MPI_Request *rq);
__must_check int MPI_Request_free(MPI_Request *rq);
__must_check int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
	int tag, MPI_Comm comm, MPI_Status *status);
__must_check int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
	int tag, MPI_Comm comm, MPI_Request *rq);
__must_check int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag,
	MPI_Status *status);
__must_check int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
__must_check int MPI_Test(MPI_Request *rq, int *flag, MPI_Status *status);
__must_check int MPI_Testall(int count, MPI_Request *request_array, int *flag,
	MPI_Status *status_array);
__must_check int MPI_Testany(int count, MPI_Request *request_array, int *index,
	int *flag, MPI_Status *status);
__must_check int MPI_Testsome(int count, MPI_Request *request_array, int *out_count,
	int *index_array, MPI_Status *status_array);
__must_check int MPI_Test_cancelled(MPI_Status *status, int *flag);
__must_check int MPI_Wait(MPI_Request *rq, MPI_Status *status);
__must_check int MPI_Waitall(int count, MPI_Request *request_array,
	MPI_Status *status_array);
__must_check int MPI_Waitany(int count, MPI_Request *request_array, int *index,
	MPI_Status *status);
__must_check int MPI_Waitsome(int count, MPI_Request *request_array, int *out_count,
	int *index_array, MPI_Status *status_array);
__must_check int MPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	int dest, int sendtag,
	void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int source, int recvtag, MPI_Comm comm, MPI_Status *status);
__must_check int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype,
	int dest, int sendtag, int source, int recvtag,
	MPI_Comm comm, MPI_Status *status);
__must_check int MPI_Barrier(MPI_Comm comm);
__must_check int MPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root,
	MPI_Comm comm);
__must_check int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm);
__must_check int MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int *recvcount, int *displs, MPI_Datatype recvtype,
	int root, MPI_Comm comm);
__must_check int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
__must_check int MPI_Scatterv(void *sendbuf, int *sendcount, int *disp,
	MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int root, MPI_Comm comm);
__must_check int MPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
__must_check int MPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int *recvcount, int *disp, MPI_Datatype recvtype, MPI_Comm comm);
__must_check int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
__must_check int MPI_Alltoallv(void *sendbuf, int *sendcount, int *sdisp,
	MPI_Datatype sendtype, void *recvbuf, int *recvcount, int *rdisp, MPI_Datatype recvtype,
	MPI_Comm comm);
__must_check int MPI_Alltoallw(void *sendbuf, int *sendcount, int *sdisp,
	MPI_Datatype *sendtype, void *recvbuf, int *recvcount, int *rdisp,
	MPI_Datatype *recvtype, MPI_Comm comm);
__must_check int MPI_Reduce(void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
__must_check int MPI_Allreduce(void *sendbuf, void *recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
__must_check int MPI_Reduce_scatter(void *sendbuf, void *recvbuf, int *counts,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
__must_check int MPI_Scan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
	MPI_Op op, MPI_Comm comm);
__must_check int MPI_Exscan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
	MPI_Op op, MPI_Comm comm);
__must_check int MPI_Init(int *argc, char ***argv);
__must_check int MPI_Initialized(int *flag);
__must_check int MPI_Finalize(void);
__must_check int MPI_Finalized(int *flag);
__must_check int MPI_Get_version(int *version, int *subversion);
__must_check int MPI_Op_create(MPI_User_function *func, int commute, MPI_Op *op);
__must_check int MPI_Op_free(MPI_Op *op);
__must_check int MPI_Get_count(MPI_Status *status, MPI_Datatype datatype, int *count);
__must_check int MPI_Get_elements(MPI_Status *status, MPI_Datatype datatype, int *elems);
__must_check int MPI_Type_size(MPI_Datatype datatype, int *size);
__must_check int MPI_Type_ub(MPI_Datatype datatype, MPI_Aint *ub);
__must_check int MPI_Type_lb(MPI_Datatype datatype, MPI_Aint *lb);
__must_check int MPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent);
__must_check int MPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent);
__must_check int MPI_Type_free(MPI_Datatype *datatype);
__must_check int MPI_Type_commit(MPI_Datatype *datatype);
__must_check int MPI_Type_struct(int count, int *b_len, MPI_Aint *disp,
	MPI_Datatype *oldtype, MPI_Datatype *newtype);
__must_check int MPI_Type_create_struct(int count, int *b_len, MPI_Aint *disp,
	MPI_Datatype *oldtype, MPI_Datatype *newtype);
__must_check int MPI_Type_contiguous(int count, MPI_Datatype old_type, MPI_Datatype *newtype);
__must_check int MPI_Type_indexed(int count, int *b_lens, int *indices,
	MPI_Datatype oldtype, MPI_Datatype *newtype);
__must_check int MPI_Type_hindexed(int count, int *b_len, MPI_Aint *disp,
	MPI_Datatype oldtype, MPI_Datatype *newtype);
__must_check int MPI_Type_create_hindexed(int count, int *b_len, MPI_Aint *disp,
	MPI_Datatype oldtype, MPI_Datatype *newtype);
__must_check int MPI_Type_vector(int count, int b_len, int stride, MPI_Datatype oldtype,
	MPI_Datatype *newtype);
__must_check int MPI_Type_hvector(int count, int b_len, MPI_Aint stride, MPI_Datatype oldtype,
	MPI_Datatype *newtype);
__must_check int MPI_Type_create_hvector(int count, int b_len, MPI_Aint stride, MPI_Datatype oldtype,
	MPI_Datatype *newtype);
__must_check int MPI_Pack_size(int count, MPI_Datatype datatype, MPI_Comm comm, int *size);
__must_check int MPI_Pack(void *buf, int count, MPI_Datatype datatype, void *pack_buf,
	int pack_size, int *pos, MPI_Comm comm);
__must_check int MPI_Unpack(void *pack_buf, int pack_size, int *pos, void *buf, int count,
	MPI_Datatype datatype, MPI_Comm comm);
__must_check int MPI_Get_processor_name(char *name, int *len);
__must_check int MPI_Abort(MPI_Comm comm, int errcode);
__must_check int MPI_Address(void *ptr, MPI_Aint *addr);
__must_check int MPI_Get_address(void *ptr, MPI_Aint *addr);
__must_check int MPI_Pcontrol(const int level, ...);

double MPI_Wtime(void);
double MPI_Wtick(void);

__must_check int MPI_Keyval_create(MPI_Copy_function *copy, MPI_Delete_function *del,
	int *keyval, void *extra_state);
__must_check int MPI_Keyval_free(int *keyval);
__must_check int MPI_Attr_put(MPI_Comm comm, int keyval, void *attr);
__must_check int MPI_Attr_get(MPI_Comm comm, int keyval, void *attr, int *flag);
__must_check int MPI_Attr_delete(MPI_Comm comm, int keyval);
__must_check int MPI_Errhandler_create(MPI_Handler_function *func, MPI_Errhandler *errh);
__must_check int MPI_Errhandler_free(MPI_Errhandler *errh);
__must_check int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errh);
__must_check int MPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errh);
__must_check int MPI_Error_string(int errcode, char *out, int *len);
__must_check int MPI_Error_class(int errcode, int *errclass);

static inline MPI_Request MPI_Request_f2c(MPI_Fint request){return request;}
static inline MPI_Fint MPI_Request_c2f(MPI_Request request){return request;}
static inline MPI_Comm MPI_Comm_f2c(MPI_Fint comm){return comm;}
static inline MPI_Fint MPI_Comm_c2f(MPI_Comm comm){return comm;}
static inline MPI_Datatype MPI_Type_f2c(MPI_Fint type){return type;}
static inline MPI_Fint MPI_Type_c2f(MPI_Datatype type){return type;}
static inline MPI_Group MPI_Group_f2c(MPI_Fint group){return group;}
static inline MPI_Fint MPI_Group_c2f(MPI_Group group){return group;}
/* static inline MPI_File MPI_File_f2c(MPI_Fint file){return file;} */
/* static inline MPI_Fint MPI_File_c2f(MIP_File file){return file;} */
/* static inline MPI_Win MPI_Win_f2c(MPI_Fint win){return win;} */
/* static inline MPI_Fint MPI_Win_c2f(MPI_Win win){return win;} */
static inline MPI_Op MPI_Op_f2c(MPI_Fint op){return op;}
static inline MPI_Fint MPI_Op_c2f(MPI_Op op){return op;}
/* static inline MPI_Info MPI_Info_f2c(MPI_Fint info){return info;} */
/* static inline MPI_Fint MPI_Info_c2f(MPI_Info info){return info;} */
__must_check int MPI_Status_f2c(MPI_Fint *f_status, MPI_Status *c_status);
__must_check int MPI_Status_c2f(MPI_Status *c_status, MPI_Fint *f_status);


/* Extensions  */
__must_check int MPI_bufcopy_set(MPI_Aint size);
__must_check int MPI_bufcopy_get(MPI_Aint *size);

#ifdef __cplusplus
}
#ifndef MPI_SKIP_CXX
#include "mpicxx.h"
#endif 
#endif

#endif /* MPI_H */
