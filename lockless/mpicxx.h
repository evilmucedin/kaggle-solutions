/* Lockless MPI C++ header */

#ifndef MPI_H
#error "Please include mpi.h and let it include mpicxx.h for you"
#endif

#include <vector>

namespace MPI {

typedef MPI_Offset Offset;
typedef MPI_Aint Aint;
typedef MPI_Fint Fint;

class Comm;
class Intracomm;
class Intercomm;
class Cartcomm;
class Graphcomm;
class Datatype;
class Errhandler;
class Group;
class Op;
class Request;
class Status;

typedef void User_function(const void *inbuf, void *outbuf, int len, const Datatype &datatype);
typedef void Errhandler_function(Comm &comm, int *err, ...);

/* This is set by the ERRORS_THROW_EXCEPTIONS handler, which then returns MPI_ERR_PENDING */
extern __thread int MPI_errno;
__attribute__((__noreturn__)) void throw_exception(void);

/* Check for MPI_ERR_PENDING, then raise the correct exception when needed */
#define M(E) if (__builtin_expect((E) == MPI_ERR_PENDING, 0)) throw_exception()

void Pcontrol(const int level, ...);

static inline int Get_error_class(int errcode);

class Exception
{
private:
	unsigned handle_;

public:

	Exception(unsigned h): handle_(h) {}
	Exception(void): handle_(0) {}

	virtual ~Exception() {}

	Exception(const Exception &ex): handle_(ex.handle_) {}

	Exception& operator=(const Exception &ex)
	{
		handle_ = ex.handle_;
		return *this;
	}

	bool operator== (const Exception &ex) const {return handle_ == ex.handle_;}
	bool operator!= (const Exception &ex) const {return handle_ != ex.handle_;}

	operator unsigned*() {return &handle_;}
	operator unsigned() const {return handle_;}
	Exception& operator=(const unsigned &h)
	{
		handle_ = h;
		return *this;
	}

	int Get_error_code(void) const {return handle_;}
	int Get_error_class(void) const {return MPI::Get_error_class(handle_);}

	const char *Get_error_string(void) const;
};

class Errhandler
{
	friend class Comm;

protected:
	MPI_Errhandler handle_;

public:
	Errhandler(MPI_Errhandler eh): handle_(eh) {}
	Errhandler(void): handle_(MPI_ERRHANDLER_NULL) {}

	virtual ~Errhandler() {}

	Errhandler(const Errhandler &eh): handle_(eh.handle_) {}

	Errhandler& operator=(const Errhandler &eh)
	{
		handle_ = eh.handle_;
		return *this;
	}

	bool operator== (const Errhandler &eh) const {return handle_ == eh.handle_;}
	bool operator!= (const Errhandler &eh) const {return handle_ != eh.handle_;}

	operator MPI_Errhandler*() {return &handle_;}
	operator MPI_Errhandler() const {return handle_;}
	Errhandler& operator=(const MPI_Errhandler &eh)
	{
		handle_ = eh;
		return *this;
	}

	virtual void Free(void) {M(MPI_Errhandler_free(&handle_));}
	virtual void Create(Errhandler_function *func)
	{
		M(MPI_Errhandler_create((MPI_Handler_function *)func, &handle_));
	}
};

class Group
{
	friend class Comm;
	friend class Intracomm;
	friend class Intercomm;

protected:
	MPI_Group handle_;

public:
	Group(MPI_Group group): handle_(group) {}
	Group(void): handle_(MPI_GROUP_NULL) {}

	virtual ~Group() {}

	Group(const Group &group): handle_(group.handle_) {}

	Group& operator=(const Group &group)
	{
		handle_ = group.handle_;
		return *this;
	}

	bool operator== (const Group &group) const {return handle_ == group.handle_;}
	bool operator!= (const Group &group) const {return handle_ != group.handle_;}

	operator MPI_Group*() {return &handle_;}
	operator MPI_Group() const {return handle_;}
	Group& operator=(const MPI_Group &group)
	{
		handle_ = group;
		return *this;
	}
	virtual void Free(void){M(MPI_Group_free(&handle_));}
	virtual Group Incl(int n, const int *ranks) const
	{
		Group group;
		M(MPI_Group_incl(handle_, n, const_cast<int *>(ranks), &group.handle_));
		return group;
	}
	static int Compare(const Group &g1, const Group &g2)
	{
		int result;
		M(MPI_Group_compare(g1.handle_, g2.handle_, &result));
		return result;
	}
	static Group Difference(const Group &g1, const Group &g2)
	{
		Group group;
		M(MPI_Group_difference(g1.handle_, g2.handle_, &group.handle_));
		return group;
	}
	virtual Group Excl(int n, const int *ranks) const
	{
		Group group;
		M(MPI_Group_excl(handle_, n, const_cast<int *>(ranks), &group.handle_));
		return group;
	}
	static Group Intersect(const Group &g1, const Group &g2)
	{
		Group group;
		M(MPI_Group_intersection(g1.handle_, g2.handle_, &group.handle_));
		return group;
	}
	virtual Group Range_excl(int n, const int ranges[][3]) const
	{
		Group group;
		M(MPI_Group_range_excl(handle_, n, const_cast<int (*)[3]>(ranges), &group.handle_));
		return group;
	}
	virtual Group Range_incl(int n, const int ranges[][3]) const
	{
		Group group;
		M(MPI_Group_range_incl(handle_, n, const_cast<int (*)[3]>(ranges), &group.handle_));
		return group;
	}
	virtual int Get_rank(void) const
	{
		int rank;
		M(MPI_Group_rank(handle_, &rank));
		return rank;
	}
	virtual int Get_size(void) const
	{
		int size;
		M(MPI_Group_size(handle_, &size));
		return size;
	}
	static void Translate_ranks(const Group &g1, int n, const int *r1, const Group &g2, int *r2)
	{
		M(MPI_Group_translate_ranks(g1.handle_, n, const_cast<int *>(r1), g2.handle_, r2));
	}

	static Group Union(const Group &g1, const Group &g2)
	{
		Group group;
		M(MPI_Group_union(g1.handle_, g2.handle_, &group.handle_));
		return group;
	}
};

class Op
{
	friend class Comm;
	friend class Intracomm;

protected:
	MPI_Op handle_;

	public:
	 Op(MPI_Op op): handle_(op) {}
	 Op(void): handle_(MPI_OP_NULL) {}

	virtual ~Op() {}

	Op(const Op &op): handle_(op.handle_) {}

	Op& operator=(const Op &op)
	{
		handle_ = op.handle_;
		return *this;
	}

	bool operator== (const Op &op) const {return handle_ == op.handle_;}
	bool operator!= (const Op &op) const {return handle_ != op.handle_;}

	operator MPI_Op*() {return &handle_;}
	operator MPI_Op() const {return handle_;}
	Op& operator=(const MPI_Op &op)
	{
		handle_ = op;
		return *this;
	}
	virtual void Free(void){M(MPI_Op_free(&handle_));}

	void Init(User_function *func, bool commute);
};

class Datatype
{
	friend class Comm;
	friend class Status;
	friend class Intracomm;
	friend class Intercomm;
	friend class Op;

protected:
	MPI_Datatype handle_;

public:
	Datatype(MPI_Datatype datatype): handle_(datatype) {}
	Datatype(void): handle_(MPI_DATATYPE_NULL) {}

	virtual ~Datatype() {}

	Datatype(const Datatype &datatype): handle_(datatype.handle_) {}

	Datatype& operator=(const Datatype &datatype)
	{
		handle_ = datatype.handle_;
		return *this;
	}

	bool operator== (const Datatype &datatype) const {return handle_ == datatype.handle_;}
	bool operator!= (const Datatype &datatype) const {return handle_ != datatype.handle_;}

	operator MPI_Datatype*() {return &handle_;}
	operator MPI_Datatype() const {return handle_;}
	Datatype& operator=(const MPI_Datatype &datatype)
	{
		handle_ = datatype;
		return *this;
	}

	virtual void Commit(void) {M(MPI_Type_commit(&handle_));}
	virtual void Free(void) {M(MPI_Type_free(&handle_));}
	virtual int Get_size(void) const
	{
		int size;
		M(MPI_Type_size(handle_, &size));
		return size;
	}
	virtual void Get_true_extent(Aint &true_lb, Aint &true_extent) const
	{
		M(MPI_Type_get_true_extent(handle_, &true_lb, &true_extent));
	}
	static Datatype Create_struct(int count, const int *blen, const Aint *disp, const Datatype *oldtype)
	{
		Datatype datatype;
		std::vector<MPI_Datatype> oldtype_c(oldtype, &oldtype[count]);

		M(MPI_Type_create_struct(count, const_cast<int *>(blen), const_cast<Aint *>(disp), &oldtype_c[0], &datatype.handle_));
		return datatype;
	}
	virtual Datatype Create_contiguous(int count) const
	{
		Datatype datatype;
		M(MPI_Type_contiguous(count, handle_, &datatype.handle_));
		return datatype;
	}
	virtual Datatype Create_indexed(int count, const int *b_len, const int *indices) const
	{
		Datatype datatype;
		M(MPI_Type_indexed(count, const_cast<int *>(b_len), const_cast<int *>(indices), handle_, &datatype.handle_));
		return datatype;
	}
	virtual Datatype Create_hindexed(int count, const int *b_len, const Aint *stride) const
	{
		Datatype datatype;
		M(MPI_Type_create_hindexed(count, const_cast<int *>(b_len), const_cast<Aint *>(stride), handle_, &datatype.handle_));
		return datatype;
	}
	virtual Datatype Create_vector(int count, int b_len, int stride) const
	{
		Datatype datatype;
		M(MPI_Type_vector(count, b_len, stride, handle_, &datatype.handle_));
		return datatype;
	}
	virtual Datatype Create_hvector(int count, int b_len, Aint stride) const
	{
		Datatype datatype;
		M(MPI_Type_create_hvector(count, b_len, stride, handle_, &datatype.handle_));
		return datatype;
	}
	virtual inline int Pack_size(int count, const Comm &comm) const;
	virtual inline void Pack(const void *buf, int count, void *pack_buf, int pack_size, int &pos, const Comm &comm) const;
	virtual inline void Unpack(const void *pack_buf, int pack_size, void *buf, int count, int &pos, const Comm &comm) const;
};

class Status
{
	friend class Request;
	friend class Comm;
protected:
	MPI_Status status_;

public:
	Status(MPI_Status s): status_(s) {}
	Status(void): status_() {}

	virtual ~Status() {}

	Status(const Status &s): status_(s.status_) {}

	Status& operator=(const Status &s)
	{
		status_ = s.status_;
		return *this;
	}

	operator MPI_Status*() {return &status_;}
	operator MPI_Status() const {return status_;}
	Status& operator=(const MPI_Status &s)
	{
		status_ = s;
		return *this;
	}
	int Get_source(void) const {return status_.MPI_SOURCE;}
	int Get_tag(void) const {return status_.MPI_TAG;}
	int Get_error(void) const {return status_.MPI_ERROR;}
	void Set_source(int source) {status_.MPI_SOURCE = source;}
	void Set_tag(int tag) {status_.MPI_TAG = tag;}
	void Set_error(int error) {status_.MPI_ERROR = error;}
	virtual bool Is_cancelled(void) const
	{
		int flag;
		M(MPI_Test_cancelled(const_cast<MPI_Status *>(&status_), &flag));
		return flag;
	}
	virtual int Get_elements(const Datatype &datatype) const
	{
		int elems;
		M(MPI_Get_elements(const_cast<MPI_Status *>(&status_), datatype.handle_, &elems));
		return elems;
	}
	virtual int Get_count(const Datatype &datatype) const
	{
		int count;
		M(MPI_Get_count(const_cast<MPI_Status *>(&status_), datatype.handle_, &count));
		return count;
	}
};

class Request
{
	friend class Comm;

protected:
	MPI_Request handle_;

public:
	Request(MPI_Request req): handle_(req) {}
	Request(void): handle_(MPI_REQUEST_NULL) {}

	virtual ~Request() {}

	Request(const Request &req): handle_(req.handle_) {}

	Request& operator=(const Request &req)
	{
		handle_ = req.handle_;
		return *this;
	}

	bool operator== (const Request &req) const {return handle_ == req.handle_;}
	bool operator!= (const Request &req) const {return handle_ != req.handle_;}

	operator MPI_Request*() {return &handle_;}
	operator MPI_Request() const {return handle_;}
	Request& operator=(const MPI_Request &req)
	{
		handle_ = req;
		return *this;
	}
	virtual void Free(void) {M(MPI_Request_free(&handle_));}
	virtual void Cancel(void) const
	{
		M(MPI_Cancel(const_cast<MPI_Request *>(&handle_)));
	}
	virtual bool Test(Status &status)
	{
		int flag;
		M(MPI_Test(&handle_, &flag, &status.status_));
		return flag;
	}
	virtual bool Test(void)
	{
		int flag;
		M(MPI_Test(&handle_, &flag, MPI_STATUS_IGNORE));
		return flag;
	}
	static bool Testall(int count, Request *request_array, Status *status_array)
	{
		int flag;

		std::vector<MPI_Request> req_c(request_array, &request_array[count]);
		std::vector<MPI_Status> status_c(count);

		M(MPI_Testall(count, &req_c[0], &flag, &status_c[0]));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		for (int i = 0; i < count; i++)
		{
			status_array[i].status_ = status_c[i];
		}

		return flag;
	}

	static bool Testall(int count, Request *request_array)
	{
		int flag;

		std::vector<MPI_Request> req_c(request_array, &request_array[count]);

		M(MPI_Testall(count, &req_c[0], &flag, MPI_STATUSES_IGNORE));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		return flag;
	}
	static bool Testany(int count, Request *request_array, int &index, Status &status)
	{
		int flag;
		std::vector<MPI_Request> req_c(request_array, &request_array[count]);

		M(MPI_Testany(count, &req_c[0], &index, &flag, &status.status_));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		return flag;
	}
	static bool Testany(int count, Request *request_array, int &index)
	{
		int flag;
		std::vector<MPI_Request> req_c(request_array, &request_array[count]);

		M(MPI_Testany(count, &req_c[0], &index, &flag, MPI_STATUS_IGNORE));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		return flag;
	}
	static int Testsome(int count, Request *request_array, int *index_array, Status *status_array)
	{
		int out_count;

		std::vector<MPI_Request> req_c(request_array, &request_array[count]);
		std::vector<MPI_Status> status_c(count);

		M(MPI_Testsome(count, &req_c[0], &out_count, index_array, &status_c[0]));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		for (int i = 0; i < out_count; i++)
		{
			status_array[i].status_ = status_c[i];
		}

		return out_count;
	}
	static int Testsome(int count, Request *request_array, int *index_array)
	{
		int out_count;

		std::vector<MPI_Request> req_c(request_array, &request_array[count]);

		M(MPI_Testsome(count, &req_c[0], &out_count, index_array, MPI_STATUSES_IGNORE));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		return out_count;
	}
	virtual void Wait(Status &status) {M(MPI_Wait(&handle_, &status.status_));}
	virtual void Wait(void) {M(MPI_Wait(&handle_, MPI_STATUS_IGNORE));}
	static void Waitall(int count, Request *request_array, Status *status_array)
	{
		std::vector<MPI_Request> req_c(request_array, &request_array[count]);
		std::vector<MPI_Status> status_c(count);

		M(MPI_Waitall(count, &req_c[0], &status_c[0]));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		for (int i = 0; i < count; i++)
		{
			status_array[i].status_ = status_c[i];
		}
	}
	static void Waitall(int count, Request *request_array)
	{
		std::vector<MPI_Request> req_c(request_array, &request_array[count]);

		M(MPI_Waitall(count, &req_c[0], MPI_STATUSES_IGNORE));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}
	}
	static int Waitany(int count, Request *request_array, Status &status)
	{
		int index;

		std::vector<MPI_Request> req_c(request_array, &request_array[count]);

		M(MPI_Waitany(count, &req_c[0], &index, &status.status_));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		return index;
	}
	static int Waitany(int count, Request *request_array)
	{
		int index;

		std::vector<MPI_Request> req_c(request_array, &request_array[count]);

		M(MPI_Waitany(count, &req_c[0], &index, MPI_STATUS_IGNORE));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		return index;
	}
	static int Waitsome(int count, Request *request_array, int *index_array, Status *status_array)
	{
		int out_count;

		std::vector<MPI_Request> req_c(request_array, &request_array[count]);
		std::vector<MPI_Status> status_c(count);

		M(MPI_Waitsome(count, &req_c[0], &out_count, index_array, &status_c[0]));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		for (int i = 0; i < out_count; i++)
		{
			status_array[i].status_ = status_c[i];
		}

		return out_count;
	}
	static int Waitsome(int count, Request *request_array, int *index_array)
	{
		int out_count;

		std::vector<MPI_Request> req_c(request_array, &request_array[count]);

		M(MPI_Waitsome(count, &req_c[0], &out_count, index_array, MPI_STATUSES_IGNORE));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}

		return out_count;
	}
};

class Prequest : public Request
{
public:
	Prequest(MPI_Request req): Request(req) {}
	Prequest(void): Request() {}

	virtual ~Prequest() {}

	Prequest(const Prequest &req): Request(req) {}

	Prequest& operator=(const Prequest &req)
	{
		handle_ = req.handle_;
		return *this;
	}

	operator MPI_Request*() {return &handle_;}
	operator MPI_Request() const {return handle_;}
	Prequest& operator=(const MPI_Request &req)
	{
		handle_ = req;
		return *this;
	}
	virtual void Start(void) {M(MPI_Start(&handle_));}
	static void Startall(int count, Prequest *request_array)
	{
		std::vector<MPI_Request> req_c(request_array, &request_array[count]);

		M(MPI_Startall(count, &req_c[0]));

		for (int i = 0; i < count; i++)
		{
			request_array[i].handle_ = req_c[i];
		}
	}
};

class Comm
{
	friend class Intercomm;
	friend class Intracomm;
	friend class Datatype;

protected:
	MPI_Comm handle_;

public:
	Comm(MPI_Comm comm) : handle_(comm) {}
	Comm(void) : handle_(MPI_COMM_NULL) {}

	virtual ~Comm() {}

	Comm(const Comm &comm) : handle_(comm.handle_) {}

	Comm& operator=(const Comm &comm)
	{
		handle_ = comm.handle_;
		return *this;
	}

	bool operator== (const Comm &comm) const {return handle_ == comm.handle_;}
	bool operator!= (const Comm &comm) const {return handle_ != comm.handle_;}

	operator MPI_Comm*() {return &handle_;}
	operator MPI_Comm() const {return handle_;}
	Comm& operator=(const MPI_Comm &comm)
	{
		handle_ = comm;
		return *this;
	}
	virtual bool Is_inter(void) const
	{
		int flag;
		M(MPI_Comm_test_inter(handle_, &flag));
		return flag;
	}
	virtual int Get_size(void) const
	{
		int size;
		M(MPI_Comm_size(handle_, &size));
		return size;
	}
	virtual Group Get_group(void) const
	{
		Group group;
		M(MPI_Comm_group(handle_, &group.handle_));
		return group;
	}
	virtual int Get_rank(void) const
	{
		int rank;
		M(MPI_Comm_rank(handle_, &rank));
		return rank;
	}
	virtual int Get_topology(void) const
	{
		int flag;
		M(MPI_Topo_test(handle_, &flag));
		return flag;
	}
	virtual void Free(void) {M(MPI_Comm_free(&handle_));}
	static int Compare(const Comm &c1, const Comm &c2) 
	{
		int result;
		M(MPI_Comm_compare(c1.handle_, c2.handle_, &result));
		return result;
	}
	/* Fake const, so can rename built-in comms */
	virtual void Set_name(const char *name) const
	{
		M(MPI_Comm_set_name(handle_, const_cast<char *>(name)));
	}
	virtual void Get_name(char *name, int &len) const
	{
		M(MPI_Comm_get_name(handle_, name, &len));
	}
	/* Fake const, so can change errhandler on builtin comms */
	virtual void Set_errhandler(const Errhandler &eh) const
	{
		M(MPI_Errhandler_set(handle_, eh.handle_));
	}
	virtual Errhandler Get_errhandler(void) const
	{
		Errhandler eh;
		M(MPI_Errhandler_get(handle_, &eh.handle_));
		return eh;
	}
	virtual void Abort(int errcode) const {MPI_Abort(handle_, errcode);}
	virtual void Probe(int source, int tag, Status &status) const
	{
		M(MPI_Probe(source, tag, handle_, &status.status_));
	}
	virtual void Probe(int source, int tag) const
	{
		M(MPI_Probe(source, tag, handle_, MPI_STATUS_IGNORE));
	}
	virtual bool Iprobe(int source, int tag, Status &status) const
	{
		int flag;
		M(MPI_Iprobe(source, tag, handle_, &flag, &status.status_));
		return flag;
	}
	virtual bool Iprobe(int source, int tag) const
	{
		int flag;
		M(MPI_Iprobe(source, tag, handle_, &flag, MPI_STATUS_IGNORE));
		return flag;
	}
	virtual void Send(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		M(MPI_Send(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_));
	}
	virtual Request Isend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		Request req;
		M(MPI_Isend(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_, &req.handle_));
		return req;
	}
	virtual void Ssend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		M(MPI_Ssend(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_));
	}
	virtual Request Issend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		Request req;
		M(MPI_Issend(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_, &req.handle_));
		return req;
	}
	virtual void Rsend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		M(MPI_Rsend(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_));
	}
	virtual Request Irsend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		Request req;
		M(MPI_Irsend(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_, &req.handle_));
		return req;
	}
	virtual void Bsend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		M(MPI_Bsend(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_));
	}
	virtual Request Ibsend(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		Request req;
		M(MPI_Ibsend(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_, &req.handle_));
		return req;
	}
	virtual void Recv(void *buf, int count, const Datatype &datatype, int source, int tag, Status &status) const
	{
		M(MPI_Recv(buf, count, datatype.handle_, source, tag, handle_, &status.status_));
	}
	virtual void Recv(void *buf, int count, const Datatype &datatype, int source, int tag) const
	{
		M(MPI_Recv(buf, count, datatype.handle_, source, tag, handle_, MPI_STATUS_IGNORE));
	}
	virtual Request Irecv(void *buf, int count, const Datatype &datatype, int source, int tag) const
	{
		Request req;
		M(MPI_Irecv(buf, count, datatype.handle_, source, tag, handle_, &req.handle_));
		return req;
	}
	virtual void Sendrecv(const void *sendbuf, int sendcount, const Datatype &sendtype, int dest, int sendtag, void *recvbuf, int recvcount, const Datatype &recvtype, int source, int recvtag, Status &status) const
	{
		M(MPI_Sendrecv(const_cast<void *>(sendbuf), sendcount, sendtype.handle_, dest, sendtag,
			recvbuf, recvcount, recvtype.handle_, source, recvtag, handle_, &status.status_));
	}
	virtual void Sendrecv(const void *sendbuf, int sendcount, const Datatype &sendtype, int dest, int sendtag, void *recvbuf, int recvcount, const Datatype &recvtype, int source, int recvtag) const
	{
		M(MPI_Sendrecv(const_cast<void *>(sendbuf), sendcount, sendtype.handle_, dest, sendtag,
			recvbuf, recvcount, recvtype.handle_, source, recvtag, handle_, MPI_STATUS_IGNORE));
	}
	virtual void Sendrecv_replace(void *buf, int count, const Datatype &datatype, int dest, int sendtag, int source, int recvtag, Status &status) const
	{
		M(MPI_Sendrecv_replace(buf, count, datatype.handle_, dest, sendtag, source, recvtag, handle_, &status.status_));
	}
	virtual void Sendrecv_replace(void *buf, int count, const Datatype &datatype, int dest, int sendtag, int source, int recvtag) const
	{
		M(MPI_Sendrecv_replace(buf, count, datatype.handle_, dest, sendtag, source, recvtag, handle_, MPI_STATUS_IGNORE));
	}

	virtual Prequest Send_init(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		Prequest req;
		M(MPI_Send_init(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_, &req.handle_));
		return req;
	}
	virtual Prequest Ssend_init(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		Prequest req;
		M(MPI_Ssend_init(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_, &req.handle_));
		return req;
	}
	virtual Prequest Rsend_init(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		Prequest req;
		M(MPI_Rsend_init(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_, &req.handle_));
		return req;
	}
	virtual Prequest Bsend_init(const void *buf, int count, const Datatype &datatype, int dest, int tag) const
	{
		Prequest req;
		M(MPI_Bsend_init(const_cast<void *>(buf), count, datatype.handle_, dest, tag, handle_, &req.handle_));
		return req;
	}
	virtual Prequest Recv_init(void *buf, int count, const Datatype &datatype, int source, int tag) const
	{
		Prequest req;
		M(MPI_Recv_init(buf, count, datatype.handle_, source, tag, handle_, &req.handle_));
		return req;
	}

	virtual void Barrier(void) const {M(MPI_Barrier(handle_));}
	virtual void Bcast(void *buf, int count, const Datatype &datatype, int root) const
	{
		M(MPI_Bcast(buf, count, datatype.handle_, root, handle_));
	}
	virtual void Scatter(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype, int root) const
	{
		M(MPI_Scatter(const_cast<void *>(sendbuf), sendcount, sendtype.handle_, recvbuf, recvcount, recvtype.handle_, root, handle_));
	}
	virtual void Scatterv(const void *sendbuf, const int *sendcount, const int *disp, const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype, int root) const
	{
		M(MPI_Scatterv(const_cast<void *>(sendbuf), const_cast<int *>(sendcount), const_cast<int *>(disp), sendtype.handle_, recvbuf, recvcount, recvtype.handle_, root, handle_));
	}
	virtual void Gather(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype, int root) const
	{
		M(MPI_Gather(const_cast<void *>(sendbuf), sendcount, sendtype.handle_, recvbuf, recvcount, recvtype.handle_, root, handle_));
	}
	virtual void Gatherv(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, const int *recvcount, const int *displs, const Datatype &recvtype, int root) const
	{
		M(MPI_Gatherv(const_cast<void *>(sendbuf), sendcount, sendtype.handle_,
			recvbuf, const_cast<int *>(recvcount), const_cast<int *>(displs), recvtype.handle_, root, handle_));
	}
	virtual void Allgather(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype) const
	{
		M(MPI_Allgather(const_cast<void *>(sendbuf), sendcount, sendtype.handle_, recvbuf, recvcount, recvtype.handle_, handle_));
	}
	virtual void Allgatherv(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, const int *recvcount, const int *disp, const Datatype &recvtype) const
	{
		M(MPI_Allgatherv(const_cast<void *>(sendbuf), sendcount, sendtype.handle_, recvbuf, const_cast<int *>(recvcount), const_cast<int *>(disp), recvtype.handle_, handle_));
	}
	virtual void Alltoall(const void *sendbuf, int sendcount, const Datatype &sendtype, void *recvbuf, int recvcount, const Datatype &recvtype) const
	{
		M(MPI_Alltoall(const_cast<void *>(sendbuf), sendcount, sendtype.handle_, recvbuf, recvcount, recvtype.handle_, handle_));
	}
	virtual void Alltoallv(const void *sendbuf, const int *sendcount, const int *sdisp, const Datatype &sendtype, void *recvbuf, const int *recvcount, const int *rdisp, const Datatype &recvtype) const
	{
		M(MPI_Alltoallv(const_cast<void *>(sendbuf), const_cast<int *>(sendcount), const_cast<int *>(sdisp), sendtype.handle_, recvbuf, const_cast<int *>(recvcount), const_cast<int *>(rdisp), recvtype.handle_, handle_));
	}
	virtual void Alltoallw(const void *sendbuf, const int *sendcount, const int *sdisp, const Datatype *sendtype, void *recvbuf, const int *recvcount, const int *rdisp, const Datatype *recvtype) const
	{
		int size = Get_size();
		std::vector<MPI_Datatype> sendtype_c(sendtype, &sendtype[size]);
		std::vector<MPI_Datatype> recvtype_c(recvtype, &recvtype[size]);

		M(MPI_Alltoallw(const_cast<void *>(sendbuf), const_cast<int *>(sendcount), const_cast<int *>(sdisp), &sendtype_c[0], recvbuf, const_cast<int *>(recvcount), const_cast<int *>(rdisp), &recvtype_c[0], handle_));
	}
	virtual void Reduce(const void *sendbuf, void *recvbuf, int count, const Datatype &datatype, const Op &op, int root) const
	{
		M(MPI_Reduce(const_cast<void *>(sendbuf), recvbuf, count, datatype.handle_, op.handle_, root, handle_));
	}
	virtual void Allreduce(const void *sendbuf, void *recvbuf, int count, const Datatype &datatype, const Op &op) const
	{
		M(MPI_Allreduce(const_cast<void *>(sendbuf), recvbuf, count, datatype.handle_, op.handle_, handle_));
	}
	virtual void Reduce_scatter(const void *sendbuf, void *recvbuf, int *counts, const Datatype &datatype, const Op &op) const
	{
		M(MPI_Reduce_scatter(const_cast<void *>(sendbuf), recvbuf, counts, datatype.handle_, op.handle_, handle_));
	}
};

class Nullcomm : public Comm
{
public:
	Nullcomm(MPI_Comm comm) : Comm(comm) {}
	Nullcomm(void) : Comm(MPI_COMM_NULL) {}

	virtual ~Nullcomm() {}

	Nullcomm(const Nullcomm &comm) : Comm(MPI_COMM_NULL) {(void) comm;}

	Nullcomm& operator=(const Nullcomm &comm)
	{
		(void) comm;
		handle_ = MPI_COMM_NULL;
		return *this;
	}

	operator MPI_Comm*() {return &handle_;}
	operator MPI_Comm() const {return MPI_COMM_NULL;}
	Nullcomm& operator=(const MPI_Comm &comm)
	{
		(void) comm;
		handle_ = MPI_COMM_NULL;
		return *this;
	}
};

class Intercomm : public Comm
{
public:
	Intercomm(MPI_Comm comm) : Comm(comm) {}
	Intercomm(void) : Comm() {}

	virtual ~Intercomm() {}

	Intercomm(const Intercomm &comm) : Comm(comm) {}

	Intercomm& operator=(const Intercomm &comm)
	{
		handle_ = comm.handle_;
		return *this;
	}

	operator MPI_Comm*() {return &handle_;}
	operator MPI_Comm() const {return handle_;}
	Intercomm& operator=(const MPI_Comm &comm)
	{
		handle_ = comm;
		return *this;
	}
	virtual inline Intracomm Merge(bool high) const;
	virtual Group Get_remote_group(void) const
	{
		Group group;
		M(MPI_Comm_remote_group(handle_, &group.handle_));
		return group;
	}
	virtual int Get_remote_size(void) const
	{
		int size;
		M(MPI_Comm_remote_size(handle_, &size));
		return size;
	}
	Intercomm Dup(void) const
	{
		Intercomm comm;
		M(MPI_Comm_dup(handle_, &comm.handle_));
		return comm;
	}
	virtual Intercomm Split(int split_key, int key) const
	{
		Intercomm comm;
		M(MPI_Comm_split(handle_, split_key, key, &comm.handle_));
		return comm;
	}
};

class Intracomm : public Comm
{
public:
	Intracomm(MPI_Comm comm) : Comm(comm) {}
	Intracomm(void) : Comm() {}

	virtual ~Intracomm() {}

	Intracomm(const Intracomm &comm) : Comm(comm) {}

	Intracomm& operator=(const Intracomm &comm)
	{
		handle_ = comm.handle_;
		return *this;
	}

	operator MPI_Comm*() {return &handle_;}
	operator MPI_Comm() const {return handle_;}
	Intracomm& operator=(const MPI_Comm &comm)
	{
		handle_ = comm;
		return *this;
	}
	virtual Intercomm Create_intercomm(int local_leader, const Comm &peer_comm, int remote_leader, int tag) const
	{
		Intercomm comm;
		M(MPI_Intercomm_create(handle_, local_leader, peer_comm.handle_, remote_leader, tag, &comm.handle_));
		return comm;
	}
	virtual Intracomm Split(int split_key, int key) const
	{
		Intracomm comm;
		M(MPI_Comm_split(handle_, split_key, key, &comm.handle_));
		return comm;
	}
	virtual inline Graphcomm Create_graph(int nnodes, const int *index, const int *edges, bool reorder) const;
	virtual inline Cartcomm Create_cart(int ndims, const int *dims, const bool *periods, bool reorder) const;
	virtual Intracomm Create(const Group &group) const
	{
		Intracomm comm;
		M(MPI_Comm_create(handle_, group.handle_, &comm.handle_));
		return comm;
	}
	Intracomm Dup(void) const
	{
		Intracomm comm;
		M(MPI_Comm_dup(handle_, &comm.handle_));
		return comm;
	}
	virtual void Scan(const void *sendbuf, void *recvbuf, int count, const Datatype &datatype, const Op &op) const
	{
		M(MPI_Scan(const_cast<void *>(sendbuf), recvbuf, count, datatype.handle_, op.handle_, handle_));
	}
	virtual void Exscan(const void *sendbuf, void *recvbuf, int count, const Datatype &datatype, const Op &op) const
	{
		M(MPI_Exscan(const_cast<void *>(sendbuf), recvbuf, count, datatype.handle_, op.handle_, handle_));
	}
};

class Graphcomm : public Intracomm
{
public:
	Graphcomm(MPI_Comm comm) : Intracomm(comm) {}
	Graphcomm(void) : Intracomm() {}

	virtual ~Graphcomm() {}

	Graphcomm(const Graphcomm &comm) : Intracomm(comm) {}

	Graphcomm& operator=(const Graphcomm &comm)
	{
		handle_ = comm.handle_;
		return *this;
	}

	operator MPI_Comm*() {return &handle_;}
	operator MPI_Comm() const {return handle_;}
	Graphcomm& operator=(const MPI_Comm &comm)
	{
		handle_ = comm;
		return *this;
	}
	virtual void Get_dims(int *nodes, int *nedges) const
	{
		M(MPI_Graphdims_get(handle_, nodes, nedges));
	}
	virtual void Get_topo(int maxindex, int maxedges, int *index, int *edges) const
	{
		M(MPI_Graph_get(handle_, maxindex, maxedges, index, edges));
	}
	virtual int Get_neighbors_count(int rank) const
	{
		int count;
		M(MPI_Graph_neighbors_count(handle_, rank, &count));
		return count;
	}
	virtual void Get_neighbors(int rank, int maxneighbors, int *neighbors) const
	{
		M(MPI_Graph_neighbors(handle_, rank, maxneighbors, neighbors));
	}
	virtual int Map(int nnodes, const int *index, const int *edges) const
	{
		int newrank;
		M(MPI_Graph_map(handle_, nnodes, const_cast<int *>(index), const_cast<int *>(edges), &newrank));
		return newrank;
	}
	Graphcomm Dup(void) const
	{
		Graphcomm comm;
		M(MPI_Comm_dup(handle_, &comm.handle_));
		return comm;
	}
};

class Cartcomm : public Intracomm
{
public:
	Cartcomm(MPI_Comm comm) : Intracomm(comm) {}
	Cartcomm(void) : Intracomm() {}

	virtual ~Cartcomm() {}

	Cartcomm(const Cartcomm &comm) : Intracomm(comm) {}

	Cartcomm& operator=(const Cartcomm &comm)
	{
		handle_ = comm.handle_;
		return *this;
	}

	operator MPI_Comm*() {return &handle_;}
	operator MPI_Comm() const {return handle_;}
	Cartcomm& operator=(const MPI_Comm &comm)
	{
		handle_ = comm;
		return *this;
	}
	virtual int Get_dim(void) const
	{
		int dim;
		M(MPI_Cartdim_get(handle_, &dim));
		return dim;
	}
	virtual void Get_topo(int maxdims, int *dims, bool *periods, int *coords) const
	{
		int size = Get_size();
		std::vector<int> c_periods(periods, &periods[size]);

		M(MPI_Cart_get(handle_, maxdims, dims, &c_periods[0], coords));

		for (int i = 0; i < size; i++)
		{
			periods[i] = c_periods[i];
		}
	}
	virtual int Get_cart_rank(const int *coords) const
	{
		int rank;
		M(MPI_Cart_rank(handle_, const_cast<int *>(coords), &rank));
		return rank;
	}
	virtual void Get_coords(int rank, int maxdims, int *coords) const
	{
		M(MPI_Cart_coords(handle_, rank, maxdims, coords));
	}
	virtual void Shift(int direction, int disp, int &rsource, int &rdest) const
	{
		M(MPI_Cart_shift(handle_, direction, disp, &rsource, &rdest));
	}
	virtual Cartcomm Sub(const bool *remain_dims) const
	{
		Cartcomm comm;
		int size = Get_size();
		std::vector<int> c_rdims(remain_dims, &remain_dims[size]);

		M(MPI_Cart_sub(handle_, &c_rdims[0], &comm.handle_));

		return comm;
	}
	virtual int Map(int ndims, const int *dims, const bool *periods) const
	{
		int rank;
		int size = Get_size();
		std::vector<int> c_periods(periods, &periods[size]);

		M(MPI_Cart_map(handle_, ndims, const_cast<int *>(dims), &c_periods[0], &rank));

		return rank;
	}
	Cartcomm Dup(void) const
	{
		Cartcomm comm;
		M(MPI_Comm_dup(handle_, &comm.handle_));
		return comm;
	}
};

static inline void Attach_buffer(void *buffer, int size)
{
	M(MPI_Buffer_attach(buffer, size));
}

static inline int Detach_buffer(void *&buffer)
{
	int size;
	M(MPI_Buffer_detach(buffer, &size));
	return size;
}

static inline void Init(void)
{
	M(MPI_Init(0, 0));
}

static inline void Init(int &argc, char **&argv)
{
	M(MPI_Init(&argc, &argv));
}

static inline bool Is_initialized(void)
{
	int flag;
	M(MPI_Initialized(&flag));
	return flag;
}

static inline void Finalize(void)
{
	M(MPI_Finalize());
}

static inline bool Is_finalized(void)
{
	int flag;
	M(MPI_Finalized(&flag));
	return flag;
}

static inline void Get_processor_name(char *name, int &len)
{
	M(MPI_Get_processor_name(name, &len));
}

static inline void Get_version(int &version, int &subversion)
{
	M(MPI_Get_version(&version, &subversion));
}

static inline void Get_error_string(int errcode, char *out, int &len)
{
	M(MPI_Error_string(errcode, out, &len));
}

static inline int Get_error_class(int errcode)
{
	int errclass;
	M(MPI_Error_class(errcode, &errclass));
	return errclass;
}

static inline Aint Get_address(void *ptr)
{
	MPI_Aint addr;
	MPI_Get_address(ptr, &addr);
	return addr;
}

static inline void Compute_dims(int nnodes, int ndims, int *dims)
{
	M(MPI_Dims_create(nnodes, ndims, dims));
}

static inline double Wtime(void)
{
	return MPI_Wtime();
}

static inline double Wtick(void)
{
	return MPI_Wtick();
}

static inline void bufcopy_set(Aint size)
{
	M(MPI_bufcopy_set(size));
}

static inline Aint bufcopy_get(void)
{
	Aint size;
	M(MPI_bufcopy_get(&size));
	return size;
}

inline int Datatype::Pack_size(int count, const Comm &comm) const
{
	int size;
	M(MPI_Pack_size(count, handle_, comm.handle_, &size));
	return size;
}
inline void Datatype::Pack(const void *buf, int count, void *pack_buf, int pack_size, int &pos, const Comm &comm) const
{
	M(MPI_Pack(const_cast<void *>(buf), count, handle_, pack_buf, pack_size, &pos, comm.handle_));
}
inline void Datatype::Unpack(const void *pack_buf, int pack_size, void *buf, int count, int &pos, const Comm &comm) const
{
	M(MPI_Unpack(const_cast<void *>(pack_buf), pack_size, &pos, buf, count, handle_, comm.handle_));
}
inline Intracomm Intercomm::Merge(bool high) const
{
	Intracomm comm;
	M(MPI_Intercomm_merge(handle_, high, &comm.handle_));
	return comm;
}
inline Cartcomm Intracomm::Create_cart(int ndims, const int *dims, const bool *periods, bool reorder) const
{
	Cartcomm comm;
	std::vector<int> c_periods(periods, &periods[Get_size()]);

	M(MPI_Cart_create(handle_, ndims, const_cast<int *>(dims), &c_periods[0], reorder, &comm.handle_));
	return comm;
}
inline Graphcomm Intracomm::Create_graph(int nnodes, const int *index, const int *edges, bool reorder) const
{
	Graphcomm comm;
	M(MPI_Graph_create(handle_, nnodes, const_cast<int *>(index), const_cast<int *>(edges), reorder, &comm.handle_));
	return comm;
}

#undef M

static const int ANY_TAG = MPI_ANY_TAG;
static const int ANY_SOURCE = MPI_ANY_SOURCE;
static const int PROC_NULL = MPI_PROC_NULL;
static const int ROOT = MPI_ROOT;

static const Nullcomm COMM_NULL(MPI_COMM_NULL);
static const Intracomm COMM_WORLD(MPI_COMM_WORLD);
static const Intracomm COMM_SELF(MPI_COMM_SELF);

static void * const IN_PLACE = MPI_IN_PLACE;
static void * const BOTTOM = MPI_BOTTOM;

static const int MAX_PROCESSOR_NAME = MPI_MAX_PROCESSOR_NAME;
static const int MAX_OBJECT_NAME = MPI_MAX_OBJECT_NAME;
static const int BSEND_OVERHEAD = MPI_BSEND_OVERHEAD;
static const int MAX_ERROR_STRING = MPI_MAX_ERROR_STRING;

static const Datatype DATATYPE_NULL(MPI_DATATYPE_NULL);
static const Datatype LB(MPI_LB);
static const Datatype UB(MPI_UB);
static const Datatype PACKED(MPI_PACKED);
static const Datatype BYTE(MPI_BYTE);
static const Datatype CHAR(MPI_CHAR);
static const Datatype SHORT(MPI_SHORT);
static const Datatype INT(MPI_INT);
static const Datatype LONG(MPI_LONG);
static const Datatype LONG_LONG_INT(MPI_LONG_LONG_INT);
static const Datatype SIGNED_CHAR(MPI_SIGNED_CHAR);
static const Datatype UNSIGNED_CHAR(MPI_UNSIGNED_CHAR);
static const Datatype WCHAR(MPI_WCHAR);
static const Datatype UNSIGNED_SHORT(MPI_UNSIGNED_SHORT);
static const Datatype UNSIGNED(MPI_UNSIGNED);
static const Datatype UNSIGNED_LONG(MPI_UNSIGNED_LONG);
static const Datatype UNSIGNED_LONG_LONG(MPI_UNSIGNED_LONG_LONG);
static const Datatype INT8_T(MPI_INT8_T);
static const Datatype INT16_T(MPI_INT16_T);
static const Datatype INT32_T(MPI_INT32_T);
static const Datatype INT64_T(MPI_INT64_T);
static const Datatype INT128_T(MPI_INT128_T);
static const Datatype UINT8_T(MPI_UINT8_T);
static const Datatype UINT16_T(MPI_UINT16_T);
static const Datatype UINT32_T(MPI_UINT32_T);
static const Datatype UINT64_T(MPI_UINT64_T);
static const Datatype UINT128_T(MPI_UINT128_T);
static const Datatype FLOAT(MPI_FLOAT);
static const Datatype DOUBLE(MPI_DOUBLE);
static const Datatype LONG_DOUBLE(MPI_LONG_DOUBLE);
static const Datatype FLOAT128(MPI_FLOAT128);
static const Datatype C_FLOAT_COMPLEX(MPI_C_FLOAT_COMPLEX);
static const Datatype C_COMPLEX(MPI_C_COMPLEX);
static const Datatype C_DOUBLE_COMPLEX(MPI_C_DOUBLE_COMPLEX);
static const Datatype C_LONG_DOUBLE_COMPLEX(MPI_C_LONG_DOUBLE_COMPLEX);
static const Datatype C_FLOAT128_COMPLEX(MPI_C_FLOAT128_COMPLEX);
static const Datatype C_BOOL(MPI_C_BOOL);
static const Datatype CHARACTER(MPI_CHARACTER);
static const Datatype LOGICAL(MPI_LOGICAL);
static const Datatype INTEGER(MPI_INTEGER);
static const Datatype LOGICAL1(MPI_LOGICAL1);
static const Datatype LOGICAL2(MPI_LOGICAL2);
static const Datatype LOGICAL4(MPI_LOGICAL4);
static const Datatype LOGICAL8(MPI_LOGICAL8);
static const Datatype INTEGER1(MPI_INTEGER1);
static const Datatype INTEGER2(MPI_INTEGER2);
static const Datatype INTEGER4(MPI_INTEGER4);
static const Datatype INTEGER8(MPI_INTEGER8);
static const Datatype REAL(MPI_REAL);
static const Datatype DOUBLE_PRECISION(MPI_DOUBLE_PRECISION);
static const Datatype COMPLEX(MPI_COMPLEX);
static const Datatype DOUBLE_COMPLEX(MPI_DOUBLE_COMPLEX);
static const Datatype LONG_DOUBLE_COMPLEX(MPI_LONG_DOUBLE_COMPLEX);
static const Datatype REAL4(MPI_REAL4);
static const Datatype REAL8(MPI_REAL8);
static const Datatype REAL16(MPI_REAL16);
static const Datatype COMPLEX8(MPI_COMPLEX8);
static const Datatype COMPLEX16(MPI_COMPLEX16);
static const Datatype COMPLEX32(MPI_COMPLEX32);
static const Datatype AINT(MPI_AINT);
static const Datatype OFFSET(MPI_OFFSET);
static const Datatype FLOAT_INT(MPI_FLOAT_INT);
static const Datatype DOUBLE_INT(MPI_DOUBLE_INT);
static const Datatype LONG_DOUBLE_INT(MPI_LONG_DOUBLE_INT);
static const Datatype SHORT_INT(MPI_SHORT_INT);
static const Datatype TWOINT(MPI_2INT);
static const Datatype LONG_INT(MPI_LONG_INT);
static const Datatype TWOINTEGER(MPI_2INTEGER);
static const Datatype TWOREAL(MPI_2REAL);
static const Datatype TWODOUBLE_PRECISION(MPI_2DOUBLE_PRECISION);

static const Group GROUP_NULL(MPI_GROUP_NULL);
static const Group GROUP_EMPTY(MPI_GROUP_EMPTY);

static const Op OP_NULL(MPI_OP_NULL);
static const Op MAX(MPI_MAX);
static const Op MIN(MPI_MIN);
static const Op SUM(MPI_SUM);
static const Op PROD(MPI_PROD);
static const Op LAND(MPI_LAND);
static const Op BAND(MPI_BAND);
static const Op LOR(MPI_LOR);
static const Op BOR(MPI_BOR);
static const Op LXOR(MPI_LXOR);
static const Op BXOR(MPI_BXOR);
static const Op MAXLOC(MPI_MAXLOC);
static const Op MINLOC(MPI_MINLOC);

static const int SUCCESS = MPI_SUCCESS;
static const int ERR_OTHER = MPI_ERR_OTHER;
static const int ERR_COMM = MPI_ERR_COMM;
static const int ERR_COUNT = MPI_ERR_COUNT;
static const int ERR_TYPE = MPI_ERR_TYPE;
static const int ERR_BUFFER = MPI_ERR_BUFFER;
static const int ERR_ROOT = MPI_ERR_ROOT;
static const int ERR_TAG = MPI_ERR_TAG;
static const int ERR_RANK = MPI_ERR_RANK;
static const int ERR_ARG = MPI_ERR_ARG;
static const int ERR_REQUEST = MPI_ERR_REQUEST;
static const int ERR_IN_STATUS = MPI_ERR_IN_STATUS;
static const int ERR_PENDING = MPI_ERR_PENDING;
static const int ERR_INTERN = MPI_ERR_INTERN;
static const int ERR_GROUP = MPI_ERR_GROUP;
static const int ERR_CANCEL = MPI_ERR_CANCEL;
static const int ERR_OP = MPI_ERR_OP;
static const int ERR_TOPOLOGY = MPI_ERR_TOPOLOGY;
static const int ERR_DIMS = MPI_ERR_DIMS;
static const int ERR_UNKNOWN = MPI_ERR_UNKNOWN;
static const int ERR_TRUNCATE = MPI_ERR_TRUNCATE;
static const int ERR_KEYVAL = MPI_ERR_KEYVAL;
static const int ERR_LASTCODE = MPI_ERR_LASTCODE;

static const Errhandler ERRHANDLER_NULL(MPI_ERRHANDLER_NULL);
static const Errhandler ERRORS_ARE_FATAL(MPI_ERRORS_ARE_FATAL);
static const Errhandler ERRORS_RETURN(MPI_ERRORS_RETURN);
static const Errhandler ERRORS_THROW_EXCEPTIONS(MPI_ERRORS_THROW_EXCEPTIONS);

static const int IDENT = MPI_IDENT;
static const int CONGRUENT = MPI_CONGRUENT;
static const int SIMILAR = MPI_SIMILAR;
static const int UNEQUAL = MPI_UNEQUAL;

static const int CART = MPI_CART;
static const int GRAPH = MPI_GRAPH;

static const int UNDEFINED = MPI_UNDEFINED;
static const Request REQUEST_NULL(MPI_REQUEST_NULL);

static const int KEYVAL_INVALID = MPI_KEYVAL_INVALID;
static const int TAG_UB = MPI_TAG_UB;
static const int HOST = MPI_HOST;
static const int IO = MPI_IO;
static const int WTIME_IS_GLOBAL = MPI_WTIME_IS_GLOBAL;


} // namespace MPI
