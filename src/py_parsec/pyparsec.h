/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * Small C helpers for the Cython bindings.
 *
 * Only public PaRSEC headers are used, plus three that PaRSEC installs for
 * the code its compilers generate: parsec_internal.h (task->task_class),
 * execution_stream.h (the context's active flag) and remote_dep.h (the flow
 * limit of a task class). The bindings read these structures directly, which
 * is why the build records which PaRSEC it compiled against and the runtime
 * module refuses to load with another one (see pyparsec_build_matches()).
 */

#ifndef PYPARSEC_H_HAS_BEEN_INCLUDED
#define PYPARSEC_H_HAS_BEEN_INCLUDED

#include <Python.h>
#include <limits.h>
#include <string.h>

#include "parsec.h"
#include "parsec/parsec_internal.h"
#include "parsec/execution_stream.h"
#include "parsec/remote_dep.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/interfaces/dtd/insert_function.h"
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

/* ------------------------------------------------------------------------
 * Variadic DTD calls.
 *
 * Cython cannot build a variadic call whose length is only known at run time.
 * PYPARSEC_ARGS_<n>(X) expands to X(0) X(1) ... X(n-1), and one switch case
 * per argument count calls the variadic function with exactly the arguments
 * of the task, so nothing is padded and the callee walks no unused slot.
 * ------------------------------------------------------------------------ */

/* Largest number of task-class parameters the helpers handle. */
#if PARSEC_DTD_MAX_PARAMS < 64
#define PYPARSEC_MAX_PARAMS PARSEC_DTD_MAX_PARAMS
#else
#define PYPARSEC_MAX_PARAMS 64
#endif

#define PYPARSEC_ARGS_0(X)
#define PYPARSEC_ARGS_1(X)  X(0)
#define PYPARSEC_ARGS_2(X)  PYPARSEC_ARGS_1(X)  X(1)
#define PYPARSEC_ARGS_3(X)  PYPARSEC_ARGS_2(X)  X(2)
#define PYPARSEC_ARGS_4(X)  PYPARSEC_ARGS_3(X)  X(3)
#define PYPARSEC_ARGS_5(X)  PYPARSEC_ARGS_4(X)  X(4)
#define PYPARSEC_ARGS_6(X)  PYPARSEC_ARGS_5(X)  X(5)
#define PYPARSEC_ARGS_7(X)  PYPARSEC_ARGS_6(X)  X(6)
#define PYPARSEC_ARGS_8(X)  PYPARSEC_ARGS_7(X)  X(7)
#define PYPARSEC_ARGS_9(X)  PYPARSEC_ARGS_8(X)  X(8)
#define PYPARSEC_ARGS_10(X) PYPARSEC_ARGS_9(X)  X(9)
#define PYPARSEC_ARGS_11(X) PYPARSEC_ARGS_10(X) X(10)
#define PYPARSEC_ARGS_12(X) PYPARSEC_ARGS_11(X) X(11)
#define PYPARSEC_ARGS_13(X) PYPARSEC_ARGS_12(X) X(12)
#define PYPARSEC_ARGS_14(X) PYPARSEC_ARGS_13(X) X(13)
#define PYPARSEC_ARGS_15(X) PYPARSEC_ARGS_14(X) X(14)
#define PYPARSEC_ARGS_16(X) PYPARSEC_ARGS_15(X) X(15)
#define PYPARSEC_ARGS_17(X) PYPARSEC_ARGS_16(X) X(16)
#define PYPARSEC_ARGS_18(X) PYPARSEC_ARGS_17(X) X(17)
#define PYPARSEC_ARGS_19(X) PYPARSEC_ARGS_18(X) X(18)
#define PYPARSEC_ARGS_20(X) PYPARSEC_ARGS_19(X) X(19)
#define PYPARSEC_ARGS_21(X) PYPARSEC_ARGS_20(X) X(20)
#define PYPARSEC_ARGS_22(X) PYPARSEC_ARGS_21(X) X(21)
#define PYPARSEC_ARGS_23(X) PYPARSEC_ARGS_22(X) X(22)
#define PYPARSEC_ARGS_24(X) PYPARSEC_ARGS_23(X) X(23)
#define PYPARSEC_ARGS_25(X) PYPARSEC_ARGS_24(X) X(24)
#define PYPARSEC_ARGS_26(X) PYPARSEC_ARGS_25(X) X(25)
#define PYPARSEC_ARGS_27(X) PYPARSEC_ARGS_26(X) X(26)
#define PYPARSEC_ARGS_28(X) PYPARSEC_ARGS_27(X) X(27)
#define PYPARSEC_ARGS_29(X) PYPARSEC_ARGS_28(X) X(28)
#define PYPARSEC_ARGS_30(X) PYPARSEC_ARGS_29(X) X(29)
#define PYPARSEC_ARGS_31(X) PYPARSEC_ARGS_30(X) X(30)
#define PYPARSEC_ARGS_32(X) PYPARSEC_ARGS_31(X) X(31)
#define PYPARSEC_ARGS_33(X) PYPARSEC_ARGS_32(X) X(32)
#define PYPARSEC_ARGS_34(X) PYPARSEC_ARGS_33(X) X(33)
#define PYPARSEC_ARGS_35(X) PYPARSEC_ARGS_34(X) X(34)
#define PYPARSEC_ARGS_36(X) PYPARSEC_ARGS_35(X) X(35)
#define PYPARSEC_ARGS_37(X) PYPARSEC_ARGS_36(X) X(36)
#define PYPARSEC_ARGS_38(X) PYPARSEC_ARGS_37(X) X(37)
#define PYPARSEC_ARGS_39(X) PYPARSEC_ARGS_38(X) X(38)
#define PYPARSEC_ARGS_40(X) PYPARSEC_ARGS_39(X) X(39)
#define PYPARSEC_ARGS_41(X) PYPARSEC_ARGS_40(X) X(40)
#define PYPARSEC_ARGS_42(X) PYPARSEC_ARGS_41(X) X(41)
#define PYPARSEC_ARGS_43(X) PYPARSEC_ARGS_42(X) X(42)
#define PYPARSEC_ARGS_44(X) PYPARSEC_ARGS_43(X) X(43)
#define PYPARSEC_ARGS_45(X) PYPARSEC_ARGS_44(X) X(44)
#define PYPARSEC_ARGS_46(X) PYPARSEC_ARGS_45(X) X(45)
#define PYPARSEC_ARGS_47(X) PYPARSEC_ARGS_46(X) X(46)
#define PYPARSEC_ARGS_48(X) PYPARSEC_ARGS_47(X) X(47)
#define PYPARSEC_ARGS_49(X) PYPARSEC_ARGS_48(X) X(48)
#define PYPARSEC_ARGS_50(X) PYPARSEC_ARGS_49(X) X(49)
#define PYPARSEC_ARGS_51(X) PYPARSEC_ARGS_50(X) X(50)
#define PYPARSEC_ARGS_52(X) PYPARSEC_ARGS_51(X) X(51)
#define PYPARSEC_ARGS_53(X) PYPARSEC_ARGS_52(X) X(52)
#define PYPARSEC_ARGS_54(X) PYPARSEC_ARGS_53(X) X(53)
#define PYPARSEC_ARGS_55(X) PYPARSEC_ARGS_54(X) X(54)
#define PYPARSEC_ARGS_56(X) PYPARSEC_ARGS_55(X) X(55)
#define PYPARSEC_ARGS_57(X) PYPARSEC_ARGS_56(X) X(56)
#define PYPARSEC_ARGS_58(X) PYPARSEC_ARGS_57(X) X(57)
#define PYPARSEC_ARGS_59(X) PYPARSEC_ARGS_58(X) X(58)
#define PYPARSEC_ARGS_60(X) PYPARSEC_ARGS_59(X) X(59)
#define PYPARSEC_ARGS_61(X) PYPARSEC_ARGS_60(X) X(60)
#define PYPARSEC_ARGS_62(X) PYPARSEC_ARGS_61(X) X(61)
#define PYPARSEC_ARGS_63(X) PYPARSEC_ARGS_62(X) X(62)
#define PYPARSEC_ARGS_64(X) PYPARSEC_ARGS_63(X) X(63)

#define PYPARSEC_FOR_EACH_COUNT(C) \
    C(0)  C(1)  C(2)  C(3)  C(4)  C(5)  C(6)  C(7)  C(8)  C(9)  C(10) C(11) C(12) \
    C(13) C(14) C(15) C(16) C(17) C(18) C(19) C(20) C(21) C(22) C(23) C(24) C(25) \
    C(26) C(27) C(28) C(29) C(30) C(31) C(32) C(33) C(34) C(35) C(36) C(37) C(38) \
    C(39) C(40) C(41) C(42) C(43) C(44) C(45) C(46) C(47) C(48) C(49) C(50) C(51) \
    C(52) C(53) C(54) C(55) C(56) C(57) C(58) C(59) C(60) C(61) C(62) C(63) C(64)

#define PYPARSEC_INSERT_PAIR(i) flags[i], ptrs[i],
#define PYPARSEC_INSERT_CASE(n)                                                   \
    case n:                                                                       \
        parsec_dtd_insert_task_with_task_class(tp, tc, priority, device_type,     \
                                               PYPARSEC_ARGS_##n(PYPARSEC_INSERT_PAIR) \
                                               PARSEC_DTD_ARG_END);               \
        return 0;

/* Insert one task of an explicit DTD task class. Each argument is a
 * (flags, pointer) pair, as for parsec_dtd_insert_task_with_task_class().
 * Returns -1 if nb_args is out of range. */
static inline int
pyparsec_dtd_insert(parsec_taskpool_t *tp, parsec_task_class_t *tc,
                    int priority, int device_type,
                    int nb_args, const int *flags, void *const *ptrs)
{
    switch( nb_args ) {
        PYPARSEC_FOR_EACH_COUNT(PYPARSEC_INSERT_CASE)
    default:
        return -1;
    }
}

#define PYPARSEC_UNPACK_OUT(i) outs[i],
#define PYPARSEC_UNPACK_CASE(n)                                                   \
    case n:                                                                       \
        (parsec_dtd_unpack_args)(task, PYPARSEC_ARGS_##n(PYPARSEC_UNPACK_OUT) (void *)0); \
        return 0;

/* Unpack the first nb_args parameters of a DTD task into outs. */
static inline int
pyparsec_dtd_unpack(parsec_task_t *task, int nb_args, void *const *outs)
{
    switch( nb_args ) {
        PYPARSEC_FOR_EACH_COUNT(PYPARSEC_UNPACK_CASE)
    default:
        return -1;
    }
}

/* Create a DTD task class from parallel arrays of parameter sizes and
 * operation flags (the pairs parsec_dtd_create_task_class() reads). */
static inline parsec_task_class_t *
pyparsec_dtd_create_task_class(const char *name, int nb_params,
                               const int *sizes, const int *ops)
{
    parsec_dtd_param_t params[PARSEC_DTD_MAX_PARAMS];

    if( nb_params < 0 || nb_params > PYPARSEC_MAX_PARAMS ) return NULL;
    for( int i = 0; i < nb_params; i++ ) {
        params[i].op = (parsec_dtd_op_t)ops[i];
        params[i].size = sizes[i];
        params[i].profile_info = NULL;
    }
    /* The DTD task class starts with its parsec_task_class_t. */
    return (parsec_task_class_t *)parsec_dtd_create_task_classv(name, nb_params, params);
}

/* A C chore that does nothing, for control tasks, tests and benchmarks. */
static inline parsec_hook_return_t
pyparsec_noop_chore(parsec_execution_stream_t *es, parsec_task_t *task)
{
    (void)es; (void)task;
    return PARSEC_HOOK_RETURN_DONE;
}

static inline const parsec_task_class_t *
pyparsec_task_class_of(const parsec_task_t *task)
{
    return task->task_class;
}

/* Largest number of data flows a DTD task class may declare. This is the
 * limit parsec_dtd_create_task_classv() enforces with parsec_fatal(): the
 * flow mask of a task must leave room for the action bits. */
static inline int
pyparsec_dtd_max_flows(void)
{
    int bits = 0;
    while( (bits < (int)(sizeof(uint32_t) * CHAR_BIT)) &&
           (((uint32_t)1 << bits) < PARSEC_ACTION_RELEASE_LOCAL_DEPS) ) {
        bits++;
    }
    bits = (MAX_PARAM_COUNT < bits) ? MAX_PARAM_COUNT : bits;
    return (PYPARSEC_MAX_PARAMS < bits) ? PYPARSEC_MAX_PARAMS : bits;
}

/* ------------------------------------------------------------------------
 * Contexts.
 * ------------------------------------------------------------------------ */

/* True between parsec_context_start() and the end of parsec_context_wait(). */
static inline int
pyparsec_context_is_active(const parsec_context_t *context)
{
    return 0 != (context->flags & PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
}

/* True if MPI has been finalized: PaRSEC finalizes MPI in parsec_fini() when
 * it initialized it, and MPI cannot be initialized again. */
static inline int
pyparsec_mpi_finalized(void)
{
#if defined(PARSEC_HAVE_MPI)
    int finalized = 0;
    MPI_Finalized(&finalized);
    return finalized;
#else
    return 0;
#endif
}

/* ------------------------------------------------------------------------
 * Build checks.
 * ------------------------------------------------------------------------ */

/* The PaRSEC build the extension was compiled against: the configure stamp
 * and the git hash recorded in parsec_config.h. */
#if defined(PARSEC_COMPILE_DATE)
#define PYPARSEC_BUILT_DATE PARSEC_COMPILE_DATE
#else
#define PYPARSEC_BUILT_DATE ""
#endif
#if defined(PARSEC_GIT_HASH)
#define PYPARSEC_BUILT_HASH PARSEC_GIT_HASH
#else
#define PYPARSEC_BUILT_HASH ""
#endif

static inline const char *pyparsec_built_date(void) { return PYPARSEC_BUILT_DATE; }
static inline const char *pyparsec_built_hash(void) { return PYPARSEC_BUILT_HASH; }

/* ------------------------------------------------------------------------
 * Data collections.
 * ------------------------------------------------------------------------ */

/* The data collection accessors are variadic function pointers. */
static inline parsec_data_key_t
pyparsec_data_key(parsec_data_collection_t *dc, int m, int n)
{
    return dc->data_key(dc, m, n);
}

static inline int
pyparsec_rank_of(parsec_data_collection_t *dc, int m, int n)
{
    return (int)dc->rank_of(dc, m, n);
}

/* Host copy of a local tile, NULL if the tile is not stored on this rank. */
static inline void *
pyparsec_local_tile(parsec_data_collection_t *dc, int m, int n)
{
    parsec_data_t *data;
    parsec_data_copy_t *copy;

    if( dc->myrank != dc->rank_of(dc, m, n) ) return NULL;
    data = dc->data_of(dc, m, n);
    if( NULL == data ) return NULL;
    copy = parsec_data_get_copy(data, 0);
    return (NULL == copy) ? NULL : parsec_data_copy_get_ptr(copy);
}

static inline size_t
pyparsec_sizeof_matrix_type(int mtype)
{
    int size = parsec_datadist_getsizeoftype((parsec_matrix_type_t)mtype);
    return (size < 0) ? 0 : (size_t)size;
}

/* Arena datatype describing a full mb x nb tile with leading dimension ld,
 * registered with the context for DTD; returns the DTD arena id or -1. */
static inline int
pyparsec_dtd_tile_arena_new(parsec_context_t *ctx, int mtype, int mb, int nb, int ld)
{
    parsec_datatype_t dtt;
    parsec_arena_datatype_t *adt;
    int id = -1;

    if( PARSEC_SUCCESS != parsec_translate_matrix_type((parsec_matrix_type_t)mtype, &dtt) ) return -1;
    adt = parsec_matrix_adt_new_rect(dtt, mb, nb, ld);
    if( NULL == adt ) return -1;
    if( PARSEC_SUCCESS != parsec_dtd_attach_arena_datatype(ctx, adt, &id) ) {
        parsec_matrix_adt_free(&adt);
        return -1;
    }
    return id;
}

/* ------------------------------------------------------------------------
 * Python thread states of the PaRSEC threads.
 *
 * A PaRSEC worker that runs a Python body takes the GIL with
 * PyGILState_Ensure(). On a thread without a Python thread state, each such
 * call would create one and the matching release destroy it. The first call
 * on a thread therefore creates a thread state and keeps it (one
 * PyGILState_Ensure() that is never released), so that later calls only take
 * and drop the GIL. Called without the GIL.
 * ------------------------------------------------------------------------ */
static inline void
pyparsec_keep_thread_state(void)
{
    static _Thread_local int kept = 0;

    if( kept ) return;
    kept = 1;
    if( NULL == PyGILState_GetThisThreadState() ) {
        (void)PyGILState_Ensure();
        (void)PyEval_SaveThread();
    }
}

#endif  /* PYPARSEC_H_HAS_BEEN_INCLUDED */
