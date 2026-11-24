/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * The C counterpart of overhead.py: the same tasks, inserted from C, so that
 * the time PaRSEC spends per task can be told apart from the binding's.
 *
 *   mpicc -O2 c_overhead.c $(pkg-config --cflags --libs parsec) -o c_overhead
 *   ./c_overhead <cores> <tasks> <c-1tile|c-3tiles|c-values>
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "parsec.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function.h"

static parsec_hook_return_t noop(parsec_execution_stream_t *es, parsec_task_t *task)
{
    (void)es; (void)task;
    return PARSEC_HOOK_RETURN_DONE;
}

/* parsec_dtd_tile_of() on tile (m, n) of a block-cyclic matrix. */
#define TILE(M, m, n) \
    parsec_dtd_tile_of(&(M).super.super, (M).super.super.data_key(&(M).super.super, (m), (n)))

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void matrix_init(parsec_matrix_block_cyclic_t *M, int side, const char *name)
{
    parsec_matrix_block_cyclic_init(M, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE, 0,
                                    1, 1, side, side, 0, 0, side, side, 1, 1, 1, 1, 0, 0);
    M->mat = parsec_data_allocate((size_t)M->super.nb_local_tiles * M->super.bsiz * sizeof(double));
    memset(M->mat, 0, (size_t)M->super.nb_local_tiles * M->super.bsiz * sizeof(double));
    parsec_data_collection_set_key(&M->super.super, (char *)name);
    parsec_dtd_data_collection_init(&M->super.super);
}

int main(int argc, char *argv[])
{
    int cores = (argc > 1) ? atoi(argv[1]) : 1;
    long ntasks = (argc > 2) ? atol(argv[2]) : 200000;
    const char *mode = (argc > 3) ? argv[3] : "c-1tile";
    int side = (int)ceil(sqrt((double)ntasks));
    int three = !strcmp(mode, "c-3tiles"), values = !strcmp(mode, "c-values");
    parsec_matrix_block_cyclic_t A, B, C;
    parsec_arena_datatype_t *adt;
    parsec_datatype_t dtt;
    int adt_id;

    parsec_context_t *ctx = parsec_init(cores, NULL, NULL);
    matrix_init(&A, side, "A");
    matrix_init(&B, side, "B");
    matrix_init(&C, side, "C");
    parsec_translate_matrix_type(PARSEC_MATRIX_DOUBLE, &dtt);
    adt = parsec_matrix_adt_new_rect(dtt, 1, 1, 1);
    parsec_dtd_attach_arena_datatype(ctx, adt, &adt_id);

    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();
    parsec_context_add_taskpool(ctx, tp);
    parsec_context_start(ctx);

    parsec_task_class_t *tc;
    if( three ) {
        tc = parsec_dtd_create_task_class(tp, "c-3tiles",
                                          PASSED_BY_REF, PARSEC_INPUT | adt_id,
                                          PASSED_BY_REF, PARSEC_INPUT | adt_id,
                                          PASSED_BY_REF, PARSEC_INOUT | PARSEC_AFFINITY | adt_id,
                                          PARSEC_DTD_ARG_END);
    } else if( values ) {
        tc = parsec_dtd_create_task_class(tp, "c-values",
                                          PASSED_BY_REF, PARSEC_INOUT | PARSEC_AFFINITY | adt_id,
                                          sizeof(int), PARSEC_VALUE, sizeof(double), PARSEC_VALUE,
                                          PARSEC_DTD_ARG_END);
    } else {
        tc = parsec_dtd_create_task_class(tp, "c-1tile",
                                          PASSED_BY_REF, PARSEC_INOUT | PARSEC_AFFINITY | adt_id,
                                          PARSEC_DTD_ARG_END);
    }
    parsec_dtd_task_class_add_chore(tp, tc, PARSEC_DEV_CPU, noop);

    double start = now();
    for( long i = 0; i < ntasks; i++ ) {
        int m = (int)(i / side), n = (int)(i % side);
        int iv = (int)i;
        double dv = 0.5;
        if( three ) {
            parsec_dtd_insert_task_with_task_class(tp, tc, 0, PARSEC_DEV_ALL,
                PARSEC_INPUT, TILE(A, m, n),
                PARSEC_INPUT, TILE(B, m, n),
                PARSEC_INOUT | PARSEC_AFFINITY, TILE(C, m, n),
                PARSEC_DTD_ARG_END);
        } else if( values ) {
            parsec_dtd_insert_task_with_task_class(tp, tc, 0, PARSEC_DEV_ALL,
                PARSEC_INOUT | PARSEC_AFFINITY, TILE(A, m, n),
                PARSEC_VALUE, &iv, PARSEC_VALUE, &dv, PARSEC_DTD_ARG_END);
        } else {
            parsec_dtd_insert_task_with_task_class(tp, tc, 0, PARSEC_DEV_ALL,
                PARSEC_INOUT | PARSEC_AFFINITY, TILE(A, m, n),
                PARSEC_DTD_ARG_END);
        }
    }
    double inserted = now();
    parsec_dtd_data_flush_all(tp, &A.super.super);
    parsec_dtd_data_flush_all(tp, &B.super.super);
    parsec_dtd_data_flush_all(tp, &C.super.super);
    parsec_taskpool_wait(tp);
    double done = now();
    printf("cores=%3d %-9s insert %6.2f us/task   insert+complete %6.2f us/task   (C)\n",
           cores, mode, (inserted - start) / ntasks * 1e6, (done - start) / ntasks * 1e6);

    parsec_dtd_task_class_release(tp, tc);
    parsec_taskpool_free(tp);
    parsec_context_wait(ctx);
    parsec_dtd_free_arena_datatype(ctx, adt_id);
    parsec_dtd_data_collection_fini(&A.super.super);
    parsec_dtd_data_collection_fini(&B.super.super);
    parsec_dtd_data_collection_fini(&C.super.super);
    parsec_data_free(A.mat);
    parsec_data_free(B.mat);
    parsec_data_free(C.mat);
    parsec_tiled_matrix_destroy(&A.super);
    parsec_tiled_matrix_destroy(&B.super);
    parsec_tiled_matrix_destroy(&C.super);
    parsec_fini(&ctx);
    return 0;
}
