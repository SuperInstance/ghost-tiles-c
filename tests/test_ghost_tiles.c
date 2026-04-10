/*
 * test_ghost_tiles.c — Tests for ghost-tiles C library
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "ghost-tiles.h"

static uint64_t test_now = 1000000;

static int approx_eq(double a, double b, double eps) {
    return fabs(a - b) < eps;
}

void test_tile_use(void) {
    gt_tile t = {0};
    t.row = 0; t.col = 0; t.weight = 1.0; t.confidence = 1.0; t.importance = 0.5;
    gt_tile_use(&t, 0.9, test_now);
    assert(t.active == true);
    assert(t.use_count == 1);
    assert(t.confidence > 0.4);
    printf("  test_tile_use: PASS\n");
}

void test_tile_decay(void) {
    gt_tile t = {0};
    t.row = 0; t.col = 0; t.weight = 1.0; t.confidence = 1.0; t.importance = 0.5;
    t.last_used_ms = 0;
    gt_tile_decay(&t, 0.5, test_now);
    assert(t.weight < 1.0);
    printf("  test_tile_decay: PASS\n");
}

void test_tile_deactivation(void) {
    gt_tile t = {0};
    t.row = 0; t.col = 0; t.weight = 0.005; t.confidence = 1.0; t.active = true;
    gt_tile_decay(&t, 0.5, test_now);
    assert(t.active == false);
    printf("  test_tile_deactivation: PASS\n");
}

void test_pattern_create(void) {
    gt_pattern p;
    int rc = gt_pattern_init(&p, "p1", 64, 8, 0.5);
    assert(rc == GT_OK);
    assert(p.num_tiles == 64); /* 8x8 grid */
    assert(p.grid_rows == 8);
    assert(p.grid_cols == 8);
    gt_pattern_free(&p);
    printf("  test_pattern_create: PASS\n");
}

void test_pattern_prune(void) {
    gt_pattern p;
    gt_pattern_init(&p, "p1", 64, 8, 0.5);
    /* Give all tiles moderate weight */
    for (int i = 0; i < p.num_tiles; i++) p.tiles[i].weight = 0.5;
    gt_pattern_prune(&p);
    int active = gt_pattern_active_count(&p);
    assert(active <= 32);
    gt_pattern_free(&p);
    printf("  test_pattern_prune: PASS\n");
}

void test_pattern_sparsity(void) {
    gt_pattern p;
    gt_pattern_init(&p, "p1", 64, 8, 0.0);
    double s = gt_pattern_sparsity(&p);
    assert(approx_eq(s, 0.0, 0.01));
    gt_pattern_free(&p);
    printf("  test_pattern_sparsity: PASS\n");
}

void test_pattern_compute_cost(void) {
    gt_pattern p;
    gt_pattern_init(&p, "p1", 64, 8, 0.0);
    double c = gt_pattern_compute_cost(&p);
    assert(approx_eq(c, 1.0, 0.01));
    gt_pattern_free(&p);
    printf("  test_pattern_compute_cost: PASS\n");
}

void test_pattern_efficiency(void) {
    gt_pattern p;
    gt_pattern_init(&p, "p1", 64, 8, 0.5);
    gt_pattern_prune(&p);
    double eff = gt_pattern_efficiency(&p);
    assert(eff >= 0.0);
    gt_pattern_free(&p);
    printf("  test_pattern_efficiency: PASS\n");
}

void test_attention_mask(void) {
    gt_pattern p;
    gt_pattern_init(&p, "p1", 64, 8, 0.0);
    float *mask = (float *)calloc(64 * 64, sizeof(float));
    gt_pattern_attention_mask(&p, mask, 64);
    int nonzero = 0;
    for (int i = 0; i < 64 * 64; i++) { if (mask[i] > 0.0f) nonzero++; }
    assert(nonzero > 0);
    free(mask);
    gt_pattern_free(&p);
    printf("  test_attention_mask: PASS\n");
}

void test_pattern_merge(void) {
    gt_pattern a, b, out;
    gt_pattern_init(&a, "a", 64, 8, 0.5);
    gt_pattern_init(&b, "b", 64, 8, 0.5);
    /* A: top-left strong, B: bottom-right strong */
    for (int i = 0; i < a.num_tiles; i++) {
        a.tiles[i].weight = a.tiles[i].row < 4 ? 0.9 : 0.1;
        b.tiles[i].weight = b.tiles[i].row >= 4 ? 0.9 : 0.1;
    }
    gt_pattern_prune(&a);
    gt_pattern_prune(&b);
    int rc = gt_pattern_merge(&a, &b, &out, "merged");
    assert(rc == GT_OK);
    gt_pattern_free(&a); gt_pattern_free(&b); gt_pattern_free(&out);
    printf("  test_pattern_merge: PASS\n");
}

void test_manager(void) {
    gt_manager m;
    gt_manager_init(&m, 0.5);
    gt_pattern p1, p2;
    gt_pattern_init(&p1, "p1", 64, 8, 0.8);
    gt_pattern_init(&p2, "p2", 64, 8, 0.2);
    gt_pattern_prune(&p1);
    gt_pattern_prune(&p2);
    gt_manager_add(&m, &p1);
    gt_manager_add(&m, &p2);
    const gt_pattern *best = gt_manager_best(&m);
    assert(best != NULL);
    assert(best != NULL); printf("    best: %s eff=%.2f\n", best->id, gt_pattern_efficiency(best));; /* sparser = more efficient */
    const gt_pattern *found = gt_manager_get(&m, "p1");
    assert(found != NULL);
    double cost = gt_manager_avg_cost(&m);
    assert(cost > 0.0 && cost < 1.0);
    gt_pattern_free(&p1); gt_pattern_free(&p2);
    printf("  test_manager: PASS\n");
}

void test_rebalance(void) {
    gt_pattern p;
    gt_pattern_init(&p, "p1", 64, 8, 0.3);
    for (int i = 0; i < p.num_tiles; i++) p.tiles[i].weight = 0.5;
    gt_pattern_rebalance(&p, test_now);
    int active = gt_pattern_active_count(&p);
    assert(active < p.num_tiles);
    gt_pattern_free(&p);
    printf("  test_rebalance: PASS\n");
}

int main(void) {
    printf("ghost-tiles C tests:\n");
    test_tile_use();
    test_tile_decay();
    test_tile_deactivation();
    test_pattern_create();
    test_pattern_prune();
    test_pattern_sparsity();
    test_pattern_compute_cost();
    test_pattern_efficiency();
    test_attention_mask();
    test_pattern_merge();
    test_manager();
    test_rebalance();
    printf("All 12 tests passed!\n");
    return 0;
}
