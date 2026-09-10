/* Lightweight alpha-beta tracker for radar detections.
 *
 * Designed for the ESP32-S3 firmware pipeline. For each radar frame
 * (up to MAX_TARGETS=3 detections), the tracker:
 *   - matches detections to existing tracks via nearest-neighbor on
 *     (x, y) within a gating radius;
 *   - updates the matched track with an alpha-beta filter;
 *   - creates new tracks for unmatched detections (with a cooldown
 *     before publishing);
 *   - drops tracks that have been unseen for > track_ttl_s.
 *
 * Each track carries: id, age_s, last_seen_s, x, y, v, class,
 * class_confidence. The dashboard reads tracks[] for rendering.
 */
#ifndef TRACKER_H
#define TRACKER_H

#include <stdint.h>
#include <stddef.h>

#define TRACKER_MAX_TRACKS    16
#define TRACKER_GATE_M        2.0f    /* matching gate (m) */
#define TRACKER_NEW_COOLDOWN  0.5f    /* ignore new tracks for this long */
#define TRACKER_TTL_S         1.0f    /* drop track if unseen > TTL */
#define TRACKER_ALPHA         0.7f    /* alpha-beta position gain */
#define TRACKER_BETA          0.5f    /* alpha-beta velocity gain */

typedef struct {
    int     id;
    float   x, y;            /* metres */
    float   vx, vy;          /* m/s (estimated) */
    float   speed_mps;       /* magnitude */
    int     class_id;        /* 0..NUM_CLASSES-1 */
    float   class_conf;      /* softmax probability */
    float   age_s;
    float   last_seen_s;
    int     hit_count;       /* consecutive frames matched (publishes >=3) */
    int     published;       /* 1 = visible to dashboard */
    int     active;
} track_t;

typedef struct {
    track_t tracks[TRACKER_MAX_TRACKS];
    int     n_tracks;
    int     next_id;
    float   now_s;           /* current time (seconds since boot) */
} tracker_t;

typedef struct {
    float   x, y;            /* metres */
    float   speed_mps;       /* magnitude, m/s */
    int     class_id;        /* 0..NUM_CLASSES-1 */
    float   class_conf;      /* 0..1 */
} detection_t;

void tracker_init(tracker_t *t);
void tracker_step(tracker_t *t, float dt_s,
                  const detection_t *dets, int n_dets);

#endif /* TRACKER_H */
