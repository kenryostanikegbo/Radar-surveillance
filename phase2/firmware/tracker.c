/* Alpha-beta tracker implementation. See tracker.h for design notes.
 *
 * Algorithm:
 *   1. Predict existing tracks forward by dt_s using their estimated velocity.
 *   2. For each detection in the new frame:
 *      a. Find the closest unmatched track within TRACKER_GATE_M.
 *      b. If matched: alpha-beta update on position + velocity.
 *      c. If unmatched: spawn a new track (with NEW_COOLDOWN gating before
 *         publishing to the dashboard).
 *   3. Increment unseen counters; drop tracks past TRACKER_TTL_S.
 *
 * This is intentionally simple: no matrix inversion, no gating on velocity.
 * For a surveillance radar at 20 Hz with <=3 targets per frame, a
 * Hungarian matcher would be overkill; nearest-neighbor is fine.
 */
#include <math.h>
#include <string.h>
#include "tracker.h"

static float dist2(float ax, float ay, float bx, float by) {
    float dx = ax - bx, dy = ay - by;
    return dx * dx + dy * dy;
}

void tracker_init(tracker_t *t) {
    memset(t, 0, sizeof(*t));
    t->next_id = 1;
}

static int find_or_create_track(tracker_t *t) {
    for (int i = 0; i < TRACKER_MAX_TRACKS; i++) {
        if (!t->tracks[i].active) return i;
    }
    return -1;
}

void tracker_step(tracker_t *t, float dt_s,
                  const detection_t *dets, int n_dets)
{
    /* Advance time and age all tracks */
    t->now_s += dt_s;
    for (int i = 0; i < TRACKER_MAX_TRACKS; i++) {
        if (!t->tracks[i].active) continue;
        t->tracks[i].age_s += dt_s;
        if (t->now_s - t->tracks[i].last_seen_s > TRACKER_TTL_S) {
            t->tracks[i].active = 0;
            t->tracks[i].published = 0;
        } else {
            /* Predict position forward */
            t->tracks[i].x  += t->tracks[i].vx * dt_s;
            t->tracks[i].y  += t->tracks[i].vy * dt_s;
        }
    }

    int det_matched[TRACKER_MAX_TRACKS] = {0};

    for (int d = 0; d < n_dets; d++) {
        const detection_t *det = &dets[d];
        float best_d2 = TRACKER_GATE_M * TRACKER_GATE_M;
        int best = -1;
        for (int i = 0; i < TRACKER_MAX_TRACKS; i++) {
            if (!t->tracks[i].active) continue;
            float d2 = dist2(det->x, det->y, t->tracks[i].x, t->tracks[i].y);
            if (d2 < best_d2) { best_d2 = d2; best = i; }
        }
        if (best >= 0) {
            track_t *tk = &t->tracks[best];
            /* alpha-beta update */
            float resid_x = det->x - tk->x;
            float resid_y = det->y - tk->y;
            tk->x  += TRACKER_ALPHA * resid_x;
            tk->y  += TRACKER_ALPHA * resid_y;
            tk->vx += (TRACKER_BETA / (dt_s > 1e-3f ? dt_s : 1e-3f)) * resid_x;
            tk->vy += (TRACKER_BETA / (dt_s > 1e-3f ? dt_s : 1e-3f)) * resid_y;
            tk->speed_mps = sqrtf(tk->vx * tk->vx + tk->vy * tk->vy);
            tk->last_seen_s = t->now_s;
            tk->class_id = det->class_id;
            tk->class_conf = det->class_conf;
            tk->hit_count++;
            /* Publish after 3 consecutive hits to suppress flicker.
               (Resetting age_s on every hit meant wall-clock never
               advanced -- which kept tracks invisible forever.) */
            if (tk->hit_count >= 3) tk->published = 1;
            det_matched[d] = 1;
        }
    }

    /* Spawn tracks for unmatched detections */
    for (int d = 0; d < n_dets; d++) {
        if (det_matched[d]) continue;
        int slot = find_or_create_track(t);
        if (slot < 0) continue;
        track_t *tk = &t->tracks[slot];
        tk->id = t->next_id++;
        tk->x = dets[d].x;
        tk->y = dets[d].y;
        tk->vx = tk->vy = 0.0f;
        tk->speed_mps = dets[d].speed_mps;
        tk->class_id = dets[d].class_id;
        tk->class_conf = dets[d].class_conf;
        tk->age_s = 0.0f;
        tk->last_seen_s = t->now_s;
        tk->hit_count = 0;
        tk->active = 1;
        tk->published = 0;
    }
    (void)0;

    /* Count active tracks for the API. */
    int n = 0;
    for (int i = 0; i < TRACKER_MAX_TRACKS; i++)
        if (t->tracks[i].active) n++;
    t->n_tracks = n;
}
