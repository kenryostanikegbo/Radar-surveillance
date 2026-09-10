/* JSON serializer for the dashboard API. Used by:
 *   - main.c inside the ESP-IDF HTTP handler (writes to httpd socket)
 *   - tests/loopback_test.py on host (writes to a Python buffer)
 *
 * Both consumers only need the JSON shape, not the HTTP transport.
 */
#include <stdio.h>
#include <string.h>
#include "dashboard_http.h"
#include "model_metadata.h"   /* CLASS_NAMES */

int dashboard_state_json(char *buf, size_t buflen,
                         const dashboard_track_t *tracks, int n_tracks,
                         uint32_t frame_count, float infer_ms,
                         uint32_t free_heap)
{
    int n = 0;
    n += snprintf(buf + n, buflen - n,
        "{\"frame_count\":%u,\"infer_ms\":%.2f,\"free_heap\":%u,\"tracks\":[",
        frame_count, infer_ms, free_heap);

    for (int i = 0; i < n_tracks; i++) {
        const dashboard_track_t *t = &tracks[i];
        n += snprintf(buf + n, buflen - n,
            "%s{\"id\":%d,\"class\":\"%s\",\"confidence\":%.3f,"
            "\"x\":%.2f,\"y\":%.2f,\"speed\":%.2f,\"age_s\":%.2f}",
            (i > 0 ? "," : ""),
            t->track_id,
            (t->class_id >= 0 && t->class_id < NUM_CLASSES) ?
                CLASS_NAMES[t->class_id] : "unknown",
            t->class_conf, t->x, t->y, t->speed_mps, t->age_s);
        if (n < 0 || (size_t)n >= buflen) return -1;
    }
    n += snprintf(buf + n, buflen - n, "]}");
    return n;
}

int dashboard_health_json(char *buf, size_t buflen,
                          uint32_t uptime_s, float fps)
{
    return snprintf(buf, buflen,
        "{\"uptime_s\":%u,\"fps\":%.2f}", uptime_s, fps);
}
