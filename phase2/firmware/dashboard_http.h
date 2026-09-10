/* HTTP server: serves the dashboard SPA and a tiny JSON API.
 *
 * Endpoints (when running on ESP32-S3 softAP):
 *   GET /            -> firmware/main/dashboard/index.html (embedded)
 *   GET /app.js      -> firmware/main/dashboard/app.js (embedded)
 *   GET /style.css   -> firmware/main/dashboard/style.css (embedded)
 *   GET /api/state   -> JSON of current tracks, frame count, uptime
 *   GET /api/health  -> JSON of {uptime_s, fps, free_heap}
 *
 * The ESP-IDF implementation lives in main.c using esp_http_server.
 * This header documents the contract that the Python loopback
 * harness (tests/loopback_test.py) will simulate.
 */
#ifndef DASHBOARD_HTTP_H
#define DASHBOARD_HTTP_H

#include <stdint.h>

#define DASHBOARD_MAX_TRACKS_JSON  16

typedef struct {
    int     track_id;
    int     class_id;
    char    class_name[16];
    float   class_conf;
    float   x, y;
    float   speed_mps;
    float   age_s;
} dashboard_track_t;

/* Serialize the current dashboard state into a fixed-size JSON buffer.
   Returns the number of bytes written, or -1 on overflow. */
int dashboard_state_json(char *buf, size_t buflen,
                         const dashboard_track_t *tracks, int n_tracks,
                         uint32_t frame_count, float infer_ms,
                         uint32_t free_heap);

/* /api/health equivalent */
int dashboard_health_json(char *buf, size_t buflen,
                          uint32_t uptime_s, float fps);

#endif /* DASHBOARD_HTTP_H */
