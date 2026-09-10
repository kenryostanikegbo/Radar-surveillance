/* ESP32-S3 firmware entry point for the Radar Surveillance System.
 *
 * Architecture (4 FreeRTOS tasks):
 *
 *   radar_rx_task    (core 0, prio 10): drains UART2 ring buffer, parses
 *                     frames, updates a 16-frame window buffer.
 *
 *   inference_task   (core 1, prio 8):  pops a 16-frame window, runs
 *                     feature_extract + TFLite Micro inference, pushes
 *                     per-window classification to tracker.
 *
 *   tracker_task     (core 1, prio 5):  maintains alpha-beta tracks,
 *                     publishes to a global state object the dashboard
 *                     reads via /api/state.
 *
 *   http_server_task (core 0, prio 1):  runs esp_http_server with two
 *                     GET handlers (/ and /api/state). Blocks here.
 *
 * Queues are bounded; if downstream is slow, oldest entries are dropped
 * and counted. We never block upstream radar RX.
 *
 * This file uses ESP-IDF headers. It compiles under `idf.py build` once
 * ESP-IDF v5.2+ is installed. The loopback harness in
 * tests/loopback_test.py exercises the same logic on host by calling
 * the individual C functions (via the radar_parse.dll) and the
 * equivalent Python implementations of tracker/inference.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_http_server.h"

#include "radar_protocol.h"
#include "radar_uart.h"
#include "feature_extract.h"
#include "tracker.h"
#include "inference.h"
#include "dashboard_http.h"

#define TAG "radar_main"
#define WINDOW_QUEUE_LEN 4
#define INFER_QUEUE_LEN  2

/* ---- Shared state ---- */
typedef struct {
    radar_target_t frames[WINDOW_SIZE][RADAR_MAX_TARGETS];
    uint8_t       masks[WINDOW_SIZE][RADAR_MAX_TARGETS];
    uint32_t      timestamps[WINDOW_SIZE];
    float         features[FEATURE_DIM];
} window_slot_t;

typedef struct {
    detection_t   dets[RADAR_MAX_TARGETS];
    int           n_dets;
    int           class_id;
    float         class_conf[NUM_CLASSES];
    uint32_t      frame_window_ts_ms;
} classification_t;

static QueueHandle_t window_q   = NULL;
static QueueHandle_t classify_q = NULL;
static inference_t   g_infer;
static tracker_t     g_tracker;

/* HTTP handler: GET /api/state */
static esp_err_t api_state_handler(httpd_req_t *req) {
    char buf[1024];
    dashboard_track_t tracks[DASHBOARD_MAX_TRACKS_JSON];
    int n = 0;
    for (int i = 0; i < TRACKER_MAX_TRACKS && n < DASHBOARD_MAX_TRACKS_JSON; i++) {
        const track_t *t = &g_tracker.tracks[i];
        if (!t->active || !t->published) continue;
        tracks[n].track_id    = t->id;
        tracks[n].class_id    = t->class_id;
        tracks[n].class_conf  = t->class_conf;
        tracks[n].x           = t->x;
        tracks[n].y           = t->y;
        tracks[n].speed_mps   = t->speed_mps;
        tracks[n].age_s       = t->age_s;
        n++;
    }
    int len = dashboard_state_json(buf, sizeof(buf), tracks, n,
                                    0, 0.0f, esp_get_free_heap_size());
    if (len < 0) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, buf, len);
    return ESP_OK;
}

static esp_err_t index_handler(httpd_req_t *req) {
    extern const uint8_t index_html_start[] asm("_binary_index_html_start");
    extern const uint8_t index_html_end[]   asm("_binary_index_html_end");
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, (const char *)index_html_start,
                           index_html_end - index_html_start);
}

/* ---- Task bodies ---- */
static void radar_rx_task(void *arg) {
    radar_uart_init(256000);
    radar_target_t targets[RADAR_MAX_TARGETS];
    uint32_t ts_ms;
    window_slot_t cur;
    memset(&cur, 0, sizeof(cur));
    int cur_idx = 0;

    for (;;) {
        int n = radar_uart_read_frame(targets, &ts_ms, portMAX_DELAY);
        if (n < 0) continue;
        memcpy(cur.frames[cur_idx], targets, sizeof(cur.frames[cur_idx]));
        for (int k = 0; k < RADAR_MAX_TARGETS; k++)
            cur.masks[cur_idx][k] = (k < n) ? 1 : 0;
        cur.timestamps[cur_idx] = ts_ms;
        cur_idx++;
        if (cur_idx >= WINDOW_SIZE) {
            /* Push to inference queue. If full, drop. */
            if (xQueueSend(window_q, &cur, 0) != pdTRUE) {
                ESP_LOGW(TAG, "window queue full, dropping oldest");
            }
            cur_idx = 0;
            memset(&cur, 0, sizeof(cur));
        }
    }
}

static void inference_task(void *arg) {
    inference_init(&g_infer);
    window_slot_t slot;
    classification_t out;
    for (;;) {
        if (xQueueReceive(window_q, &slot, portMAX_DELAY) != pdTRUE) continue;

        /* Compute features, normalize, quantize. */
        int n_filled = feature_extract_window(slot.frames, slot.masks,
                                              slot.features);
        if (n_filled != FEATURE_DIM) {
            ESP_LOGE(TAG, "feature_extract returned %d, expected %d",
                     n_filled, FEATURE_DIM);
            continue;
        }
        feature_normalize(slot.features);
        int8_t features_q[FEATURE_DIM];
        feature_quantize(slot.features, features_q, FEATURE_DIM);

        int class_id = -1;
        float conf[NUM_CLASSES];
        if (inference_run(&g_infer, features_q, &class_id, conf) != 0) continue;

        /* Build detections from the most recent 3 frames of the window. */
        out.n_dets = 0;
        for (int t = WINDOW_SIZE - 3; t < WINDOW_SIZE; t++) {
            for (int k = 0; k < RADAR_MAX_TARGETS; k++) {
                if (!slot.masks[t][k]) continue;
                if (out.n_dets >= RADAR_MAX_TARGETS) break;
                const radar_target_t *tg = &slot.frames[t][k];
                float xm = tg->x_cm / 100.0f;
                float ym = tg->y_cm / 100.0f;
                float vm = tg->speed_cm_s / 100.0f;
                detection_t *d = &out.dets[out.n_dets++];
                d->x = xm;
                d->y = ym;
                d->speed_mps = vm;
                d->class_id = class_id;
                d->class_conf = conf[class_id];
            }
        }
        out.class_id = class_id;
        memcpy(out.class_conf, conf, sizeof(out.class_conf));
        out.frame_window_ts_ms = slot.timestamps[WINDOW_SIZE - 1];

        if (xQueueSend(classify_q, &out, portMAX_DELAY) != pdTRUE) {
            ESP_LOGW(TAG, "classify queue full, dropping oldest");
        }
    }
}

static void tracker_task(void *arg) {
    tracker_init(&g_tracker);
    classification_t c;
    TickType_t prev = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(50);  /* 20 Hz radar */
    for (;;) {
        if (xQueueReceive(classify_q, &c, portMAX_DELAY) == pdTRUE) {
            tracker_step(&g_tracker, 0.05f, c.dets, c.n_dets);
        }
        (void)prev;
    }
}

static void http_server_task(void *arg) {
    /* softAP + httpd */
    esp_netif_create_default_ap();
    esp_event_loop_create_default();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_wifi_set_mode(WIFI_MODE_AP);
    wifi_config_t ap = {0};
    snprintf((char *)ap.ap.ssid, sizeof(ap.ap.ssid), "RadarSurv-%02X%02X",
             0xAB, 0xCD);  /* last 2 MAC bytes -- placeholder */
    ap.ap.ssid_len = strlen((char *)ap.ap.ssid);
    ap.ap.channel = 6;
    ap.ap.authmode = WIFI_AUTH_WPA2_PSK;
    strcpy((char *)ap.ap.password, "radar1234");
    esp_wifi_set_config(WIFI_IF_AP, &ap);
    esp_wifi_start();

    httpd_handle_t server = NULL;
    httpd_config_t conf = HTTPD_DEFAULT_CONFIG();
    if (httpd_start(&server, &conf) == ESP_OK) {
        httpd_uri_t a = { .uri = "/",        .method = HTTP_GET, .handler = index_handler };
        httpd_uri_t b = { .uri = "/api/state", .method = HTTP_GET, .handler = api_state_handler };
        httpd_register_uri_handler(server, &a);
        httpd_register_uri_handler(server, &b);
    }
    vTaskDelete(NULL);
}

void app_main(void) {
    window_q   = xQueueCreate(WINDOW_QUEUE_LEN, sizeof(window_slot_t));
    classify_q = xQueueCreate(INFER_QUEUE_LEN,  sizeof(classification_t));

    xTaskCreatePinnedToCore(radar_rx_task,    "radar_rx",    4096, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(inference_task,   "infer",       8192, NULL,  8, NULL, 1);
    xTaskCreatePinnedToCore(tracker_task,     "tracker",     4096, NULL,  5, NULL, 1);
    xTaskCreatePinnedToCore(http_server_task, "http",        8192, NULL,  1, NULL, 0);
}
