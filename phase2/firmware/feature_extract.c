/* Feature extraction -- C port of phase2/scripts/compute_features.py.
 *
 * Notes:
 *  - The Python reference uses numpy; this file uses plain C with
 *    simple loops. Floating-point order should match exactly.
 *  - The WINDOW_SIZE target slots are 0..2 in input order; the model
 *    doesn't care about slot order, only that the same per-target
 *    statistics are computed for each slot independently.
 *  - Output is normalized via z-score, then quantized to int8 using
 *    the model's input scale/zero_point. The ESP32-S3 TFLite Micro
 *    invocation consumes the int8_t buffer directly.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "feature_extract.h"
#include "model_metadata.h"   /* FEATURE_MEAN, FEATURE_STD, INPUT_SCALE, INPUT_ZP */

#define MAX_FEATS  (3 * FEATS_PER_TARGET + GLOBAL_FEATS)
/* MAX_FEATS == FEATURE_DIM == 72 by construction; see model_metadata.h. */

/* Helper: smallest valid value of an array, NaN-safe. */
static float arr_min(const float *a, int n) {
    float m = a[0];
    for (int i = 1; i < n; i++) if (a[i] < m) m = a[i];
    return m;
}
static float arr_max(const float *a, int n) {
    float m = a[0];
    for (int i = 1; i < n; i++) if (a[i] > m) m = a[i];
    return m;
}
static float arr_mean(const float *a, int n) {
    if (n == 0) return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i];
    return s / (float)n;
}
static float arr_std(const float *a, int n, float mean) {
    if (n <= 1) return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - mean;
        s += d * d;
    }
    return sqrtf(s / (float)n);
}
static int arr_argmin(const float *a, int n) {
    int idx = 0; float mv = a[0];
    for (int i = 1; i < n; i++) if (a[i] < mv) { mv = a[i]; idx = i; }
    return idx;
}
static int arr_argmax(const float *a, int n) {
    int idx = 0; float mv = a[0];
    for (int i = 1; i < n; i++) if (a[i] > mv) { mv = a[i]; idx = i; }
    return idx;
}
static float arr_percentile(float *a, int n, float q) {
    if (n == 0) return 0.0f;
    /* insertion sort -- n is small (up to 16) so this is fine */
    for (int i = 1; i < n; i++) {
        float v = a[i]; int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j+1] = a[j]; j--; }
        a[j+1] = v;
    }
    float idx = q * (float)(n - 1);
    int lo = (int)idx; int hi = lo + 1;
    if (hi >= n) hi = n - 1;
    float frac = idx - (float)lo;
    return a[lo] * (1.0f - frac) + a[hi] * frac;
}

int feature_extract_window(const radar_target_t (*window)[RADAR_MAX_TARGETS],
                           const uint8_t (*valid_mask)[RADAR_MAX_TARGETS],
                           float *out)
{
    /* Per-target present arrays and (d, v, x, y) series. */
    float present[3][WINDOW_SIZE];
    float d_series[3][WINDOW_SIZE];
    float v_series[3][WINDOW_SIZE];
    float x_series[3][WINDOW_SIZE];
    float y_series[3][WINDOW_SIZE];
    int   present_n[3] = {0, 0, 0};
    /* present_pos[k][t] = index into d_series[k] for time t's entry,
       or -1 if masked. Used for time-major global flattening. */
    int   present_pos[3][WINDOW_SIZE];
    for (int k = 0; k < 3; k++)
        for (int t = 0; t < WINDOW_SIZE; t++)
            present_pos[k][t] = -1;

    for (int k = 0; k < 3; k++) {
        for (int t = 0; t < WINDOW_SIZE; t++) {
            int valid = valid_mask ? valid_mask[t][k] : 1;
            if (!valid) continue;
            const radar_target_t *tg = &window[t][k];
            /* d_m from x_cm, y_cm */
            float xm = (float)tg->x_cm / 100.0f;
            float ym = (float)tg->y_cm / 100.0f;
            float dm = sqrtf(xm * xm + ym * ym);
            float vm = (float)tg->speed_cm_s / 100.0f;
            present[k][present_n[k]] = 1.0f;
            d_series[k][present_n[k]] = dm;
            v_series[k][present_n[k]] = vm;
            x_series[k][present_n[k]] = xm;
            y_series[k][present_n[k]] = ym;
            present_pos[k][t] = present_n[k];
            present_n[k]++;
        }
    }

    /* Per-target 17 features -- must match Python's order in
       scripts/compute_features.py:window_to_features (target_feats.append):
         1  d_mean
         2  v_mean
         3  x_mean
         4  y_mean
         5  d_std
         6  v_std
         7  x_std
         8  y_std
         9  slope_d
        10  slope_v
        11  max_abs_v
        12  persistence
        13  sign_changes
        14  smoothness (mean squared speed delta)
        15  extent_x
        16  extent_y
        17  avg_distance_to_origin */
    int idx = 0;
    for (int k = 0; k < 3; k++) {
        if (present_n[k] < 1) {
            for (int f = 0; f < FEATS_PER_TARGET; f++) out[idx++] = 0.0f;
            continue;
        }
        const float *d = d_series[k], *v = v_series[k];
        const float *xs = x_series[k], *ys = y_series[k];
        int n = present_n[k];

        float d_mean = arr_mean(d, n);
        float v_mean = arr_mean(v, n);
        float x_mean = arr_mean(xs, n);
        float y_mean = arr_mean(ys, n);
        float d_std  = arr_std(d, n, d_mean);
        float v_std  = arr_std(v, n, v_mean);
        float x_std  = arr_std(xs, n, x_mean);
        float y_std  = arr_std(ys, n, y_mean);

        /* 1-4: means */
        out[idx++] = d_mean;
        out[idx++] = v_mean;
        out[idx++] = x_mean;
        out[idx++] = y_mean;

        /* 5-8: stds (only if more than 1 sample, else 0) */
        if (n > 1) {
            out[idx++] = d_std;
            out[idx++] = v_std;
            out[idx++] = x_std;
            out[idx++] = y_std;
        } else {
            out[idx++] = 0.0f; out[idx++] = 0.0f;
            out[idx++] = 0.0f; out[idx++] = 0.0f;
        }

        /* 9-10: slope of d, v (linear trend) */
        if (n > 1) {
            float t_mean = (float)(n - 1) / 2.0f;
            float num_d = 0.0f, num_v = 0.0f, den = 0.0f;
            for (int i = 0; i < n; i++) {
                float dt = (float)i - t_mean;
                den += dt * dt;
                num_d += dt * (d[i] - d_mean);
                num_v += dt * (v[i] - v_mean);
            }
            out[idx++] = (den > 0.0f) ? (num_d / den) : 0.0f;
            out[idx++] = (den > 0.0f) ? (num_v / den) : 0.0f;
        } else {
            out[idx++] = 0.0f;
            out[idx++] = 0.0f;
        }

        /* 11: max abs speed */
        float max_abs_v = 0.0f;
        for (int i = 0; i < n; i++) {
            float av = v[i] < 0 ? -v[i] : v[i];
            if (av > max_abs_v) max_abs_v = av;
        }
        out[idx++] = max_abs_v;

        /* 12: persistence ratio */
        out[idx++] = (float)present_n[k] / (float)WINDOW_SIZE;

        /* 13: speed sign-change rate */
        if (n > 1) {
            int changes = 0;
            for (int i = 1; i < n; i++) {
                float s1 = v[i-1] < 0 ? -1.0f : (v[i-1] > 0 ? 1.0f : 0.0f);
                float s2 = v[i]   < 0 ? -1.0f : (v[i]   > 0 ? 1.0f : 0.0f);
                if (s1 != s2) changes++;
            }
            out[idx++] = (float)changes / (float)(n - 1);
        } else {
            out[idx++] = 0.0f;
        }

        /* 14: trajectory smoothness -- mean squared speed delta */
        if (n > 2) {
            float s = 0.0f;
            for (int i = 1; i < n; i++) {
                float dv = v[i] - v[i-1];
                s += dv * dv;
            }
            out[idx++] = s / (float)(n - 1);
        } else {
            out[idx++] = 0.0f;
        }

        /* 15-16: spatial extent in x, y */
        if (n > 1) {
            out[idx++] = arr_max(xs, n) - arr_min(xs, n);
            out[idx++] = arr_max(ys, n) - arr_min(ys, n);
        } else {
            out[idx++] = 0.0f;
            out[idx++] = 0.0f;
        }

        /* 17: average distance to origin */
        out[idx++] = sqrtf(x_mean * x_mean + y_mean * y_mean);
    }

    /* Global features (21) -- order matches Python's window_to_features
       (scripts/compute_features.py:window_to_features).
         1   total detection density (mask.sum() / 48)
         2   primary target persistence (mask[:,0].sum()/16)
         3   secondary target persistence
         4   tertiary target persistence
         5   global d_mean
         6   global d_std
         7   global v_mean
         8   global v_std
         9   global max abs v
        10   frames_with_target / 16
        11   total_detections / 16 / 3
        12   v IQR (q75 - q25)
        13   d change rate (mean abs diff)
        14   v change rate
        15   frac of detections with |v| > 0.5
        16   frac with |v| > 2.0
        17   frac with |v| > 5.0
        18   frac with |v| < 0.1
        19   max simultaneous targets / 3
        20   fraction of empty frames
        21   mean abs v */
    float d_all[WINDOW_SIZE * 3];
    float v_all[WINDOW_SIZE * 3];
    int dn = 0, vn = 0;
    /* Iterate time-major (t outer, k inner) so the order of entries in
       d_all/v_all matches numpy's tensor[:,:,0][mask>0.5] flattening. */
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int k = 0; k < 3; k++) {
            if (valid_mask && !valid_mask[t][k]) continue;
            d_all[dn++] = d_series[k][present_pos[k][t]];
            v_all[vn++] = v_series[k][present_pos[k][t]];
        }
    }

    /* 1: total detection density */
    int total_det = 0;
    for (int t = 0; t < WINDOW_SIZE; t++)
        for (int k = 0; k < 3; k++)
            if (valid_mask && valid_mask[t][k]) total_det++;
    out[idx++] = (float)total_det / (float)(WINDOW_SIZE * 3);

    /* 2-4: per-target persistence ratios */
    for (int k = 0; k < 3; k++) {
        int n_k = 0;
        for (int t = 0; t < WINDOW_SIZE; t++) if (valid_mask && valid_mask[t][k]) n_k++;
        out[idx++] = (float)n_k / (float)WINDOW_SIZE;
    }

    /* 5-6: d_mean, d_std */
    if (dn > 0) {
        float d_mean = arr_mean(d_all, dn);
        out[idx++] = d_mean;
        out[idx++] = (dn > 1) ? arr_std(d_all, dn, d_mean) : 0.0f;
    } else {
        out[idx++] = 0.0f;
        out[idx++] = 0.0f;
    }

    /* 7-9: v_mean, v_std, max abs v */
    if (vn > 0) {
        float v_mean = arr_mean(v_all, vn);
        out[idx++] = v_mean;
        out[idx++] = (vn > 1) ? arr_std(v_all, vn, v_mean) : 0.0f;
        float max_av = 0.0f;
        for (int i = 0; i < vn; i++) {
            float av = v_all[i] < 0 ? -v_all[i] : v_all[i];
            if (av > max_av) max_av = av;
        }
        out[idx++] = max_av;
    } else {
        out[idx++] = 0.0f; out[idx++] = 0.0f; out[idx++] = 0.0f;
    }

    /* 10: frames_with_target / 16 */
    {
        int fwt = 0;
        for (int t = 0; t < WINDOW_SIZE; t++) {
            int any = 0;
            for (int k = 0; k < 3; k++) if (valid_mask && valid_mask[t][k]) { any = 1; break; }
            if (any) fwt++;
        }
        out[idx++] = (float)fwt / (float)WINDOW_SIZE;
    }

    /* 11: average targets per frame / 3 */
    out[idx++] = (float)total_det / (float)WINDOW_SIZE / 3.0f;

    /* 12: v IQR */
    if (vn > 1) {
        float v_sorted[48];
        memcpy(v_sorted, v_all, sizeof(float) * (size_t)vn);
        out[idx++] = arr_percentile(v_sorted, vn, 0.75f) - arr_percentile(v_sorted, vn, 0.25f);
    } else out[idx++] = 0.0f;

    /* 13: d change rate */
    if (dn > 1) {
        float s = 0.0f;
        for (int i = 1; i < dn; i++) s += (d_all[i] - d_all[i-1]) < 0 ?
                                              (d_all[i-1] - d_all[i]) : (d_all[i] - d_all[i-1]);
        out[idx++] = s / (float)(dn - 1);
    } else out[idx++] = 0.0f;

    /* 14: v change rate */
    if (vn > 1) {
        float s = 0.0f;
        for (int i = 1; i < vn; i++) s += (v_all[i] - v_all[i-1]) < 0 ?
                                              (v_all[i-1] - v_all[i]) : (v_all[i] - v_all[i-1]);
        out[idx++] = s / (float)(vn - 1);
    } else out[idx++] = 0.0f;

    /* 15-18: speed-band fractions */
    if (vn > 0) {
        int c1 = 0, c2 = 0, c3 = 0, c4 = 0;
        for (int i = 0; i < vn; i++) {
            float av = v_all[i] < 0 ? -v_all[i] : v_all[i];
            if (av > 0.5f) c1++;
            if (av > 2.0f) c2++;
            if (av > 5.0f) c3++;
            if (av < 0.1f) c4++;
        }
        out[idx++] = (float)c1 / (float)vn;
        out[idx++] = (float)c2 / (float)vn;
        out[idx++] = (float)c3 / (float)vn;
        out[idx++] = (float)c4 / (float)vn;
    } else {
        for (int f = 0; f < 4; f++) out[idx++] = 0.0f;
    }

    /* 19: max simultaneous targets / 3 */
    {
        float m_per_frame[WINDOW_SIZE];
        for (int t = 0; t < WINDOW_SIZE; t++) {
            int cnt = 0;
            for (int k = 0; k < 3; k++) if (valid_mask && valid_mask[t][k]) cnt++;
            m_per_frame[t] = (float)cnt / 3.0f;
        }
        out[idx++] = arr_max(m_per_frame, WINDOW_SIZE);
    }

    /* 20: empty frame fraction */
    {
        int empty = 0;
        for (int t = 0; t < WINDOW_SIZE; t++) {
            int any = 0;
            for (int k = 0; k < 3; k++) if (valid_mask && valid_mask[t][k]) { any = 1; break; }
            if (!any) empty++;
        }
        out[idx++] = (float)empty / (float)WINDOW_SIZE;
    }

    /* 21: mean abs v */
    if (vn > 0) {
        float s = 0.0f;
        for (int i = 0; i < vn; i++) s += v_all[i] < 0 ? -v_all[i] : v_all[i];
        out[idx++] = s / (float)vn;
    } else out[idx++] = 0.0f;

    return idx;
}

void feature_normalize(float *x)
{
    for (int i = 0; i < FEATURE_DIM; i++) {
        x[i] = (x[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
    }
}

void feature_quantize(const float *in, int8_t *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float q = in[i] / INPUT_SCALE + INPUT_ZP;
        if (q < -128.0f) q = -128.0f;
        if (q >  127.0f) q =  127.0f;
        out[i] = (int8_t)q;
    }
}
