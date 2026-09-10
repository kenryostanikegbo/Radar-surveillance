/* Feature extraction: 16-frame window -> 72-dim feature vector.
 * Direct C port of phase2/scripts/compute_features.py:window_to_features.
 *
 * 17 features per target x 3 targets + 21 global = 72.
 * Inputs: arrays of recent radar frames (max MAX_TARGETS=3 per frame).
 * Output: int8_t feature vector quantized with the same scale/zero_point
 *         the model expects (see model_metadata.h FEATURE_MEAN/FEATURE_STD).
 */
#ifndef FEATURE_EXTRACT_H
#define FEATURE_EXTRACT_H

#include <stdint.h>
#include <stddef.h>
#include "radar_protocol.h"

#define WINDOW_SIZE       16
#define FEATS_PER_TARGET  17
#define GLOBAL_FEATS      21
/* FEATURE_DIM is defined in model_metadata.h (single source of truth). */

/* Source layout: the firmware maintains a ring buffer of the last
   WINDOW_SIZE frame arrays. Each array holds up to MAX_TARGETS targets
   in cartesian-cm form (we ignore range_cm / angle_cdeg / rcs -- the
   features only use d_m, speed_mps, x_m, y_m).

   The d/speed/x/y conversion from frame units to float metres:
       d_m      = sqrt(x_cm^2 + y_cm^2) / 100
       speed_m  = speed_cm_s / 100
       x_m      = x_cm / 100
       y_m      = y_cm / 100
*/

/* Compute features for a single window.
   window   : array of WINDOW_SIZE frames, each frame up to MAX_TARGETS targets
   out      : output buffer of length FEATURE_DIM floats (z-score normalized)
   valid_mask: per-frame, per-target mask (1=present, 0=padded). May be NULL.
   Returns number of features written (FEATURE_DIM on success).
*/
int feature_extract_window(const radar_target_t (*window)[RADAR_MAX_TARGETS],
                           const uint8_t (*valid_mask)[RADAR_MAX_TARGETS],
                           float *out);

/* Quantize a float feature vector into int8_t using the model's input scale
   and zero_point (input_scale = 0.07973, input_zero_point = 8). */
void feature_quantize(const float *in, int8_t *out, size_t n);

/* Apply z-score normalization: (x - mean) / std, using values baked into
   model_metadata.h (FEATURE_MEAN, FEATURE_STD). */
void feature_normalize(float *x);

#endif /* FEATURE_EXTRACT_H */
