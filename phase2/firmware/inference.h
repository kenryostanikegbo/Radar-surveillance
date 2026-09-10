/* TFLite Micro inference wrapper.
 *
 * Loads model_data[] (see models/model_data.cc) once at boot into a
 * 20 KB arena, sets up the resolver for the INT8 quantized MLP, and
 * exposes `inference_run()` to map a 72-dim feature vector through
 * the model and return the predicted class + softmax confidences.
 *
 * Input: int8_t[72]   (already quantized via feature_quantize())
 * Output: int8_t[3]    (logits; softmax applied in C via int_softmax)
 *
 * This file is the firmware counterpart of scripts/test_model_offline.py.
 * The Python harness loads the same TFLite model with tflite_runtime
 * for parity validation.
 */
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stddef.h>
#include "model_metadata.h"   /* FEATURE_DIM, NUM_CLASSES, INPUT_* */

#define INFERENCE_ARENA_BYTES 24576    /* 24 KB, fits 6.6KB model + intermediates */

typedef struct {
    void    *interpreter;   /* tflite::MicroInterpreter* (opaque) */
    void    *model;         /* tflite::Model* (opaque) */
    uint8_t arena[INFERENCE_ARENA_BYTES];
} inference_t;

/* One-time setup. Allocates arena, loads model, prepares interpreter.
   Returns 0 on success, -1 on failure. */
int  inference_init(inference_t *ctx);

/* Run inference on a 72-dim int8 feature vector. Writes the predicted
   class index to *out_class and the per-class softmax probabilities
   to out_conf[NUM_CLASSES]. Returns 0 on success. */
int  inference_run(inference_t *ctx, const int8_t *features_q,
                   int *out_class, float *out_conf);

#endif /* INFERENCE_H */
