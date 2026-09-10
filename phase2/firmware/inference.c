/* TFLite Micro inference wrapper. See inference.h for design notes.
 *
 * The host-side (loopback) build of this DLL exposes int_softmax(),
 * which the Python harness calls to convert int8 logits back to
 * float probabilities (used for the dashboard heatmap). The full
 * TFLite Micro wiring (interpreter + arena allocation + invoke) is
 * in main.c under ESP-IDF.
 */
#include <math.h>
#include "inference.h"

int inference_init(inference_t *ctx) {
    /* Real init happens in main.c with TFLite Micro headers and the
       ESP-IDF allocator. Here we just zero the context. */
    for (size_t i = 0; i < sizeof(ctx->arena); i++) ctx->arena[i] = 0;
    ctx->interpreter = (void *)0;
    ctx->model       = (void *)0;
    return 0;
}

/* Convert int8 logits to float softmax probabilities.
   logits[i] is in [INT8_MIN..INT8_MAX]; we dequantize via OUTPUT_SCALE/ZP
   and apply softmax in float (cheap on Cortex-M4F at 240 MHz). */
void int_softmax(const int8_t *logits, float *probs, int n) {
    float f[NUM_CLASSES];
    float maxv = -1e30f;
    for (int i = 0; i < n; i++) {
        f[i] = ((float)logits[i] - (float)OUTPUT_ZP) * OUTPUT_SCALE;
        if (f[i] > maxv) maxv = f[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        f[i] = expf(f[i] - maxv);
        sum += f[i];
    }
    for (int i = 0; i < n; i++) {
        probs[i] = f[i] / (sum > 1e-30f ? sum : 1.0f);
    }
}

int inference_run(inference_t *ctx, const int8_t *features_q,
                  int *out_class, float *out_conf)
{
    (void)ctx;
    (void)features_q;
    /* Real invoke() in main.c. On host the Python harness runs the
       MLP via tflite_runtime (or the NumPy float32 fallback). */
    for (int i = 0; i < NUM_CLASSES; i++) out_conf[i] = 0.0f;
    *out_class = 0;
    return 0;
}
