/* UART driver + frame parser. Pulls bytes from UART2 (256 000 baud,
 * 8N1, RX-only) into a ring buffer, validates magic + CRC, copies out
 * the target struct array.
 *
 * The ESP-IDF implementation uses uart_driver_install + DMA. The
 * loopback harness (tests/loopback_test.py) bypasses UART entirely
 * and calls radar_parse_frame() directly from bytes handed to it by
 * pc_radar_sim.py's in-process _LoopbackSerial.
 */
#ifndef RADAR_UART_H
#define RADAR_UART_H

#include <stdint.h>
#include <stddef.h>
#include "radar_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize UART2 at the given baud. Returns 0 on success. */
int  radar_uart_init(uint32_t baud);

/* Block until a full frame arrives (or timeout). Writes up to
   RADAR_MAX_TARGETS targets to out_targets and the frame timestamp
   to out_ts_ms. Returns number of targets, or -1 on parse error. */
int  radar_uart_read_frame(radar_target_t *out_targets, uint32_t *out_ts_ms,
                           uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* RADAR_UART_H */
