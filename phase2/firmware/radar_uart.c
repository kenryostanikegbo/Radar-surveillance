/* UART driver stub. The actual implementation lives in main.c /
 * ESP-IDF uart_driver_install() with DMA. This file exists so the
 * firmware module list is complete and the loopback harness can
 * reason about the contract.
 *
 * On host (loopback harness), radar_parse_frame() in parse_frame.c
 * is called directly on the bytes read from the loopback serial,
 * bypassing this stub.
 */
#include "radar_uart.h"

int radar_uart_init(uint32_t baud) {
    (void)baud;
    return 0;
}

int radar_uart_read_frame(radar_target_t *out_targets, uint32_t *out_ts_ms,
                          uint32_t timeout_ms)
{
    (void)out_targets; (void)out_ts_ms; (void)timeout_ms;
    return -1;
}
