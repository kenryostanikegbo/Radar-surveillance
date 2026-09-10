/* Compile radar_protocol.c-like functions alongside the parse glue.
   The Python loopback harness loads this DLL via ctypes.

   Exposes:
     radar_crc16(buf, len) -> uint16
     radar_parse_frame(buf, len, out_targets_array, out_ts_ms) -> n_targets
     radar_frame_size() -> int
*/
#include <stdio.h>
#include <string.h>
#include "radar_protocol.h"

/* CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no xor-out.
   Same algorithm used in scripts/protocol_encode.py:crc16_ccitt().
   Reference: CRC of "123456789" = 0x29B1. */
uint16_t radar_crc16(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= ((uint16_t)data[i]) << 8;
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000) crc = (uint16_t)((crc << 1) ^ 0x1021);
            else              crc = (uint16_t)(crc << 1);
        }
    }
    return crc;
}

/* Decode one frame from buf. Writes up to MAX_TARGETS target structs into out_targets.
   Validates magic and CRC. Returns number of targets on success, -1 on error. */
int radar_parse_frame(const uint8_t *buf, size_t len,
                      radar_target_t *out_targets, uint32_t *out_ts_ms) {
    const size_t fsize = RADAR_HEADER_SIZE + RADAR_MAX_TARGETS * RADAR_TARGET_SIZE + RADAR_CRC_SIZE;
    if (len < fsize) return -1;
    if (!radar_magic_ok(buf)) return -1;

    radar_header_t hdr;
    memcpy(&hdr, buf, sizeof(hdr));
    if (hdr.n_targets > RADAR_MAX_TARGETS) return -1;

    /* CRC covers header + body (i.e. bytes [0, fsize-2]) */
    uint16_t expected = (uint16_t)buf[fsize - 2] | ((uint16_t)buf[fsize - 1] << 8);
    uint16_t actual   = radar_crc16(buf, fsize - RADAR_CRC_SIZE);
    if (expected != actual) return -1;

    *out_ts_ms = hdr.timestamp_ms;
    memcpy(out_targets, buf + RADAR_HEADER_SIZE, hdr.n_targets * sizeof(radar_target_t));
    return (int)hdr.n_targets;
}

int radar_frame_size(void) {
    return (int)(RADAR_HEADER_SIZE + RADAR_MAX_TARGETS * RADAR_TARGET_SIZE + RADAR_CRC_SIZE);
}
