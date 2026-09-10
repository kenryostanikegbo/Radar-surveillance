/* Radar wire protocol -- must match scripts/protocol_encode.py exactly. */
#ifndef RADAR_PROTOCOL_H
#define RADAR_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

/* MSVC and GCC disagree on packed structs. Use #pragma pack on MSVC,
   __attribute__((packed)) on GCC; pick at compile time. */
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif

#define RADAR_MAGIC          0xAA55u
#define RADAR_VERSION        0x01
#define RADAR_MAX_TARGETS    3
#define RADAR_HEADER_SIZE    16
#define RADAR_TARGET_SIZE    16
#define RADAR_CRC_SIZE       2
#define RADAR_FRAME_SIZE(n)  (RADAR_HEADER_SIZE + RADAR_MAX_TARGETS * RADAR_TARGET_SIZE + RADAR_CRC_SIZE)

typedef struct {
    uint16_t range_cm;
    int16_t  speed_cm_s;
    int16_t  angle_cdeg;
    uint8_t  rcs;
    uint8_t  flags;
    int16_t  x_cm;
    int16_t  y_cm;
    int16_t  z_cm;
    uint16_t reserved;
}
#ifdef __GNUC__
__attribute__((packed))
#endif
radar_target_t;

typedef struct {
    uint16_t magic;
    uint8_t  version;
    uint8_t  n_targets;
    uint8_t  reserved0;
    uint32_t timestamp_ms;
    uint8_t  reserved1[7];
}
#ifdef __GNUC__
__attribute__((packed))
#endif
radar_header_t;

#ifdef _MSC_VER
#pragma pack(pop)
#endif

#ifdef __cplusplus
extern "C" {
#endif

uint16_t radar_crc16(const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif

static inline int radar_magic_ok(const uint8_t *buf) {
    return buf[0] == (RADAR_MAGIC & 0xFF) && buf[1] == ((RADAR_MAGIC >> 8) & 0xFF);
}

#endif
