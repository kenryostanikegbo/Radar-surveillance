# RD-03D AI Radar Surveillance — Project Status

## Current Progress

**Completed / available**
- Machine learning model training (MLP and 1D-CNN)
- INT8 TFLite model export
- Radar data simulation
- PC dashboard and simulated inference
- Feature extraction and model evaluation scripts
- ESP32-S3 firmware framework and target tracking logic

**Not yet completed or hardware-verified**
- Actual RD-03D UART data reception
- Native RD-03D packet parsing
- TFLite Micro inference on ESP32-S3
- Live radar classification validation
- LCD, buzzer and LED integration
- ESP32-hosted dashboard validation
- microSD event logging

## Hardware

- RD-03D radar sensor
- ESP32-S3 development board
- I²C character LCD (as shown in the schematic)
- Buzzer and LED
- microSD module

Verify exact hardware variants and wiring before flashing firmware.

## Next Priority

1. Build and flash the ESP32-S3 firmware.
2. Connect the RD-03D and capture real UART data.
3. Implement native radar-frame decoding.
4. Integrate the trained INT8 model with TFLite Micro.
5. Validate predictions using real radar measurements.
6. Integrate the LCD, alerts and dashboard.

## Important Technical Notes

- The current inference implementation is a placeholder and does not yet execute the trained model.
- The current UART read function is also a placeholder.
- Offline model accuracy does not establish real-world radar classification accuracy.
- The 72-feature preprocessing pipeline must remain consistent between Python and ESP32 firmware.

## Team Workflow

- Create a separate branch for each feature.
- Open pull requests before merging changes into `main`.
- Record hardware wiring changes and test results.
- Update this file whenever a milestone is completed.
- Never commit passwords, private credentials, local environments or generated build directories.
