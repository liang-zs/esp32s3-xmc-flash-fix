# ESP32-S3 XMC Flash `0x207018` experimental boot workaround

> **Experimental, not a production fix.** This narrowly scoped workaround was tested on one ESP32-S3 N16R8 board. No claim of manufacturer support or long-term flash reliability is made.

## Symptom

On an ESP32-S3 with a 16 MiB XMC flash reporting JEDEC ID `0x207018` (manufacturer `0x20`, device `0x7018`), the stock bootloader can fail before the application starts:

```text
E (26) bootloader_flash: XMC flash startup fail
E (26) boot:esp32s3: failed when running XMC startup flow, reboot!
```

## Experimental change

In **ESP-IDF v5.5.5**, `is_xmc_chip_strict()` in `components/bootloader_support/bootloader_flash/src/bootloader_flash.c` recognizes several XMC ID families but not MFID `0x70` / CPID `0x18`. The [patch](patches/esp-idf-v5.5.5-xmc-207018.patch) accepts **only this exact ID**, leaving all existing families unchanged.

**Important:** When the initial ROM flash ID matches, this skips the existing XMC special startup flow. The correct initialization sequence for this part has not been confirmed against its datasheet. This may not be safe for all devices or power conditions.

## Apply and build a minimal test

```bash
git clone --branch v5.5.5 --depth 1 https://github.com/espressif/esp-idf.git
cd esp-idf
git submodule update --init --recursive
./install.sh esp32s3
# Activate the installed ESP-IDF environment; supply a compatible Python interpreter.
. ./export.sh
git apply /path/to/esp32s3-xmc-flash-fix/patches/esp-idf-v5.5.5-xmc-207018.patch

cp -R examples/get-started/hello_world ../xmc_flash_test
cd ../xmc_flash_test
idf.py set-target esp32s3
idf.py menuconfig  # DIO, 40 MHz, 16 MB; XMC bootloader support enabled
idf.py build
```

**Back up the entire flash before writing anything.** Inspect `build/flasher_args.json` and the actual partition table. The test image used bootloader `0x0`, partition table `0x8000`, and app `0x10000`. Avoid blanket chip erasure.

## Validation on one board

- ESP32-S3 N16R8: **16 MiB flash**, **8 MiB PSRAM**; chip rev. v0.2.
- ESP-IDF v5.5.5 with patched bootloader successfully ran `hello_world`.
- Five complete power-cycle starts: **5/5 successful**.
- A fresh 64 KiB read at flash offset `0x900000` matched the original pre-flash backup.
- An Arduino ESP32 Core 3.3.12 HID Bridge app (~942 KiB) was written **only at app offset `0x10000`**, preserving the patched bootloader and the tested ESP-IDF factory partition table. Write verification and app boot succeeded.
- With that app, an actual iPhone showed a USB HID pointer and accepted a mouse click. Mac-to-board Wi-Fi TCP and iPhone-to-Mac AirPlay mirroring worked; one DeepSeek API-driven navigation task succeeded.

**Not validated:** long-term flash stability/endurance, comprehensive flash erase/write testing, extensive PSRAM/Wi-Fi testing, or repeatability on other boards. The Arduino-generated partition table was different from the active ESP-IDF table. **Do not flash Arduino's `merged.bin` or use IDE Upload without preserving/rebuilding the patched bootloader.** The app-only flash procedure applies only to a confirmed compatible partition layout.

## Privacy

Never commit `secrets.h`, `.env`, Wi-Fi passwords, model API keys, device flash dumps, or locally generated binaries. This repository contains no such data.

## Related work and attribution

- [ESP-IDF issue #19396](https://github.com/espressif/esp-idf/issues/19396): initial investigation.
- [ESP-IDF issue #18947](https://github.com/espressif/esp-idf/issues/18947): similar flash ID.
- Original iPhone control software: **[ESP32 iPhone Agent by Henry Yuan](https://github.com/onehumanbeing/esp32-iphone-agent)** (MIT license). The original HID Bridge and Python Agent source are **not** redistributed here.

The patch targets ESP-IDF code; respect the original project's license and copyright. This repository is not endorsed by Espressif or XMC.
