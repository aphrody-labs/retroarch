/*  RetroArch - A frontend for libretro.
 *  Copyright (C) 2026 - Aphrody
 *
 *  RetroArch is free software: you can redistribute it and/or modify it under the terms
 *  of the GNU General Public License as published by the Free Software Found-
 *  ation, either version 3 of the License, or (at your option) any later version.
 *
 *  RetroArch is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with RetroArch.
 *  If not, see <http://www.gnu.org/licenses/>.
 */

/* RetroArch as a library, driven by a host (a bun:ffi binding, a
 * WebAssembly page) instead of its own main loop. The host selects
 * the "host" video, audio and joypad drivers in the configuration and
 * makes every call from one thread. */

#ifndef __LIBRETROARCH_H
#define __LIBRETROARCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define LIBRETROARCH_API __declspec(dllexport)
#elif defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define LIBRETROARCH_API EMSCRIPTEN_KEEPALIVE
#else
#define LIBRETROARCH_API __attribute__((visibility("default")))
#endif

#define LIBRETROARCH_API_VERSION 1

enum libretroarch_pixel_format
{
   LIBRETROARCH_PIXEL_XRGB8888 = 0,
   /* 0RGB1555 cores arrive converted to this */
   LIBRETROARCH_PIXEL_RGB565,
   /* the menu's own framebuffer */
   LIBRETROARCH_PIXEL_RGBA4444
};

enum libretroarch_status_flags
{
   LIBRETROARCH_STATUS_CORE_RUNNING = (1 << 0),
   LIBRETROARCH_STATUS_PAUSED       = (1 << 1),
   LIBRETROARCH_STATUS_MENU_ALIVE   = (1 << 2),
   LIBRETROARCH_STATUS_FASTFORWARD  = (1 << 3)
};

/* Valid until the next libretroarch_iterate(). serial grows by one
 * with each new picture, so a host uploads only when it changed. */
typedef struct libretroarch_frame
{
   uint64_t serial;
   const void *data;
   unsigned width;
   unsigned height;
   unsigned pitch;
   unsigned format;
} libretroarch_frame_t;

typedef struct libretroarch_status
{
   double fps;
   double core_sample_rate;
   unsigned audio_rate;
   unsigned flags;
} libretroarch_status_t;

LIBRETROARCH_API unsigned libretroarch_api_version(void);

/* Takes RetroArch's own command line (-L core content,
 * --appendconfig ...). Returns 0 on success. */
LIBRETROARCH_API int libretroarch_init(int argc, char *argv[]);

/* Runs one frame. Returns 0, or -1 once RetroArch wants to quit. */
LIBRETROARCH_API int libretroarch_iterate(void);

LIBRETROARCH_API void libretroarch_deinit(void);

LIBRETROARCH_API void libretroarch_status(libretroarch_status_t *status);

/* Returns 0 while no picture has been presented yet. */
LIBRETROARCH_API int libretroarch_frame(libretroarch_frame_t *frame);

/* Interleaved stereo float frames at libretroarch_status().audio_rate.
 * Returns the frames copied, at most max_frames. */
LIBRETROARCH_API size_t libretroarch_audio_read(float *out,
      size_t max_frames);

/* buttons: one bit per RETRO_DEVICE_ID_JOYPAD_*. axes: left x, left y,
 * right x, right y, then L2 and R2, in [-0x7fff, 0x7fff]. */
LIBRETROARCH_API void libretroarch_joypad(unsigned port,
      uint32_t buttons, const int16_t *axes, unsigned num_axes);

/* A network command line ("PAUSE_TOGGLE", "SAVE_STATE",
 * "READ_CORE_MEMORY 0x100 16"), run at the next iterate. Returns 0
 * when the queue is full. */
LIBRETROARCH_API int libretroarch_command(const char *cmd);

/* Drains the replies of earlier commands into buf, NUL-terminated.
 * Returns the bytes written, excluding the terminator. */
LIBRETROARCH_API size_t libretroarch_command_reply(char *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif
