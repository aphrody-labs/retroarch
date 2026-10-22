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

/* Float stereo for the libretroarch host to drain. The host's device
 * clock sets the drain rate, and the fill reported here lets rate
 * control follow it. */

#include <stdint.h>
#include <stdlib.h>

#include <boolean.h>
#include <retro_spsc.h>

#include "../audio_driver.h"
#include "../../frontend/libretroarch.h"

#define HOST_AUDIO_FRAME_SIZE (2 * sizeof(float))

typedef struct
{
   retro_spsc_t ring;
   size_t size;
   bool is_paused;
} host_audio_t;

static host_audio_t *host_audio = NULL;

static void *host_audio_init(const char *device, unsigned rate,
      unsigned latency, unsigned *new_rate)
{
   host_audio_t *host = (host_audio_t*)calloc(1, sizeof(*host));
   size_t frames      = (size_t)rate * latency / 1000;

   if (!host)
      return NULL;
   if (frames < 256)
      frames = 256;
   if (!retro_spsc_init(&host->ring, frames * HOST_AUDIO_FRAME_SIZE))
   {
      free(host);
      return NULL;
   }
   host->size = frames * HOST_AUDIO_FRAME_SIZE;
   host_audio = host;
   return host;
}

static ssize_t host_audio_write(void *data, const void *buf, size_t len)
{
   host_audio_t *host = (host_audio_t*)data;
   size_t avail       = retro_spsc_write_avail(&host->ring);

   len -= len % HOST_AUDIO_FRAME_SIZE;
   if (len > avail)
      len = avail - avail % HOST_AUDIO_FRAME_SIZE;
   if (!len)
      return 0;
   return (ssize_t)retro_spsc_write(&host->ring, buf, len);
}

static bool host_audio_stop(void *data)
{
   host_audio_t *host = (host_audio_t*)data;
   if (host)
      host->is_paused = true;
   return true;
}

static bool host_audio_start(void *data, bool is_shutdown)
{
   host_audio_t *host = (host_audio_t*)data;
   if (host)
      host->is_paused = false;
   return true;
}

static bool host_audio_alive(void *data)
{
   host_audio_t *host = (host_audio_t*)data;
   return host && !host->is_paused;
}

/* never blocks: write takes what fits either way */
static void host_audio_set_nonblock_state(void *data, bool state) { }

static void host_audio_free(void *data)
{
   host_audio_t *host = (host_audio_t*)data;
   if (!host)
      return;
   if (host_audio == host)
      host_audio = NULL;
   retro_spsc_free(&host->ring);
   free(host);
}

static bool host_audio_use_float(void *data) { return true; }

static size_t host_audio_write_avail(void *data)
{
   host_audio_t *host = (host_audio_t*)data;
   size_t avail       = retro_spsc_write_avail(&host->ring);
   if (avail > host->size)
      avail = host->size;
   return avail;
}

static size_t host_audio_buffer_size(void *data)
{
   host_audio_t *host = (host_audio_t*)data;
   return host->size;
}

size_t libretroarch_audio_read(float *out, size_t max_frames)
{
   size_t bytes;
   if (!host_audio || host_audio->is_paused)
      return 0;
   bytes = retro_spsc_read_avail(&host_audio->ring);
   if (bytes > max_frames * HOST_AUDIO_FRAME_SIZE)
      bytes = max_frames * HOST_AUDIO_FRAME_SIZE;
   bytes -= bytes % HOST_AUDIO_FRAME_SIZE;
   if (!bytes)
      return 0;
   return retro_spsc_read(&host_audio->ring, out, bytes)
      / HOST_AUDIO_FRAME_SIZE;
}

audio_driver_t audio_host = {
   host_audio_init,
   host_audio_write,
   host_audio_stop,
   host_audio_start,
   host_audio_alive,
   host_audio_set_nonblock_state,
   host_audio_free,
   host_audio_use_float,
   "host",
   NULL,
   NULL,
   host_audio_write_avail,
   host_audio_buffer_size,
   NULL  /* write_raw */
};
