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

#include <boolean.h>
#include <features/features_cpu.h>
#include <queues/task_queue.h>

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

#ifdef HAVE_MENU
#include "../menu/menu_driver.h"
#endif

#include "../audio/audio_driver.h"
#include "../configuration.h"
#include "../gfx/video_driver.h"
#include "../input/input_driver.h"
#include "../runloop.h"
#include "frontend.h"
#include "libretroarch.h"

static bool libretroarch_inited = false;

unsigned libretroarch_api_version(void)
{
   return LIBRETROARCH_API_VERSION;
}

int libretroarch_init(int argc, char *argv[])
{
   if (libretroarch_inited)
      return -1;
   if (rarch_main(argc, argv, NULL) != 0)
   {
      main_exit(NULL);
      return -1;
   }
   libretroarch_inited = true;
   return 0;
}

int libretroarch_iterate(void)
{
   int ret;
   retro_time_t start;
   input_driver_state_t *input_st = input_state_get_ptr();
   if (!libretroarch_inited)
      return -1;
   start = cpu_features_get_time_usec();
   ret   = runloop_iterate();
   task_queue_check();
   if (ret == -1)
      return -1;

   /* Fast-forward runs extra frames within one host frame: the host
    * paces, so the runloop's own timer cannot. */
   if (input_st->flags & INP_FLAG_NONBLOCKING)
   {
      settings_t *settings = config_get_ptr();
      float ratio          = settings->floats.fastforward_ratio;
      double fps           = video_state_get_ptr()->av_info.timing.fps;
      retro_time_t budget  = (retro_time_t)(750000.0 / (fps > 0.0 ? fps : 60.0));
      unsigned max         = ratio >= 1.0f ? (unsigned)ratio : 64;
      unsigned count       = 1;

      while (     count < max
               && (input_st->flags & INP_FLAG_NONBLOCKING)
               && cpu_features_get_time_usec() - start < budget)
      {
         if (runloop_iterate() == -1)
            return -1;
         task_queue_check();
         count++;
      }
   }
   return 0;
}

void libretroarch_deinit(void)
{
   if (!libretroarch_inited)
      return;
   main_exit(NULL);
   libretroarch_inited = false;
}

void libretroarch_status(libretroarch_status_t *status)
{
   video_driver_state_t *video_st = video_state_get_ptr();
   audio_driver_state_t *audio_st = audio_state_get_ptr();
   input_driver_state_t *input_st = input_state_get_ptr();
   uint32_t runloop_flags         = runloop_get_flags();
   unsigned flags                 = 0;

   if (runloop_flags & RUNLOOP_FLAG_CORE_RUNNING)
      flags |= LIBRETROARCH_STATUS_CORE_RUNNING;
   if (runloop_flags & RUNLOOP_FLAG_PAUSED)
      flags |= LIBRETROARCH_STATUS_PAUSED;
   if (input_st->flags & INP_FLAG_NONBLOCKING)
      flags |= LIBRETROARCH_STATUS_FASTFORWARD;
#ifdef HAVE_MENU
   if (menu_state_get_ptr()->flags & MENU_ST_FLAG_ALIVE)
      flags |= LIBRETROARCH_STATUS_MENU_ALIVE;
#endif

   status->fps              = video_st->av_info.timing.fps;
   status->core_sample_rate = video_st->av_info.timing.sample_rate;
   status->audio_rate       = audio_st->out_rate;
   status->flags            = flags;
}
