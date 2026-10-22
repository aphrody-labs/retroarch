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

/* Hands each picture to the libretroarch host as it is, in the core's
 * pixel format: the host converts and scales on its GPU. */

#include <stdlib.h>
#include <string.h>

#ifdef HAVE_CONFIG_H
#include "../../config.h"
#endif

#ifdef HAVE_MENU
#include "../../menu/menu_driver.h"
#endif

#include "../video_driver.h"
#include "../../frontend/libretroarch.h"

typedef struct
{
   uint8_t *buf;
   size_t buf_cap;
   unsigned width;
   unsigned height;
   unsigned pitch;
   unsigned format;
} host_picture_t;

typedef struct
{
   uint64_t serial;
   host_picture_t core;
   host_picture_t menu;
   const host_picture_t *shown;
   unsigned core_format;
   bool menu_dirty;
} host_video_t;

static host_video_t *host_video = NULL;

static bool host_picture_store(host_picture_t *pic, const void *frame,
      unsigned width, unsigned height, unsigned pitch, unsigned format)
{
   size_t required = (size_t)pitch * height;
   if (required > pic->buf_cap)
   {
      uint8_t *tmp = (uint8_t*)realloc(pic->buf, required);
      if (!tmp)
         return false;
      pic->buf     = tmp;
      pic->buf_cap = required;
   }
   memcpy(pic->buf, frame, required);
   pic->width  = width;
   pic->height = height;
   pic->pitch  = pitch;
   pic->format = format;
   return true;
}

static void *host_gfx_init(const video_info_t *video)
{
   host_video_t *host = (host_video_t*)calloc(1, sizeof(*host));
   if (!host)
      return NULL;
   host->core_format = video->rgb32
      ? LIBRETROARCH_PIXEL_XRGB8888
      : LIBRETROARCH_PIXEL_RGB565;
   host_video = host;
   return host;
}

static bool host_gfx_frame(void *data, const void *frame,
      unsigned dims, uint64_t frame_count,
      unsigned pitch, const char *msg, video_frame_info_t *video_info)
{
   host_video_t *host = (host_video_t*)data;
   unsigned width     = VIDEO_SCALE_W(dims);
   unsigned height    = VIDEO_SCALE_H(dims);
#ifdef HAVE_MENU
   bool menu_is_alive = (video_info->menu_st_flags & MENU_ST_FLAG_ALIVE)
      ? true : false;

   menu_driver_frame(menu_is_alive, video_info);

   if (menu_is_alive && host->menu.buf)
   {
      if (host->menu_dirty || host->shown != &host->menu)
         host->serial++;
      host->shown      = &host->menu;
      host->menu_dirty = false;
      return true;
   }
#endif

   /* NULL is a dupe: the last picture stands */
   if (!frame || !width || !height)
      return true;

   if (host_picture_store(&host->core, frame, width, height, pitch,
            host->core_format))
   {
      host->shown = &host->core;
      host->serial++;
   }
   return true;
}

static void host_gfx_set_nonblock_state(void *a, bool b, bool c,
      unsigned d) { }
static bool host_gfx_alive(void *data) { return true; }
static bool host_gfx_focus(void *data) { return true; }
static bool host_gfx_suppress_screensaver(void *data, bool enable) { return false; }
static bool host_gfx_has_windowed(void *data) { return true; }
static bool host_gfx_set_shader(void *data,
      enum rarch_shader_type type, const char *path) { return false; }
static void host_gfx_set_viewport(void *data, unsigned dims,
      bool force_full, bool allow_rotate) { }
static void host_gfx_set_rotation(void *data, unsigned rotation) { }

static void host_gfx_free(void *data)
{
   host_video_t *host = (host_video_t*)data;
   if (!host)
      return;
   if (host_video == host)
      host_video = NULL;
   free(host->core.buf);
   free(host->menu.buf);
   free(host);
}

#ifdef HAVE_MENU
static void host_set_texture_frame(void *data,
      const void *frame, bool rgb32, unsigned dims,
      float alpha)
{
   host_video_t *host = (host_video_t*)data;
   unsigned width     = VIDEO_SCALE_W(dims);
   unsigned height    = VIDEO_SCALE_H(dims);

   if (!frame || !width || !height)
      return;

   if (host_picture_store(&host->menu, frame, width, height,
         width * (rgb32 ? 4 : 2),
         rgb32 ? LIBRETROARCH_PIXEL_XRGB8888 : LIBRETROARCH_PIXEL_RGBA4444))
      host->menu_dirty = true;
}
#endif

static const video_poke_interface_t host_poke_interface = {
   NULL, /* get_flags */
   NULL, /* load_texture */
   NULL, /* unload_texture */
   NULL, /* set_video_mode */
   NULL, /* get_refresh_rate */
   NULL, /* set_filtering */
   NULL, /* get_video_output_size */
   NULL, /* get_video_output_prev */
   NULL, /* get_video_output_next */
   NULL, /* get_current_framebuffer */
   NULL, /* get_proc_address */
   NULL, /* set_aspect_ratio */
   NULL, /* apply_state_changes */
#ifdef HAVE_MENU
   host_set_texture_frame,
#else
   NULL, /* set_texture_frame */
#endif
   NULL, /* set_texture_enable */
   NULL, /* set_osd_msg */
   NULL, /* show_mouse */
   NULL, /* grab_mouse_toggle */
   NULL, /* get_current_shader */
   NULL, /* get_current_software_framebuffer */
   NULL, /* get_hw_render_interface */
   NULL, /* set_hdr_menu_nits */
   NULL, /* set_hdr_paper_white_nits */
   NULL, /* set_hdr_expand_gamut */
   NULL, /* set_hdr_scanlines */
   NULL  /* set_hdr_subpixel_layout */
};

static void host_gfx_get_poke_interface(void *data,
      const video_poke_interface_t **iface)
{
   *iface = &host_poke_interface;
}

int libretroarch_frame(libretroarch_frame_t *out)
{
   const host_picture_t *pic;
   if (!host_video || !(pic = host_video->shown))
      return 0;
   out->serial = host_video->serial;
   out->data   = pic->buf;
   out->width  = pic->width;
   out->height = pic->height;
   out->pitch  = pic->pitch;
   out->format = pic->format;
   return 1;
}

video_driver_t video_host = {
   host_gfx_init,
   host_gfx_frame,
   host_gfx_set_nonblock_state,
   host_gfx_alive,
   host_gfx_focus,
   host_gfx_suppress_screensaver,
   host_gfx_has_windowed,
   host_gfx_set_shader,
   host_gfx_free,
   "host",
   host_gfx_set_viewport,
   host_gfx_set_rotation,
   NULL, /* viewport_info */
   NULL, /* read_viewport */
#ifdef HAVE_OVERLAY
   NULL, /* overlay_interface */
#endif
   host_gfx_get_poke_interface,
   NULL, /* wrap_type_to_enum */
   NULL, /* shader_load_begin */
   NULL, /* shader_load_step */
#ifdef HAVE_GFX_WIDGETS
   NULL  /* gfx_widgets_enabled */
#endif
};
