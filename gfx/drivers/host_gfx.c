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
#include "../../verbosity.h"

#ifdef HAVE_EGL
#define HOST_HW_GL
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

typedef struct
{
   uint8_t *buf;
   size_t buf_cap;
   unsigned width;
   unsigned height;
   unsigned pitch;
   unsigned format;
} host_picture_t;

#ifdef HOST_HW_GL
/* GL through eglGetProcAddress: no GL headers or library are linked */
#define HOST_GL_FRAMEBUFFER              0x8D40
#define HOST_GL_RENDERBUFFER             0x8D41
#define HOST_GL_COLOR_ATTACHMENT0        0x8CE0
#define HOST_GL_DEPTH_STENCIL_ATTACHMENT 0x821A
#define HOST_GL_FRAMEBUFFER_COMPLETE     0x8CD5
#define HOST_GL_RGBA8                    0x8058
#define HOST_GL_DEPTH24_STENCIL8         0x88F0
#define HOST_GL_RGBA                     0x1908
#define HOST_GL_BGRA                     0x80E1
#define HOST_GL_UNSIGNED_BYTE            0x1401
#define HOST_GL_PACK_ALIGNMENT           0x0D05
#define HOST_GL_PIXEL_PACK_BUFFER        0x88EB
#define HOST_GL_STREAM_READ              0x88E1
#define HOST_GL_MAP_READ_BIT             0x0001

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA    0x31DD
#endif

typedef void (*host_gl_gen_t)(int n, unsigned *ids);
typedef void (*host_gl_bind_t)(unsigned target, unsigned id);
typedef void (*host_gl_storage_t)(unsigned target, unsigned format,
      int width, int height);
typedef void (*host_gl_attach_t)(unsigned target, unsigned attachment,
      unsigned rb_target, unsigned rb);
typedef unsigned (*host_gl_status_t)(unsigned target);
typedef void (*host_gl_pixel_store_t)(unsigned name, int value);
typedef void (*host_gl_read_t)(int x, int y, int width, int height,
      unsigned format, unsigned type, void *pixels);
typedef void (*host_gl_buffer_data_t)(unsigned target, intptr_t size,
      const void *data, unsigned usage);
typedef void *(*host_gl_map_t)(unsigned target, intptr_t offset,
      intptr_t length, unsigned access);
typedef unsigned char (*host_gl_unmap_t)(unsigned target);

typedef struct
{
   EGLDisplay dpy;
   EGLContext ctx;
   EGLSurface surf;
   unsigned fbo;
   unsigned rb[2];
   unsigned pbo[2];
   unsigned pbo_width[2];
   unsigned pbo_height[2];
   unsigned pbo_index;
   unsigned width;
   unsigned height;
   bool bgra;
   bool flip;
   uint8_t *rows;
   size_t rows_cap;
   host_gl_gen_t gen_framebuffers;
   host_gl_gen_t gen_renderbuffers;
   host_gl_gen_t delete_framebuffers;
   host_gl_gen_t delete_renderbuffers;
   host_gl_bind_t bind_framebuffer;
   host_gl_bind_t bind_renderbuffer;
   host_gl_storage_t renderbuffer_storage;
   host_gl_attach_t framebuffer_renderbuffer;
   host_gl_status_t check_framebuffer_status;
   host_gl_pixel_store_t pixel_store;
   host_gl_read_t read_pixels;
   host_gl_gen_t gen_buffers;
   host_gl_gen_t delete_buffers;
   host_gl_bind_t bind_buffer;
   host_gl_buffer_data_t buffer_data;
   host_gl_map_t map_buffer_range;
   host_gl_unmap_t unmap_buffer;
} host_gl_t;
#endif

typedef struct
{
#ifdef HOST_HW_GL
   host_gl_t *gl;
#endif
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

#ifdef HOST_HW_GL
bool video_host_hw_context_supported(enum retro_hw_context_type type)
{
   switch (type)
   {
      case RETRO_HW_CONTEXT_OPENGL:
      case RETRO_HW_CONTEXT_OPENGL_CORE:
      case RETRO_HW_CONTEXT_OPENGLES2:
      case RETRO_HW_CONTEXT_OPENGLES3:
      case RETRO_HW_CONTEXT_OPENGLES_VERSION:
         return true;
      default:
         break;
   }
   return false;
}

static void host_gl_free(host_gl_t *gl)
{
   if (gl->ctx != EGL_NO_CONTEXT)
   {
      eglMakeCurrent(gl->dpy, gl->surf, gl->surf, gl->ctx);
      if (gl->pbo[0])
         gl->delete_buffers(2, gl->pbo);
      if (gl->fbo)
         gl->delete_framebuffers(1, &gl->fbo);
      if (gl->rb[0])
         gl->delete_renderbuffers(2, gl->rb);
      eglMakeCurrent(gl->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE,
            EGL_NO_CONTEXT);
      eglDestroyContext(gl->dpy, gl->ctx);
   }
   if (gl->surf != EGL_NO_SURFACE)
      eglDestroySurface(gl->dpy, gl->surf);
   if (gl->dpy != EGL_NO_DISPLAY)
      eglTerminate(gl->dpy);
   free(gl->rows);
   free(gl);
}

static EGLDisplay host_gl_display(void)
{
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)
      eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay dpy = EGL_NO_DISPLAY;

   /* no window system: Mesa's surfaceless platform first */
   if (get_platform_display)
      dpy = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA,
            EGL_DEFAULT_DISPLAY, NULL);
   if (dpy == EGL_NO_DISPLAY)
      dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   return dpy;
}

/* A headless context for the core, left current for its context_reset,
 * rendering into an FBO the host reads back. */
static host_gl_t *host_gl_init(const struct retro_hw_render_callback *hwr,
      unsigned width, unsigned height)
{
   EGLint major, minor, num;
   EGLConfig config;
   EGLint attribs[16];
   EGLint config_attribs[] = {
      EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE,   8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE,  8,
      EGL_NONE
   };
   EGLint pbuffer_attribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
   int n                    = 0;
   bool gles                = hwr->context_type != RETRO_HW_CONTEXT_OPENGL
      && hwr->context_type != RETRO_HW_CONTEXT_OPENGL_CORE;
   host_gl_t *gl            = (host_gl_t*)calloc(1, sizeof(*gl));

   if (!gl)
      return NULL;
   gl->dpy  = EGL_NO_DISPLAY;
   gl->ctx  = EGL_NO_CONTEXT;
   gl->surf = EGL_NO_SURFACE;

   if (     (gl->dpy = host_gl_display()) == EGL_NO_DISPLAY
         || !eglInitialize(gl->dpy, &major, &minor))
      goto error;

   if (gles)
      config_attribs[3] = hwr->context_type == RETRO_HW_CONTEXT_OPENGLES2
         ? EGL_OPENGL_ES2_BIT : EGL_OPENGL_ES3_BIT_KHR;
   if (     !eglBindAPI(gles ? EGL_OPENGL_ES_API : EGL_OPENGL_API)
         || !eglChooseConfig(gl->dpy, config_attribs, &config, 1, &num)
         || num < 1)
      goto error;

   if (gles)
   {
      attribs[n++] = EGL_CONTEXT_MAJOR_VERSION_KHR;
      attribs[n++] = hwr->context_type == RETRO_HW_CONTEXT_OPENGLES2 ? 2
         : hwr->context_type == RETRO_HW_CONTEXT_OPENGLES3 ? 3
         : (EGLint)hwr->version_major;
      attribs[n++] = EGL_CONTEXT_MINOR_VERSION_KHR;
      attribs[n++] = hwr->context_type == RETRO_HW_CONTEXT_OPENGLES_VERSION
         ? (EGLint)hwr->version_minor : 0;
   }
   else if (hwr->context_type == RETRO_HW_CONTEXT_OPENGL_CORE)
   {
      attribs[n++] = EGL_CONTEXT_MAJOR_VERSION_KHR;
      attribs[n++] = (EGLint)hwr->version_major;
      attribs[n++] = EGL_CONTEXT_MINOR_VERSION_KHR;
      attribs[n++] = (EGLint)hwr->version_minor;
      attribs[n++] = EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR;
      attribs[n++] = EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR;
   }
   if (hwr->debug_context)
   {
      attribs[n++] = EGL_CONTEXT_FLAGS_KHR;
      attribs[n++] = EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR;
   }
   attribs[n] = EGL_NONE;

   if ((gl->ctx = eglCreateContext(gl->dpy, config, EGL_NO_CONTEXT,
               attribs)) == EGL_NO_CONTEXT)
      goto error;

   if (!eglMakeCurrent(gl->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, gl->ctx))
   {
      gl->surf = eglCreatePbufferSurface(gl->dpy, config, pbuffer_attribs);
      if (     gl->surf == EGL_NO_SURFACE
            || !eglMakeCurrent(gl->dpy, gl->surf, gl->surf, gl->ctx))
         goto error;
   }

   gl->gen_framebuffers         = (host_gl_gen_t)
      eglGetProcAddress("glGenFramebuffers");
   gl->gen_renderbuffers        = (host_gl_gen_t)
      eglGetProcAddress("glGenRenderbuffers");
   gl->delete_framebuffers      = (host_gl_gen_t)
      eglGetProcAddress("glDeleteFramebuffers");
   gl->delete_renderbuffers     = (host_gl_gen_t)
      eglGetProcAddress("glDeleteRenderbuffers");
   gl->bind_framebuffer         = (host_gl_bind_t)
      eglGetProcAddress("glBindFramebuffer");
   gl->bind_renderbuffer        = (host_gl_bind_t)
      eglGetProcAddress("glBindRenderbuffer");
   gl->renderbuffer_storage     = (host_gl_storage_t)
      eglGetProcAddress("glRenderbufferStorage");
   gl->framebuffer_renderbuffer = (host_gl_attach_t)
      eglGetProcAddress("glFramebufferRenderbuffer");
   gl->check_framebuffer_status = (host_gl_status_t)
      eglGetProcAddress("glCheckFramebufferStatus");
   gl->pixel_store              = (host_gl_pixel_store_t)
      eglGetProcAddress("glPixelStorei");
   gl->read_pixels              = (host_gl_read_t)
      eglGetProcAddress("glReadPixels");
   if (     !gl->gen_framebuffers || !gl->gen_renderbuffers
         || !gl->delete_framebuffers || !gl->delete_renderbuffers
         || !gl->bind_framebuffer || !gl->bind_renderbuffer
         || !gl->renderbuffer_storage || !gl->framebuffer_renderbuffer
         || !gl->check_framebuffer_status || !gl->pixel_store
         || !gl->read_pixels)
      goto error;

   gl->width  = width;
   gl->height = height;
   gl->bgra   = !gles;
   gl->flip   = hwr->bottom_left_origin;
   gl->gen_framebuffers(1, &gl->fbo);
   gl->gen_renderbuffers(2, gl->rb);
   gl->bind_framebuffer(HOST_GL_FRAMEBUFFER, gl->fbo);
   gl->bind_renderbuffer(HOST_GL_RENDERBUFFER, gl->rb[0]);
   gl->renderbuffer_storage(HOST_GL_RENDERBUFFER, HOST_GL_RGBA8,
         width, height);
   gl->framebuffer_renderbuffer(HOST_GL_FRAMEBUFFER,
         HOST_GL_COLOR_ATTACHMENT0, HOST_GL_RENDERBUFFER, gl->rb[0]);
   if (hwr->depth || hwr->stencil)
   {
      gl->bind_renderbuffer(HOST_GL_RENDERBUFFER, gl->rb[1]);
      gl->renderbuffer_storage(HOST_GL_RENDERBUFFER,
            HOST_GL_DEPTH24_STENCIL8, width, height);
      gl->framebuffer_renderbuffer(HOST_GL_FRAMEBUFFER,
            HOST_GL_DEPTH_STENCIL_ATTACHMENT, HOST_GL_RENDERBUFFER,
            gl->rb[1]);
   }
   gl->bind_renderbuffer(HOST_GL_RENDERBUFFER, 0);
   if (gl->check_framebuffer_status(HOST_GL_FRAMEBUFFER)
         != HOST_GL_FRAMEBUFFER_COMPLETE)
      goto error;

   /* two pack buffers so readback never waits on the GPU: each frame
    * lands while the one before it is copied out */
   gl->gen_buffers      = (host_gl_gen_t)eglGetProcAddress("glGenBuffers");
   gl->delete_buffers   = (host_gl_gen_t)eglGetProcAddress("glDeleteBuffers");
   gl->bind_buffer      = (host_gl_bind_t)eglGetProcAddress("glBindBuffer");
   gl->buffer_data      = (host_gl_buffer_data_t)
      eglGetProcAddress("glBufferData");
   gl->map_buffer_range = (host_gl_map_t)
      eglGetProcAddress("glMapBufferRange");
   gl->unmap_buffer     = (host_gl_unmap_t)
      eglGetProcAddress("glUnmapBuffer");
   if (     gl->gen_buffers && gl->delete_buffers && gl->bind_buffer
         && gl->buffer_data && gl->map_buffer_range && gl->unmap_buffer
         && (!gles || hwr->context_type != RETRO_HW_CONTEXT_OPENGLES2))
   {
      unsigned i;
      gl->gen_buffers(2, gl->pbo);
      for (i = 0; i < 2; i++)
      {
         gl->bind_buffer(HOST_GL_PIXEL_PACK_BUFFER, gl->pbo[i]);
         gl->buffer_data(HOST_GL_PIXEL_PACK_BUFFER,
               (intptr_t)width * height * 4, NULL, HOST_GL_STREAM_READ);
      }
      gl->bind_buffer(HOST_GL_PIXEL_PACK_BUFFER, 0);
   }

   RARCH_LOG("[Host] Headless OpenGL%s context, EGL %d.%d, %ux%u%s.\n",
         gles ? " ES" : "", major, minor, width, height,
         gl->pbo[0] ? ", async readback" : "");
   return gl;

error:
   RARCH_ERR("[Host] No headless OpenGL context (EGL 0x%x).\n",
         eglGetError());
   host_gl_free(gl);
   return NULL;
}

/* Rows of the read picture into the core picture, top first, XRGB8888 */
static bool host_gl_store(host_gl_t *gl, host_picture_t *pic,
      const uint8_t *src, unsigned width, unsigned height)
{
   unsigned y;
   unsigned pitch = width * 4;
   size_t size    = (size_t)pitch * height;

   if (size > pic->buf_cap)
   {
      uint8_t *tmp = (uint8_t*)realloc(pic->buf, size);
      if (!tmp)
         return false;
      pic->buf     = tmp;
      pic->buf_cap = size;
   }
   for (y = 0; y < height; y++)
   {
      const uint8_t *row = src
         + (size_t)pitch * (gl->flip ? height - 1 - y : y);
      uint8_t *dst       = pic->buf + (size_t)pitch * y;
      if (gl->bgra)
         memcpy(dst, row, pitch);
      else
      {
         unsigned x;
         for (x = 0; x < pitch; x += 4)
         {
            dst[x + 0] = row[x + 2];
            dst[x + 1] = row[x + 1];
            dst[x + 2] = row[x + 0];
            dst[x + 3] = 0xff;
         }
      }
   }
   pic->width  = width;
   pic->height = height;
   pic->pitch  = pitch;
   pic->format = LIBRETROARCH_PIXEL_XRGB8888;
   return true;
}

static bool host_gl_read(host_video_t *host, unsigned width,
      unsigned height)
{
   host_gl_t *gl = host->gl;
   unsigned fmt  = gl->bgra ? HOST_GL_BGRA : HOST_GL_RGBA;
   size_t size;

   if (width > gl->width)
      width = gl->width;
   if (height > gl->height)
      height = gl->height;
   size = (size_t)width * height * 4;

   gl->bind_framebuffer(HOST_GL_FRAMEBUFFER, gl->fbo);
   gl->pixel_store(HOST_GL_PACK_ALIGNMENT, 4);

   if (gl->pbo[0])
   {
      unsigned cur  = gl->pbo_index;
      unsigned prev = cur ^ 1;
      bool stored   = false;

      gl->bind_buffer(HOST_GL_PIXEL_PACK_BUFFER, gl->pbo[cur]);
      gl->read_pixels(0, 0, width, height, fmt, HOST_GL_UNSIGNED_BYTE,
            NULL);
      gl->pbo_width[cur]  = width;
      gl->pbo_height[cur] = height;
      gl->pbo_index       = prev;

      if (gl->pbo_width[prev])
      {
         const uint8_t *src;
         gl->bind_buffer(HOST_GL_PIXEL_PACK_BUFFER, gl->pbo[prev]);
         if ((src = (const uint8_t*)gl->map_buffer_range(
                     HOST_GL_PIXEL_PACK_BUFFER, 0,
                     (intptr_t)gl->pbo_width[prev]
                     * gl->pbo_height[prev] * 4,
                     HOST_GL_MAP_READ_BIT)))
         {
            stored = host_gl_store(gl, &host->core, src,
                  gl->pbo_width[prev], gl->pbo_height[prev]);
            gl->unmap_buffer(HOST_GL_PIXEL_PACK_BUFFER);
         }
      }
      gl->bind_buffer(HOST_GL_PIXEL_PACK_BUFFER, 0);
      return stored;
   }

   if (size > gl->rows_cap)
   {
      uint8_t *tmp = (uint8_t*)realloc(gl->rows, size);
      if (!tmp)
         return false;
      gl->rows     = tmp;
      gl->rows_cap = size;
   }
   gl->read_pixels(0, 0, width, height, fmt, HOST_GL_UNSIGNED_BYTE,
         gl->rows);
   return host_gl_store(gl, &host->core, gl->rows, width, height);
}

static uintptr_t host_gl_get_current_framebuffer(void *data)
{
   host_video_t *host = (host_video_t*)data;
   return host->gl ? host->gl->fbo : 0;
}

static retro_proc_address_t host_gl_get_proc_address(void *data,
      const char *sym)
{
   return (retro_proc_address_t)eglGetProcAddress(sym);
}
#endif

static void *host_gfx_init(const video_info_t *video)
{
   host_video_t *host = (host_video_t*)calloc(1, sizeof(*host));
#ifdef HOST_HW_GL
   struct retro_hw_render_callback *hwr = video_driver_get_hw_context();
#endif
   if (!host)
      return NULL;
   host->core_format = video->rgb32
      ? LIBRETROARCH_PIXEL_XRGB8888
      : LIBRETROARCH_PIXEL_RGB565;
#ifdef HOST_HW_GL
   if (hwr && video_host_hw_context_supported(hwr->context_type))
   {
      const struct retro_game_geometry *geom =
         &video_state_get_ptr()->av_info.geometry;
      if (!(host->gl = host_gl_init(hwr,
                  geom->max_width  ? geom->max_width  : geom->base_width,
                  geom->max_height ? geom->max_height : geom->base_height)))
      {
         free(host);
         return NULL;
      }
   }
#endif
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

#ifdef HOST_HW_GL
   if (frame == RETRO_HW_FRAME_BUFFER_VALID)
   {
      if (host->gl && host_gl_read(host, width, height))
      {
         host->shown = &host->core;
         host->serial++;
      }
      return true;
   }
#endif

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
#ifdef HOST_HW_GL
   if (host->gl)
      host_gl_free(host->gl);
#endif
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
#ifdef HOST_HW_GL
   host_gl_get_current_framebuffer,
   host_gl_get_proc_address,
#else
   NULL, /* get_current_framebuffer */
   NULL, /* get_proc_address */
#endif
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
