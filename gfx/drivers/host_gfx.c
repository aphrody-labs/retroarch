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

#ifdef HAVE_VULKAN
#define HOST_HW_VK
#include <libretro_vulkan.h>
#ifdef HAVE_THREADS
#include <rthreads/rthreads.h>
#endif
#include "../common/vulkan_common.h"
#endif

#if defined(HOST_HW_GL) || defined(HOST_HW_VK)
#define HOST_HW
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

#ifdef HOST_HW_VK
/* Frames in flight: the one being copied and the one being read */
#define HOST_VK_SYNC 2

typedef struct
{
   gfx_ctx_vulkan_data_t ctx;
   struct retro_hw_render_interface_vulkan iface;
   VkCommandPool pool;
   VkCommandBuffer cmd[HOST_VK_SYNC];
   VkFence fence[HOST_VK_SYNC];
   VkBuffer buf[HOST_VK_SYNC];
   VkDeviceMemory mem[HOST_VK_SYNC];
   const uint8_t *map[HOST_VK_SYNC];
   unsigned buf_width[HOST_VK_SYNC];
   unsigned buf_height[HOST_VK_SYNC];
   VkFormat buf_format[HOST_VK_SYNC];
   bool pending[HOST_VK_SYNC];
   bool coherent;
   unsigned index;
   unsigned width;
   unsigned height;
   /* what the core handed over for this frame */
   const struct retro_vulkan_image *image;
   VkSemaphore *wait;
   VkPipelineStageFlags *wait_stages;
   unsigned num_wait;
   unsigned cap_wait;
   uint32_t src_queue_family;
   VkCommandBuffer *cmds;
   unsigned num_cmd;
   unsigned cap_cmd;
   VkSemaphore signal;
} host_vk_t;
#endif

typedef struct
{
#ifdef HOST_HW_GL
   host_gl_t *gl;
#endif
#ifdef HOST_HW_VK
   host_vk_t *vk;
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

#ifdef HOST_HW
bool video_host_hw_context_supported(enum retro_hw_context_type type)
{
   switch (type)
   {
#ifdef HOST_HW_GL
      case RETRO_HW_CONTEXT_OPENGL:
      case RETRO_HW_CONTEXT_OPENGL_CORE:
      case RETRO_HW_CONTEXT_OPENGLES2:
      case RETRO_HW_CONTEXT_OPENGLES3:
      case RETRO_HW_CONTEXT_OPENGLES_VERSION:
         return true;
#endif
#ifdef HOST_HW_VK
      case RETRO_HW_CONTEXT_VULKAN:
         return true;
#endif
      default:
         break;
   }
   return false;
}

/* 32-bit rows into the core picture as XRGB8888, top first */
static bool host_picture_store_rgba32(host_picture_t *pic,
      const uint8_t *src, unsigned width, unsigned height,
      bool flip, bool bgra)
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
         + (size_t)pitch * (flip ? height - 1 - y : y);
      uint8_t *dst       = pic->buf + (size_t)pitch * y;
      if (bgra)
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
#endif

#ifdef HOST_HW_GL

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
            stored = host_picture_store_rgba32(&host->core, src,
                  gl->pbo_width[prev], gl->pbo_height[prev],
                  gl->flip, gl->bgra);
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
   return host_picture_store_rgba32(&host->core, gl->rows, width, height,
         gl->flip, gl->bgra);
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

#ifdef HOST_HW_VK
static void host_vk_free(host_vk_t *vk)
{
   unsigned i;
   VkDevice device = vk->ctx.context.device;

   if (device)
   {
      vkDeviceWaitIdle(device);
      for (i = 0; i < HOST_VK_SYNC; i++)
      {
         if (vk->map[i])
            vkUnmapMemory(device, vk->mem[i]);
         if (vk->buf[i])
            vkDestroyBuffer(device, vk->buf[i], NULL);
         if (vk->mem[i])
            vkFreeMemory(device, vk->mem[i], NULL);
         if (vk->fence[i])
            vkDestroyFence(device, vk->fence[i], NULL);
      }
      if (vk->pool)
         vkDestroyCommandPool(device, vk->pool, NULL);
   }
   vulkan_context_destroy(&vk->ctx, false);
   free(vk->wait);
   free(vk->wait_stages);
   free(vk->cmds);
   free(vk);
}

static uint32_t host_vk_get_sync_index(void *handle)
{
   return ((host_vk_t*)handle)->index;
}

static uint32_t host_vk_get_sync_index_mask(void *handle)
{
   return (1 << HOST_VK_SYNC) - 1;
}

/* The frame before reused this index's slot; its fence was waited then. */
static void host_vk_wait_sync_index(void *handle) { }

static void host_vk_set_image(void *handle,
      const struct retro_vulkan_image *image,
      uint32_t num_semaphores, const VkSemaphore *semaphores,
      uint32_t src_queue_family)
{
   host_vk_t *vk = (host_vk_t*)handle;
   unsigned i;

   vk->image            = image;
   vk->src_queue_family = src_queue_family;
   vk->num_wait         = 0;
   if (num_semaphores > vk->cap_wait)
   {
      VkPipelineStageFlags *stages;
      VkSemaphore *wait = (VkSemaphore*)realloc(vk->wait,
            num_semaphores * sizeof(*wait));
      if (wait)
         vk->wait = wait;
      stages = (VkPipelineStageFlags*)realloc(vk->wait_stages,
            num_semaphores * sizeof(*stages));
      if (stages)
         vk->wait_stages = stages;
      if (!wait || !stages)
         return;
      vk->cap_wait = num_semaphores;
   }
   for (i = 0; i < num_semaphores; i++)
   {
      vk->wait[i]        = semaphores[i];
      vk->wait_stages[i] = VK_PIPELINE_STAGE_TRANSFER_BIT;
   }
   vk->num_wait = num_semaphores;
}

static void host_vk_set_command_buffers(void *handle, uint32_t num_cmd,
      const VkCommandBuffer *cmd)
{
   host_vk_t *vk = (host_vk_t*)handle;

   vk->num_cmd = 0;
   /* one more for the host's own copy, submitted after the core's */
   if (num_cmd + 1 > vk->cap_cmd)
   {
      VkCommandBuffer *cmds = (VkCommandBuffer*)realloc(vk->cmds,
            (num_cmd + 1) * sizeof(*cmds));
      if (!cmds)
         return;
      vk->cmds    = cmds;
      vk->cap_cmd = num_cmd + 1;
   }
   if (num_cmd)
      memcpy(vk->cmds, cmd, num_cmd * sizeof(*cmd));
   vk->num_cmd = num_cmd;
}

static void host_vk_lock_queue(void *handle)
{
#ifdef HAVE_THREADS
   host_vk_t *vk = (host_vk_t*)handle;
   if (vk->ctx.context.queue_lock)
      slock_lock(vk->ctx.context.queue_lock);
#endif
}

static void host_vk_unlock_queue(void *handle)
{
#ifdef HAVE_THREADS
   host_vk_t *vk = (host_vk_t*)handle;
   if (vk->ctx.context.queue_lock)
      slock_unlock(vk->ctx.context.queue_lock);
#endif
}

static void host_vk_set_signal_semaphore(void *handle, VkSemaphore semaphore)
{
   ((host_vk_t*)handle)->signal = semaphore;
}

/* A headless device the core renders with; its images are copied into
 * host-visible buffers, one per sync index. */
static host_vk_t *host_vk_init(unsigned width, unsigned height)
{
   unsigned i;
   VkDevice device;
   VkCommandPoolCreateInfo pool_info;
   VkCommandBufferAllocateInfo cmd_info;
   VkFenceCreateInfo fence_info;
   VkBufferCreateInfo buf_info;
   struct retro_hw_render_interface_vulkan *iface;
   const VkPhysicalDeviceMemoryProperties *mp;
   host_vk_t *vk = (host_vk_t*)calloc(1, sizeof(*vk));

   if (!vk)
      return NULL;
   if (     !vulkan_context_init(&vk->ctx, VULKAN_WSI_NONE)
         || !vulkan_context_init_headless_device(&vk->ctx))
      goto error;

   device     = vk->ctx.context.device;
   mp         = &vk->ctx.context.memory_properties;
   vk->width  = width;
   vk->height = height;

   memset(&pool_info, 0, sizeof(pool_info));
   pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
   pool_info.flags            =
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
   pool_info.queueFamilyIndex = vk->ctx.context.graphics_queue_index;
   if (vkCreateCommandPool(device, &pool_info, NULL, &vk->pool)
         != VK_SUCCESS)
      goto error;

   memset(&cmd_info, 0, sizeof(cmd_info));
   cmd_info.sType              =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
   cmd_info.commandPool        = vk->pool;
   cmd_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   cmd_info.commandBufferCount = HOST_VK_SYNC;
   if (vkAllocateCommandBuffers(device, &cmd_info, vk->cmd) != VK_SUCCESS)
      goto error;

   memset(&fence_info, 0, sizeof(fence_info));
   fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

   memset(&buf_info, 0, sizeof(buf_info));
   buf_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
   buf_info.size        = (VkDeviceSize)width * height * 4;
   buf_info.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
   buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

   for (i = 0; i < HOST_VK_SYNC; i++)
   {
      VkMemoryRequirements reqs;
      VkMemoryAllocateInfo alloc;
      void *map = NULL;

      if (     vkCreateFence(device, &fence_info, NULL, &vk->fence[i])
            != VK_SUCCESS
            || vkCreateBuffer(device, &buf_info, NULL, &vk->buf[i])
            != VK_SUCCESS)
         goto error;
      vkGetBufferMemoryRequirements(device, vk->buf[i], &reqs);

      memset(&alloc, 0, sizeof(alloc));
      alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      alloc.allocationSize  = reqs.size;
      /* cached memory is far faster for the CPU to read */
      alloc.memoryTypeIndex = vulkan_find_memory_type_fallback(mp,
            reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      vk->coherent = (mp->memoryTypes[alloc.memoryTypeIndex].propertyFlags
            & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? true : false;
      if (     vkAllocateMemory(device, &alloc, NULL, &vk->mem[i])
            != VK_SUCCESS
            || vkBindBufferMemory(device, vk->buf[i], vk->mem[i], 0)
            != VK_SUCCESS
            || vkMapMemory(device, vk->mem[i], 0, VK_WHOLE_SIZE, 0, &map)
            != VK_SUCCESS)
         goto error;
      vk->map[i] = (const uint8_t*)map;
   }

   iface                         = &vk->iface;
   iface->interface_type         = RETRO_HW_RENDER_INTERFACE_VULKAN;
   iface->interface_version      = RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION;
   iface->handle                 = vk;
   iface->instance               = vk->ctx.context.instance;
   iface->gpu                    = vk->ctx.context.gpu;
   iface->device                 = device;
   iface->get_device_proc_addr   = vkGetDeviceProcAddr;
   iface->get_instance_proc_addr =
      vulkan_symbol_wrapper_instance_proc_addr();
   iface->queue                  = vk->ctx.context.queue;
   iface->queue_index            = vk->ctx.context.graphics_queue_index;
   iface->set_image              = host_vk_set_image;
   iface->get_sync_index         = host_vk_get_sync_index;
   iface->get_sync_index_mask    = host_vk_get_sync_index_mask;
   iface->set_command_buffers    = host_vk_set_command_buffers;
   iface->wait_sync_index        = host_vk_wait_sync_index;
   iface->lock_queue             = host_vk_lock_queue;
   iface->unlock_queue           = host_vk_unlock_queue;
   iface->set_signal_semaphore   = host_vk_set_signal_semaphore;

   RARCH_LOG("[Host] Headless Vulkan device, %ux%u, async readback.\n",
         width, height);
   return vk;

error:
   RARCH_ERR("[Host] No headless Vulkan device.\n");
   host_vk_free(vk);
   return NULL;
}

static bool host_vk_format(VkFormat format, bool *bgra)
{
   switch (format)
   {
      case VK_FORMAT_B8G8R8A8_UNORM:
      case VK_FORMAT_B8G8R8A8_SRGB:
         *bgra = true;
         return true;
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_R8G8B8A8_SRGB:
      case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
      case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
         *bgra = false;
         return true;
      default:
         break;
   }
   return false;
}

/* Records the copy of this frame's image and publishes the one before
 * it, whose copy has had a frame to land. */
static void host_vk_copy(host_vk_t *vk, unsigned cur, unsigned width,
      unsigned height)
{
   VkCommandBufferBeginInfo begin;
   VkImageMemoryBarrier barrier;
   VkBufferMemoryBarrier buf_barrier;
   VkBufferImageCopy region;
   VkSubmitInfo submit;
   VkCommandBuffer cmd = vk->cmd[cur];
   uint32_t family     = vk->ctx.context.graphics_queue_index;
   const VkImageSubresourceRange *range =
      &vk->image->create_info.subresourceRange;

   vkResetCommandBuffer(cmd, 0);
   memset(&begin, 0, sizeof(begin));
   begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
   begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   vkBeginCommandBuffer(cmd, &begin);

   memset(&barrier, 0, sizeof(barrier));
   barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
   barrier.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
   barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
   barrier.oldLayout           = vk->image->image_layout;
   barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   if (     vk->src_queue_family != VK_QUEUE_FAMILY_IGNORED
         && vk->src_queue_family != family)
   {
      barrier.srcQueueFamilyIndex = vk->src_queue_family;
      barrier.dstQueueFamilyIndex = family;
   }
   barrier.image                           = vk->image->create_info.image;
   barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
   barrier.subresourceRange.baseMipLevel   = range->baseMipLevel;
   barrier.subresourceRange.levelCount     = 1;
   barrier.subresourceRange.baseArrayLayer = range->baseArrayLayer;
   barrier.subresourceRange.layerCount     = 1;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
         1, &barrier);

   memset(&region, 0, sizeof(region));
   region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
   region.imageSubresource.mipLevel       = range->baseMipLevel;
   region.imageSubresource.baseArrayLayer = range->baseArrayLayer;
   region.imageSubresource.layerCount     = 1;
   region.imageExtent.width               = width;
   region.imageExtent.height              = height;
   region.imageExtent.depth               = 1;
   vkCmdCopyImageToBuffer(cmd, barrier.image,
         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk->buf[cur], 1, &region);

   /* back to where the core left it */
   barrier.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
   barrier.dstAccessMask       = 0;
   barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   barrier.newLayout           = vk->image->image_layout;
   barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

   memset(&buf_barrier, 0, sizeof(buf_barrier));
   buf_barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
   buf_barrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
   buf_barrier.dstAccessMask       = VK_ACCESS_HOST_READ_BIT;
   buf_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   buf_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   buf_barrier.buffer              = vk->buf[cur];
   buf_barrier.size                = VK_WHOLE_SIZE;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
         0, 0, NULL, 1, &buf_barrier, 1, &barrier);
   vkEndCommandBuffer(cmd);

   memset(&submit, 0, sizeof(submit));
   submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
   submit.waitSemaphoreCount = vk->num_wait;
   submit.pWaitSemaphores    = vk->wait;
   submit.pWaitDstStageMask  = vk->wait_stages;
   if (vk->num_cmd && vk->num_cmd < vk->cap_cmd)
   {
      vk->cmds[vk->num_cmd]     = cmd;
      submit.commandBufferCount = vk->num_cmd + 1;
      submit.pCommandBuffers    = vk->cmds;
   }
   else
   {
      submit.commandBufferCount = 1;
      submit.pCommandBuffers    = &cmd;
   }
   if (vk->signal != VK_NULL_HANDLE)
   {
      submit.signalSemaphoreCount = 1;
      submit.pSignalSemaphores    = &vk->signal;
   }

   host_vk_lock_queue(vk);
   if (vkQueueSubmit(vk->ctx.context.queue, 1, &submit, vk->fence[cur])
         == VK_SUCCESS)
   {
      vk->pending[cur]    = true;
      vk->buf_width[cur]  = width;
      vk->buf_height[cur] = height;
      vk->buf_format[cur] = vk->image->create_info.format;
   }
   host_vk_unlock_queue(vk);
}

static bool host_vk_read(host_video_t *host, unsigned width,
      unsigned height)
{
   bool bgra;
   host_vk_t *vk   = host->vk;
   VkDevice device = vk->ctx.context.device;
   unsigned cur    = vk->index;
   unsigned prev   = cur ^ 1;
   bool stored     = false;

   if (width > vk->width)
      width = vk->width;
   if (height > vk->height)
      height = vk->height;

   if (vk->image && host_vk_format(vk->image->create_info.format, &bgra))
      host_vk_copy(vk, cur, width, height);
   /* the core hands these over again for every frame */
   vk->num_wait = 0;
   vk->num_cmd  = 0;
   vk->signal   = VK_NULL_HANDLE;

   if (vk->pending[prev])
   {
      vkWaitForFences(device, 1, &vk->fence[prev], VK_TRUE, UINT64_MAX);
      vkResetFences(device, 1, &vk->fence[prev]);
      vk->pending[prev] = false;
      if (!vk->coherent)
      {
         VkMappedMemoryRange mapped;
         memset(&mapped, 0, sizeof(mapped));
         mapped.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
         mapped.memory = vk->mem[prev];
         mapped.size   = VK_WHOLE_SIZE;
         vkInvalidateMappedMemoryRanges(device, 1, &mapped);
      }
      if (host_vk_format(vk->buf_format[prev], &bgra))
         stored = host_picture_store_rgba32(&host->core, vk->map[prev],
               vk->buf_width[prev], vk->buf_height[prev], false, bgra);
   }
   vk->index = prev;
   return stored;
}

static bool host_vk_get_hw_render_interface(void *data,
      const struct retro_hw_render_interface **iface)
{
   host_video_t *host = (host_video_t*)data;
   if (!host->vk)
      return false;
   *iface = (const struct retro_hw_render_interface*)&host->vk->iface;
   return true;
}
#endif

static void *host_gfx_init(const video_info_t *video)
{
   host_video_t *host = (host_video_t*)calloc(1, sizeof(*host));
#ifdef HOST_HW
   struct retro_hw_render_callback *hwr = video_driver_get_hw_context();
#endif
   if (!host)
      return NULL;
   host->core_format = video->rgb32
      ? LIBRETROARCH_PIXEL_XRGB8888
      : LIBRETROARCH_PIXEL_RGB565;
#ifdef HOST_HW
   if (hwr && video_host_hw_context_supported(hwr->context_type))
   {
      const struct retro_game_geometry *geom =
         &video_state_get_ptr()->av_info.geometry;
      unsigned width  = geom->max_width  ? geom->max_width  : geom->base_width;
      unsigned height = geom->max_height ? geom->max_height : geom->base_height;
      bool ok         = false;
#ifdef HOST_HW_VK
      if (hwr->context_type == RETRO_HW_CONTEXT_VULKAN)
         ok = (host->vk = host_vk_init(width, height)) != NULL;
#endif
#ifdef HOST_HW_GL
      if (hwr->context_type != RETRO_HW_CONTEXT_VULKAN)
         ok = (host->gl = host_gl_init(hwr, width, height)) != NULL;
#endif
      if (!ok)
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

#ifdef HOST_HW
   if (frame == RETRO_HW_FRAME_BUFFER_VALID)
   {
      bool read = false;
#ifdef HOST_HW_GL
      if (host->gl)
         read = host_gl_read(host, width, height);
#endif
#ifdef HOST_HW_VK
      if (host->vk)
         read = host_vk_read(host, width, height);
#endif
      if (read)
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
#ifdef HOST_HW_VK
   if (host->vk)
      host_vk_free(host->vk);
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
#ifdef HOST_HW_VK
   host_vk_get_hw_render_interface,
#else
   NULL, /* get_hw_render_interface */
#endif
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
