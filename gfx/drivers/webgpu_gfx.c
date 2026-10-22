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

/* WebGPU through the standard webgpu.h: wgpu-native or Dawn natively,
 * emdawnwebgpu on the web. The frame is uploaded to a texture and drawn
 * in the viewport, the menu's texture blended over it. */

#include <stdlib.h>
#include <string.h>

#include <webgpu/webgpu.h>

#ifdef HAVE_CONFIG_H
#include "../../config.h"
#endif

#ifdef HAVE_MENU
#include "../../menu/menu_driver.h"
#endif

#ifdef __EMSCRIPTEN__
#include "../../frontend/drivers/platform_emscripten.h"
#else
#ifdef _WIN32
#include "../common/win32_common.h"
#include "../../input/input_driver.h"
#endif

#ifdef HAVE_X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include "../common/x11_common.h"
#endif
#include "../../frontend/frontend_driver.h"
#ifdef HAVE_WAYLAND
#include "../common/wayland_common.h"
#include "../common/wayland_resize.h"
#include "../../input/common/wayland_common.h"
#include "../../input/input_driver.h"
#endif
#endif

#ifdef HAVE_WEBGPU_SLANG
#include <compat/strl.h>
#include <string/stdstring.h>
#include "../drivers_shader/shader_webgpu.h"
#endif

#include "../video_driver.h"
#include "../../configuration.h"
#include "../../verbosity.h"

typedef struct
{
   WGPUTexture tex;
   WGPUTextureView view;
   WGPUBindGroup group;
   unsigned dims;
} webgpu_image_t;

typedef struct
{
   WGPUInstance instance;
   WGPUSurface surface;
   WGPUAdapter adapter;
   WGPUDevice device;
   WGPUQueue queue;
   WGPURenderPipeline core_pipe;
   WGPURenderPipeline menu_pipe;
   WGPUSampler samplers[2];
   WGPUBuffer menu_alpha;
   WGPUTextureFormat format;
   WGPUPresentMode present_modes[2];

   webgpu_image_t core;
   webgpu_image_t menu;
#ifdef HAVE_WEBGPU_SLANG
   webgpu_filter_chain_t *chain;
   /* a preset set before the device came */
   char *shader_path;
#endif

   video_viewport_t vp;

#ifdef HAVE_WAYLAND
   /* The Wayland window, NULL while the X11 one is used */
   gfx_ctx_wayland_data_t *wl;
#endif

   /* RGB565 frames as XRGB8888, menu textures as RGBA8 */
   uint32_t *convert;
   size_t convert_size;
   uint32_t *menu_frame;
   size_t menu_frame_size;
   unsigned menu_dims;
   float menu_alpha_value;

   unsigned surface_dims;
   bool ready;
   bool failed;
   bool configured;
   bool rgb32;
   bool smooth;
   bool core_smooth;
   bool vsync;
   bool fullscreen;
   bool menu_enable;
   bool menu_dirty;
} webgpu_t;

static const char webgpu_shader[] =
   "struct Out { @builtin(position) pos : vec4f, @location(0) uv : vec2f };\n"
   "@vertex fn vs(@builtin(vertex_index) i : u32) -> Out {\n"
   "   var o : Out;\n"
   "   let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));\n"
   "   o.pos = vec4f(p * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);\n"
   "   o.uv  = p;\n"
   "   return o;\n"
   "}\n"
   "@group(0) @binding(0) var smp : sampler;\n"
   "@group(0) @binding(1) var tex : texture_2d<f32>;\n"
   "@group(0) @binding(2) var<uniform> alpha : vec4f;\n"
   "@fragment fn fs_core(i : Out) -> @location(0) vec4f {\n"
   "   return vec4f(textureSample(tex, smp, i.uv).rgb, 1.0);\n"
   "}\n"
   "@fragment fn fs_menu(i : Out) -> @location(0) vec4f {\n"
   "   let c = textureSample(tex, smp, i.uv);\n"
   "   return vec4f(c.rgb, c.a * alpha.x);\n"
   "}\n";

static WGPUStringView webgpu_str(const char *s)
{
   WGPUStringView v;
   v.data   = s;
   v.length = WGPU_STRLEN;
   return v;
}

static void webgpu_image_free(webgpu_image_t *img)
{
   if (img->group)
      wgpuBindGroupRelease(img->group);
   if (img->view)
      wgpuTextureViewRelease(img->view);
   if (img->tex)
   {
      wgpuTextureDestroy(img->tex);
      wgpuTextureRelease(img->tex);
   }
   memset(img, 0, sizeof(*img));
}

/* The bind group follows the texture and the filter */
static bool webgpu_image_group(webgpu_t *wg, webgpu_image_t *img,
      WGPURenderPipeline pipe, bool smooth, bool with_alpha)
{
   WGPUBindGroupEntry entries[3];
   WGPUBindGroupDescriptor desc;
   WGPUBindGroupLayout layout;

   if (img->group)
      wgpuBindGroupRelease(img->group);
   img->group = NULL;
   if (!img->view)
      return false;

   memset(entries, 0, sizeof(entries));
   entries[0].binding     = 0;
   entries[0].sampler     = wg->samplers[smooth ? 1 : 0];
   entries[1].binding     = 1;
   entries[1].textureView = img->view;
   entries[2].binding     = 2;
   entries[2].buffer      = wg->menu_alpha;
   entries[2].size        = 16;

   layout = wgpuRenderPipelineGetBindGroupLayout(pipe, 0);
   memset(&desc, 0, sizeof(desc));
   desc.layout     = layout;
   desc.entryCount = with_alpha ? 3 : 2;
   desc.entries    = entries;
   img->group      = wgpuDeviceCreateBindGroup(wg->device, &desc);
   wgpuBindGroupLayoutRelease(layout);
   return img->group != NULL;
}

static bool webgpu_image_make(webgpu_t *wg, webgpu_image_t *img,
      unsigned dims, WGPUTextureFormat format)
{
   WGPUTextureDescriptor desc;

   webgpu_image_free(img);
   memset(&desc, 0, sizeof(desc));
   desc.usage                   = WGPUTextureUsage_TextureBinding
      | WGPUTextureUsage_CopyDst;
   desc.dimension               = WGPUTextureDimension_2D;
   desc.size.width              = VIDEO_SCALE_W(dims);
   desc.size.height             = VIDEO_SCALE_H(dims);
   desc.size.depthOrArrayLayers = 1;
   desc.format                  = format;
   desc.mipLevelCount           = 1;
   desc.sampleCount             = 1;
   if (!(img->tex = wgpuDeviceCreateTexture(wg->device, &desc)))
      return false;
   img->view = wgpuTextureCreateView(img->tex, NULL);
   img->dims = dims;
   return img->view != NULL;
}

static void webgpu_image_write(webgpu_t *wg, webgpu_image_t *img,
      const void *data, unsigned pitch)
{
   WGPUTexelCopyTextureInfo dst;
   WGPUTexelCopyBufferLayout layout;
   WGPUExtent3D size;
   unsigned width  = VIDEO_SCALE_W(img->dims);
   unsigned height = VIDEO_SCALE_H(img->dims);

   memset(&dst, 0, sizeof(dst));
   dst.texture             = img->tex;
   dst.aspect              = WGPUTextureAspect_All;
   layout.offset           = 0;
   layout.bytesPerRow      = pitch;
   layout.rowsPerImage     = height;
   size.width              = width;
   size.height             = height;
   size.depthOrArrayLayers = 1;
   wgpuQueueWriteTexture(wg->queue, &dst, data,
         (size_t)pitch * (height - 1) + width * 4, &layout, &size);
}

static WGPURenderPipeline webgpu_pipeline(webgpu_t *wg,
      WGPUShaderModule module, const char *entry, bool blend)
{
   WGPURenderPipelineDescriptor desc;
   WGPUFragmentState fragment;
   WGPUColorTargetState target;
   WGPUBlendState state;

   memset(&state, 0, sizeof(state));
   state.color.operation = WGPUBlendOperation_Add;
   state.color.srcFactor = WGPUBlendFactor_SrcAlpha;
   state.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
   state.alpha.operation = WGPUBlendOperation_Add;
   state.alpha.srcFactor = WGPUBlendFactor_One;
   state.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;

   memset(&target, 0, sizeof(target));
   target.format    = wg->format;
   target.blend     = blend ? &state : NULL;
   target.writeMask = WGPUColorWriteMask_All;

   memset(&fragment, 0, sizeof(fragment));
   fragment.module      = module;
   fragment.entryPoint  = webgpu_str(entry);
   fragment.targetCount = 1;
   fragment.targets     = &target;

   memset(&desc, 0, sizeof(desc));
   desc.vertex.module      = module;
   desc.vertex.entryPoint  = webgpu_str("vs");
   desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
   desc.multisample.count  = 1;
   desc.multisample.mask   = 0xFFFFFFFF;
   desc.fragment           = &fragment;
   return wgpuDeviceCreateRenderPipeline(wg->device, &desc);
}

static void webgpu_configure(webgpu_t *wg, unsigned dims)
{
   WGPUSurfaceConfiguration cfg;

   if (!VIDEO_SCALE_W(dims) || !VIDEO_SCALE_H(dims))
      return;
   memset(&cfg, 0, sizeof(cfg));
   cfg.device      = wg->device;
   cfg.format      = wg->format;
   cfg.usage       = WGPUTextureUsage_RenderAttachment;
   cfg.width       = VIDEO_SCALE_W(dims);
   cfg.height      = VIDEO_SCALE_H(dims);
   cfg.alphaMode   = WGPUCompositeAlphaMode_Auto;
   cfg.presentMode = wg->present_modes[wg->vsync ? 0 : 1];
   wgpuSurfaceConfigure(wg->surface, &cfg);
   wg->surface_dims       = dims;
   wg->configured         = true;
#ifdef HAVE_WAYLAND
   if (wg->wl)
   {
      wg->wl->last_buffer_scale         = wg->wl->buffer_scale;
      wg->wl->last_fractional_scale_num = wg->wl->fractional_scale_num;
      wl_surface_resized(wg->wl->surface, wg->wl->fractional_scale != NULL,
            wg->wl->buffer_scale, &wg->wl->ignore_configuration);
   }
#endif
   wg->vp.full_dims       = dims;
   video_driver_set_output_dims(dims);
}

static void webgpu_on_error(WGPUDevice const *device, WGPUErrorType type,
      WGPUStringView message, void *userdata1, void *userdata2)
{
   RARCH_ERR("[WebGPU] %.*s\n", (int)(message.length == WGPU_STRLEN
            ? strlen(message.data) : message.length), message.data);
}

/* Everything that needs the device, once it came */
static void webgpu_setup(webgpu_t *wg)
{
   unsigned i;
   WGPUSurfaceCapabilities caps;
   WGPUShaderSourceWGSL wgsl;
   WGPUShaderModuleDescriptor module_desc;
   WGPUShaderModule module;
   WGPUSamplerDescriptor sampler;
   WGPUBufferDescriptor buffer;
   bool immediate = false;
   bool mailbox   = false;

   wg->queue = wgpuDeviceGetQueue(wg->device);

   memset(&caps, 0, sizeof(caps));
   wg->format = WGPUTextureFormat_BGRA8Unorm;
   if (wgpuSurfaceGetCapabilities(wg->surface, wg->adapter, &caps)
         == WGPUStatus_Success)
   {
      bool found = false;
      for (i = 0; i < caps.formatCount; i++)
         if (caps.formats[i] == WGPUTextureFormat_BGRA8Unorm
               || caps.formats[i] == WGPUTextureFormat_RGBA8Unorm)
         {
            wg->format = caps.formats[i];
            found      = true;
            break;
         }
      if (!found && caps.formatCount)
         wg->format = caps.formats[0];
      for (i = 0; i < caps.presentModeCount; i++)
      {
         if (caps.presentModes[i] == WGPUPresentMode_Immediate)
            immediate = true;
         else if (caps.presentModes[i] == WGPUPresentMode_Mailbox)
            mailbox = true;
      }
      wgpuSurfaceCapabilitiesFreeMembers(caps);
   }
   wg->present_modes[0] = WGPUPresentMode_Fifo;
   wg->present_modes[1] = immediate ? WGPUPresentMode_Immediate
      : mailbox ? WGPUPresentMode_Mailbox : WGPUPresentMode_Fifo;

   memset(&wgsl, 0, sizeof(wgsl));
   wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
   wgsl.code        = webgpu_str(webgpu_shader);
   memset(&module_desc, 0, sizeof(module_desc));
   module_desc.nextInChain = &wgsl.chain;
   module = wgpuDeviceCreateShaderModule(wg->device, &module_desc);
   wg->core_pipe = webgpu_pipeline(wg, module, "fs_core", false);
   wg->menu_pipe = webgpu_pipeline(wg, module, "fs_menu", true);
   wgpuShaderModuleRelease(module);

   for (i = 0; i < 2; i++)
   {
      memset(&sampler, 0, sizeof(sampler));
      sampler.addressModeU  = WGPUAddressMode_ClampToEdge;
      sampler.addressModeV  = WGPUAddressMode_ClampToEdge;
      sampler.addressModeW  = WGPUAddressMode_ClampToEdge;
      sampler.magFilter     = i ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
      sampler.minFilter     = sampler.magFilter;
      sampler.mipmapFilter  = WGPUMipmapFilterMode_Nearest;
      sampler.lodMaxClamp   = 32.0f;
      sampler.maxAnisotropy = 1;
      wg->samplers[i] = wgpuDeviceCreateSampler(wg->device, &sampler);
   }

   memset(&buffer, 0, sizeof(buffer));
   buffer.usage   = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
   buffer.size    = 16;
   wg->menu_alpha = wgpuDeviceCreateBuffer(wg->device, &buffer);

   if (!wg->core_pipe || !wg->menu_pipe || !wg->menu_alpha
         || !wg->samplers[0] || !wg->samplers[1])
   {
      wg->failed = true;
      return;
   }
   wg->ready = true;
#ifdef HAVE_WEBGPU_SLANG
   if (wg->shader_path)
   {
      wg->chain = webgpu_filter_chain_new(wg->instance, wg->device,
            wg->queue, wg->format, wg->shader_path);
      free(wg->shader_path);
      wg->shader_path = NULL;
   }
#endif
}

static void webgpu_on_device(WGPURequestDeviceStatus status,
      WGPUDevice device, WGPUStringView message,
      void *userdata1, void *userdata2)
{
   webgpu_t *wg = (webgpu_t*)userdata1;
   if (status != WGPURequestDeviceStatus_Success)
   {
      RARCH_ERR("[WebGPU] No device.\n");
      wg->failed = true;
      return;
   }
   wg->device = device;
   webgpu_setup(wg);
}

static void webgpu_on_adapter(WGPURequestAdapterStatus status,
      WGPUAdapter adapter, WGPUStringView message,
      void *userdata1, void *userdata2)
{
   WGPUDeviceDescriptor desc;
   WGPURequestDeviceCallbackInfo cb;
   webgpu_t *wg = (webgpu_t*)userdata1;

   if (status != WGPURequestAdapterStatus_Success)
   {
      RARCH_ERR("[WebGPU] No adapter.\n");
      wg->failed = true;
      return;
   }
   wg->adapter = adapter;

   memset(&desc, 0, sizeof(desc));
   desc.uncapturedErrorCallbackInfo.callback = webgpu_on_error;
   memset(&cb, 0, sizeof(cb));
   cb.mode      = WGPUCallbackMode_AllowProcessEvents;
   cb.callback  = webgpu_on_device;
   cb.userdata1 = wg;
   wgpuAdapterRequestDevice(adapter, &desc, cb);
}

static void webgpu_free(void *data);

#if defined(HAVE_WAYLAND) && !defined(__EMSCRIPTEN__)
static bool webgpu_window_wayland(webgpu_t *wg, const video_info_t *video,
      WGPUSurfaceDescriptor *desc, unsigned *dims)
{
   static WGPUSurfaceSourceWaylandSurface wayland;
   gfx_ctx_wayland_data_t *wl = NULL;
   unsigned width;
   unsigned height;

   if (!gfx_ctx_wl_init_common(NULL, &wl))
   {
      RARCH_ERR("[WebGPU] Cannot connect to the Wayland compositor.\n");
      if (wl)
         gfx_ctx_wl_free_common(wl, false);
      return false;
   }
   wg->wl = wl;
   wl->max_swapchain_images = config_get_ptr()->uints.video_max_swapchain_images;

   /* Fullscreen at the desktop's size is asked for as 0x0 */
   *dims  = video->dims;
   width  = VIDEO_SCALE_W(*dims);
   height = VIDEO_SCALE_H(*dims);
   if (!gfx_ctx_wl_set_video_mode_common_size(wl, width, height,
            video->fullscreen))
      goto error;

   if (!wl->fractional_scale &&
       wl_compositor_get_version(wl->compositor) >=
       WL_SURFACE_SET_BUFFER_SCALE_SINCE_VERSION)
      wl_surface_set_buffer_scale(wl->surface, wl->buffer_scale);

   if (!gfx_ctx_wl_set_video_mode_common_fullscreen(wl, video->fullscreen))
      goto error;

   /* The buffer is sized in pixels, which the scale and a fullscreen
    * configure may have changed from what was asked for */
   if (video->fullscreen)
      gfx_ctx_wl_get_video_size_common(wl, dims);
   if (!video->fullscreen || !VIDEO_SCALE_W(*dims) || !VIDEO_SCALE_H(*dims))
      *dims = wl->buffer_dims;

   wl->input.gfx = wl;
   input_driver_video_window(INPUT_WINDOW_WAYLAND, &wl->input);
   frontend_driver_install_signal_handler();

   memset(&wayland, 0, sizeof(wayland));
   wayland.chain.sType = WGPUSType_SurfaceSourceWaylandSurface;
   wayland.display     = wl->input.dpy;
   wayland.surface     = wl->surface;
   desc->nextInChain   = &wayland.chain;
   return true;

error:
   gfx_ctx_wl_free_common(wl, false);
   wg->wl = NULL;
   return false;
}
#endif

static bool webgpu_window(webgpu_t *wg, const video_info_t *video,
      WGPUSurfaceDescriptor *desc, unsigned *dims)
{
#ifdef __EMSCRIPTEN__
   static WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas;
   memset(&canvas, 0, sizeof(canvas));
   canvas.chain.sType  = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
   canvas.selector     = webgpu_str("#canvas");
   desc->nextInChain   = &canvas.chain;
   *dims               = platform_emscripten_get_canvas_dims();
   input_driver_video_window(INPUT_WINDOW_PLATFORM, NULL);
   return true;
#elif defined(_WIN32)
   static WGPUSurfaceSourceWindowsHWND win32;
   WNDCLASSEX wndclass = {0};
   unsigned win_dims;
   HWND hwnd;

   win32_window_reset();
   win32_monitor_init();
   /* The same window procedure as the other Windows video drivers */
   wndclass.lpfnWndProc = win32_window_proc;
   win32_window_proc_setup(WIN32_WINDOW_D3D);
   if (!win32_window_init(&wndclass, true, NULL))
      return false;

   win_dims = video->dims;
   /* Fullscreen at the desktop's size is asked for as 0x0 */
   if (video->fullscreen
         && (!VIDEO_SCALE_W(win_dims) || !VIDEO_SCALE_H(win_dims)))
   {
      MONITORINFOEX mon;
      HMONITOR hm = NULL;
      unsigned mon_id = 0;

      win32_monitor_info(&mon, &hm, &mon_id);
      win_dims = VIDEO_SCALE_PACK(
            mon.rcMonitor.right - mon.rcMonitor.left,
            mon.rcMonitor.bottom - mon.rcMonitor.top);
   }
   if (!win32_set_video_mode(NULL, win_dims, video->fullscreen))
   {
      win32_destroy_window();
      return false;
   }
   if (!(hwnd = win32_get_window()))
      return false;

   *dims = VIDEO_SCALE_PACK(g_win32_resize_width, g_win32_resize_height);
   if (!VIDEO_SCALE_W(*dims) || !VIDEO_SCALE_H(*dims))
      *dims = win_dims;
   input_driver_video_window(INPUT_WINDOW_WINDOWS, NULL);

   memset(&win32, 0, sizeof(win32));
   win32.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
   win32.hinstance   = GetModuleHandle(NULL);
   win32.hwnd        = hwnd;
   desc->nextInChain = &win32.chain;
   return true;
#else
#ifdef HAVE_X11
   static WGPUSurfaceSourceXlibWindow xlib;
   char title[128];
   XEvent event;
   XSetWindowAttributes attributes;
#endif

#ifdef HAVE_WAYLAND
   {
      /* A Wayland session is served natively, whatever X it also has */
      const char *wl_display_name = getenv("WAYLAND_DISPLAY");
      if (wl_display_name && *wl_display_name)
      {
         if (webgpu_window_wayland(wg, video, desc, dims))
            return true;
         RARCH_WARN("[WebGPU] No Wayland window, trying X11.\n");
      }
   }
#endif

#ifdef HAVE_X11
   XInitThreads();
   if (!(g_x11_dpy = XOpenDisplay(NULL)))
   {
      RARCH_ERR("[WebGPU] Cannot connect to the X server.\n");
      return false;
   }

   /* Fullscreen at the desktop's size is asked for as 0x0 */
   *dims = video->dims;
   if (!VIDEO_SCALE_W(*dims) || !VIDEO_SCALE_H(*dims))
      *dims = VIDEO_SCALE_PACK(
            DisplayWidth(g_x11_dpy,  DefaultScreen(g_x11_dpy)),
            DisplayHeight(g_x11_dpy, DefaultScreen(g_x11_dpy)));

   memset(&attributes, 0, sizeof(attributes));
   attributes.border_pixel = 0;
   attributes.event_mask   = StructureNotifyMask | KeyPressMask
      | KeyReleaseMask | ButtonReleaseMask | ButtonPressMask
      | ExposureMask | FocusChangeMask | EnterWindowMask | LeaveWindowMask;
   g_x11_win = XCreateWindow(g_x11_dpy, DefaultRootWindow(g_x11_dpy),
         0, 0, VIDEO_SCALE_W(*dims), VIDEO_SCALE_H(*dims), 0,
         CopyFromParent, InputOutput, CopyFromParent,
         CWBorderPixel | CWEventMask, &attributes);
   XSetWindowBackground(g_x11_dpy, g_x11_win, 0);
   if (video->fullscreen)
      x11_set_net_wm_fullscreen_hint(g_x11_dpy, g_x11_win);
   XMapWindow(g_x11_dpy, g_x11_win);

   title[0] = '\0';
   video_driver_get_window_title(title, sizeof(title));
   if (title[0])
      XStoreName(g_x11_dpy, g_x11_win, title);
   x11_set_window_attr(g_x11_dpy, g_x11_win);
   if (video->fullscreen)
   {
      x11_event_queue_check(&event);
      x11_set_net_wm_fullscreen(g_x11_dpy, g_x11_win);
      x11_show_mouse(wg, false);
   }
   XSync(g_x11_dpy, False);

   x11_install_quit_atom();
   frontend_driver_install_signal_handler();
   if (!x11_input_ctx_new(true))
      return false;
   input_driver_video_window(INPUT_WINDOW_X11, NULL);

   memset(&xlib, 0, sizeof(xlib));
   xlib.chain.sType  = WGPUSType_SurfaceSourceXlibWindow;
   xlib.display      = g_x11_dpy;
   xlib.window       = (uint64_t)g_x11_win;
   desc->nextInChain = &xlib.chain;
   return true;
#else
   RARCH_ERR("[WebGPU] No window system.\n");
   return false;
#endif
#endif
}

static void *webgpu_init(const video_info_t *video)
{
   unsigned dims;
   WGPUSurfaceDescriptor surface;
   WGPURequestAdapterOptions options;
   WGPURequestAdapterCallbackInfo cb;
   webgpu_t *wg = (webgpu_t*)calloc(1, sizeof(webgpu_t));

   if (!wg)
      return NULL;
   wg->rgb32      = video->rgb32;
   wg->smooth     = video->smooth;
   wg->vsync      = video->vsync;
   wg->fullscreen = video->fullscreen;

   if (!(wg->instance = wgpuCreateInstance(NULL)))
   {
      RARCH_ERR("[WebGPU] No instance.\n");
      goto error;
   }

   memset(&surface, 0, sizeof(surface));
   if (!webgpu_window(wg, video, &surface, &dims))
      goto error;
   if (!(wg->surface = wgpuInstanceCreateSurface(wg->instance, &surface)))
   {
      RARCH_ERR("[WebGPU] No surface.\n");
      goto error;
   }
   wg->surface_dims = dims;

   memset(&options, 0, sizeof(options));
   options.featureLevel      = WGPUFeatureLevel_Core;
   options.powerPreference   = WGPUPowerPreference_HighPerformance;
   options.compatibleSurface = wg->surface;
   memset(&cb, 0, sizeof(cb));
   cb.mode      = WGPUCallbackMode_AllowProcessEvents;
   cb.callback  = webgpu_on_adapter;
   cb.userdata1 = wg;
   wgpuInstanceRequestAdapter(wg->instance, &options, cb);

#ifndef __EMSCRIPTEN__
   {
      /* Natively the device comes within a few polls; on the web it
       * comes once the browser runs again, and frames wait for it */
      unsigned polls;
      for (polls = 0; polls < 10000 && !wg->ready && !wg->failed; polls++)
         wgpuInstanceProcessEvents(wg->instance);
      if (!wg->ready)
         goto error;
   }
#endif

   RARCH_LOG("[WebGPU] Up, %ux%u.\n", VIDEO_SCALE_W(dims),
         VIDEO_SCALE_H(dims));
   return wg;

error:
   webgpu_free(wg);
   return NULL;
}

static void webgpu_upload_core(webgpu_t *wg, const void *frame,
      unsigned dims, unsigned pitch)
{
   unsigned width  = VIDEO_SCALE_W(dims);
   unsigned height = VIDEO_SCALE_H(dims);

   if (wg->core.dims != dims || !wg->core.tex)
   {
      if (!webgpu_image_make(wg, &wg->core, dims,
               WGPUTextureFormat_BGRA8Unorm))
         return;
   }
   if (!wg->core.group || wg->core_smooth != wg->smooth)
   {
      webgpu_image_group(wg, &wg->core, wg->core_pipe, wg->smooth, false);
      wg->core_smooth = wg->smooth;
   }

   /* XRGB8888 is bgra8unorm already; RGB565 is widened on the CPU */
   if (wg->rgb32)
      webgpu_image_write(wg, &wg->core, frame, pitch);
   else
   {
      unsigned x, y;
      size_t size = (size_t)width * height * sizeof(uint32_t);
      if (size > wg->convert_size)
      {
         uint32_t *grown = (uint32_t*)realloc(wg->convert, size);
         if (!grown)
            return;
         wg->convert      = grown;
         wg->convert_size = size;
      }
      for (y = 0; y < height; y++)
      {
         const uint16_t *src = (const uint16_t*)
            ((const uint8_t*)frame + (size_t)y * pitch);
         uint32_t *dst       = wg->convert + (size_t)y * width;
         for (x = 0; x < width; x++)
         {
            uint32_t p = src[x];
            uint32_t r = (p >> 11) & 0x1f;
            uint32_t g = (p >>  5) & 0x3f;
            uint32_t b =  p        & 0x1f;
            dst[x]     = 0xff000000u
               | (((r << 3) | (r >> 2)) << 16)
               | (((g << 2) | (g >> 4)) <<  8)
               |  ((b << 3) | (b >> 2));
         }
      }
      webgpu_image_write(wg, &wg->core, wg->convert, width * 4);
   }
}

static void webgpu_upload_menu(webgpu_t *wg)
{
   float alpha[4];

   if (!wg->menu_dirty || !wg->menu_frame)
      return;
   wg->menu_dirty = false;
   if (wg->menu.dims != wg->menu_dims || !wg->menu.tex)
   {
      if (!webgpu_image_make(wg, &wg->menu, wg->menu_dims,
               WGPUTextureFormat_RGBA8Unorm))
         return;
      webgpu_image_group(wg, &wg->menu, wg->menu_pipe, true, true);
   }
   webgpu_image_write(wg, &wg->menu, wg->menu_frame,
         VIDEO_SCALE_W(wg->menu_dims) * 4);
   alpha[0] = wg->menu_alpha_value;
   alpha[1] = alpha[2] = alpha[3] = 0.0f;
   wgpuQueueWriteBuffer(wg->queue, wg->menu_alpha, 0, alpha, sizeof(alpha));
}

/* @vp clipped to the surface, which the viewport may not leave */
static bool webgpu_viewport(WGPURenderPassEncoder pass,
      const video_viewport_t *vp, unsigned surface)
{
   int x  = VIDEO_POS_X(vp->pos);
   int y  = VIDEO_POS_Y(vp->pos);
   int w  = (int)VIDEO_SCALE_W(vp->dims);
   int h  = (int)VIDEO_SCALE_H(vp->dims);
   int sw = (int)VIDEO_SCALE_W(surface);
   int sh = (int)VIDEO_SCALE_H(surface);
   if (x < 0)
   {
      w += x;
      x  = 0;
   }
   if (y < 0)
   {
      h += y;
      y  = 0;
   }
   if (x + w > sw)
      w = sw - x;
   if (y + h > sh)
      h = sh - y;
   if (w <= 0 || h <= 0)
      return false;
   wgpuRenderPassEncoderSetViewport(pass, (float)x, (float)y,
         (float)w, (float)h, 0.0f, 1.0f);
   return true;
}

static bool webgpu_frame(void *data, const void *frame,
      unsigned dims, uint64_t frame_count,
      unsigned pitch, const char *msg, video_frame_info_t *video_info)
{
   WGPUSurfaceTexture current;
   WGPUTextureView target;
   WGPUCommandEncoder encoder;
   WGPURenderPassEncoder pass;
   WGPURenderPassDescriptor pass_desc;
   WGPURenderPassColorAttachment color;
   WGPUCommandBuffer commands;
   unsigned window;
   bool draw_core     = false;
   webgpu_t *wg       = (webgpu_t*)data;
#ifdef HAVE_MENU
   bool menu_is_alive = (video_info->menu_st_flags & MENU_ST_FLAG_ALIVE)
      ? true : false;
#endif

   if (!wg || wg->failed)
      return false;
   wgpuInstanceProcessEvents(wg->instance);

   window = wg->surface_dims;
#ifdef __EMSCRIPTEN__
   window = platform_emscripten_get_canvas_dims();
#elif defined(_WIN32)
   {
      bool quit   = false;
      bool resize = false;
      win32_check_window(NULL, &quit, &resize, &window);
      if (quit)
         return false;
   }
#else
   {
      bool quit   = false;
      bool resize = false;
#ifdef HAVE_WAYLAND
      if (wg->wl)
         gfx_ctx_wl_check_window_common(wg->wl,
               gfx_ctx_wl_get_video_size_common, &quit, &resize, &window);
#ifdef HAVE_X11
      else
#endif
#endif
#ifdef HAVE_X11
      x11_check_window(wg, &quit, &resize, &window);
#endif
      if (quit)
         return false;
   }
#endif
   /* On the web the device may still be on its way */
   if (!wg->ready)
      return true;
   if (!wg->configured || window != wg->surface_dims)
      webgpu_configure(wg, window);
   if (!wg->configured)
      return true;

   if (frame && VIDEO_SCALE_W(dims) && VIDEO_SCALE_H(dims))
   {
      webgpu_upload_core(wg, frame, dims, pitch);
      draw_core = wg->core.group != NULL;
   }
   video_driver_update_viewport(&wg->vp, false, true, true);

#ifdef HAVE_MENU
   menu_driver_frame(menu_is_alive, video_info);
#endif
   if (wg->menu_enable)
      webgpu_upload_menu(wg);

   wgpuSurfaceGetCurrentTexture(wg->surface, &current);
   switch (current.status)
   {
      case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
      case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
         break;
      case WGPUSurfaceGetCurrentTextureStatus_Timeout:
         if (current.texture)
            wgpuTextureRelease(current.texture);
         return true;
      default:
         /* Outdated or lost: configured again next frame */
         if (current.texture)
            wgpuTextureRelease(current.texture);
         wg->configured = false;
         return true;
   }
   target  = wgpuTextureCreateView(current.texture, NULL);
   encoder = wgpuDeviceCreateCommandEncoder(wg->device, NULL);
#ifdef HAVE_WEBGPU_SLANG
   if (draw_core && wg->chain)
      webgpu_filter_chain_record(wg->chain, encoder, wg->core.view,
            wg->core.dims, wg->smooth, wg->vp.dims, frame_count, 0);
#endif

   memset(&color, 0, sizeof(color));
   color.view       = target;
   color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
   color.loadOp     = WGPULoadOp_Clear;
   color.storeOp    = WGPUStoreOp_Store;
   color.clearValue.a = 1.0;
   memset(&pass_desc, 0, sizeof(pass_desc));
   pass_desc.colorAttachmentCount = 1;
   pass_desc.colorAttachments     = &color;

   pass    = wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
#ifdef HAVE_WEBGPU_SLANG
   if (     draw_core && wg->chain
         && webgpu_viewport(pass, &wg->vp, wg->surface_dims))
      webgpu_filter_chain_draw(wg->chain, pass);
   else
#endif
   if (draw_core && webgpu_viewport(pass, &wg->vp, wg->surface_dims))
   {
      wgpuRenderPassEncoderSetPipeline(pass, wg->core_pipe);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, wg->core.group, 0, NULL);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
   }
   if (     wg->menu_enable && wg->menu.group
         && webgpu_viewport(pass, &wg->vp, wg->surface_dims))
   {
      wgpuRenderPassEncoderSetPipeline(pass, wg->menu_pipe);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, wg->menu.group, 0, NULL);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
   }
   wgpuRenderPassEncoderEnd(pass);
   wgpuRenderPassEncoderRelease(pass);
   commands = wgpuCommandEncoderFinish(encoder, NULL);
   wgpuCommandEncoderRelease(encoder);
   wgpuQueueSubmit(wg->queue, 1, &commands);
   wgpuCommandBufferRelease(commands);
   /* The browser shows the canvas once control returns to it */
#ifndef __EMSCRIPTEN__
   wgpuSurfacePresent(wg->surface);
#endif
   wgpuTextureViewRelease(target);
   wgpuTextureRelease(current.texture);

#ifndef __EMSCRIPTEN__
#ifdef HAVE_WAYLAND
   if (wg->wl)
      gfx_ctx_wl_update_title_common(wg->wl);
#ifdef HAVE_X11
   else
#endif
#endif
#ifdef HAVE_X11
   x11_update_title(NULL);
#endif
#endif
   return true;
}

static void webgpu_set_nonblock_state(void *data, bool toggle,
      bool adaptive_vsync, unsigned swap_interval)
{
   webgpu_t *wg = (webgpu_t*)data;
   if (!wg || wg->vsync == !toggle)
      return;
   wg->vsync      = !toggle;
   wg->configured = false;
}

#ifdef __EMSCRIPTEN__
static bool webgpu_alive(void *data) { return true; }
static bool webgpu_focus(void *data)
{
   return !platform_emscripten_is_window_hidden();
}
static bool webgpu_suppress_screensaver(void *data, bool enable)
{
   platform_emscripten_set_wake_lock(enable);
   return true;
}
#endif

#ifdef _WIN32
static bool webgpu_alive(void *data)
{
   return !(g_win32_flags & WIN32_CMN_FLAG_QUIT);
}
static bool webgpu_focus(void *data) { return win32_has_focus(data); }
static bool webgpu_suppress_screensaver(void *data, bool enable)
{
   return win32_suspend_screensaver(data, enable);
}
#elif !defined(__EMSCRIPTEN__)
static bool webgpu_alive(void *data)
{
#ifdef HAVE_WAYLAND
   webgpu_t *wg = (webgpu_t*)data;
   /* The compositor's events are pumped by the frame's window check */
   if (wg && wg->wl)
      return !frontend_driver_get_signal_handler_state();
#endif
#ifdef HAVE_X11
   return x11_alive(data);
#else
   return true;
#endif
}

static bool webgpu_focus(void *data)
{
#ifdef HAVE_WAYLAND
   webgpu_t *wg = (webgpu_t*)data;
   if (wg && wg->wl)
      return gfx_ctx_wl_has_focus(wg->wl);
#endif
#ifdef HAVE_X11
   return x11_has_focus_internal(data);
#else
   return false;
#endif
}

static bool webgpu_suppress_screensaver(void *data, bool enable)
{
#ifdef HAVE_WAYLAND
   webgpu_t *wg = (webgpu_t*)data;
   if (wg && wg->wl)
      return gfx_ctx_wl_suppress_screensaver(wg->wl, enable);
#endif
#ifdef HAVE_X11
   return x11_suspend_screensaver(data, enable);
#else
   return false;
#endif
}
#endif

static bool webgpu_has_windowed(void *data) { return true; }

static bool webgpu_set_shader(void *data,
      enum rarch_shader_type type, const char *path)
{
#ifdef HAVE_WEBGPU_SLANG
   webgpu_t *wg = (webgpu_t*)data;
   if (!wg)
      return false;
   webgpu_filter_chain_free(wg->chain);
   wg->chain = NULL;
   free(wg->shader_path);
   wg->shader_path = NULL;
   if (string_is_empty(path))
      return true;
   if (type != RARCH_SHADER_SLANG)
   {
      RARCH_WARN("[WebGPU] Only slang presets run here.\n");
      return false;
   }
   if (!wg->ready)
   {
      wg->shader_path = strldup(path, strlen(path));
      return wg->shader_path != NULL;
   }
   wg->chain = webgpu_filter_chain_new(wg->instance, wg->device,
         wg->queue, wg->format, path);
   return wg->chain != NULL;
#else
   return false;
#endif
}

static void webgpu_free(void *data)
{
   unsigned i;
   webgpu_t *wg = (webgpu_t*)data;
   if (!wg)
      return;
#ifdef HAVE_WEBGPU_SLANG
   webgpu_filter_chain_free(wg->chain);
   free(wg->shader_path);
#endif
   webgpu_image_free(&wg->core);
   webgpu_image_free(&wg->menu);
   if (wg->menu_alpha)
   {
      wgpuBufferDestroy(wg->menu_alpha);
      wgpuBufferRelease(wg->menu_alpha);
   }
   for (i = 0; i < 2; i++)
      if (wg->samplers[i])
         wgpuSamplerRelease(wg->samplers[i]);
   if (wg->core_pipe)
      wgpuRenderPipelineRelease(wg->core_pipe);
   if (wg->menu_pipe)
      wgpuRenderPipelineRelease(wg->menu_pipe);
   if (wg->surface)
   {
      if (wg->configured)
         wgpuSurfaceUnconfigure(wg->surface);
      wgpuSurfaceRelease(wg->surface);
   }
   if (wg->queue)
      wgpuQueueRelease(wg->queue);
   if (wg->device)
      wgpuDeviceRelease(wg->device);
   if (wg->adapter)
      wgpuAdapterRelease(wg->adapter);
   if (wg->instance)
      wgpuInstanceRelease(wg->instance);
#ifdef _WIN32
   if (win32_get_window() && !win32_window_keep())
   {
      win32_monitor_from_window();
      win32_destroy_window();
   }
   g_win32_flags &= ~WIN32_CMN_FLAG_INITED;
#elif !defined(__EMSCRIPTEN__)
#ifdef HAVE_WAYLAND
   if (wg->wl)
   {
      gfx_ctx_wl_free_common(wg->wl, true);
      wg->wl = NULL;
   }
#endif
#ifdef HAVE_X11
   if (g_x11_dpy)
   {
      x11_input_ctx_destroy();
      x11_window_destroy(wg->fullscreen);
      XCloseDisplay(g_x11_dpy);
      g_x11_dpy = NULL;
   }
#endif
#endif
   free(wg->convert);
   free(wg->menu_frame);
   free(wg);
}

static void webgpu_poke_set_filtering(void *data, unsigned index,
      bool smooth, bool ctx_scaling)
{
   webgpu_t *wg = (webgpu_t*)data;
   if (wg)
      wg->smooth = smooth;
}

/* The viewport is laid out every frame */
static void webgpu_poke_set_aspect_ratio(void *data,
      unsigned aspect_ratio_idx) { }
static void webgpu_poke_apply_state_changes(void *data) { }

/* The menu's texture as RGBA8, uploaded with the next frame */
static void webgpu_poke_set_texture_frame(void *data,
      const void *frame, bool rgb32, unsigned dims, float alpha)
{
   size_t i, count;
   webgpu_t *wg = (webgpu_t*)data;
   count        = VIDEO_SCALE_AREA(dims);
   if (!wg || !frame || !count)
      return;
   if (count * 4 > wg->menu_frame_size)
   {
      uint32_t *grown = (uint32_t*)realloc(wg->menu_frame, count * 4);
      if (!grown)
         return;
      wg->menu_frame      = grown;
      wg->menu_frame_size = count * 4;
   }
   if (rgb32)
   {
      const uint32_t *src = (const uint32_t*)frame;
      for (i = 0; i < count; i++)
      {
         uint32_t p        = src[i];
         wg->menu_frame[i] = (p & 0xff00ff00u)
            | ((p >> 16) & 0xffu) | ((p & 0xffu) << 16);
      }
   }
   else
   {
      const uint16_t *src = (const uint16_t*)frame;
      for (i = 0; i < count; i++)
      {
         uint32_t p        = src[i];
         wg->menu_frame[i] = (((p >> 12) & 0xf) * 0x11)
            | ((((p >> 8) & 0xf) * 0x11) <<  8)
            | ((((p >> 4) & 0xf) * 0x11) << 16)
            | (( (p       & 0xf) * 0x11) << 24);
      }
   }
   wg->menu_dims        = dims;
   wg->menu_alpha_value = alpha;
   wg->menu_dirty       = true;
}

static void webgpu_poke_texture_enable(void *data,
      bool enable, bool full_screen)
{
   webgpu_t *wg = (webgpu_t*)data;
   if (wg)
      wg->menu_enable = enable;
}

static void webgpu_viewport_info(void *data, struct video_viewport *vp)
{
   webgpu_t *wg = (webgpu_t*)data;
   if (wg)
      *vp = wg->vp;
}

static void webgpu_show_mouse(void *data, bool state)
{
#ifdef __EMSCRIPTEN__
   platform_emscripten_set_pointer_visibility(state);
#elif defined(_WIN32)
   win32_show_cursor(data, state);
#else
#ifdef HAVE_WAYLAND
   webgpu_t *wg = (webgpu_t*)data;
   if (wg && wg->wl)
   {
      gfx_ctx_wl_show_mouse(wg->wl, state);
      return;
   }
#endif
#ifdef HAVE_X11
   x11_show_mouse(data, state);
#endif
#endif
}

static void webgpu_grab_mouse_toggle(void *data) { }

static uint32_t webgpu_get_flags(void *data)
{
   uint32_t flags = 0;
#ifdef HAVE_WEBGPU_SLANG
   BIT32_SET(flags, GFX_CTX_FLAGS_SHADERS_SLANG);
#endif
   return flags;
}

#ifdef HAVE_WEBGPU_SLANG
static struct video_shader *webgpu_get_current_shader(void *data)
{
   webgpu_t *wg = (webgpu_t*)data;
   return wg ? webgpu_filter_chain_get_preset(wg->chain) : NULL;
}
#endif

static video_poke_interface_t webgpu_poke_interface = {
   webgpu_get_flags,
   NULL, /* load_texture */
   NULL, /* unload_texture */
   NULL, /* set_video_mode */
   NULL, /* get_refresh_rate */
   webgpu_poke_set_filtering,
   NULL, /* get_video_output_size */
   NULL, /* get_video_output_prev */
   NULL, /* get_video_output_next */
   NULL, /* get_current_framebuffer */
   NULL, /* get_proc_address */
   webgpu_poke_set_aspect_ratio,
   webgpu_poke_apply_state_changes,
   webgpu_poke_set_texture_frame,
   webgpu_poke_texture_enable,
   NULL, /* set_osd_msg */
   webgpu_show_mouse,
   webgpu_grab_mouse_toggle,
#ifdef HAVE_WEBGPU_SLANG
   webgpu_get_current_shader,
#else
   NULL, /* get_current_shader */
#endif
   NULL, /* get_current_software_framebuffer */
   NULL, /* get_hw_render_interface */
   NULL, /* set_hdr_menu_nits */
   NULL, /* set_hdr_paper_white_nits */
   NULL, /* set_hdr_expand_gamut */
   NULL, /* set_hdr_scanlines */
   NULL  /* set_hdr_subpixel_layout */
};

static void webgpu_get_poke_interface(void *data,
      const video_poke_interface_t **iface)
{
   *iface = &webgpu_poke_interface;
}

video_driver_t video_webgpu = {
   webgpu_init,
   webgpu_frame,
   webgpu_set_nonblock_state,
   webgpu_alive,
   webgpu_focus,
   webgpu_suppress_screensaver,
   webgpu_has_windowed,
   webgpu_set_shader,
   webgpu_free,
   "webgpu",
   NULL, /* set_viewport */
   NULL, /* set_rotation */
   webgpu_viewport_info,
   NULL, /* read_viewport */
#ifdef HAVE_OVERLAY
   NULL, /* get_overlay_interface */
#endif
   webgpu_get_poke_interface,
   NULL, /* wrap_type_to_enum */
   NULL, /* shader_load_begin */
   NULL, /* shader_load_step */
#ifdef HAVE_GFX_WIDGETS
   NULL  /* gfx_widgets_enabled */
#endif
};
