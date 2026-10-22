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

#include <stdlib.h>
#include <string.h>

#include <spirv-tools/libspirv.h>

#include <compat/strl.h>
#include <formats/image.h>
#include <string/stdstring.h>

#ifdef HAVE_CONFIG_H
#include "../../config.h"
#endif

#include "shader_webgpu.h"
#include "glslang_util.h"
#include "slang_process.h"

#include "../video_driver.h"
#include "../../verbosity.h"
#ifdef HAVE_REWIND
#include "../../state_manager.h"
#endif

/* Each slang binding b becomes the texture at 2b and its sampler at
 * 2b + 1; the push constant block becomes a uniform buffer here */
#define WEBGPU_PUSH_BINDING (2 * SLANG_NUM_BINDINGS)
#define WEBGPU_MAX_ENTRIES  (2 * SLANG_NUM_BINDINGS + 1)

#define SPV_OP_TYPE_IMAGE           25
#define SPV_OP_TYPE_SAMPLER         26
#define SPV_OP_TYPE_ARRAY           28
#define SPV_OP_TYPE_RUNTIME_ARRAY   29
#define SPV_OP_TYPE_POINTER         32
#define SPV_OP_VARIABLE             59
#define SPV_OP_DECORATE             71
#define SPV_DECORATION_BINDING      33
#define SPV_DECORATION_SET          34
#define SPV_STORAGE_UNIFORM         2
#define SPV_STORAGE_PUSH_CONSTANT   9

typedef struct
{
   WGPUTexture tex;
   WGPUTextureView view;
   WGPUTextureFormat format;
   unsigned dims;
} webgpu_target_t;

typedef struct
{
   slang_reflection reflection;
   webgpu_target_t fb;
   webgpu_target_t feedback;
   WGPURenderPipeline pipe;
   WGPUBindGroupLayout layout;
   WGPUBuffer ubo;
   WGPUBuffer push;
   uint8_t *ubo_data;
   uint8_t *push_data;
   size_t ubo_size;
   size_t push_size;
   unsigned *params;
   unsigned num_params;
   uint32_t *vertex;
   uint32_t *fragment;
   size_t vertex_len;
   size_t fragment_len;
   unsigned dims;
   WGPUTextureFormat format;
   bool needs_feedback;
} webgpu_pass_t;

typedef struct
{
   webgpu_target_t tex;
   unsigned filter;
   enum gfx_wrap_type wrap;
} webgpu_lut_t;

struct webgpu_filter_chain
{
   WGPUInstance instance;
   WGPUDevice device;
   WGPUQueue queue;
   WGPUTextureFormat format;
   struct video_shader *shader;
   webgpu_pass_t *passes;
   unsigned num_passes;
   webgpu_lut_t luts[GFX_MAX_TEXTURES];
   unsigned num_luts;
   /* nearest, linear; edge, repeat, mirrored */
   WGPUSampler samplers[2][3];
   WGPUBuffer vbo;
   webgpu_target_t dummy;
   WGPUBindGroup final_group;
   WGPUTextureView original;
   unsigned original_dims;
   unsigned viewport_dims;
   bool smooth;
   slang_texture_semantic_name_map texture_map;
   slang_texture_semantic_name_map texture_uniform_map;
};

typedef struct
{
   WGPUTextureView view;
   WGPUTextureFormat format;
   unsigned dims;
   unsigned filter;
   enum gfx_wrap_type wrap;
} webgpu_input_t;

static WGPUStringView webgpu_chain_str(const char *s)
{
   WGPUStringView v;
   v.data   = s;
   v.length = WGPU_STRLEN;
   return v;
}

/* Push constants into a uniform block, and the bindings spread out
 * for the samplers split from their images */
static uint32_t *webgpu_spirv_rebind(const uint32_t *in, size_t len,
      size_t *out_len)
{
   size_t i, o;
   uint32_t bound;
   uint32_t push_var  = 0;
   bool inserted      = false;
   uint8_t *sampler   = NULL;
   uint32_t *pointee  = NULL;
   uint32_t *var_type = NULL;
   uint32_t *out      = NULL;

   if (len < 5)
      return NULL;
   bound    = in[3];
   sampler  = (uint8_t*)calloc(bound, 1);
   pointee  = (uint32_t*)calloc(bound, sizeof(uint32_t));
   var_type = (uint32_t*)calloc(bound, sizeof(uint32_t));
   out      = (uint32_t*)malloc((len + 8) * sizeof(uint32_t));
   if (!sampler || !pointee || !var_type || !out)
      goto error;

   for (i = 5; i < len; )
   {
      unsigned count  = in[i] >> 16;
      unsigned opcode = in[i] & 0xffff;
      if (!count || i + count > len)
         goto error;
      switch (opcode)
      {
         case SPV_OP_TYPE_SAMPLER:
            if (in[i + 1] < bound)
               sampler[in[i + 1]] = 1;
            break;
         case SPV_OP_TYPE_ARRAY:
         case SPV_OP_TYPE_RUNTIME_ARRAY:
            if (in[i + 1] < bound && in[i + 2] < bound)
               sampler[in[i + 1]] = sampler[in[i + 2]];
            break;
         case SPV_OP_TYPE_POINTER:
            if (in[i + 1] < bound)
               pointee[in[i + 1]] = in[i + 3];
            break;
         case SPV_OP_VARIABLE:
            if (in[i + 2] < bound)
               var_type[in[i + 2]] = in[i + 1];
            if (in[i + 3] == SPV_STORAGE_PUSH_CONSTANT)
               push_var = in[i + 2];
            break;
         default:
            break;
      }
      i += count;
   }

   memcpy(out, in, 5 * sizeof(uint32_t));
   for (i = o = 5; i < len; )
   {
      unsigned count  = in[i] >> 16;
      unsigned opcode = in[i] & 0xffff;

      if (opcode == SPV_OP_DECORATE && push_var && !inserted)
      {
         out[o++] = (4 << 16) | SPV_OP_DECORATE;
         out[o++] = push_var;
         out[o++] = SPV_DECORATION_SET;
         out[o++] = 0;
         out[o++] = (4 << 16) | SPV_OP_DECORATE;
         out[o++] = push_var;
         out[o++] = SPV_DECORATION_BINDING;
         out[o++] = WEBGPU_PUSH_BINDING;
         inserted = true;
      }

      memcpy(out + o, in + i, count * sizeof(uint32_t));
      switch (opcode)
      {
         case SPV_OP_DECORATE:
            if (count >= 4 && out[o + 2] == SPV_DECORATION_BINDING)
            {
               uint32_t target = out[o + 1];
               uint32_t type   = target < bound ? var_type[target] : 0;
               uint32_t inner  = type < bound ? pointee[type] : 0;
               out[o + 3]      = out[o + 3] * 2
                  + ((inner < bound && sampler[inner]) ? 1 : 0);
            }
            break;
         case SPV_OP_TYPE_POINTER:
            if (out[o + 2] == SPV_STORAGE_PUSH_CONSTANT)
               out[o + 2] = SPV_STORAGE_UNIFORM;
            break;
         case SPV_OP_VARIABLE:
            if (out[o + 3] == SPV_STORAGE_PUSH_CONSTANT)
               out[o + 3] = SPV_STORAGE_UNIFORM;
            break;
         default:
            break;
      }
      o += count;
      i += count;
   }

   if (push_var && !inserted)
      goto error;
   free(sampler);
   free(pointee);
   free(var_type);
   *out_len = o;
   return out;

error:
   free(sampler);
   free(pointee);
   free(var_type);
   free(out);
   return NULL;
}

/* WebGPU has no combined image samplers, and naga's SPIR-V reader
 * trips on some of glslang's control flow until it is optimized */
static uint32_t *webgpu_spirv_prepare(const uint32_t *words, size_t len,
      size_t *out_len)
{
   uint32_t *out                 = NULL;
   spv_binary split              = NULL;
   spv_optimizer_options options = NULL;
   spv_optimizer_t *opt          = spvOptimizerCreate(SPV_ENV_VULKAN_1_0);

   if (!opt)
      return NULL;
   options = spvOptimizerOptionsCreate();
   if (options)
      spvOptimizerOptionsSetRunValidator(options, false);
   if (     options
         && spvOptimizerRegisterPassFromFlag(opt,
            "--split-combined-image-sampler"))
      spvOptimizerRegisterPerformancePasses(opt);
   if (     options
         && spvOptimizerRun(opt, words, len, &split, options) == SPV_SUCCESS
         && split)
      out = webgpu_spirv_rebind(split->code, split->wordCount, out_len);
   if (split)
      spvBinaryDestroy(split);
   if (options)
      spvOptimizerOptionsDestroy(options);
   spvOptimizerDestroy(opt);
   return out;
}

typedef struct
{
   bool done;
   bool failed;
} webgpu_scope_t;

static void webgpu_chain_on_scope(WGPUPopErrorScopeStatus status,
      WGPUErrorType type, WGPUStringView message,
      void *userdata1, void *userdata2)
{
   webgpu_scope_t *scope = (webgpu_scope_t*)userdata1;
   if (type != WGPUErrorType_NoError)
   {
      RARCH_ERR("[WebGPU] %s: %.*s\n", (const char*)userdata2,
            (int)(message.length == WGPU_STRLEN
               ? strlen(message.data) : message.length), message.data);
      scope->failed = true;
   }
   scope->done = true;
}

/* At build time only: an invalid object would abort at its first use */
static bool webgpu_chain_scope_end(webgpu_filter_chain_t *chain,
      const char *what)
{
   unsigned polls;
   webgpu_scope_t scope;
   WGPUPopErrorScopeCallbackInfo cb;

   scope.done   = false;
   scope.failed = false;
   memset(&cb, 0, sizeof(cb));
   cb.mode      = WGPUCallbackMode_AllowProcessEvents;
   cb.callback  = webgpu_chain_on_scope;
   cb.userdata1 = &scope;
   cb.userdata2 = (void*)what;
   wgpuDevicePopErrorScope(chain->device, cb);
   for (polls = 0; polls < 10000 && !scope.done; polls++)
      wgpuInstanceProcessEvents(chain->instance);
   return scope.done && !scope.failed;
}

static void webgpu_target_free(webgpu_target_t *t)
{
   if (t->view)
      wgpuTextureViewRelease(t->view);
   if (t->tex)
   {
      wgpuTextureDestroy(t->tex);
      wgpuTextureRelease(t->tex);
   }
   memset(t, 0, sizeof(*t));
}

static bool webgpu_target_make(webgpu_filter_chain_t *chain,
      webgpu_target_t *t, unsigned dims, WGPUTextureFormat format,
      WGPUTextureUsage usage)
{
   WGPUTextureDescriptor desc;

   webgpu_target_free(t);
   memset(&desc, 0, sizeof(desc));
   desc.usage                   = WGPUTextureUsage_TextureBinding | usage;
   desc.dimension               = WGPUTextureDimension_2D;
   desc.size.width              = VIDEO_SCALE_W(dims);
   desc.size.height             = VIDEO_SCALE_H(dims);
   desc.size.depthOrArrayLayers = 1;
   desc.format                  = format;
   desc.mipLevelCount           = 1;
   desc.sampleCount             = 1;
   if (!(t->tex = wgpuDeviceCreateTexture(chain->device, &desc)))
      return false;
   t->view   = wgpuTextureCreateView(t->tex, NULL);
   t->format = format;
   t->dims   = dims;
   return t->view != NULL;
}

static void webgpu_target_write(webgpu_filter_chain_t *chain,
      webgpu_target_t *t, const void *data, unsigned pitch)
{
   WGPUTexelCopyTextureInfo dst;
   WGPUTexelCopyBufferLayout layout;
   WGPUExtent3D size;

   memset(&dst, 0, sizeof(dst));
   dst.texture             = t->tex;
   dst.aspect              = WGPUTextureAspect_All;
   layout.offset           = 0;
   layout.bytesPerRow      = pitch;
   layout.rowsPerImage     = VIDEO_SCALE_H(t->dims);
   size.width              = VIDEO_SCALE_W(t->dims);
   size.height             = VIDEO_SCALE_H(t->dims);
   size.depthOrArrayLayers = 1;
   wgpuQueueWriteTexture(chain->queue, &dst, data,
         (size_t)pitch * size.height, &layout, &size);
}

static WGPUTextureFormat webgpu_chain_format(enum glslang_format fmt,
      uint8_t flags)
{
   switch (fmt)
   {
      case SLANG_FORMAT_R8_UNORM:
         return WGPUTextureFormat_R8Unorm;
      case SLANG_FORMAT_R8_UINT:
         return WGPUTextureFormat_R8Uint;
      case SLANG_FORMAT_R8_SINT:
         return WGPUTextureFormat_R8Sint;
      case SLANG_FORMAT_R8G8_UNORM:
         return WGPUTextureFormat_RG8Unorm;
      case SLANG_FORMAT_R8G8_UINT:
         return WGPUTextureFormat_RG8Uint;
      case SLANG_FORMAT_R8G8_SINT:
         return WGPUTextureFormat_RG8Sint;
      case SLANG_FORMAT_R8G8B8A8_UNORM:
         return WGPUTextureFormat_RGBA8Unorm;
      case SLANG_FORMAT_R8G8B8A8_UINT:
         return WGPUTextureFormat_RGBA8Uint;
      case SLANG_FORMAT_R8G8B8A8_SINT:
         return WGPUTextureFormat_RGBA8Sint;
      case SLANG_FORMAT_R8G8B8A8_SRGB:
         return WGPUTextureFormat_RGBA8UnormSrgb;
      case SLANG_FORMAT_A2B10G10R10_UNORM_PACK32:
         return WGPUTextureFormat_RGB10A2Unorm;
      case SLANG_FORMAT_A2B10G10R10_UINT_PACK32:
         return WGPUTextureFormat_RGB10A2Uint;
      case SLANG_FORMAT_R16_UINT:
         return WGPUTextureFormat_R16Uint;
      case SLANG_FORMAT_R16_SINT:
         return WGPUTextureFormat_R16Sint;
      case SLANG_FORMAT_R16_SFLOAT:
         return WGPUTextureFormat_R16Float;
      case SLANG_FORMAT_R16G16_UINT:
         return WGPUTextureFormat_RG16Uint;
      case SLANG_FORMAT_R16G16_SINT:
         return WGPUTextureFormat_RG16Sint;
      case SLANG_FORMAT_R16G16_SFLOAT:
         return WGPUTextureFormat_RG16Float;
      case SLANG_FORMAT_R16G16B16A16_UINT:
         return WGPUTextureFormat_RGBA16Uint;
      case SLANG_FORMAT_R16G16B16A16_SINT:
         return WGPUTextureFormat_RGBA16Sint;
      case SLANG_FORMAT_R16G16B16A16_SFLOAT:
         return WGPUTextureFormat_RGBA16Float;
      case SLANG_FORMAT_R32_UINT:
         return WGPUTextureFormat_R32Uint;
      case SLANG_FORMAT_R32_SINT:
         return WGPUTextureFormat_R32Sint;
      case SLANG_FORMAT_R32_SFLOAT:
         return WGPUTextureFormat_R32Float;
      case SLANG_FORMAT_R32G32_UINT:
         return WGPUTextureFormat_RG32Uint;
      case SLANG_FORMAT_R32G32_SINT:
         return WGPUTextureFormat_RG32Sint;
      case SLANG_FORMAT_R32G32_SFLOAT:
         return WGPUTextureFormat_RG32Float;
      case SLANG_FORMAT_R32G32B32A32_UINT:
         return WGPUTextureFormat_RGBA32Uint;
      case SLANG_FORMAT_R32G32B32A32_SINT:
         return WGPUTextureFormat_RGBA32Sint;
      case SLANG_FORMAT_R32G32B32A32_SFLOAT:
         return WGPUTextureFormat_RGBA32Float;
      default:
         break;
   }
   if (flags & FBO_SCALE_FLAG_FP_FBO)
      return WGPUTextureFormat_RGBA16Float;
   if (flags & FBO_SCALE_FLAG_SRGB_FBO)
      return WGPUTextureFormat_RGBA8UnormSrgb;
   if (flags & FBO_SCALE_FLAG_RGB10_FBO)
      return WGPUTextureFormat_RGB10A2Unorm;
   return WGPUTextureFormat_RGBA8Unorm;
}

static WGPUTextureSampleType webgpu_chain_sample_type(WGPUTextureFormat f)
{
   switch (f)
   {
      case WGPUTextureFormat_R32Float:
      case WGPUTextureFormat_RG32Float:
      case WGPUTextureFormat_RGBA32Float:
         return WGPUTextureSampleType_UnfilterableFloat;
      case WGPUTextureFormat_R8Uint:
      case WGPUTextureFormat_RG8Uint:
      case WGPUTextureFormat_RGBA8Uint:
      case WGPUTextureFormat_RGB10A2Uint:
      case WGPUTextureFormat_R16Uint:
      case WGPUTextureFormat_RG16Uint:
      case WGPUTextureFormat_RGBA16Uint:
      case WGPUTextureFormat_R32Uint:
      case WGPUTextureFormat_RG32Uint:
      case WGPUTextureFormat_RGBA32Uint:
         return WGPUTextureSampleType_Uint;
      case WGPUTextureFormat_R8Sint:
      case WGPUTextureFormat_RG8Sint:
      case WGPUTextureFormat_RGBA8Sint:
      case WGPUTextureFormat_R16Sint:
      case WGPUTextureFormat_RG16Sint:
      case WGPUTextureFormat_RGBA16Sint:
      case WGPUTextureFormat_R32Sint:
      case WGPUTextureFormat_RG32Sint:
      case WGPUTextureFormat_RGBA32Sint:
         return WGPUTextureSampleType_Sint;
      default:
         break;
   }
   return WGPUTextureSampleType_Float;
}

/* What pass @p reads for texture semantic @sem, index @k; history
 * beyond the current frame is not kept */
static void webgpu_chain_input(webgpu_filter_chain_t *chain, unsigned p,
      enum slang_texture_semantic sem, unsigned k, webgpu_input_t *in)
{
   const struct video_shader *shader = chain->shader;
   unsigned last                     = chain->num_passes - 1;

   in->view   = chain->dummy.view;
   in->format = chain->dummy.format;
   in->dims   = chain->dummy.dims;
   in->filter = shader->pass[p].filter;
   in->wrap   = shader->pass[p].wrap;

   switch (sem)
   {
      case SLANG_TEXTURE_SEMANTIC_ORIGINAL:
      case SLANG_TEXTURE_SEMANTIC_ORIGINAL_HISTORY:
         if (k == 0)
         {
            in->view   = chain->original;
            in->format = WGPUTextureFormat_BGRA8Unorm;
            in->dims   = chain->original_dims;
            in->filter = shader->pass[0].filter;
            in->wrap   = shader->pass[0].wrap;
         }
         break;
      case SLANG_TEXTURE_SEMANTIC_SOURCE:
         if (p == 0)
         {
            in->view   = chain->original;
            in->format = WGPUTextureFormat_BGRA8Unorm;
            in->dims   = chain->original_dims;
         }
         else
         {
            in->view   = chain->passes[p - 1].fb.view;
            in->format = chain->passes[p - 1].format;
            in->dims   = chain->passes[p - 1].dims;
         }
         break;
      case SLANG_TEXTURE_SEMANTIC_PASS_OUTPUT:
         if (k < p)
         {
            in->view   = chain->passes[k].fb.view;
            in->format = chain->passes[k].format;
            in->dims   = chain->passes[k].dims;
            in->filter = shader->pass[k + 1].filter;
            in->wrap   = shader->pass[k + 1].wrap;
         }
         break;
      case SLANG_TEXTURE_SEMANTIC_PASS_FEEDBACK:
         if (k < last && chain->passes[k].needs_feedback)
         {
            in->view   = chain->passes[k].feedback.view;
            in->format = chain->passes[k].format;
            in->dims   = chain->passes[k].dims;
            in->filter = shader->pass[k + 1].filter;
            in->wrap   = shader->pass[k + 1].wrap;
         }
         break;
      case SLANG_TEXTURE_SEMANTIC_USER:
         if (k < chain->num_luts)
         {
            in->view   = chain->luts[k].tex.view;
            in->format = chain->luts[k].tex.format;
            in->dims   = chain->luts[k].tex.dims;
            in->filter = chain->luts[k].filter;
            in->wrap   = chain->luts[k].wrap;
         }
         break;
      default:
         break;
   }
}

static WGPUSampler webgpu_chain_sampler(webgpu_filter_chain_t *chain,
      const webgpu_input_t *in)
{
   unsigned wrap;
   bool linear = in->filter == RARCH_FILTER_LINEAR
      || (in->filter == RARCH_FILTER_UNSPEC && chain->smooth);
   if (webgpu_chain_sample_type(in->format) != WGPUTextureSampleType_Float)
      linear = false;
   switch (in->wrap)
   {
      case RARCH_WRAP_REPEAT:
         wrap = 1;
         break;
      case RARCH_WRAP_MIRRORED_REPEAT:
         wrap = 2;
         break;
      default:
         wrap = 0;
         break;
   }
   return chain->samplers[linear ? 1 : 0][wrap];
}

static WGPUShaderStage webgpu_chain_stages(uint32_t mask)
{
   WGPUShaderStage stages = WGPUShaderStage_None;
   if (mask & SLANG_STAGE_VERTEX_MASK)
      stages |= WGPUShaderStage_Vertex;
   if (mask & SLANG_STAGE_FRAGMENT_MASK)
      stages |= WGPUShaderStage_Fragment;
   return stages;
}

static WGPUShaderModule webgpu_chain_module(webgpu_filter_chain_t *chain,
      const uint32_t *words, size_t len)
{
   WGPUShaderSourceSPIRV spirv;
   WGPUShaderModuleDescriptor desc;

   memset(&spirv, 0, sizeof(spirv));
   spirv.chain.sType = WGPUSType_ShaderSourceSPIRV;
   spirv.codeSize    = (uint32_t)len;
   spirv.code        = words;
   memset(&desc, 0, sizeof(desc));
   desc.nextInChain  = &spirv.chain;
   return wgpuDeviceCreateShaderModule(chain->device, &desc);
}

static bool webgpu_chain_build_pass(webgpu_filter_chain_t *chain,
      unsigned p)
{
   unsigned i, k;
   size_t vs_len, fs_len;
   WGPUBindGroupLayoutEntry entries[WEBGPU_MAX_ENTRIES];
   WGPUBindGroupLayoutDescriptor layout_desc;
   WGPUPipelineLayoutDescriptor pipe_layout_desc;
   WGPUPipelineLayout pipe_layout;
   WGPURenderPipelineDescriptor desc;
   WGPUVertexAttribute attributes[2];
   WGPUVertexBufferLayout vbo;
   WGPUFragmentState fragment;
   WGPUColorTargetState target;
   WGPUBufferDescriptor buffer;
   WGPUShaderModule vs        = NULL;
   WGPUShaderModule fs        = NULL;
   uint32_t *vs_words         = NULL;
   uint32_t *fs_words         = NULL;
   unsigned count             = 0;
   bool ok                    = false;
   webgpu_pass_t *pass        = &chain->passes[p];
   slang_reflection *refl     = &pass->reflection;

   memset(entries, 0, sizeof(entries));
   if (refl->ubo_size)
   {
      entries[count].binding           = refl->ubo_binding * 2;
      entries[count].visibility        = webgpu_chain_stages(refl->ubo_stage_mask);
      entries[count].buffer.type       = WGPUBufferBindingType_Uniform;
      count++;
   }
   if (refl->push_constant_size)
   {
      entries[count].binding           = WEBGPU_PUSH_BINDING;
      entries[count].visibility        = webgpu_chain_stages(refl->push_constant_stage_mask);
      entries[count].buffer.type       = WGPUBufferBindingType_Uniform;
      count++;
   }
   for (i = 0; i < SLANG_NUM_TEXTURE_SEMANTICS; i++)
   {
      for (k = 0; k < refl->semantic_textures[i].size; k++)
      {
         webgpu_input_t in;
         WGPUTextureSampleType type;
         const slang_texture_semantic_meta *meta =
            &refl->semantic_textures[i].data[k];
         if (!meta->texture || count + 2 > WEBGPU_MAX_ENTRIES)
            continue;
         webgpu_chain_input(chain, p, (enum slang_texture_semantic)i, k, &in);
         type = webgpu_chain_sample_type(in.format);
         entries[count].binding               = meta->binding * 2;
         entries[count].visibility            = webgpu_chain_stages(meta->stage_mask);
         entries[count].texture.sampleType    = type;
         entries[count].texture.viewDimension = WGPUTextureViewDimension_2D;
         count++;
         entries[count].binding               = meta->binding * 2 + 1;
         entries[count].visibility            = webgpu_chain_stages(meta->stage_mask);
         entries[count].sampler.type          = type == WGPUTextureSampleType_Float
            ? WGPUSamplerBindingType_Filtering
            : WGPUSamplerBindingType_NonFiltering;
         count++;
      }
   }

   if (!(vs_words = webgpu_spirv_prepare(pass->vertex, pass->vertex_len, &vs_len))
         || !(fs_words = webgpu_spirv_prepare(pass->fragment, pass->fragment_len, &fs_len)))
   {
      RARCH_ERR("[WebGPU] Pass %u: SPIR-V could not be made WebGPU's.\n", p);
      goto end;
   }

   wgpuDevicePushErrorScope(chain->device, WGPUErrorFilter_Validation);
   vs = webgpu_chain_module(chain, vs_words, vs_len);
   fs = webgpu_chain_module(chain, fs_words, fs_len);
   if (!webgpu_chain_scope_end(chain, "shader"))
      goto end;

   memset(&layout_desc, 0, sizeof(layout_desc));
   layout_desc.entryCount = count;
   layout_desc.entries    = entries;
   pass->layout           = wgpuDeviceCreateBindGroupLayout(chain->device, &layout_desc);
   memset(&pipe_layout_desc, 0, sizeof(pipe_layout_desc));
   pipe_layout_desc.bindGroupLayoutCount = 1;
   pipe_layout_desc.bindGroupLayouts     = &pass->layout;

   memset(attributes, 0, sizeof(attributes));
   attributes[0].format         = WGPUVertexFormat_Float32x4;
   attributes[0].shaderLocation = 0;
   attributes[1].format         = WGPUVertexFormat_Float32x2;
   attributes[1].offset         = 4 * sizeof(float);
   attributes[1].shaderLocation = 1;
   memset(&vbo, 0, sizeof(vbo));
   vbo.arrayStride    = 6 * sizeof(float);
   vbo.stepMode       = WGPUVertexStepMode_Vertex;
   vbo.attributeCount = 2;
   vbo.attributes     = attributes;

   memset(&target, 0, sizeof(target));
   target.format    = p + 1 == chain->num_passes ? chain->format : pass->format;
   target.writeMask = WGPUColorWriteMask_All;
   memset(&fragment, 0, sizeof(fragment));
   fragment.module      = fs;
   fragment.entryPoint  = webgpu_chain_str("main");
   fragment.targetCount = 1;
   fragment.targets     = &target;

   wgpuDevicePushErrorScope(chain->device, WGPUErrorFilter_Validation);
   pipe_layout = wgpuDeviceCreatePipelineLayout(chain->device, &pipe_layout_desc);
   memset(&desc, 0, sizeof(desc));
   desc.layout             = pipe_layout;
   desc.vertex.module      = vs;
   desc.vertex.entryPoint  = webgpu_chain_str("main");
   desc.vertex.bufferCount = 1;
   desc.vertex.buffers     = &vbo;
   desc.primitive.topology = WGPUPrimitiveTopology_TriangleStrip;
   desc.multisample.count  = 1;
   desc.multisample.mask   = 0xFFFFFFFF;
   desc.fragment           = &fragment;
   pass->pipe              = wgpuDeviceCreateRenderPipeline(chain->device, &desc);
   wgpuPipelineLayoutRelease(pipe_layout);
   if (!webgpu_chain_scope_end(chain, "pipeline") || !pass->pipe)
   {
      if (pass->pipe)
         wgpuRenderPipelineRelease(pass->pipe);
      pass->pipe = NULL;
      goto end;
   }

   pass->ubo_size  = (refl->ubo_size + 15) & ~(size_t)15;
   pass->push_size = (refl->push_constant_size + 15) & ~(size_t)15;
   memset(&buffer, 0, sizeof(buffer));
   buffer.usage    = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
   if (pass->ubo_size)
   {
      buffer.size    = pass->ubo_size;
      pass->ubo      = wgpuDeviceCreateBuffer(chain->device, &buffer);
      pass->ubo_data = (uint8_t*)calloc(1, pass->ubo_size);
      if (!pass->ubo || !pass->ubo_data)
         goto end;
   }
   if (pass->push_size)
   {
      buffer.size     = pass->push_size;
      pass->push      = wgpuDeviceCreateBuffer(chain->device, &buffer);
      pass->push_data = (uint8_t*)calloc(1, pass->push_size);
      if (!pass->push || !pass->push_data)
         goto end;
   }
   ok = true;

end:
   if (vs)
      wgpuShaderModuleRelease(vs);
   if (fs)
      wgpuShaderModuleRelease(fs);
   free(vs_words);
   free(fs_words);
   return ok;
}

/* Names passes and LUTs go by, before any pass is reflected */
static bool webgpu_chain_names(webgpu_filter_chain_t *chain,
      char (*names)[64])
{
   unsigned i;
   for (i = 0; i < chain->num_passes; i++)
   {
      if (!*names[i])
         continue;
      if (     !slang_texture_semantic_name_map_set_unique(&chain->texture_map,
               names[i], NULL, SLANG_TEXTURE_SEMANTIC_PASS_OUTPUT, i)
            || !slang_texture_semantic_name_map_set_unique(&chain->texture_uniform_map,
               names[i], "Size", SLANG_TEXTURE_SEMANTIC_PASS_OUTPUT, i)
            || !slang_texture_semantic_name_map_set_unique(&chain->texture_map,
               names[i], "Feedback", SLANG_TEXTURE_SEMANTIC_PASS_FEEDBACK, i)
            || !slang_texture_semantic_name_map_set_unique(&chain->texture_uniform_map,
               names[i], "FeedbackSize", SLANG_TEXTURE_SEMANTIC_PASS_FEEDBACK, i))
         return false;
   }
   for (i = 0; i < chain->shader->luts; i++)
   {
      if (     !slang_texture_semantic_name_map_set_unique(&chain->texture_map,
               chain->shader->lut[i].id, NULL, SLANG_TEXTURE_SEMANTIC_USER, i)
            || !slang_texture_semantic_name_map_set_unique(&chain->texture_uniform_map,
               chain->shader->lut[i].id, "Size", SLANG_TEXTURE_SEMANTIC_USER, i))
         return false;
   }
   return true;
}

static bool webgpu_chain_load_luts(webgpu_filter_chain_t *chain)
{
   unsigned i;
   for (i = 0; i < chain->shader->luts; i++)
   {
      struct texture_image image;
      const struct video_shader_lut *lut = &chain->shader->lut[i];

      memset(&image, 0, sizeof(image));
      if (!image_texture_load(&image, lut->path))
      {
         RARCH_ERR("[WebGPU] Cannot load LUT \"%s\".\n", lut->path);
         return false;
      }
      if (webgpu_target_make(chain, &chain->luts[i].tex,
               VIDEO_SCALE_PACK(image.width, image.height),
               WGPUTextureFormat_BGRA8Unorm, WGPUTextureUsage_CopyDst))
         webgpu_target_write(chain, &chain->luts[i].tex, image.pixels,
               image.width * 4);
      image_texture_free(&image);
      if (!chain->luts[i].tex.view)
         return false;
      chain->luts[i].filter = lut->filter;
      chain->luts[i].wrap   = lut->wrap;
      chain->num_luts++;
   }
   return true;
}

static bool webgpu_chain_common(webgpu_filter_chain_t *chain)
{
   unsigned i, j;
   uint32_t black = 0;
   /* x, y, z, w, u, v: a strip over the unit square */
   static const float quad[] = {
      0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
      1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f
   };
   static const WGPUAddressMode modes[3] = {
      WGPUAddressMode_ClampToEdge,
      WGPUAddressMode_Repeat,
      WGPUAddressMode_MirrorRepeat
   };
   WGPUBufferDescriptor buffer;

   for (i = 0; i < 2; i++)
   {
      for (j = 0; j < 3; j++)
      {
         WGPUSamplerDescriptor desc;
         memset(&desc, 0, sizeof(desc));
         desc.addressModeU  = modes[j];
         desc.addressModeV  = modes[j];
         desc.addressModeW  = modes[j];
         desc.magFilter     = i ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
         desc.minFilter     = desc.magFilter;
         desc.mipmapFilter  = WGPUMipmapFilterMode_Nearest;
         desc.lodMaxClamp   = 32.0f;
         desc.maxAnisotropy = 1;
         if (!(chain->samplers[i][j] = wgpuDeviceCreateSampler(chain->device, &desc)))
            return false;
      }
   }

   memset(&buffer, 0, sizeof(buffer));
   buffer.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
   buffer.size  = sizeof(quad);
   if (!(chain->vbo = wgpuDeviceCreateBuffer(chain->device, &buffer)))
      return false;
   wgpuQueueWriteBuffer(chain->queue, chain->vbo, 0, quad, sizeof(quad));

   if (!webgpu_target_make(chain, &chain->dummy, VIDEO_SCALE_PACK(1, 1),
            WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_CopyDst))
      return false;
   webgpu_target_write(chain, &chain->dummy, &black, 4);
   return true;
}

/* Every pass's SPIR-V, parameters and output format, from its source */
static bool webgpu_chain_compile(webgpu_filter_chain_t *chain,
      char (*names)[64])
{
   unsigned i;
   bool ok             = true;
   void *include_cache = glslang_include_cache_new();
   struct video_shader *shader = chain->shader;

   for (i = 0; i < chain->num_passes && ok; i++)
   {
      size_t j;
      glslang_output output;
      webgpu_pass_t *pass = &chain->passes[i];

      glslang_output_init(&output);
      if (!glslang_compile_shader_cached(shader->pass[i].source.path,
               &output, include_cache))
      {
         RARCH_ERR("[WebGPU] Cannot compile \"%s\".\n",
               shader->pass[i].source.path);
         ok = false;
         break;
      }

      if (output.meta.num_parameters)
         pass->params = (unsigned*)calloc(output.meta.num_parameters,
               sizeof(unsigned));
      for (j = 0; j < output.meta.num_parameters; j++)
      {
         unsigned k;
         for (k = 0; k < shader->num_parameters; k++)
            if (string_is_equal(shader->parameters[k].id,
                     output.meta.parameters[j].id))
               break;
         /* an unknown one keeps its slot, pointing nowhere */
         pass->params[pass->num_params++] = k;
      }
      chain->passes[i].format = webgpu_chain_format(
            glslang_hdr_output_format(shader, i, output.meta.rt_format),
            shader->pass[i].fbo.flags);

      if (*shader->pass[i].alias)
         strlcpy(names[i], shader->pass[i].alias, sizeof(names[i]));
      else
         strlcpy(names[i], output.meta.name, sizeof(names[i]));

      pass->vertex       = output.vertex;
      pass->vertex_len   = output.vertex_len;
      pass->fragment     = output.fragment;
      pass->fragment_len = output.fragment_len;
      output.vertex      = NULL;
      output.fragment    = NULL;
      glslang_output_free(&output);

      /* the name maps are filled before any pass is reflected */
      if (!(ok = slang_reflection_init(&pass->reflection)))
         break;
      pass->reflection.pass_number                  = i;
      pass->reflection.texture_semantic_map         = &chain->texture_map;
      pass->reflection.texture_semantic_uniform_map = &chain->texture_uniform_map;
   }

   glslang_include_cache_free(include_cache);
   return ok;
}

static bool webgpu_chain_reflect(webgpu_filter_chain_t *chain)
{
   unsigned i;
   for (i = 0; i < chain->num_passes; i++)
   {
      bool ok;
      size_t j;
      slang_semantic_name_map map;
      webgpu_pass_t *pass = &chain->passes[i];

      memset(&map, 0, sizeof(map));
      for (j = 0; j < pass->num_params; j++)
         if (pass->params[j] < chain->shader->num_parameters
               && !slang_semantic_name_map_set_unique(&map,
                  chain->shader->parameters[pass->params[j]].id, NULL,
                  SLANG_SEMANTIC_FLOAT_PARAMETER, (unsigned)j))
         {
            slang_semantic_name_map_free(&map);
            return false;
         }
      pass->reflection.semantic_map = &map;
      ok = slang_reflect_spirv(pass->vertex, pass->vertex_len,
            pass->fragment, pass->fragment_len, &pass->reflection);
      pass->reflection.semantic_map = NULL;
      slang_semantic_name_map_free(&map);
      if (!ok)
      {
         RARCH_ERR("[WebGPU] Pass %u failed reflection.\n", i);
         return false;
      }
   }

   /* PassFeedback# reads what pass # drew the frame before */
   for (i = 0; i < chain->num_passes; i++)
   {
      unsigned k;
      const slang_texture_semantic_array *fb =
         &chain->passes[i].reflection.semantic_textures[SLANG_TEXTURE_SEMANTIC_PASS_FEEDBACK];
      for (k = 0; k < fb->size && k + 1 < chain->num_passes; k++)
         if (fb->data[k].texture)
            chain->passes[k].needs_feedback = true;
   }
   return true;
}

webgpu_filter_chain_t *webgpu_filter_chain_new(WGPUInstance instance,
      WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format,
      const char *path)
{
   unsigned i;
   char (*names)[64]             = NULL;
   webgpu_filter_chain_t *chain  = (webgpu_filter_chain_t*)
      calloc(1, sizeof(*chain));

   if (!chain)
      return NULL;
   chain->instance = instance;
   chain->device   = device;
   chain->queue    = queue;
   chain->format   = format;

   if (     !(chain->shader = (struct video_shader*)calloc(1, sizeof(*chain->shader)))
         || !video_shader_load_preset_into_shader(path, chain->shader)
         || !chain->shader->passes)
   {
      RARCH_ERR("[WebGPU] Cannot load preset \"%s\".\n", path);
      goto error;
   }

   chain->num_passes = chain->shader->passes;
   chain->passes     = (webgpu_pass_t*)calloc(chain->num_passes,
         sizeof(*chain->passes));
   names             = (char(*)[64])calloc(chain->num_passes, 64);
   if (!chain->passes || !names)
      goto error;

   if (     !webgpu_chain_common(chain)
         || !webgpu_chain_load_luts(chain)
         || !webgpu_chain_compile(chain, names)
         || !webgpu_chain_names(chain, names)
         || !webgpu_chain_reflect(chain))
      goto error;

   for (i = 0; i < chain->num_passes; i++)
   {
      if (!webgpu_chain_build_pass(chain, i))
         goto error;
      free(chain->passes[i].vertex);
      free(chain->passes[i].fragment);
      chain->passes[i].vertex   = NULL;
      chain->passes[i].fragment = NULL;
   }

   free(names);
   RARCH_LOG("[WebGPU] Preset \"%s\", %u pass(es).\n", path,
         chain->num_passes);
   return chain;

error:
   free(names);
   webgpu_filter_chain_free(chain);
   return NULL;
}

void webgpu_filter_chain_free(webgpu_filter_chain_t *chain)
{
   unsigned i, j;
   if (!chain)
      return;
   for (i = 0; i < chain->num_passes; i++)
   {
      webgpu_pass_t *pass = &chain->passes[i];
      slang_reflection_free(&pass->reflection);
      webgpu_target_free(&pass->fb);
      webgpu_target_free(&pass->feedback);
      if (pass->pipe)
         wgpuRenderPipelineRelease(pass->pipe);
      if (pass->layout)
         wgpuBindGroupLayoutRelease(pass->layout);
      if (pass->ubo)
      {
         wgpuBufferDestroy(pass->ubo);
         wgpuBufferRelease(pass->ubo);
      }
      if (pass->push)
      {
         wgpuBufferDestroy(pass->push);
         wgpuBufferRelease(pass->push);
      }
      free(pass->ubo_data);
      free(pass->push_data);
      free(pass->params);
      free(pass->vertex);
      free(pass->fragment);
   }
   for (i = 0; i < chain->num_luts; i++)
      webgpu_target_free(&chain->luts[i].tex);
   for (i = 0; i < 2; i++)
      for (j = 0; j < 3; j++)
         if (chain->samplers[i][j])
            wgpuSamplerRelease(chain->samplers[i][j]);
   if (chain->final_group)
      wgpuBindGroupRelease(chain->final_group);
   if (chain->vbo)
   {
      wgpuBufferDestroy(chain->vbo);
      wgpuBufferRelease(chain->vbo);
   }
   webgpu_target_free(&chain->dummy);
   slang_texture_semantic_name_map_free(&chain->texture_map);
   slang_texture_semantic_name_map_free(&chain->texture_uniform_map);
   free(chain->passes);
   free(chain->shader);
   free(chain);
}

struct video_shader *webgpu_filter_chain_get_preset(
      webgpu_filter_chain_t *chain)
{
   return chain ? chain->shader : NULL;
}

static void webgpu_pass_put(webgpu_pass_t *pass, bool ubo, size_t ubo_offset,
      bool push, size_t push_offset, const void *data, size_t size)
{
   if (ubo && pass->ubo_data && ubo_offset + size <= pass->ubo_size)
      memcpy(pass->ubo_data + ubo_offset, data, size);
   if (push && pass->push_data && push_offset + size <= pass->push_size)
      memcpy(pass->push_data + push_offset, data, size);
}

static void webgpu_pass_semantic(webgpu_pass_t *pass,
      enum slang_semantic sem, const void *data, size_t size)
{
   const slang_semantic_meta *meta = &pass->reflection.semantics[sem];
   webgpu_pass_put(pass, meta->uniform, meta->ubo_offset,
         meta->push_constant, meta->push_constant_offset, data, size);
}

static void webgpu_size_vec4(float *v, unsigned dims)
{
   v[0] = (float)VIDEO_SCALE_W(dims);
   v[1] = (float)VIDEO_SCALE_H(dims);
   v[2] = v[0] ? 1.0f / v[0] : 0.0f;
   v[3] = v[1] ? 1.0f / v[1] : 0.0f;
}

static unsigned webgpu_pass_dims(const struct gfx_fbo_scale *s,
      unsigned source, unsigned viewport)
{
   unsigned w, h;
   if (!(s->flags & FBO_SCALE_FLAG_VALID))
      return source;
   switch (s->type_x)
   {
      case RARCH_SCALE_ABSOLUTE:
         w = s->abs_x;
         break;
      case RARCH_SCALE_VIEWPORT:
         w = (unsigned)(VIDEO_SCALE_W(viewport) * s->scale_x + 0.5f);
         break;
      default:
         w = (unsigned)(VIDEO_SCALE_W(source) * s->scale_x + 0.5f);
         break;
   }
   switch (s->type_y)
   {
      case RARCH_SCALE_ABSOLUTE:
         h = s->abs_y;
         break;
      case RARCH_SCALE_VIEWPORT:
         h = (unsigned)(VIDEO_SCALE_H(viewport) * s->scale_y + 0.5f);
         break;
      default:
         h = (unsigned)(VIDEO_SCALE_H(source) * s->scale_y + 0.5f);
         break;
   }
   return VIDEO_SCALE_PACK(w ? w : 1, h ? h : 1);
}

static void webgpu_pass_uniforms(webgpu_filter_chain_t *chain,
      unsigned p, uint64_t frame_count, unsigned rotation)
{
   unsigned i, k;
   float v[4];
   uint32_t u;
   int32_t s;
   float aspect;
   /* the unit square onto clip space, y down */
   static const float mvp[16] = {
      2.0f,  0.0f, 0.0f, 0.0f,
      0.0f, -2.0f, 0.0f, 0.0f,
      0.0f,  0.0f, 1.0f, 0.0f,
     -1.0f,  1.0f, 0.0f, 1.0f
   };
   webgpu_pass_t *pass     = &chain->passes[p];
   slang_reflection *refl  = &pass->reflection;
   unsigned mod            = chain->shader->pass[p].frame_count_mod;

   webgpu_pass_semantic(pass, SLANG_SEMANTIC_MVP, mvp, sizeof(mvp));
   webgpu_size_vec4(v, pass->dims);
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_OUTPUT, v, sizeof(v));
   webgpu_size_vec4(v, chain->viewport_dims);
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_FINAL_VIEWPORT, v, sizeof(v));
   u = mod ? (uint32_t)(frame_count % mod) : (uint32_t)frame_count;
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_FRAME_COUNT, &u, sizeof(u));
   s = 1;
#ifdef HAVE_REWIND
   if (state_manager_frame_is_reversed())
      s = -1;
#endif
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_FRAME_DIRECTION, &s, sizeof(s));
   u = 1;
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_TOTAL_SUBFRAMES, &u, sizeof(u));
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_CURRENT_SUBFRAME, &u, sizeof(u));
   u = (uint32_t)frame_count;
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_SWAP_COUNT, &u, sizeof(u));
   u = (uint32_t)video_driver_get_frame_time_delta_usec();
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_FRAME_TIME_DELTA, &u, sizeof(u));
   v[0] = video_driver_get_original_fps();
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_ORIGINAL_FPS, v, sizeof(float));
   u = rotation;
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_ROTATION, &u, sizeof(u));
   aspect = video_driver_get_core_aspect();
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_CORE_ASPECT, &aspect, sizeof(aspect));
   if ((rotation & 1) && aspect)
      aspect = 1.0f / aspect;
   webgpu_pass_semantic(pass, SLANG_SEMANTIC_CORE_ASPECT_ROT, &aspect, sizeof(aspect));

   for (i = 0; i < SLANG_NUM_TEXTURE_SEMANTICS; i++)
   {
      for (k = 0; k < refl->semantic_textures[i].size; k++)
      {
         webgpu_input_t in;
         const slang_texture_semantic_meta *meta =
            &refl->semantic_textures[i].data[k];
         if (!meta->uniform && !meta->push_constant)
            continue;
         webgpu_chain_input(chain, p, (enum slang_texture_semantic)i, k, &in);
         webgpu_size_vec4(v, in.dims);
         webgpu_pass_put(pass, meta->uniform, meta->ubo_offset,
               meta->push_constant, meta->push_constant_offset, v, sizeof(v));
      }
   }

   for (i = 0; i < pass->num_params && i < refl->num_float_parameters; i++)
   {
      const slang_semantic_meta *meta = &refl->semantic_float_parameters[i];
      if (pass->params[i] >= chain->shader->num_parameters)
         continue;
      v[0] = video_shader_param_get(&chain->shader->parameters[pass->params[i]]);
      webgpu_pass_put(pass, meta->uniform, meta->ubo_offset,
            meta->push_constant, meta->push_constant_offset, v, sizeof(float));
   }

   if (pass->ubo)
      wgpuQueueWriteBuffer(chain->queue, pass->ubo, 0, pass->ubo_data, pass->ubo_size);
   if (pass->push)
      wgpuQueueWriteBuffer(chain->queue, pass->push, 0, pass->push_data, pass->push_size);
}

static WGPUBindGroup webgpu_pass_group(webgpu_filter_chain_t *chain,
      unsigned p)
{
   unsigned i, k;
   WGPUBindGroupEntry entries[WEBGPU_MAX_ENTRIES];
   WGPUBindGroupDescriptor desc;
   unsigned count         = 0;
   webgpu_pass_t *pass    = &chain->passes[p];
   slang_reflection *refl = &pass->reflection;

   memset(entries, 0, sizeof(entries));
   if (pass->ubo)
   {
      entries[count].binding = refl->ubo_binding * 2;
      entries[count].buffer  = pass->ubo;
      entries[count].size    = pass->ubo_size;
      count++;
   }
   if (pass->push)
   {
      entries[count].binding = WEBGPU_PUSH_BINDING;
      entries[count].buffer  = pass->push;
      entries[count].size    = pass->push_size;
      count++;
   }
   for (i = 0; i < SLANG_NUM_TEXTURE_SEMANTICS; i++)
   {
      for (k = 0; k < refl->semantic_textures[i].size; k++)
      {
         webgpu_input_t in;
         const slang_texture_semantic_meta *meta =
            &refl->semantic_textures[i].data[k];
         if (!meta->texture || count + 2 > WEBGPU_MAX_ENTRIES)
            continue;
         webgpu_chain_input(chain, p, (enum slang_texture_semantic)i, k, &in);
         entries[count].binding     = meta->binding * 2;
         entries[count].textureView = in.view;
         count++;
         entries[count].binding     = meta->binding * 2 + 1;
         entries[count].sampler     = webgpu_chain_sampler(chain, &in);
         count++;
      }
   }

   memset(&desc, 0, sizeof(desc));
   desc.layout     = pass->layout;
   desc.entryCount = count;
   desc.entries    = entries;
   return wgpuDeviceCreateBindGroup(chain->device, &desc);
}

static void webgpu_pass_draw(webgpu_filter_chain_t *chain,
      WGPURenderPassEncoder rp, unsigned p, WGPUBindGroup group)
{
   wgpuRenderPassEncoderSetPipeline(rp, chain->passes[p].pipe);
   wgpuRenderPassEncoderSetBindGroup(rp, 0, group, 0, NULL);
   wgpuRenderPassEncoderSetVertexBuffer(rp, 0, chain->vbo, 0,
         WGPU_WHOLE_SIZE);
   wgpuRenderPassEncoderDraw(rp, 4, 1, 0, 0);
}

void webgpu_filter_chain_record(webgpu_filter_chain_t *chain,
      WGPUCommandEncoder encoder,
      WGPUTextureView original, unsigned original_dims,
      bool smooth, unsigned viewport_dims,
      uint64_t frame_count, unsigned rotation)
{
   unsigned p;
   unsigned source;
   unsigned last;

   if (!chain)
      return;
   last                 = chain->num_passes - 1;
   chain->original      = original;
   chain->original_dims = original_dims;
   chain->viewport_dims = viewport_dims;
   chain->smooth        = smooth;
   if (chain->final_group)
      wgpuBindGroupRelease(chain->final_group);
   chain->final_group   = NULL;

   source = original_dims;
   for (p = 0; p < last; p++)
   {
      WGPUBindGroup group;
      WGPURenderPassEncoder rp;
      WGPURenderPassDescriptor rp_desc;
      WGPURenderPassColorAttachment color;
      webgpu_pass_t *pass = &chain->passes[p];
      unsigned dims       = webgpu_pass_dims(&chain->shader->pass[p].fbo,
            source, viewport_dims);

      if (pass->needs_feedback && pass->fb.dims == dims)
      {
         webgpu_target_t t = pass->feedback;
         pass->feedback    = pass->fb;
         pass->fb          = t;
      }
      if (pass->fb.dims != dims || !pass->fb.view)
      {
         if (!webgpu_target_make(chain, &pass->fb, dims, pass->format,
                  WGPUTextureUsage_RenderAttachment))
            return;
         if (     pass->needs_feedback
               && !webgpu_target_make(chain, &pass->feedback, dims,
                  pass->format, WGPUTextureUsage_RenderAttachment))
            return;
      }
      pass->dims = dims;
      source     = dims;

      webgpu_pass_uniforms(chain, p, frame_count, rotation);
      if (!(group = webgpu_pass_group(chain, p)))
         return;

      memset(&color, 0, sizeof(color));
      color.view       = pass->fb.view;
      color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
      color.loadOp     = WGPULoadOp_Clear;
      color.storeOp    = WGPUStoreOp_Store;
      memset(&rp_desc, 0, sizeof(rp_desc));
      rp_desc.colorAttachmentCount = 1;
      rp_desc.colorAttachments     = &color;
      rp = wgpuCommandEncoderBeginRenderPass(encoder, &rp_desc);
      wgpuRenderPassEncoderSetViewport(rp, 0.0f, 0.0f,
            (float)VIDEO_SCALE_W(dims), (float)VIDEO_SCALE_H(dims),
            0.0f, 1.0f);
      webgpu_pass_draw(chain, rp, p, group);
      wgpuRenderPassEncoderEnd(rp);
      wgpuRenderPassEncoderRelease(rp);
      wgpuBindGroupRelease(group);
   }

   chain->passes[last].dims = viewport_dims;
   webgpu_pass_uniforms(chain, last, frame_count, rotation);
   chain->final_group = webgpu_pass_group(chain, last);
}

void webgpu_filter_chain_draw(webgpu_filter_chain_t *chain,
      WGPURenderPassEncoder pass)
{
   if (chain && chain->final_group)
      webgpu_pass_draw(chain, pass, chain->num_passes - 1,
            chain->final_group);
}
