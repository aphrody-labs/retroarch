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

#ifndef __SHADER_WEBGPU_H
#define __SHADER_WEBGPU_H

#include <stdint.h>

#include <boolean.h>
#include <retro_common_api.h>

#include <webgpu/webgpu.h>

#include "../video_shader_parse.h"

RETRO_BEGIN_DECLS

/* Slang presets on WebGPU: SPIR-V from the slang pipeline, combined
 * samplers split and push constants turned into a uniform buffer */
typedef struct webgpu_filter_chain webgpu_filter_chain_t;

/* @instance is polled for the build's error scopes */
webgpu_filter_chain_t *webgpu_filter_chain_new(WGPUInstance instance,
      WGPUDevice device, WGPUQueue queue, WGPUTextureFormat format,
      const char *path);

void webgpu_filter_chain_free(webgpu_filter_chain_t *chain);

struct video_shader *webgpu_filter_chain_get_preset(
      webgpu_filter_chain_t *chain);

/* Records every pass but the last into @encoder, @original being the
 * frame and @viewport_dims what the last pass draws into */
void webgpu_filter_chain_record(webgpu_filter_chain_t *chain,
      WGPUCommandEncoder encoder,
      WGPUTextureView original, unsigned original_dims,
      bool smooth, unsigned viewport_dims,
      uint64_t frame_count, unsigned rotation);

/* The last pass, into @pass with its viewport already set */
void webgpu_filter_chain_draw(webgpu_filter_chain_t *chain,
      WGPURenderPassEncoder pass);

RETRO_END_DECLS

#endif
