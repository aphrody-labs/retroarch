/* Drives a HAVE_LIBRETROARCH=1 build from the page: one
 * libretroarch_iterate per animation frame, frames drawn from the heap
 * (WebGPU when available, else a 2D canvas), audio pulled by an
 * AudioWorklet. */

const CFG = "/home/web_user/retroarch/userdata/retroarch.cfg";

const SETTINGS = {
   video_driver: "host",
   audio_driver: "host",
   input_driver: "null",
   input_joypad_driver: "host",
   menu_driver: "rgui",
   video_vsync: "false",
   audio_sync: "false",
   audio_out_rate: "48000",
   config_save_on_exit: "false"
};

const XRGB8888 = 0, RGB565 = 1;

function lut(format)
{
   const table = new Uint32Array(65536);
   for (let p = 0; p < 65536; p++)
   {
      let r, g, b, a;
      if (format === RGB565)
      {
         r = ((p >> 11) & 0x1f) * 255 / 31;
         g = ((p >> 5) & 0x3f) * 255 / 63;
         b = (p & 0x1f) * 255 / 31;
         a = 255;
      }
      else
      {
         r = ((p >> 12) & 0xf) * 17;
         g = ((p >> 8) & 0xf) * 17;
         b = ((p >> 4) & 0xf) * 17;
         a = (p & 0xf) * 17;
      }
      table[p] = ((a << 24) | (Math.round(b) << 16) | (Math.round(g) << 8) | Math.round(r)) >>> 0;
   }
   return table;
}

/* ---- 2D canvas ---- */

function create2d(canvas)
{
   const ctx = canvas.getContext("2d");
   const luts = [];
   let image;

   return {
      kind: "2d",
      draw(heap, f)
      {
         const { data, width, height, pitch, format } = f;
         if (!image || image.width !== width || image.height !== height)
         {
            canvas.width = width;
            canvas.height = height;
            image = new ImageData(width, height);
         }
         const dst = new Uint32Array(image.data.buffer);
         for (let y = 0; y < height; y++)
         {
            if (format === XRGB8888)
            {
               const row = new Uint32Array(heap, data + y * pitch, width);
               for (let x = 0; x < width; x++)
               {
                  const p = row[x];
                  dst[y * width + x] = (0xff000000 | ((p & 0xff) << 16) | (p & 0xff00) | ((p >> 16) & 0xff)) >>> 0;
               }
            }
            else
            {
               const table = luts[format] ??= lut(format);
               const row = new Uint16Array(heap, data + y * pitch, width);
               for (let x = 0; x < width; x++)
                  dst[y * width + x] = table[row[x]];
            }
         }
         ctx.putImageData(image, 0, 0);
      },
      /* RGBA8 of the last picture */
      async capture()
      {
         return new Uint8Array(image.data);
      }
   };
}

/* ---- WebGPU: same shaders and upload rules as bun-engine's
 * libretroarch-gpu.ts. XRGB8888 is already bgra8unorm bytes; RGB565 and
 * the menu's RGBA4444 go up as r16uint and are unpacked in WGSL. ---- */

const VERTEX = `
struct Out { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return Out(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
}
`;

const RGBA = `${VERTEX}
@group(0) @binding(0) var frame: texture_2d<f32>;
@fragment fn fs(in: Out) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(frame));
  let texel = vec2u(min(in.uv * size, size - 1.0));
  return vec4f(textureLoad(frame, texel, 0).rgb, 1.0);
}
`;

const PACKED = `${VERTEX}
@group(0) @binding(0) var frame: texture_2d<u32>;
@group(0) @binding(1) var<uniform> format: u32;
@fragment fn fs(in: Out) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(frame));
  let p = textureLoad(frame, vec2u(min(in.uv * size, size - 1.0)), 0).r;
  if (format == ${RGB565}u) {
    return vec4f(f32(p >> 11u) / 31.0, f32((p >> 5u) & 63u) / 63.0, f32(p & 31u) / 31.0, 1.0);
  }
  return vec4f(f32(p >> 12u) / 15.0, f32((p >> 8u) & 15u) / 15.0, f32((p >> 4u) & 15u) / 15.0, 1.0);
}
`;

async function createWebGpu(canvas)
{
   if (!globalThis.navigator?.gpu)
      return null;
   const adapter = await navigator.gpu.requestAdapter();
   const device = adapter && await adapter.requestDevice();
   const context = device && canvas.getContext("webgpu");
   if (!context)
      return null;
   const target = navigator.gpu.getPreferredCanvasFormat();
   context.configure({ device, format: target, alphaMode: "opaque" });

   const pipeline = (code) =>
   {
      const module = device.createShaderModule({ code });
      return device.createRenderPipeline({
         layout: "auto",
         vertex: { module, entryPoint: "vs" },
         fragment: { module, entryPoint: "fs", targets: [{ format: target }] },
         primitive: { topology: "triangle-list" }
      });
   };
   const rgba = pipeline(RGBA), packed = pipeline(PACKED);
   const uniform = device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
   });
   let texture, group, current = XRGB8888;

   /* aspect-preserving viewport inside a w x h target */
   function drawTo(encoder, view, w, h)
   {
      const pass = encoder.beginRenderPass({
         colorAttachments: [{
            view, loadOp: "clear", storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 1 }
         }]
      });
      if (texture && group && w && h)
      {
         const scale = Math.min(w / texture.width, h / texture.height);
         const vw = Math.max(1, Math.round(texture.width * scale));
         const vh = Math.max(1, Math.round(texture.height * scale));
         pass.setViewport((w - vw) >> 1, (h - vh) >> 1, vw, vh, 0, 1);
         pass.setPipeline(current === XRGB8888 ? rgba : packed);
         pass.setBindGroup(0, group);
         pass.draw(3);
      }
      pass.end();
   }

   return {
      kind: "webgpu",
      draw(heap, f)
      {
         const { data, width, height, pitch, format } = f;
         if (!texture || texture.width !== width || texture.height !== height
               || format !== current)
         {
            texture?.destroy();
            const pack = format !== XRGB8888;
            texture = device.createTexture({
               size: { width, height },
               format: pack ? "r16uint" : "bgra8unorm",
               usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
            });
            current = format;
            const entries = [{ binding: 0, resource: texture.createView() }];
            if (pack)
            {
               device.queue.writeBuffer(uniform, 0, new Uint32Array([format, 0, 0, 0]));
               entries.push({ binding: 1, resource: { buffer: uniform } });
            }
            group = device.createBindGroup({
               layout: (pack ? packed : rgba).getBindGroupLayout(0), entries
            });
         }
         device.queue.writeTexture(
            { texture },
            new Uint8Array(heap, data, pitch * height),
            { offset: 0, bytesPerRow: pitch, rowsPerImage: height },
            { width, height });
         /* back buffer follows the CSS box, else the frame size */
         const dpr = globalThis.devicePixelRatio || 1;
         const w = Math.round(canvas.clientWidth * dpr) || width;
         const h = Math.round(canvas.clientHeight * dpr) || height;
         if (canvas.width !== w || canvas.height !== h)
         {
            canvas.width = w;
            canvas.height = h;
         }
         const encoder = device.createCommandEncoder();
         drawTo(encoder, context.getCurrentTexture().createView(), w, h);
         device.queue.submit([encoder.finish()]);
      },
      /* renders the last picture offscreen at w x h and reads it back as RGBA8 */
      async capture(w, h)
      {
         const out = device.createTexture({
            size: { width: w, height: h }, format: target,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
         });
         const bytesPerRow = Math.ceil(w * 4 / 256) * 256;
         const buffer = device.createBuffer({
            size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
         });
         const encoder = device.createCommandEncoder();
         drawTo(encoder, out.createView(), w, h);
         encoder.copyTextureToBuffer({ texture: out }, { buffer, bytesPerRow }, { width: w, height: h });
         device.queue.submit([encoder.finish()]);
         await buffer.mapAsync(GPUMapMode.READ);
         const mapped = new Uint8Array(buffer.getMappedRange());
         const px = new Uint8Array(w * h * 4);
         for (let y = 0; y < h; y++)
            px.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + w * 4), y * w * 4);
         buffer.unmap();
         buffer.destroy();
         out.destroy();
         if (target.startsWith("bgra"))
            for (let i = 0; i < px.length; i += 4)
            {
               const b = px[i];
               px[i] = px[i + 2];
               px[i + 2] = b;
            }
         return px;
      }
   };
}

/* ---- AudioWorklet pull audio. The page drains the host audio driver
 * (libretroarch_audio_read) after each iterate into a power-of-two
 * float stereo ring the worklet consumes at the device clock. The ring
 * is a SharedArrayBuffer when the page is cross-origin isolated, else
 * chunks travel by postMessage. ---- */

const WORKLET = `
class LibretroArchSink extends AudioWorkletProcessor {
   constructor(options) {
      super();
      const o = options.processorOptions;
      this.mask = o.cap - 1;
      this.prefill = o.prefill;
      this.ring = new Float32Array(o.ring || o.cap * 2 * 4);
      this.idx = new Int32Array(o.idx || 4); /* write, read, underruns, 0 */
      this.shared = !!o.ring;
      this.primed = false;
      this.tick = 0;
      this.port.onmessage = (e) => {
         const c = e.data, n = c.length >> 1;
         let w = this.idx[0];
         for (let i = 0; i < n; i++, w++) {
            const at = (w & this.mask) * 2;
            this.ring[at] = c[2 * i];
            this.ring[at + 1] = c[2 * i + 1];
         }
         this.idx[0] = w;
      };
   }
   process(inputs, outputs) {
      const out = outputs[0], n = out[0].length, idx = this.idx;
      const w = Atomics.load(idx, 0);
      let r = idx[1];
      let avail = (w - r) | 0;
      if (!this.primed && avail >= this.prefill) this.primed = true;
      let got = this.primed ? Math.min(avail, n) : 0;
      for (let i = 0; i < got; i++, r++) {
         const at = (r & this.mask) * 2;
         out[0][i] = this.ring[at];
         if (out[1]) out[1][i] = this.ring[at + 1];
      }
      for (let i = got; i < n; i++) {
         out[0][i] = 0;
         if (out[1]) out[1][i] = 0;
      }
      if (this.primed && got < n) { idx[2]++; this.primed = false; }
      Atomics.store(idx, 1, r);
      if (!this.shared && (++this.tick & 7) === 0)
         this.port.postMessage({ read: r, underruns: idx[2] });
      return true;
   }
}
registerProcessor("libretroarch-sink", LibretroArchSink);
`;

const RING_FRAMES = 1 << 14;
const CHUNK = 2048;

async function createAudio(Module, rate, { latency = 0.12 } = {})
{
   const context = new AudioContext({ sampleRate: rate, latencyHint: "interactive" });
   const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
   await context.audioWorklet.addModule(url);
   URL.revokeObjectURL(url);

   const shared = typeof SharedArrayBuffer === "function" && globalThis.crossOriginIsolated;
   const options = { cap: RING_FRAMES, prefill: Math.round(rate * 0.04) };
   let ring, idx, remote = { read: 0, underruns: 0 };
   if (shared)
   {
      options.ring = new SharedArrayBuffer(RING_FRAMES * 2 * 4);
      options.idx = new SharedArrayBuffer(16);
      ring = new Float32Array(options.ring);
      idx = new Int32Array(options.idx);
   }
   const node = new AudioWorkletNode(context, "libretroarch-sink", {
      numberOfInputs: 0, outputChannelCount: [2], processorOptions: options
   });
   node.port.onmessage = (e) => { remote = e.data; };
   node.connect(context.destination);

   const scratch = Module._malloc(CHUNK * 8);
   const target = Math.round(rate * latency);
   let written = 0;
   const stats = { drained: 0 };

   const read = () => shared ? Atomics.load(idx, 1) : remote.read;
   const buffered = () => (written - read()) | 0;

   function push(samples)
   {
      const n = samples.length >> 1;
      if (shared)
      {
         let w = written;
         for (let i = 0; i < n; i++, w++)
         {
            const at = (w & (RING_FRAMES - 1)) * 2;
            ring[at] = samples[2 * i];
            ring[at + 1] = samples[2 * i + 1];
         }
         written = w;
         Atomics.store(idx, 0, w);
      }
      else
      {
         const copy = samples.slice();
         node.port.postMessage(copy, [copy.buffer]);
         written = (written + n) | 0;
      }
      stats.drained += n;
   }

   return {
      context,
      node,
      shared,
      push,
      /* pull what the core produced, keeping about `latency` s queued */
      pump()
      {
         while (buffered() < target)
         {
            const n = Module._libretroarch_audio_read(scratch, CHUNK);
            if (!n)
               break;
            push(new Float32Array(Module.HEAPF32.buffer, scratch, n * 2));
         }
      },
      stats()
      {
         return {
            shared,
            state: context.state,
            rate: context.sampleRate,
            drained: stats.drained,
            consumed: read() >>> 0,
            underruns: shared ? Atomics.load(idx, 2) : remote.underruns
         };
      },
      resume: () => context.resume()
   };
}

/* ---- driver ---- */

/* options: video "auto" (WebGPU, else 2D) | "webgpu" | "2d" | "none";
 * audio true | false; files {path: url} are fetched into the core's
 * filesystem before init (content, system files) */
export async function start(factory, { canvas, args = [], config = {}, module = {},
      video = "auto", audio = false, files = {} } = {})
{
   const Module = await factory({ canvas, print: console.log, printErr: console.warn, ...module });
   const { FS } = Module;
   const cfg = Object.entries({ ...SETTINGS, ...config })
      .map(([key, value]) => `${key} = "${value}"\n`).join("");
   FS.mkdirTree("/home/web_user/retroarch/userdata");
   FS.writeFile(CFG, cfg);
   for (const [path, url] of Object.entries(files))
   {
      const dir = path.slice(0, path.lastIndexOf("/"));
      if (dir)
         FS.mkdirTree(dir);
      FS.writeFile(path, new Uint8Array(await (await fetch(url)).arrayBuffer()));
   }

   const argv = ["retroarch", "-c", CFG, ...args];
   const ptrs = Module._malloc(4 * (argv.length + 1));
   argv.forEach((arg, i) => Module.HEAPU32[(ptrs >> 2) + i] = Module.stringToNewUTF8(arg));
   Module.HEAPU32[(ptrs >> 2) + argv.length] = 0;
   if (await Module._libretroarch_init(argv.length, ptrs) !== 0)
      throw new Error("libretroarch_init failed");

   const frame = Module._malloc(32);
   const status = Module._malloc(32);
   let sink = null, serial = -1n;

   if (video === "webgpu" || video === "auto")
      sink = await createWebGpu(canvas);
   if (!sink && video === "webgpu")
      throw new Error("WebGPU is not available");
   if (!sink && video !== "none")
      sink = create2d(canvas);

   let out = null;
   if (audio)
   {
      Module._libretroarch_status(status);
      const rate = Module.HEAPU32[(status >> 2) + 4] || 48000;
      out = await createAudio(Module, rate);
      /* browsers start the context on a user gesture */
      for (const type of ["pointerdown", "keydown"])
         addEventListener(type, () => out.resume(), { once: true });
   }

   function draw()
   {
      if (!sink || !Module._libretroarch_frame(frame))
         return;
      const u32 = Module.HEAPU32, at = frame >> 2;
      const s = (BigInt(u32[at + 1]) << 32n) | BigInt(u32[at]);
      if (s === serial)
         return;
      serial = s;
      sink.draw(Module.HEAPU8.buffer, {
         data: u32[at + 2], width: u32[at + 3], height: u32[at + 4],
         pitch: u32[at + 5], format: u32[at + 6]
      });
   }

   return {
      Module,
      /* "webgpu", "2d" or "none" */
      video: sink ? sink.kind : "none",
      audio: out,
      capture: (w, h) => sink.capture(w, h),
      /* one frame; false once RetroArch quits */
      iterate()
      {
         if (Module._libretroarch_iterate() !== 0)
            return false;
         draw();
         out?.pump();
         return true;
      },
      run()
      {
         const loop = () => this.iterate() && requestAnimationFrame(loop);
         requestAnimationFrame(loop);
      }
   };
}
