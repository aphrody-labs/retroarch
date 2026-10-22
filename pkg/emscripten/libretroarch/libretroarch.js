/* Drives a HAVE_LIBRETROARCH=1 build from the page: one
 * libretroarch_iterate per animation frame, frames drawn from the heap. */

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

export async function start(factory, { canvas, args = [], config = {}, module = {} } = {})
{
   const Module = await factory({ canvas, print: console.log, printErr: console.warn, ...module });
   const { FS } = Module;
   const cfg = Object.entries({ ...SETTINGS, ...config })
      .map(([key, value]) => `${key} = "${value}"\n`).join("");
   FS.mkdirTree("/home/web_user/retroarch/userdata");
   FS.writeFile(CFG, cfg);

   const argv = ["retroarch", "-c", CFG, ...args];
   const ptrs = Module._malloc(4 * (argv.length + 1));
   argv.forEach((arg, i) => Module.HEAPU32[(ptrs >> 2) + i] = Module.stringToNewUTF8(arg));
   Module.HEAPU32[(ptrs >> 2) + argv.length] = 0;
   if (Module._libretroarch_init(argv.length, ptrs) !== 0)
      throw new Error("libretroarch_init failed");

   const frame = Module._malloc(32);
   const ctx = canvas.getContext("2d");
   const luts = [];
   let image, serial = -1n;

   function draw()
   {
      if (!Module._libretroarch_frame(frame))
         return;
      const u32 = Module.HEAPU32, at = frame >> 2;
      const s = (BigInt(u32[at + 1]) << 32n) | BigInt(u32[at]);
      if (s === serial)
         return;
      serial = s;
      const data = u32[at + 2], width = u32[at + 3], height = u32[at + 4];
      const pitch = u32[at + 5], format = u32[at + 6];
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
            const row = new Uint32Array(Module.HEAPU8.buffer, data + y * pitch, width);
            for (let x = 0; x < width; x++)
            {
               const p = row[x];
               dst[y * width + x] = (0xff000000 | ((p & 0xff) << 16) | (p & 0xff00) | ((p >> 16) & 0xff)) >>> 0;
            }
         }
         else
         {
            const table = luts[format] ??= lut(format);
            const row = new Uint16Array(Module.HEAPU8.buffer, data + y * pitch, width);
            for (let x = 0; x < width; x++)
               dst[y * width + x] = table[row[x]];
         }
      }
      ctx.putImageData(image, 0, 0);
   }

   return {
      Module,
      /* one frame; false once RetroArch quits */
      iterate()
      {
         if (Module._libretroarch_iterate() !== 0)
            return false;
         draw();
         return true;
      },
      run()
      {
         const loop = () => this.iterate() && requestAnimationFrame(loop);
         requestAnimationFrame(loop);
      }
   };
}
