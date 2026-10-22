// Static server with the COOP/COEP headers pthreads and SharedArrayBuffer need.
// bun serve.ts [root] ; PORT sets the port. POST bodies are logged (test pages report here).
const root = process.argv[2] ?? ".";
const server = Bun.serve({
  port: Number(process.env.PORT ?? 8099),
  async fetch(req) {
    const path = decodeURIComponent(new URL(req.url).pathname);
    if (req.method === "POST") {
      console.log(await req.text());
      return new Response("ok");
    }
    if (path.includes("..")) return new Response("forbidden", { status: 403 });
    const file = Bun.file(root + (path.endsWith("/") ? path + "index.html" : path));
    if (!(await file.exists())) return new Response("not found", { status: 404 });
    return new Response(file, {
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
        ...(path.endsWith(".wasm") ? { "Content-Type": "application/wasm" } : {}),
      },
    });
  },
});
console.log(server.url.href);
