#!/usr/bin/env node
/**
 * JSON bridge to the installed `pi-knowledge` package.
 *
 * Command layout (arg is the mode; JSON object arrives on stdin, one JSON
 * response object is printed to stdout):
 *   search  {queries, category, top_k, mode, profile, kb_name}
 *   list    {}
 *   add     {source, name, include_suggested_text}
 *   update  {target}
 *   doctor  {}
 *
 * Env:
 *   PI_KNOWLEDGE_ROOT   path to the pi-knowledge checkout/package (optional)
 *   PI_KNOWLEDGE_MODULE module id/path to import; default resolves `pi-knowledge`
 */
import { createRequire } from "node:module";
import { join } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const request = JSON.parse(await readStdin());

function log(message) {
  process.stderr.write(`${message}\n`);
}

function readStdin() {
  return new Promise((resolveP, rejectP) => {
    let data = "";
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (chunk) => (data += chunk));
    process.stdin.on("end", () => resolveP(data || "{}"));
    process.stdin.on("error", rejectP);
  });
}

async function resolveModuleRoot() {
  if (process.env.PI_KNOWLEDGE_ROOT) {
    const root = process.env.PI_KNOWLEDGE_ROOT;
    if (root.endsWith(".js") || root.endsWith(".ts") || root.endsWith(".mjs")) return root;
    return root;
  }
  try {
    return require.resolve("pi-knowledge/package.json").replace(/\/package\.json$/, "");
  } catch (error) {
    log(`pi-knowledge not found; install it into pi_agent/ (npm install pi-knowledge)`);
    throw error;
  }
}

function engineModule(root) {
  if (root.endsWith(".js") || root.endsWith(".ts")) return root;
  return pathToFileURL(join(root, "dist", "src", "engine.js")).href;
}

function storageModule(root) {
  if (root.endsWith(".js") || root.endsWith(".ts")) return root;
  return pathToFileURL(join(root, "dist", "src", "storage", "sqlite.js")).href;
}

async function run(command, payload, modeName, knowledgeDir, root) {
  const { KnowledgeEngine } = await import(engineModule(root));
  const storage = await import(storageModule(root));
  const dir = knowledgeDir || storage.getDefaultKnowledgeDir();
  const engine = new KnowledgeEngine();
  await engine.initialize(dir);
  try {
    if (command === "list") {
      return { ok: true, kbs: engine.list(), knowledge_dir: dir };
    }
    if (command === "add") {
      const { kb, chunkCount } = await engine.add(
        payload.source,
        payload.name || modeName,
        (message) => log(message),
        undefined,
        { include_suggested_text: !!payload.include_suggested_text },
      );
      return { ok: true, kb, chunk_count: chunkCount };
    }
    if (command === "update") {
      const result = await engine.update(payload.target || modeName, (message) => log(message));
      return { ok: true, ...result };
    }
    if (command === "doctor") {
      return { ok: true, report: engine.doctor() };
    }
    if (command === "remove") {
      const result = engine.remove(payload.target || modeName);
      const existed = !!result;
      return { ok: true, removed: existed, target: payload.target || modeName };
    }
    if (command === "search") {
      const queries = Array.isArray(payload.queries) ? payload.queries : [payload.queries];
      const filters =
        payload.category && payload.category !== 'any'
          ? { path_pattern: `${payload.category}/` }
          : undefined;
      const seen = new Set();
      const results = [];
      let lastResponse = null;
      for (const query of queries) {
        const response = await engine.search(String(query), {
          mode: payload.mode,
          profile: payload.profile,
          limit: payload.top_k,
          kb_id: payload.kb_name,
          filters,
        });
        lastResponse = response;
        for (const hit of response.results || []) {
          const id = hit.provenance?.chunk_id || `${hit.file_path}:${hit.start_line}`;
          if (seen.has(id)) continue;
          seen.add(id);
          results.push({ ...hit, kb_name: hit.kb_name || modeName });
        }
      }
      results.sort((a, b) => b.score - a.score);
      return {
        ok: true,
        results: results.slice(0, Math.max(1, (Number(payload.top_k) || 1) * queries.length)),
        mode_used: lastResponse?.mode_used,
        warnings: lastResponse?.warnings || lastResponse?.suggestions,
      };
    }
    throw new Error(`unknown bridge command ${command}`);
  } finally {
    await engine.dispose({ disposeModels: false }).catch(() => {});
  }
}

(async () => {
  try {
    const root = await resolveModuleRoot();
    const result = await run(
      process.argv[2],
      request,
      String(request.kb_name || "gen9ou"),
      process.env.PI_KNOWLEDGE_DIR,
      root,
    );
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (error) {
    process.stdout.write(
      `${JSON.stringify({ ok: false, error: String(error?.message || error) })}\n`
    );
    process.exit(1);
  }
})();
